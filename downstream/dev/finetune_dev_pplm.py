import os
import argparse
import json
import math
import random
import re
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.stats
import torch
import wandb
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit, cross_val_predict
from sklearn.metrics import make_scorer
from sklearn.preprocessing import PowerTransformer
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import EsmTokenizer, AutoConfig
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

# import mint
from byprot.models.lm.modules.mint.model.esm2 import ESM2
from byprot.models.lm.modules.pplm import PPLM
import csv

warnings.filterwarnings("ignore")

class DevDataset(Dataset):
    def __init__(self, csv_path, target_col):
        super().__init__()

        df = pd.read_csv(csv_path)
        # Filter to samples with non-null values
        not_na_mask = df[target_col].notna()
        self.data = df[not_na_mask]

        self.heavy = self.data["vh_protein_sequence"].str.replace("J", "L").tolist()
        self.light = self.data["vl_protein_sequence"].str.replace("J", "L").tolist()
        self.target = self.data[target_col].tolist()
        self.fold = self.data['hierarchical_cluster_IgG_isotype_stratified_fold'].tolist()

    def __len__(self):
        return len(self.heavy)

    def __getitem__(self, index):
        return self.heavy[index], self.light[index], self.target[index], self.fold[index]


class DesautelsCollateFn:
    def __init__(self, tokenizer_path=None, truncation_seq_length=None):
        # by default we use the EsmTokenizer and the esm vocab. 
        # if you want to use the different vocab, 
        # please set the vocab path to the tokenizer_path
        if tokenizer_path is None:
            self.alphabet = EsmTokenizer.from_pretrained('/vepfs-mlp2/mlp-public/zhuyiheng/hub/checkpoints/esm2_t33_650M_UR50D')
        else:
            self.alphabet = EsmTokenizer.from_pretrained(tokenizer_path)
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, batches):
        heavy_chain, light_chain, labels, folds = zip(*batches)
        chains = [self.alphabet.batch_encode_plus(c,
                                                  add_special_tokens=True,
                                                  padding="longest",
                                                  return_tensors='pt')['input_ids']
                  for c in [heavy_chain, light_chain]]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]
        chains = torch.cat(chains, -1)
        chain_ids = torch.cat(chain_ids, -1)
        labels = torch.from_numpy(np.stack(labels, 0))
        folds = torch.from_numpy(np.stack(folds, 0))
        return chains, chain_ids, labels, folds


class MyCollateFn:
    def __init__(self, tokenizer_path=None, truncation_seq_length=None):
        # by default we use the EsmTokenizer and the esm vocab. 
        # if you want to use the different vocab, 
        # please set the vocab path to the tokenizer_path
        if tokenizer_path is None:
            self.alphabet = EsmTokenizer.from_pretrained('/vepfs-mlp2/mlp-public/zhuyiheng/hub/checkpoints/esm2_t33_650M_UR50D')
        else:
            self.alphabet = EsmTokenizer.from_pretrained(tokenizer_path)
        self.truncation_seq_length = truncation_seq_length
        self.chain_lengths = {
            'fv_heavy': 150,
            'fv_light': 128
        }

    def __call__(self, batches):
        heavy_chains, light_chains, labels, folds = zip(*batches)
        chains = [self.convert(heavy_chains, 'fv_heavy'), self.convert(light_chains, 'fv_light')]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]
        inter_chain_mask = torch.ones((self.chain_lengths['fv_heavy'] + self.chain_lengths['fv_light'], 
                                       self.chain_lengths['fv_heavy'] + self.chain_lengths['fv_light']))
        inter_chain_mask[:self.chain_lengths['fv_heavy'], :self.chain_lengths['fv_heavy']] = 0
        inter_chain_mask[self.chain_lengths['fv_heavy']:, self.chain_lengths['fv_heavy']:] = 0
        
        chains = torch.cat(chains, -1)
        chain_ids = torch.cat(chain_ids, -1)
        labels = torch.from_numpy(np.stack(labels, 0))
        folds = torch.from_numpy(np.stack(folds, 0))
        return chains, chain_ids, inter_chain_mask, labels, folds

    def convert(self, seq_str_list, chain=None):
        batch_size = len(seq_str_list)
        seq_encoded_list = [
            self.alphabet.encode(seq_str.replace("J", "L"))
            for seq_str in seq_str_list
        ]
        if self.truncation_seq_length:
            for i in range(batch_size):
                seq = seq_encoded_list[i]
                if len(seq) > self.truncation_seq_length:
                    start = random.randint(0, len(seq) - self.truncation_seq_length + 1)
                    seq_encoded_list[i] = seq[start : start + self.truncation_seq_length]
        if chain:
            max_len = self.chain_lengths[chain]
        else:
            max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        if self.truncation_seq_length:
            assert max_len <= self.truncation_seq_length
        tokens = torch.empty((batch_size, max_len), dtype=torch.int64)
        tokens.fill_(self.alphabet.eos_token_id)

        for i, seq_encoded in enumerate(seq_encoded_list):
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, : len(seq_encoded)] = seq
        return tokens
    
    
def upgrade_state_dict(state_dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


class FlabWrapper(nn.Module):
    def __init__(
        self, cfg, checkpoint_path, freeze_percent=0.0, use_multimer=True, sep_chains=False, device="cuda:0"
    ):
        super().__init__()
        self.cfg = cfg
        self.sep_chains = sep_chains
        self.model = PPLM(
            num_layers=cfg.num_hidden_layers,
            embed_dim=cfg.hidden_size,
            attention_heads=cfg.num_attention_heads,
            token_dropout=cfg.token_dropout,
        )
        
        if os.path.isdir(checkpoint_path): # load from zero checkpoint
            checkpoint = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
            if use_multimer:
                # remove 'model.net.model.' in keys
                new_checkpoint = OrderedDict(
                    (key.replace("model.net.model.", ""), value)
                    for key, value in checkpoint.items()
                )
                self.model.load_state_dict(new_checkpoint)
            self.model.to(device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
            if use_multimer:
                if os.path.basename(checkpoint_path) == 'mint.ckpt':
                    # MINT: remove 'model.' in keys 
                    new_checkpoint = OrderedDict(
                        (key.replace("model.", ""), value)
                        for key, value in checkpoint["state_dict"].items()
                    )
                else:
                    new_checkpoint = OrderedDict(
                        (key.replace("model.net.model.", ""), value)
                        for key, value in checkpoint["state_dict"].items()
                    )
                self.model.load_state_dict(new_checkpoint)
            else:
                new_checkpoint = upgrade_state_dict(checkpoint["model"])
                self.model.load_state_dict(new_checkpoint)
                
        total_layers = 33
        for name, param in self.model.named_parameters():
            if "embed_tokens.weight" in name or "_norm_after" in name or "lm_head" in name or "contact_head" in name:
                param.requires_grad = False
            else:
                layer_num = name.split(".")[1]
                if int(layer_num) <= math.floor(total_layers * freeze_percent):
                    param.requires_grad = False

    def get_one_chain(self, chain_out, mask_expanded, mask):
        masked_chain_out = chain_out * mask_expanded
        sum_masked = masked_chain_out.sum(dim=1)
        mask_counts = mask.sum(dim=1, keepdim=True).float()  # Convert to float for division
        mean_chain_out = sum_masked / mask_counts
        return mean_chain_out
    
    def forward(self, chains, chain_ids, inter_chain_mask):
        mask = (
            (~chains.eq(self.model.cls_idx))
            & (~chains.eq(self.model.eos_idx))
            & (~chains.eq(self.model.padding_idx))
        )
        
        # chain_out = self.model(chains, chain_ids, repr_layers=[33])["representations"][33]
        chain_out = self.model(chains, inter_chain_mask, repr_layers=[33])["representations"][33]
        
        if self.sep_chains:
            max_chain_id = chain_ids.max().item()
            mean_chain_outs = []
    
            for chain_id in range(max_chain_id + 1):
                chain_mask = (chain_ids == chain_id) & mask
                chain_mask_exp = chain_mask.unsqueeze(-1).expand_as(chain_out)
                mean_chain_out = self.get_one_chain(chain_out, chain_mask_exp, chain_mask)
                mean_chain_outs.append(mean_chain_out)
    
            # Concatenate outputs from each chain along the last dimension
            return torch.cat(mean_chain_outs, dim=-1)
        else:
            mask_expanded = mask.unsqueeze(-1).expand_as(chain_out)
            masked_chain_out = chain_out * mask_expanded
            sum_masked = masked_chain_out.sum(dim=1)
            mask_counts = mask.sum(dim=1, keepdim=True).float()  # Convert to float for division
            mean_chain_out = sum_masked / mask_counts
            return mean_chain_out


@torch.no_grad()
def get_embeddings(model, loader, device="cuda"):

    model.to(device)

    embeddings = []
    targets = []
    folds = []
    
    for step, eval_batch in enumerate(loader):

        chains, chain_ids, inter_chain_mask, target, fold = eval_batch
        chains = chains.to(device)
        chain_ids = chain_ids.to(device)
        inter_chain_mask = inter_chain_mask.to(device)
        target = target.to(device).float()
        fold = fold.to(device)

        embedding = model(chains, chain_ids, inter_chain_mask)

        embeddings.append(embedding.detach().cpu().numpy())
        targets.append(target.cpu().numpy())
        folds.append(fold.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    targets = np.concatenate(targets)
    folds = np.concatenate(folds)

    return embeddings, targets, folds


def gaussian_transform(y):
    y = PowerTransformer().fit_transform(y.reshape(-1, 1))
    return y


def spearman_corr(y_true, y_pred):
    return scipy.stats.spearmanr(y_true, y_pred).correlation


def cross_validate(embeddings, targets, folds, scale_all=True):
    lambda_grid = np.logspace(
        3, -6, num=10
    ).tolist()  # creates [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    lambda_grid.append(0)  # Append 0 to the list of lambdas
    param_grid = {"alpha": lambda_grid}

    cv = PredefinedSplit(folds)
    spearman_scorer = make_scorer(spearman_corr, greater_is_better=True)
    
    X_scaled = embeddings

    if scale_all:
        targets = gaussian_transform(targets)

    # Initialize the Ridge Regression model
    ridge_model = Ridge()

    # 5-fold cross-validation
    clf = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=cv, scoring=spearman_scorer)
    clf.fit(X_scaled, targets)
    best_model = clf.best_estimator_
    # predictions = cross_val_predict(
    #     best_model, # 传入最佳模型
    #     X_scaled, targets,
    #     cv=cv, # 关键！使用GridSearchCV内部的拆分器，确保拆分方式完全一致
    #     method='predict' # 获取类别预测，也可以用 'predict_proba' 或 'decision_function'
    # )

    # Output the performance
    print("Average Spearman correlation across all folds:", round(clf.best_score_, 3))

    return best_model


def calculate_scores(model, dataset_file, holdout_file, property, device):
    dataset = DevDataset(dataset_file, property)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, collate_fn=MyCollateFn(), shuffle=False
    )
    embeddings, targets, folds = get_embeddings(model, loader, device=device)
    targets = np.array(dataset.target)
    
    print(f"Evaluating for property {property} with {len(dataset)} samples.")
    best_model = cross_validate(embeddings, targets, folds, scale_all=True)
    
    # test_dataset = DevDataset(holdout_file, property)
    # test_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=64, collate_fn=MyCollateFn(), shuffle=False
    # )
    # test_embeddings, test_targets, folds = get_embeddings(model, loader, device=device)
    # targets = np.array(dataset.target)


def main(args):
    dataset_file = "downstream/dev/GDPa1_v1.2_20250814.csv"
    heldout_file = 'downstream/dev/heldout-set-sequences.csv'
    valid_property_names = ['AC-SINS_pH7.4', 'PR_CHO', 'HIC', 'Tm2', 'Titer']

    cfg = AutoConfig.from_pretrained('/vepfs-mlp2/mlp-public/zhuyiheng/hub/checkpoints/esm2_t33_650M_UR50D')
    model = FlabWrapper(cfg, args.checkpoint_path, 1.0, args.use_multimer, args.sep_chains, args.device)

    for property in valid_property_names:
        calculate_scores(model, dataset_file, heldout_file, property, args.device)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuning on Flab dataset")

    parser.add_argument(
        "--checkpoint_path",
        type=str
    )
    parser.add_argument(
        "--result_path",
        type=str
    )
    parser.add_argument("--use_multimer", action="store_true", default=False)
    parser.add_argument("--sep_chains", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    main(args)
