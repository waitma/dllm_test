import os
import math
import argparse
from datetime import date
import random
import re
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
)

import sys
# MINT 相关
import json
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import EsmTokenizer, AutoConfig
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from tqdm import tqdm
from datasets import load_dataset, ClassLabel
from transformers import Trainer, TrainingArguments 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from byprot.models.lm.modules.mint.model.esm2 import ESM2


# 一个简单的三分类头：2560 -> 3
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=2560, hidden_dim=512, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class EsmClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_dim=2560, num_classes=3):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
    
class PairClsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, h_col="h_sequence", l_col="l_sequence", y_col="label"):
        df = pd.read_csv(csv_path)
        self.h = df[h_col].astype(str).tolist()
        self.l = df[l_col].astype(str).tolist()
        self.y = df[y_col].astype(int).tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.h[idx], self.l[idx], self.y[idx]


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
        heavy_chains, light_chains, labels = zip(*batches)
        chains = [self.convert(heavy_chains, 'fv_heavy'), self.convert(light_chains, 'fv_light')]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]
        inter_chain_mask = torch.ones((self.chain_lengths['fv_heavy'] + self.chain_lengths['fv_light'], 
                                       self.chain_lengths['fv_heavy'] + self.chain_lengths['fv_light']))
        inter_chain_mask[:self.chain_lengths['fv_heavy'], :self.chain_lengths['fv_heavy']] = 0
        inter_chain_mask[self.chain_lengths['fv_heavy']:, self.chain_lengths['fv_heavy']:] = 0
        
        chains = torch.cat(chains, -1)
        chain_ids = torch.cat(chain_ids, -1)
        labels = torch.from_numpy(np.stack(labels, 0))
        return chains, chain_ids, inter_chain_mask, labels

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
        # tokens.fill_(self.alphabet.pad_token_id) # MINT

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
        self.model = ESM2(
            num_layers=cfg.num_hidden_layers,
            embed_dim=cfg.hidden_size,
            attention_heads=cfg.num_attention_heads,
            token_dropout=cfg.token_dropout,
            use_multimer=use_multimer,
        )
        
        # self.model = PPLM(
        #     num_layers=cfg.num_hidden_layers,
        #     embed_dim=cfg.hidden_size,
        #     attention_heads=cfg.num_attention_heads,
        #     token_dropout=cfg.token_dropout,
        # )
        
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
                
        total_layers = self.cfg.num_hidden_layers
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
        
        chain_out = self.model(chains, chain_ids, repr_layers=[self.cfg.num_hidden_layers])["representations"][self.cfg.num_hidden_layers]
        # chain_out = self.model(chains, inter_chain_mask, repr_layers=[self.cfg.num_hidden_layers])["representations"][self.cfg.num_hidden_layers]
        
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
        
         
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "test_accuracy": acc,
        "test_macro-precision": macro_p,
        "test_micro-precision": micro_p,
        "test_macro-recall": macro_r,
        "test_micro-recall": micro_r,
        "test_macro-f1": macro_f1,
        "test_micro-f1": micro_f1,
        "test_mcc": mcc,
    }


def def_training_args(run_name, batch_size=8, lr=5e-5):
    # 保持你原来的参数设置
    return TrainingArguments(
        run_name=run_name,
        seed=42,
        fp16=True,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=5,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=250,
        per_device_eval_batch_size=batch_size,
        logging_steps=50,
        save_strategy="no",
        output_dir=f"./checkpoints/{run_name}",
        report_to="wandb",
        logging_dir=f"./logs/{run_name}",
        logging_first_step=True,
        # 必须把 remove_unused_columns 设为 False，否则 Trainer 会把 chain_ids 过滤掉
        remove_unused_columns=False 
    )

def get_embeddings(csv_path, wrapper, device, batch_size=8, crop_len=512):
    ds = PairClsDataset(csv_path)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=MyCollateFn(truncation_seq_length=crop_len),
        shuffle=False,
    )
    wrapper = wrapper.to(device)
    wrapper.eval()

    all_embs, all_labels = [], []
    for step, eval_batch in enumerate(loader):
        chains, chain_ids, inter_chain_mask, labels = eval_batch
        chains = chains.to(device, non_blocking=True)
        chain_ids = chain_ids.to(device, non_blocking=True)
        inter_chain_mask = inter_chain_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            embs = wrapper(chains, chain_ids, inter_chain_mask)  # (B, D)

        all_embs.append(embs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_embs = torch.cat(all_embs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_embs, all_labels


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 准备 Labels
    class_labels = ClassLabel(names=["Healthy-donor", "Flu-specific", "Sars-specific"])
    n_classes = len(class_labels.names)

    results = pd.DataFrame({
        "epoch": [],
        "itr": [],
        "test_accuracy": [],
        "test_macro-precision": [],
        "test_micro-precision": [],
        "test_macro-recall": [],
        "test_micro-recall": [],
        "test_macro-f1": [],
        "test_micro-f1": [],
        "test_mcc": [],
    })

    cfg = AutoConfig.from_pretrained('/vepfs-mlp2/mlp-public/zhuyiheng/hub/checkpoints/esm2_t33_650M_UR50D')
    model = FlabWrapper(cfg, args.checkpoint_path, 1.0, args.use_multimer, args.sep_chains, args.device)
    
    for fold in range(5):
        # run_name = f"{args.model_name}_MINT_HD-Flu-CoV-paired_itr{fold}_{date.today().isoformat()}"
        # wandb.init(project="mxd-data", name=run_name, job_type=args.model_name)

        print(f"=== Fold {fold} ===")
        
        train_csv = f"downstream/specificity/TTE/hd-0_flu-1_cov-2_train{fold}.csv"
        test_csv = f"downstream/specificity/TTE/hd-0_flu-1_cov-2_test{fold}.csv"

        # 1) 预先算好 MINT embedding（这样训练分类头很快）
        train_embs, train_labels = get_embeddings(
            train_csv, model, device, batch_size=64, crop_len=512
        )
        test_embs, test_labels = get_embeddings(
            test_csv, model, device, batch_size=64, crop_len=512
        )

        # 2) 三分类头
        # clf = MLPClassifier(in_dim=train_embs.shape[1], hidden_dim=512, num_classes=3)
        clf = EsmClassificationHead(hidden_dim=train_embs.shape[-1], num_classes=3)
        clf.to(device)

        # === 超参数：照论文 3-class 设定 ===
        lr = args.lr # default 5e-5
        num_epochs = args.epoch # default 5
        batch_size = args.bs # default 8

        train_dataset = TensorDataset(train_embs, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(clf.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # 计算总步数 & warmup 步数，做 linear scheduler
        num_training_steps = num_epochs * (len(train_dataset) // batch_size + 1)
        num_warmup_steps = int(0.1 * num_training_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # 3) 训练分类头
        global_step = 0
        for epoch in range(1, num_epochs + 1):
            clf.train()
            total_loss = 0.0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits = clf(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item() * x.size(0)
                global_step += 1
                
                # if global_step % 250 == 0:
                #     print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_dataset)
            print(f"Epoch {epoch}: Train loss = {avg_loss:.4f}")
            
            if epoch % 5 == 0:
                clf.eval()
                with torch.no_grad():
                    logits = clf(test_embs.to(device))
                    test_loss = criterion(logits, test_labels.to(device)).item()
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    preds = np.argmax(probs, axis=-1)
                metrics = compute_metrics(test_labels.numpy(), preds)
                metrics["test_loss"] = test_loss
                metrics["itr"] = fold
                metrics["epoch"] = epoch
                print(f"Test metrics: {metrics}")
                results.loc[len(results)] = metrics

        # 4) 测试集评估
        # clf.eval()
        # with torch.no_grad():
        #     logits = clf(test_embs.to(device))
        #     test_loss = criterion(logits, test_labels.to(device)).item()
        #     probs = torch.softmax(logits, dim=-1).cpu().numpy()
        #     preds = np.argmax(probs, axis=-1)

        # metrics = compute_metrics(test_labels.numpy(), preds)
        # metrics["test_loss"] = test_loss
        # metrics["itr"] = fold
        # metrics["epoch"] = epoch
        # print(f"Test metrics: {metrics}")
        # results.loc[len(results)] = metrics

    print("=== Final Results ===")
    print(results['test_accuracy'].mean(), results['test_accuracy'].std())
    print(results['test_macro-f1'].mean(), results['test_macro-f1'].std())
    print(results['test_mcc'].mean(), results['test_mcc'].std())

    # 保存最终结果
    res_path = f"downstream/specificity/HD_Flu_Cov-paired_results_{args.lr}_{args.epoch}_wd_0.csv"
    results.to_csv(res_path, index=False)
    print(f"Done! Results saved to {res_path}")

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
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=5)
    
    args = parser.parse_args()
    main(args)
