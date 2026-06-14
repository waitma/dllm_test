# The get_motif function of this code is highly motivated by EvoDiff:
# https://github.com/microsoft/evodiff

import argparse
import os
import random
from pathlib import Path
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import Dataset
from byprot import utils
from byprot.models.lm.dplm_multichain import MultiChainDiffusionProteinLanguageModel
from byprot.models.lm.cond_dplm import ConditionalDPLM
from byprot.datamodules.dataset.data_utils import PDBDataProcessor
from copy import deepcopy
import esm
from transformers import EsmTokenizer, AutoConfig
import pandas as pd


class SAb23H2Dataset(Dataset):
    def __init__(self, file_path, mode):
        super().__init__()
        
        self.design_path = os.path.join(file_path, 'fasta.files.design', mode)
        self.native_path = os.path.join(file_path, 'fasta.files.native')
        with open(os.path.join(file_path, 'prot_ids.txt'), 'r', encoding='utf-8') as file:
            self.prot_ids = [line.strip() for line in file.readlines()]

    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, index):
        prot_id = self.prot_ids[index]
        with open(os.path.join(self.design_path, f'{prot_id}.fasta'), 'r', encoding='utf-8') as file:
            lines = file.readlines()
            heavy_chain = lines[1].strip()
            light_chain = lines[3].strip()
        
        with open(os.path.join(self.native_path, f'{prot_id}.fasta'), 'r', encoding='utf-8') as file:
            lines = file.readlines()
            heavy_chain_target = lines[1].strip()
            light_chain_target = lines[3].strip()
        
        return heavy_chain, light_chain, heavy_chain_target, light_chain_target
    

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
        heavy_chains, light_chains, heavy_chain_labels, light_chain_labels = zip(*batches)
        chains = [self.convert(heavy_chains, 'fv_heavy'), self.convert(light_chains, 'fv_light')]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]
        labels = [self.convert(heavy_chain_labels, 'fv_heavy'), self.convert(light_chain_labels, 'fv_light')]
        
        chains = torch.cat(chains, -1)
        chain_ids = torch.cat(chain_ids, -1)
        labels = torch.cat(labels, -1)
        
        return chains, chain_ids, labels

    def convert(self, seq_str_list, chain=None):
        batch_size = len(seq_str_list)
        seq_encoded_list = [
            self.alphabet.encode(seq_str.replace("J", "L").replace("X", "<mask>"))
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
    
    
def prepare_data(pdb_path, alphabet, collator, num_seqs, device):
    def _full_mask(target_tokens, coord_mask, alphabet):
        target_mask = (
            target_tokens.ne(alphabet.padding_idx)  # & mask
            & target_tokens.ne(alphabet.cls_idx)
            & target_tokens.ne(alphabet.eos_idx)
        )
        _tokens = target_tokens.masked_fill(target_mask, alphabet.mask_idx)
        _mask = _tokens.eq(alphabet.mask_idx) & coord_mask
        return _tokens, _mask

    pdb_id = Path(pdb_path).stem
    structure = PDBDataProcessor().parse_PDB(pdb_path)
    batch = collator([deepcopy(structure) for idx in range(num_seqs)])
    prev_tokens, prev_token_mask = _full_mask(
        batch["tokens"], batch["coord_mask"], alphabet
    )
    batch["prev_tokens"] = prev_tokens
    batch["prev_token_mask"] = prev_tokens.eq(alphabet.mask_idx)
    batch = utils.recursive_to(batch, device=device)
    return batch, structure["seq"]


def get_intervals(list, single_res_domain=False):
    "Given a list (Tensor) of non-masked residues get new start and end index for motif placed in scaffold"
    if single_res_domain:
        start = [l.item() for l in list]
        stop = start
    else:
        start = []
        stop = []
        for i, item in enumerate(list):
            if i == 0:
                start.append(item.item())
            elif i == (len(list) - 1):
                stop.append(item.item())
            elif i != len(list) and (item + 1) != list[i + 1]:
                stop.append(item.item())
                start.append(list[i + 1].item())
    return start, stop


def generate(args, saveto):
    model = MultiChainDiffusionProteinLanguageModel.from_pretrained(args.checkpoint_path, from_huggingface=False)
    tokenizer = model.tokenizer
    model = model.eval()
    model = model.cuda()
    device = next(model.parameters()).device
    results = {}

    for mode in ['cdrh3', 'cdrh2', 'cdrh1', 'cdrl3', 'cdrl2', 'cdrl1']:
        dataset = SAb23H2Dataset(args.test_set, mode)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, collate_fn=MyCollateFn(), shuffle=False
        )

        inner_aars = []
        for step, eval_batch in enumerate(loader):
            chains, chain_ids, labels = eval_batch
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            labels = labels.to(device)
            batch = {
                "input_ids": chains,
                "chain_ids": chain_ids,
                }
            partial_mask = (
                batch["input_ids"].ne(tokenizer.mask_token_id)
                & batch["input_ids"].ne(tokenizer.cls_token_id)
            )
            # type_as(batch["input_mask"])
            
            # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                    batch=batch,
                    temperature=args.temperature,
                    max_iter=args.max_iter,
                    sampling_strategy=args.sampling_strategy,
                    partial_masks=partial_mask,
                    cfg_scale=args.cfg_scale,
                )
            output_tokens = outputs[0]
            
            correct = (output_tokens == labels) * chains.eq(tokenizer.mask_token_id)
            aar = correct.sum(1) / chains.eq(tokenizer.mask_token_id).sum(1)
            inner_aars += aar.tolist()
        
        # Output the performance
        print(f"Average AAR for {mode}:", np.round(np.mean(inner_aars), 4) * 100.)
        print(f"AAR Standard deviation for {mode}:", np.round(np.std(inner_aars), 4) * 100.)
        print("\n")
        
        results[mode] = np.round(np.mean(inner_aars), 4) * 100.
    
    print(results)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--test_set", type=str, required=True)
    parser.add_argument("--num_seqs", type=int, default=40)
    parser.add_argument("--saveto", type=str, default="gen.fasta")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling_strategy", type=str, default="argmax")
    parser.add_argument("--max_iter", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    args = parser.parse_args()
    pprint(args)

    generate(args, args.saveto)


if __name__ == "__main__":
    main()
