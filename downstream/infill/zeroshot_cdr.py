from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import CdrInfillCollator, load_model, run_generate
from dllm.pipelines.bioseq import Esm2ProteinTokenizer, ophiuchus_ab_checkpoint_path


class SAbDabDataset(Dataset):
    def __init__(self, file_path: str, mode: str, fold: int = 0):
        self.mode = mode
        data = pd.read_json(os.path.join(file_path, mode, f"fold_{fold}", "test.json"), lines=True)
        self.heavy = data["heavy_chain_seq"].astype(str).str.replace("J", "L").tolist()
        self.light = data["light_chain_seq"].astype(str).str.replace("J", "L").tolist()
        self.target = data[f"{mode}_seq"].tolist()
        self.pos = data[f"{mode}_pos"].tolist()

    def __len__(self) -> int:
        return len(self.heavy)

    def __getitem__(self, index: int):
        return self.heavy[index], self.light[index], self.target[index], self.pos[index], self.mode


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint_path, device=device)
    tokenizer = model.tokenizer
    collator = CdrInfillCollator(tokenizer=tokenizer)

    outer_aars = []
    all_aars = []

    for fold in range(10):
        print(f"Evaluating fold {fold}")
        dataset = SAbDabDataset(args.test_set, args.mode, fold=fold)
        loader = DataLoader(dataset, batch_size=64, collate_fn=collator, shuffle=False)
        inner_aars = []

        for chains, chain_ids, labels in loader:
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            labels = labels.to(device)
            output_tokens, _ = run_generate(
                model,
                chains,
                chain_ids,
                tokenizer,
                max_iter=args.max_iter,
                sampling_strategy=args.sampling_strategy,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
            )
            correct = (output_tokens == labels) * chains.eq(tokenizer.mask_token_id)
            aar = correct.sum(1) / chains.eq(tokenizer.mask_token_id).sum(1).clamp_min(1)
            inner_aars.extend(aar.tolist())

        outer_aars.append(float(np.mean(inner_aars) * 100.0))
        all_aars.extend(inner_aars)

    print("Average AAR:", float(np.mean(all_aars) * 100.0))
    print("Average AAR all folds:", float(np.mean(outer_aars)))
    print("AAR Standard deviation across all folds:", float(np.std(outer_aars)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=str(ophiuchus_ab_checkpoint_path()))
    parser.add_argument("--test-set", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-strategy", type=str, default="argmax")
    parser.add_argument("--max-iter", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    args = parser.parse_args()
    pprint(vars(args))
    evaluate(args)


if __name__ == "__main__":
    main()
