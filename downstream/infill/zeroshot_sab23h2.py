"""Historical port. Official-ckpt eval lives in ``downstream/ophiuchus_eval/cdr_sab23h2.py``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import Sab23H2Collator, load_model, run_generate
from dllm.pipelines.bioseq import ophiuchus_ab_checkpoint_path


# Released IgGM/SAb23H2 tree uses h_cdr*; AirGen-Dev original script used cdrh*.
SAB23H2_MODE_DIRS = {
    "cdrh1": "h_cdr1",
    "cdrh2": "h_cdr2",
    "cdrh3": "h_cdr3",
    "cdrl1": "l_cdr1",
    "cdrl2": "l_cdr2",
    "cdrl3": "l_cdr3",
}


class SAb23H2Dataset(Dataset):
    def __init__(self, file_path: str, mode: str):
        candidates = []
        mapped = SAB23H2_MODE_DIRS.get(mode)
        if mapped:
            candidates.append(mapped)
        candidates.append(mode)
        design_root = os.path.join(file_path, "fasta.files.design")
        design_mode = next((name for name in candidates if os.path.isdir(os.path.join(design_root, name))), mode)
        self.design_path = os.path.join(design_root, design_mode)
        self.native_path = os.path.join(file_path, "fasta.files.native")
        with open(os.path.join(file_path, "prot_ids.txt"), encoding="utf-8") as handle:
            self.prot_ids = [line.strip() for line in handle.readlines()]

    def __len__(self) -> int:
        return len(self.prot_ids)

    def __getitem__(self, index: int):
        prot_id = self.prot_ids[index]
        with open(os.path.join(self.design_path, f"{prot_id}.fasta"), encoding="utf-8") as handle:
            lines = handle.readlines()
            heavy_chain = lines[1].strip()
            light_chain = lines[3].strip()
        with open(os.path.join(self.native_path, f"{prot_id}.fasta"), encoding="utf-8") as handle:
            lines = handle.readlines()
            heavy_chain_target = lines[1].strip()
            light_chain_target = lines[3].strip()
        return heavy_chain, light_chain, heavy_chain_target, light_chain_target


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint_path, device=device)
    tokenizer = model.tokenizer
    collator = Sab23H2Collator(tokenizer=tokenizer)
    results = {}

    for mode in ["cdrh3", "cdrh2", "cdrh1", "cdrl3", "cdrl2", "cdrl1"]:
        dataset = SAb23H2Dataset(args.test_set, mode)
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

        mean_aar = float(np.round(np.mean(inner_aars), 4) * 100.0)
        std_aar = float(np.round(np.std(inner_aars), 4) * 100.0)
        print(f"Average AAR for {mode}: {mean_aar}")
        print(f"AAR Standard deviation for {mode}: {std_aar}")
        results[mode] = {
            "average_aar": mean_aar,
            "aar_std": std_aar,
            "n": int(len(inner_aars)),
        }

    metrics = {
        "baseline": "Ophiuchus-Ab",
        "dataset": "SAb23H2",
        "protocol": "official_checkpoint_mask_cdr_span_diffusion_decode_AAR",
        "n_proteins": int(len(SAb23H2Dataset(args.test_set, "cdrh3"))),
        "by_cdr": results,
        "baseline_provenance": {
            "evidence_type": "official_code_rerun",
            "protocol_alignment": "compatible",
            "paper_comparable": True,
            "citation": "doi:10.64898/2026.02.02.703197",
            "source_location": "Ophiuchus-Ab official checkpoint on SAb23H2",
            "notes": "paper Table 1 remains a separate paper_reported row",
        },
    }
    print(json.dumps(metrics, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=str(ophiuchus_ab_checkpoint_path()))
    parser.add_argument("--test-set", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-strategy", type=str, default="argmax")
    parser.add_argument("--max-iter", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="", help="optional metrics JSON path")
    args = parser.parse_args()
    pprint(vars(args))
    evaluate(args)


if __name__ == "__main__":
    main()
