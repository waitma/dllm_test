"""Smoke-test downstream decoding configs on GPU (light pairing + humanization)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import ChainPaddingCollator, load_model, run_generate
from downstream.comp_chain.generate_light_from_csv import generate_for_batch
from downstream.humanization.humanize import HumanizationCollator, HumanizationDataset
from dllm.pipelines.bioseq import ophiuchus_ab_checkpoint_path


def sequence_identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n


def diversity_score(sequences: list[str]) -> float:
    if len(sequences) <= 1:
        return 0.0
    uniq = len(set(sequences))
    return (uniq - 1) / (len(sequences) - 1)


def test_light_pairing(model, tokenizer, collator, device, df: pd.DataFrame, configs: list[dict]) -> list[dict]:
    rows = []
    for cfg in configs:
        cfg_rows = []
        for _, row in df.iterrows():
            heavy = str(row["h_sequence"]).replace("-", "")
            light = str(row["l_sequence"]).replace("-", "")
            samples = [(heavy, light, {"row_idx": int(row.name)})]
            torch.manual_seed(cfg.get("seed", 42))
            np.random.seed(cfg.get("seed", 42))
            gen_rows = generate_for_batch(
                samples=samples,
                num_seqs=cfg["num_seqs"],
                model=model,
                tokenizer=tokenizer,
                collator=collator,
                device=device,
                temperature=1.0,
                max_iter=cfg["max_iter"],
                cfg_scale=cfg["cfg_scale"],
                sampling_strategy=cfg["sampling_strategy"],
                light_prompt_tokens=cfg["light_prompt_tokens"],
            )
            gen_seqs = [item["gen_l_sequence"] for item in gen_rows]
            cfg_rows.append(
                {
                    "row_idx": int(row.name),
                    "mean_identity": float(np.mean([sequence_identity(g, light) for g in gen_seqs])),
                    "diversity": diversity_score(gen_seqs),
                    "gen_example": gen_seqs[0][:60],
                }
            )
        rows.append(
            {
                "task": "light_pairing",
                "config": cfg,
                "n_pairs": len(cfg_rows),
                "mean_identity": float(np.mean([r["mean_identity"] for r in cfg_rows])),
                "mean_diversity": float(np.mean([r["diversity"] for r in cfg_rows])),
                "examples": cfg_rows[:3],
            }
        )
    return rows


def test_humanization(model, tokenizer, device, pdb_dir: str, chain_csv: str, configs: list[dict]) -> list[dict]:
    dataset = HumanizationDataset(pdb_dir, chain_csv, start_index=0, end_index=3)
    collator = HumanizationCollator(tokenizer=tokenizer)
    rows = []
    for cfg in configs:
        heavy_aars = []
        light_aars = []
        examples = []
        for sample in dataset:
            heavy_masked, light_masked, heavy_seq, light_seq, pdb_id = sample
            batch_data = [sample[:4] for _ in range(cfg["n_sequences"])]
            torch.manual_seed(cfg.get("seed", 42))
            chains, chain_ids, labels, heavy_len = collator(batch_data)
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            labels = labels.to(device)
            output_tokens, _ = run_generate(
                model,
                chains,
                chain_ids,
                tokenizer,
                max_iter=cfg["max_iter"],
                sampling_strategy=cfg["sampling_strategy"],
                temperature=1.0,
                cfg_scale=cfg["cfg_scale"],
            )
            correct = (output_tokens == labels) * chains.eq(tokenizer.mask_token_id)
            heavy_mask = chains[:, :heavy_len].eq(tokenizer.mask_token_id)
            light_mask = chains[:, heavy_len:].eq(tokenizer.mask_token_id)
            heavy_aar = (correct[:, :heavy_len] & heavy_mask).sum(1) / heavy_mask.sum(1).clamp_min(1)
            light_aar = (correct[:, heavy_len:] & light_mask).sum(1) / light_mask.sum(1).clamp_min(1)
            heavy_aars.extend(heavy_aar.detach().cpu().tolist())
            light_aars.extend(light_aar.detach().cpu().tolist())
            heavy_out = tokenizer.decode(output_tokens[0, 1 : heavy_len - 1].tolist(), skip_special_tokens=True)
            light_out = tokenizer.decode(output_tokens[0, heavy_len + 1 : -1].tolist(), skip_special_tokens=True)
            gen_variants = []
            for i in range(cfg["n_sequences"]):
                h = tokenizer.decode(output_tokens[i, 1 : heavy_len - 1].tolist(), skip_special_tokens=True)
                l = tokenizer.decode(output_tokens[i, heavy_len + 1 : -1].tolist(), skip_special_tokens=True)
                gen_variants.append((h, l))
            examples.append(
                {
                    "pdb_id": pdb_id,
                    "heavy_fr_aar": float(heavy_aar.mean().item()),
                    "light_fr_aar": float(light_aar.mean().item()),
                    "variant_diversity_heavy": diversity_score([x[0] for x in gen_variants]),
                    "variant_diversity_light": diversity_score([x[1] for x in gen_variants]),
                    "generated_heavy_head": heavy_out[:60],
                    "generated_light_head": light_out[:60],
                }
            )
        rows.append(
            {
                "task": "humanization",
                "config": cfg,
                "n_structures": len(examples),
                "heavy_fr_aar_percent": float(np.mean(heavy_aars) * 100),
                "light_fr_aar_percent": float(np.mean(light_aars) * 100),
                "examples": examples,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=str(ophiuchus_ab_checkpoint_path()))
    parser.add_argument("--holdout-csv", type=str, default=str(PROJECT_ROOT / "data/downstream/comp_chain/test_data_oas_holdout.csv"))
    parser.add_argument("--humanization-pdb-dir", type=str, default=str(PROJECT_ROOT / "data/downstream/humanization/humanisation/test-pdb"))
    parser.add_argument("--humanization-chain-csv", type=str, default=str(PROJECT_ROOT / "data/downstream/humanization/humanisation/test_chains.csv"))
    parser.add_argument("--n-light-pairs", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model = load_model(args.checkpoint_path, device=device)
    tokenizer = model.tokenizer
    collator = ChainPaddingCollator(tokenizer=tokenizer)

    holdout = pd.read_csv(args.holdout_csv).head(args.n_light_pairs)
    old_cfg = {
        "name": "old_cdr_style",
        "sampling_strategy": "argmax",
        "max_iter": 4,
        "cfg_scale": 0.0,
        "light_prompt_tokens": 3,
        "num_seqs": 8,
        "seed": 42,
    }
    new_cfg = {
        "name": "airgen_default",
        "sampling_strategy": "gumbel_argmax",
        "max_iter": 32,
        "cfg_scale": 0.0,
        "light_prompt_tokens": 3,
        "num_seqs": 8,
        "seed": 42,
    }

    results = {
        "checkpoint": args.checkpoint_path,
        "device": str(device),
        "light_pairing": test_light_pairing(model, tokenizer, collator, device, holdout, [old_cfg, new_cfg]),
        "humanization": test_humanization(
            model,
            tokenizer,
            device,
            args.humanization_pdb_dir,
            args.humanization_chain_csv,
            [
                {**old_cfg, "n_sequences": old_cfg["num_seqs"]},
                {**new_cfg, "n_sequences": new_cfg["num_seqs"]},
            ],
        ),
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
