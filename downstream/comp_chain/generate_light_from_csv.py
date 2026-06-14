from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import ChainPaddingCollator, load_model, run_generate
from dllm.pipelines.bioseq import Esm2ProteinTokenizer, ophiuchus_ab_checkpoint_path


class PairDataset(Dataset):
    def __init__(self, csv_path: str, heavy_col: str = "h_sequence", light_col: str = "l_sequence"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[~self.df[heavy_col].isna()].copy()
        self.heavy_col = heavy_col
        self.light_col = light_col
        self.heavy_list = self.df[heavy_col].astype(str).str.replace("-", "").tolist()
        if light_col in self.df.columns:
            self.light_list = self.df[light_col].fillna("").astype(str).str.replace("-", "").tolist()
        else:
            self.light_list = [""] * len(self.heavy_list)

    def __len__(self) -> int:
        return len(self.heavy_list)

    def __getitem__(self, idx: int):
        return self.heavy_list[idx], self.light_list[idx]


def save_results(rows: list[dict], output_fpath: str, num_seqs: int) -> str:
    result_df = pd.DataFrame(rows)
    if "." in output_fpath:
        base, ext = output_fpath.rsplit(".", 1)
        output_with_n = f"{base}_n{num_seqs}.{ext}"
    else:
        output_with_n = f"{output_fpath}_n{num_seqs}"
    result_df.to_csv(output_with_n, index=False)
    return output_with_n


def generate_for_single_sequence(
    heavy_str: str,
    raw_light_str: str,
    num_seqs: int,
    model,
    tokenizer: Esm2ProteinTokenizer,
    collator: ChainPaddingCollator,
    device: torch.device,
    temperature: float,
    max_iter: int,
    cfg_scale: float,
    sampling_strategy: str,
    light_prompt_tokens: int,
):
    batch = [(heavy_str, raw_light_str) for _ in range(num_seqs)]
    heavy_sequences, light_sequences = zip(*batch)
    input_ids, chain_ids, meta = collator.stack_chains(
        heavy_sequences,
        light_sequences,
        mask_light_from=light_prompt_tokens,
    )
    input_ids = input_ids.to(device)
    chain_ids = chain_ids.to(device)

    output_tokens, _ = run_generate(
        model,
        input_ids,
        chain_ids,
        tokenizer,
        max_iter=max_iter,
        sampling_strategy=sampling_strategy,
        temperature=temperature,
        cfg_scale=cfg_scale,
    )

    heavy_max_len = meta["heavy_max_len"]
    results = []
    for i in range(num_seqs):
        light_core = output_tokens[i, heavy_max_len + 1 :]
        gen_light = tokenizer.decode(light_core.tolist(), skip_special_tokens=True)
        results.append(
            {
                "h_sequence": heavy_str,
                "gen_l_sequence": gen_light,
                "raw_l_sequence": raw_light_str,
            }
        )
    return results


def run_generation(
    df_path: str,
    checkpoint_path: str,
    output_file: str,
    num_seqs: int = 1,
    heavy_col: str = "h_sequence",
    light_col: str = "l_sequence",
    temperature: float = 1.0,
    sampling_strategy: str = "gumbel_argmax",
    max_iter: int = 124,
    cfg_scale: float = 1.5,
    device: str = "auto",
    seed: int | None = 42,
    light_prompt_tokens: int = 4,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    model = load_model(checkpoint_path, device=device_obj)
    tokenizer = model.tokenizer
    collator = ChainPaddingCollator(tokenizer=tokenizer)
    dataset = PairDataset(df_path, heavy_col=heavy_col, light_col=light_col)

    all_rows = []
    for seq_idx in tqdm(range(len(dataset)), desc="heavy2light"):
        heavy_str, raw_light_str = dataset[seq_idx]
        all_rows.extend(
            generate_for_single_sequence(
                heavy_str=heavy_str,
                raw_light_str=raw_light_str,
                num_seqs=num_seqs,
                model=model,
                tokenizer=tokenizer,
                collator=collator,
                device=device_obj,
                temperature=temperature,
                max_iter=max_iter,
                cfg_scale=cfg_scale,
                sampling_strategy=sampling_strategy,
                light_prompt_tokens=light_prompt_tokens,
            )
        )

    saved_path = save_results(all_rows, output_file, num_seqs)
    print(f"Saved results to {saved_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate light chains from heavy chains (Ophiuchus-Ab).")
    parser.add_argument("--df-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=str(ophiuchus_ab_checkpoint_path()))
    parser.add_argument("--output-file", type=str, default="paired_results.csv")
    parser.add_argument("--heavy-col", type=str, default="h_sequence")
    parser.add_argument("--light-col", type=str, default="l_sequence")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-strategy", type=str, default="gumbel_argmax")
    parser.add_argument("--max-iter", type=int, default=124)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--light-prompt-tokens", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    run_generation(
        df_path=args.df_path,
        checkpoint_path=args.checkpoint_path,
        output_file=args.output_file,
        num_seqs=args.num_seqs,
        heavy_col=args.heavy_col,
        light_col=args.light_col,
        temperature=args.temperature,
        sampling_strategy=args.sampling_strategy,
        max_iter=args.max_iter,
        cfg_scale=args.cfg_scale,
        device=args.device,
        seed=args.seed,
        light_prompt_tokens=args.light_prompt_tokens,
    )


if __name__ == "__main__":
    main()
