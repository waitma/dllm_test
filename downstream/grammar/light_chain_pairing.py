"""Light-chain pairing eval adapter for grammar-v1 BioSeq models.

Generate + evaluate (ImmunoMatch, diversity, ANARCI chain/V/J metrics):
  conda activate protenix_abtcr
  python -m downstream.grammar.light_chain_pairing \\
    --csv-path /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/comp_chain/test_data_oas_holdout.csv \\
    --checkpoint-path /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v1_esmc300m/latest.pt \\
    --output-csv /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/grammar_v1_esmc300m_light_pairing.csv \\
    --device cuda --num-seqs 8 --light-prompt-tokens 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DEFAULT_HOLDOUT_CSV = PROJECT_ROOT / "data" / "downstream" / "comp_chain" / "test_data_oas_holdout.csv"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.grammar.common import (
    antibody_pair_record,
    build_grammar_collator,
    build_grammar_tokenizer,
    collate_records,
    load_grammar_checkpoint,
    load_sample_oas_record,
    load_untrained_no_encoder,
    run_grammar_generate,
)
from downstream.grammar.masks import light_chain_generation_partial_mask
from downstream.grammar.metrics import extract_chain_sequence


class HeavyLightCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        heavy_col: str = "h_sequence",
        light_col: str = "l_sequence",
        start_index: int = 0,
        end_index: int | None = None,
    ):
        frame = pd.read_csv(csv_path)
        frame = frame[~frame[heavy_col].isna()].copy()
        frame["_input_row_idx"] = frame.index
        frame = frame.iloc[start_index:end_index].copy()
        self.heavy_col = heavy_col
        self.light_col = light_col
        self.heavy = frame[heavy_col].astype(str).str.replace("-", "").tolist()
        self.light = frame.get(light_col, "").fillna("").astype(str).str.replace("-", "").tolist()
        self.metadata = frame.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.heavy)

    def __getitem__(self, index: int):
        return self.heavy[index], self.light[index], self.metadata[index]


def save_generation_csv(rows: list[dict], output_path: Path, num_seqs: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix:
        base, ext = output_path.stem, output_path.suffix
        saved_path = output_path.with_name(f"{base}_n{num_seqs}{ext}")
    else:
        saved_path = Path(f"{output_path}_n{num_seqs}.csv")
    pd.DataFrame(rows).to_csv(saved_path, index=False)
    return saved_path


def generate_for_batch(
    samples: list[tuple[str, str, dict]],
    *,
    model,
    collator,
    tokenizer,
    device: torch.device,
    num_seqs: int,
    max_iter: int,
    sampling_strategy: str,
    temperature: float,
    light_prompt_tokens: int,
) -> list[dict]:
    records = []
    metadata_rows: list[tuple[str, str, int, dict]] = []
    for heavy, light, metadata in samples:
        for variant_idx in range(num_seqs):
            records.append(antibody_pair_record(heavy, light))
            metadata_rows.append((heavy, light, variant_idx, metadata))

    batch = collator(records)
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    partial_mask = light_chain_generation_partial_mask(
        batch,
        tokenizer,
        prompt_residues=light_prompt_tokens,
    )
    output_tokens, _ = run_grammar_generate(
        model,
        batch,
        partial_mask=partial_mask,
        max_iter=max_iter,
        sampling_strategy=sampling_strategy,
        temperature=temperature,
    )

    rows: list[dict] = []
    for row_idx, (heavy, light, variant_idx, metadata) in enumerate(metadata_rows):
        generated_light = extract_chain_sequence(
            output_tokens[row_idx],
            batch["attention_mask"][row_idx],
            batch["residue_mask"][row_idx],
            tokenizer,
            chain="light",
        )
        result = {
            "h_sequence": heavy,
            "gen_l_sequence": generated_light,
            "raw_l_sequence": light,
            "variant_idx": variant_idx,
        }
        for key, value in metadata.items():
            if key not in result:
                result[key] = value
        rows.append(result)
    return rows


def run_comp_chain_eval(generated_csv: Path, num_seqs: int) -> dict:
    eval_dir = PROJECT_ROOT / "downstream" / "comp_chain" / "eval_scripts"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from generation_eval import main as generation_eval_main

    return generation_eval_main(
        generate_results_file=str(generated_csv),
        gen_col_name="gen_l_sequence",
        ref_col_name="raw_l_sequence",
        detailed_comparison=True,
        heavy_col_name="h_sequence",
        expected_count=num_seqs,
    )


def run_generation(args) -> Path:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.checkpoint_path:
        model, tokenizer = load_grammar_checkpoint(args.checkpoint_path, device=device)
    else:
        tokenizer = build_grammar_tokenizer()
        model = load_untrained_no_encoder(tokenizer.vocab_size, mask_token_id=tokenizer.mask_token_id).to(device)

    collator = build_grammar_collator(tokenizer)
    end_index = None if args.max_samples is None else args.start_index + args.max_samples
    dataset = HeavyLightCsvDataset(
        args.csv_path,
        heavy_col=args.heavy_col,
        light_col=args.light_col,
        start_index=args.start_index,
        end_index=end_index,
    )
    heavy_batch_size = max(1, args.heavy_batch_size)
    all_rows: list[dict] = []
    for start_idx in tqdm(range(0, len(dataset), heavy_batch_size), desc="grammar-heavy2light"):
        batch_samples = [
            dataset[seq_idx]
            for seq_idx in range(start_idx, min(start_idx + heavy_batch_size, len(dataset)))
        ]
        all_rows.extend(
            generate_for_batch(
                batch_samples,
                model=model,
                collator=collator,
                tokenizer=tokenizer,
                device=device,
                num_seqs=args.num_seqs,
                max_iter=args.max_iter,
                sampling_strategy=args.sampling_strategy,
                temperature=args.temperature,
                light_prompt_tokens=args.light_prompt_tokens,
            )
        )

    output_path = Path(args.output_csv)
    saved_path = save_generation_csv(all_rows, output_path, args.num_seqs)
    print(f"Saved generation CSV: {saved_path} ({len(all_rows)} rows)")
    return saved_path


def smoke_eval(device: str = "cpu") -> dict[str, float | list[str]]:
    tokenizer = build_grammar_tokenizer()
    record = load_sample_oas_record(split="valid", index=0)
    batch = collate_records([record])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    model = load_untrained_no_encoder(tokenizer.vocab_size, mask_token_id=tokenizer.mask_token_id).to(device)
    partial_mask = light_chain_generation_partial_mask(batch, tokenizer, prompt_residues=3)
    output_tokens, _ = run_grammar_generate(
        model,
        batch,
        partial_mask=partial_mask,
        max_iter=8,
        sampling_strategy="argmax",
        temperature=1.0,
    )
    generated_light = extract_chain_sequence(
        output_tokens[0],
        batch["attention_mask"][0],
        batch["residue_mask"][0],
        tokenizer,
        chain="light",
    )
    return {"generated_light_sequences": [generated_light]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grammar light-chain pairing generate + comp_chain eval.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_HOLDOUT_CSV))
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--heavy-col", type=str, default="h_sequence")
    parser.add_argument("--light-col", type=str, default="l_sequence")
    parser.add_argument("--heavy-batch-size", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=32)
    parser.add_argument("--sampling-strategy", type=str, default="gumbel_argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--light-prompt-tokens", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-json", type=str, default="")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        result = smoke_eval(device="cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
        print(f"sample light={result['generated_light_sequences'][0][:80]}")
        return

    if not args.output_csv:
        args.output_csv = str(
            PROJECT_ROOT / "output" / "downstream_generation" / "grammar_v1_esmc300m_light_pairing.csv"
        )

    saved_csv = run_generation(args)
    if args.skip_eval:
        return

    metrics = run_comp_chain_eval(saved_csv, args.num_seqs)
    metrics_path = Path(args.metrics_json) if args.metrics_json else saved_csv.with_name(saved_csv.stem + "_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved comp_chain metrics: {metrics_path}")


if __name__ == "__main__":
    main()
