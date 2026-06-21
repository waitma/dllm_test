"""Light-chain pairing eval adapter for grammar-v1 BioSeq models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.grammar.common import (
    antibody_pair_record,
    build_grammar_collator,
    build_grammar_tokenizer,
    collate_records,
    load_sample_oas_record,
    load_untrained_no_encoder,
    run_grammar_generate,
)
from downstream.grammar.masks import light_chain_generation_partial_mask
from downstream.grammar.metrics import extract_chain_sequence, masked_token_accuracy


class HeavyLightCsvDataset(Dataset):
    def __init__(self, csv_path: str, heavy_col: str = "h_sequence", light_col: str = "l_sequence"):
        import pandas as pd

        frame = pd.read_csv(csv_path)
        self.heavy = frame[heavy_col].astype(str).str.replace("-", "").tolist()
        self.light = frame.get(light_col, "").fillna("").astype(str).str.replace("-", "").tolist()

    def __len__(self) -> int:
        return len(self.heavy)

    def __getitem__(self, index: int):
        return self.heavy[index], self.light[index]


def evaluate_batch(
    model,
    batch: dict[str, torch.Tensor],
    tokenizer,
    *,
    max_iter: int,
    sampling_strategy: str,
    temperature: float,
) -> dict[str, float | list[str]]:
    partial_mask = light_chain_generation_partial_mask(batch, tokenizer)
    labels = batch["labels"]
    generation_mask = partial_mask.logical_not() & batch["attention_mask"] & batch["residue_mask"]
    output_tokens, _ = run_grammar_generate(
        model,
        batch,
        partial_mask=partial_mask,
        max_iter=max_iter,
        sampling_strategy=sampling_strategy,
        temperature=temperature,
    )
    aar = masked_token_accuracy(output_tokens, labels, generation_mask)
    generated_lights = [
        extract_chain_sequence(
            output_tokens[row],
            batch["attention_mask"][row],
            batch["residue_mask"][row],
            tokenizer,
            chain="light",
        )
        for row in range(output_tokens.size(0))
    ]
    return {
        "aar": float(aar.mean().item()),
        "generated_light_sequences": generated_lights,
    }


def smoke_eval(device: str = "cpu") -> dict[str, float | list[str]]:
    tokenizer = build_grammar_tokenizer()
    record = load_sample_oas_record(split="valid", index=0)
    batch = collate_records([record])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    model = load_untrained_no_encoder(tokenizer.vocab_size, mask_token_id=tokenizer.mask_token_id).to(device)
    return evaluate_batch(
        model,
        batch,
        tokenizer,
        max_iter=8,
        sampling_strategy="argmax",
        temperature=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grammar light-chain pairing smoke/full eval.")
    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=32)
    parser.add_argument("--sampling-strategy", type=str, default="gumbel_argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke or not args.csv_path:
        result = smoke_eval(device=args.device)
        print(f"smoke AAR={result['aar']:.4f}")
        print(f"sample light={result['generated_light_sequences'][0][:80]}")
        return

    if args.checkpoint_path:
        raise NotImplementedError(
            "Checkpoint loading for grammar BioSeq models is not wired yet; use --smoke for sampler validation."
        )

    tokenizer = build_grammar_tokenizer()
    collator = build_grammar_collator(tokenizer)
    dataset = HeavyLightCsvDataset(args.csv_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = load_untrained_no_encoder(tokenizer.vocab_size, mask_token_id=tokenizer.mask_token_id).to(args.device)

    scores = []
    for heavy_rows, light_rows in loader:
        records = [antibody_pair_record(heavy, light) for heavy, light in zip(heavy_rows, light_rows)]
        batch = collator(records)
        batch = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        result = evaluate_batch(
            model,
            batch,
            tokenizer,
            max_iter=args.max_iter,
            sampling_strategy=args.sampling_strategy,
            temperature=args.temperature,
        )
        scores.append(float(result["aar"]))
    print(f"mean AAR={sum(scores) / max(len(scores), 1):.4f}")


if __name__ == "__main__":
    main()
