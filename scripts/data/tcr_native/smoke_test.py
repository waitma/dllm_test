#!/usr/bin/env python3
"""Offline smoke test for prepared immune JSONL and grammar-v2 batching.

Example::

    python scripts/data/tcr_native/smoke_test.py \
        --prepared-data-dir data/prepared/immune_v3_heterotypic \
        --dataset-args tcr_native --max-rows 5

The command reads only prepared semantic records, builds a real grammar batch,
and never imports the trainer, parses raw CSV, loads model weights, or launches
GPU work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
sys.path.insert(0, str(PROJECT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-args", default="oas+ots+tcr_native")
    ap.add_argument(
        "--prepared-data-dir",
        default=str(PROJECT / "data/prepared/immune_v3_heterotypic"),
        help="Prepared semantic JSONL directory",
    )
    ap.add_argument("--max-rows", type=int, default=5)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    from dllm.pipelines.immune_llada.data import (
        GrammarBioSeqCollator,
        GrammarTokenizer,
        load_prepared_dataset,
    )
    from dllm.pipelines.immune_llada.data.registry import parse_sources

    if args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    sources = parse_sources(args.dataset_args)
    records = []
    for source in sources:
        try:
            dataset = load_prepared_dataset(
                args.prepared_data_dir,
                split=args.split,
                source=source,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  [{source}] BUILD FAILED: {type(exc).__name__}: {exc}")
            continue
        selected = [dataset[i] for i in range(min(args.max_rows, len(dataset)))]
        records.extend(selected)
        print(f"  [{source}] rows={len(dataset)} selected={len(selected)}")
        for i, record in enumerate(selected[:2]):
            shown = [
                (chain.role, (chain.sequence[:24] + "..." if len(chain.sequence) > 24 else chain.sequence))
                for chain in record.chains
            ]
            print(
                f"      #{i} task={record.task_type} relation={record.labels.get('relation', '-')} "
                f"source={record.source} chains={shown}"
            )

    if not records:
        raise RuntimeError("No prepared records selected; check --prepared-data-dir/--split/--dataset-args")
    batch = GrammarBioSeqCollator(GrammarTokenizer())(records)
    if batch["input_ids"].shape[0] != len(records):
        raise AssertionError("grammar batch size does not match prepared records")
    if not bool(batch["encoder_chain_mask"].any()):
        raise AssertionError("grammar batch has no real encoder chains")
    if not bool(batch["diffusion_eligible_mask"].any()):
        raise AssertionError("grammar batch has no diffusion-eligible tokens")
    print(
        f"grammar_batch={tuple(batch['input_ids'].shape)} "
        f"encoder={tuple(batch['encoder_input_ids'].shape)}"
    )
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
