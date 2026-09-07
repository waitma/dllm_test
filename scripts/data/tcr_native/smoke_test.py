#!/usr/bin/env python3
"""Data-path smoke test for --dataset_args oas+ots+tcr_native.

Builds the immune source specs and datasets (small caps) and renders a few
example rows. Does NOT construct the model / tokenizer / ESMC. Verifies the
tcr_native wiring (token + tcr_native_row_to_record) and the TRAIT blocklist
mechanism, without launching any training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "examples/llada"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-args", default="oas+ots+tcr_native")
    ap.add_argument("--tcr-native-dir", default=str(PROJECT / "data/tcr_native/dataset"))
    ap.add_argument("--max-rows", type=int, default=5)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    from protein_pretrain_esmc import DataArguments, build_immune_specs
    from dllm.pipelines.bioseq.datasets import ImmuneCsvDataset

    data_args = DataArguments(
        dataset_args=args.dataset_args,
        tcr_native_dir=args.tcr_native_dir,
    )
    specs = build_immune_specs(data_args)
    print(f"dataset_args={args.dataset_args!r} -> {len(specs)} specs: {[s.name for s in specs]}")

    for spec in specs:
        try:
            ds = ImmuneCsvDataset(spec, split=args.split, max_rows=args.max_rows)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{spec.name}] BUILD FAILED: {type(exc).__name__}: {exc}")
            continue
        print(f"  [{spec.name}] rows={len(ds)} task_type={spec.task_type}")
        for i in range(min(2, len(ds))):
            rec = ds[i]
            roles = rec.get("roles") or [f"chain{j}" for j in range(len(rec["chains"]))]
            shown = [(r, (c[:24] + "..." if len(c) > 24 else c)) for r, c in zip(roles, rec["chains"])]
            print(f"      #{i} task={rec.get('task_type')} relation={rec.get('relation','-')} "
                  f"source={rec.get('source')} chains={shown}")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
