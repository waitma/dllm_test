"""Clean the nanobody training corpus before grammar-shard building.

Two fixes, both required before the integrated-data run:
  1. Drop every ``nbbench_*`` source row. These are the NbBench benchmark
     splits that were mixed into training (step0 parser pulled them in), so
     leaving them creates direct train/test leakage against the downstream
     nanobody benchmark.
  2. Apply a VHH length window on ``cleaned_seq`` (default 90-160 aa) to remove
     truncated / fusion artifacts. Real VHH domains sit at ~110-130 aa
     (p1=97, p99=132 in step6_final/train.csv), so [90,160] keeps 99.9% of
     genuine nanobodies and drops only outliers.

Input : /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step6_final/{train,valid,holdout}.csv
Output: /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step7_clean/{train,valid,holdout}.csv

Run (conda env ``pllm``)::

    conda activate pllm
    python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/clean_nanobody_training.py

This does NOT do downstream sequence-similarity dedup (that is handled by the
separate cross-test dedup tool); it only removes the copied benchmark rows and
length outliers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_IN = Path(
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step6_final"
)
DEFAULT_OUT = Path(
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step7_clean"
)
SEQ_COL = "cleaned_seq"
SOURCE_COL = "source"
DROP_SOURCE_PREFIX = "nbbench"
CHUNK = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits", default="train,valid,holdout")
    parser.add_argument("--min-len", type=int, default=90)
    parser.add_argument("--max-len", type=int, default=160)
    return parser.parse_args()


def clean_split(
    in_csv: Path, out_csv: Path, min_len: int, max_len: int
) -> dict[str, int]:
    total = 0
    dropped_nbbench = 0
    dropped_len = 0
    dropped_empty = 0
    kept = 0
    header_written = False
    dropped_sources: dict[str, int] = {}

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    for chunk in pd.read_csv(in_csv, chunksize=CHUNK, low_memory=False):
        total += len(chunk)
        src = chunk[SOURCE_COL].astype(str)
        nb_mask = src.str.startswith(DROP_SOURCE_PREFIX)
        for name, cnt in src[nb_mask].value_counts().items():
            dropped_sources[name] = dropped_sources.get(name, 0) + int(cnt)
        dropped_nbbench += int(nb_mask.sum())
        chunk = chunk[~nb_mask]

        seq = chunk[SEQ_COL].astype(str)
        empty_mask = seq.isin(["", "nan", "None"]) | seq.isna()
        dropped_empty += int(empty_mask.sum())
        chunk = chunk[~empty_mask]

        seq_len = chunk[SEQ_COL].astype(str).str.len()
        len_mask = (seq_len >= min_len) & (seq_len <= max_len)
        dropped_len += int((~len_mask).sum())
        chunk = chunk[len_mask]

        kept += len(chunk)
        chunk.to_csv(out_csv, mode="a", index=False, header=not header_written)
        header_written = True

    return {
        "input": str(in_csv),
        "output": str(out_csv),
        "total": total,
        "dropped_nbbench": dropped_nbbench,
        "dropped_empty": dropped_empty,
        "dropped_len": dropped_len,
        "kept": kept,
        "dropped_sources": dropped_sources,
    }


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    report = {
        "min_len": args.min_len,
        "max_len": args.max_len,
        "drop_source_prefix": DROP_SOURCE_PREFIX,
        "splits": {},
    }
    for split in splits:
        in_csv = args.input_dir / f"{split}.csv"
        out_csv = args.output_dir / f"{split}.csv"
        if not in_csv.is_file():
            print(f"skip {split}: {in_csv} not found", flush=True)
            continue
        stats = clean_split(in_csv, out_csv, args.min_len, args.max_len)
        report["splits"][split] = stats
        print(
            f"[{split}] {stats['total']:,} -> kept {stats['kept']:,} "
            f"(drop nbbench={stats['dropped_nbbench']:,}, "
            f"empty={stats['dropped_empty']:,}, len={stats['dropped_len']:,})",
            flush=True,
        )

    report_path = args.output_dir / "clean_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
