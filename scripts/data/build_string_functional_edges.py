#!/usr/bin/env python3
"""Sample STRING functional-channel edges from protein.links.detailed.v12.0.txt.gz.

STRING ships functional evidence as channel subscores inside the detailed links file
(~190GB). This script streams the gzip file and writes a manageable CSV subset for
audit and future grammar shard builders.

Prerequisites::

    bash scripts/data/download_stringdb_assets.sh --with-detailed

Example::

    python scripts/data/build_string_functional_edges.py --max-records 1000000
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.data.string_channels import parse_string_detailed_link  # noqa: E402

RAW_ROOT = PROJECT_ROOT / "data/ppi_task_raw/raw/stringdb_mint"
DEFAULT_INPUT = RAW_ROOT / "protein.links.detailed.v12.0.txt.gz"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/ppi_task_raw/processed/string_functional_edges_sample.csv"

FIELDNAMES = [
    "protein1",
    "protein2",
    "grammar_relation",
    "string_channel",
    "source_id",
    "task_family",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-gz", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-records", type=int, default=0, help="0 = scan entire file.")
    parser.add_argument("--min-combined-score", type=float, default=400.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_gz.exists():
        raise FileNotFoundError(
            f"Missing {args.input_gz}. Run download_stringdb_assets.sh --with-detailed."
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    scanned = 0
    relation_counts: dict[str, int] = {}

    with gzip.open(args.input_gz, "rt") as handle, args.output_csv.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
        writer.writeheader()
        header = next(handle)
        if "protein1" not in header:
            raise ValueError(f"Unexpected header: {header!r}")
        for line in handle:
            scanned += 1
            parts = line.strip().split()
            if len(parts) >= 3:
                combined = float(parts[2]) if parts[2].replace(".", "", 1).isdigit() else 0.0
                if combined < args.min_combined_score:
                    continue
            parsed = parse_string_detailed_link(line)
            if parsed is None:
                continue
            name1, name2, grammar_relation, channel = parsed
            writer.writerow(
                {
                    "protein1": name1,
                    "protein2": name2,
                    "grammar_relation": grammar_relation,
                    "string_channel": channel,
                    "source_id": "stringdb_mint",
                    "task_family": "ppi_pretraining",
                }
            )
            relation_counts[grammar_relation] = relation_counts.get(grammar_relation, 0) + 1
            kept += 1
            if args.max_records and kept >= args.max_records:
                break
            if scanned % 5_000_000 == 0:
                print(f"scanned={scanned:,} kept={kept:,}", flush=True)

    manifest = {
        "input_gz": str(args.input_gz),
        "output_csv": str(args.output_csv),
        "scanned": scanned,
        "kept": kept,
        "relation_counts": relation_counts,
    }
    manifest_path = args.output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
