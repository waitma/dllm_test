"""Prepare immune semantic records before training.

Examples::

    python scripts/data/preprocess_immune_dataset.py \
        --config configs/data/immune_v3.yaml \
        --output-dir data/prepared/immune_v3

Use ``--dry-run --max-rows 100`` to inspect source/filter statistics without
writing shards. This command never creates model-ready token caches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dllm.pipelines.immune_llada.data.preprocessing.pipeline import (
    load_preprocess_config,
    preprocess_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline-prepare immune semantic JSONL shards")
    parser.add_argument("--config", required=True, help="YAML preprocessing config")
    parser.add_argument("--output-dir", required=True, help="Prepared dataset directory")
    parser.add_argument("--source", default=None, help="Optional source selection, e.g. oas+ots")
    parser.add_argument("--split", default=None, help="Optional split; default is all configured splits or train")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum raw rows per source and split")
    parser.add_argument("--dry-run", action="store_true", help="Read and report without writing shards")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reserved for future resumable preprocessing; currently fails fast",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_preprocess_config(args.config)
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    configured_splits = list(config.splits)
    if args.split:
        splits = [args.split]
    else:
        splits = configured_splits or ["train"]
    result = preprocess_dataset(
        config,
        args.output_dir,
        splits=splits,
        source=args.source,
        max_rows=args.max_rows,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    print(json.dumps(result["report"], ensure_ascii=False, indent=2, sort_keys=True))
    if result["manifest"]:
        print(f"prepared manifest: {result['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
