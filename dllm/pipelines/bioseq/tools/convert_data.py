from __future__ import annotations

import argparse
from pathlib import Path

from ..adapters import (
    iter_csv_examples,
    iter_processed_jsonl_examples,
    nanobody_row_to_example,
    oas_paired_row_to_example,
    ots_paired_row_to_example,
    write_jsonl,
)


CSV_ADAPTERS = {
    "oas": oas_paired_row_to_example,
    "ots": ots_paired_row_to_example,
    "nanobody": nanobody_row_to_example,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local immune sequence data to BioSeq JSONL.")
    parser.add_argument(
        "--source-type",
        choices=("oas", "ots", "nanobody", "processed"),
        required=True,
        help="Input schema to normalize.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Absolute input CSV or JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Absolute output JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum examples to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_absolute() or not args.output.is_absolute():
        raise ValueError("--input and --output must be absolute paths")

    if args.source_type == "processed":
        examples = iter_processed_jsonl_examples(args.input, limit=args.limit)
    else:
        examples = iter_csv_examples(args.input, CSV_ADAPTERS[args.source_type], limit=args.limit)

    written = write_jsonl(examples, args.output)
    print(f"wrote {written} examples to {args.output}")


if __name__ == "__main__":
    main()
