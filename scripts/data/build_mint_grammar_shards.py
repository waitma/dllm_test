#!/usr/bin/env python3
"""Build grammar-v1 Arrow shards from MINT STRING pretraining splits.

Prerequisites::

    bash scripts/data/download_stringdb_assets.sh
    # MMseqs2 clu50.tsv on protein.sequences.v12.0.fa
    python scripts/data/build_mint_string_splits.py

Then build shards (streaming, supports ~96M train pairs)::

    python scripts/data/build_mint_grammar_shards.py --split train
    python scripts/data/build_mint_grammar_shards.py --split valid

Outputs under ``data/bioseq_grammar_v1/mint_ppi/{train,valid}/``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

_data = PROJECT_ROOT / "dllm/pipelines/qwen3_vl_arch/data"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_grammar_builders = _load_module("grammar_builders", _data / "grammar_builders.py")
_ppi_splits = _load_module("ppi_splits", _data / "ppi_splits.py")
iter_semantic_rows = _grammar_builders.iter_semantic_rows
ppi_record = _grammar_builders.ppi_record
semantic_row = _grammar_builders.semantic_row
MINT_STRING_PRETRAIN = _ppi_splits.MINT_STRING_PRETRAIN
validate_split = _ppi_splits.validate_split

DEFAULT_MINT_DIR = PROJECT_ROOT / "data/ppi_task_raw/processed/mint_string_pretrain_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/bioseq_grammar_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mint-dir", type=Path, default=DEFAULT_MINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--max-protein-length", type=int, default=1024)
    parser.add_argument("--max-records", type=int, default=None, help="Debug cap.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_paths(mint_dir: Path, split: str) -> tuple[Path, Path]:
    if split == "train":
        return (
            mint_dir / "training_filtered.links.txt.gz",
            mint_dir / "training_filtered.seqs.txt.gz",
        )
    return mint_dir / "validation.links.txt.gz", mint_dir / "validation.seqs.txt.gz"


def load_sequence_map(seqs_gz: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    with gzip.open(seqs_gz, "rt") as handle:
        for line in handle:
            name, sequence = line.strip().split(None, 1)
            seqs[name] = sequence
    return seqs


def iter_mint_ppi_rows(
    links_gz: Path,
    seqs: dict[str, str],
    split: str,
    max_protein_length: int,
    max_records: int | None,
) -> Iterator[dict[str, Any]]:
    kept = 0
    with gzip.open(links_gz, "rt") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name1, name2 = parts[0], parts[1]
            seq_a = seqs.get(name1, "")
            seq_b = seqs.get(name2, "")
            record = ppi_record(
                seq_a,
                seq_b,
                split=split,
                relation="binding",
                source="mint_string_ppi",
                pair_key=tuple(sorted((name1, name2))),
                max_protein_length=max_protein_length,
            )
            if record is None:
                continue
            yield semantic_row(record, split)
            kept += 1
            if max_records is not None and kept >= max_records:
                break


def main() -> None:
    args = parse_args()
    validate_split("stringdb_mint", args.split, policy_id=MINT_STRING_PRETRAIN.policy_id)
    links_gz, seqs_gz = split_paths(args.mint_dir, args.split)
    for path in (links_gz, seqs_gz):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run build_mint_string_splits.py first "
                f"(requires MMseqs2 clu50.tsv)."
            )

    target = args.output_dir / "mint_ppi" / args.split
    if target.exists():
        if not args.force:
            from datasets import Dataset

            existing = Dataset.load_from_disk(str(target))
            print(json.dumps({"source": "mint_ppi", "split": args.split, "rows": len(existing), "path": str(target)}))
            return
        shutil.rmtree(target)

    print(f"Loading sequences from {seqs_gz}", flush=True)
    seqs = load_sequence_map(seqs_gz)
    print(f"Loaded {len(seqs):,} sequences; streaming links from {links_gz}", flush=True)

    from datasets import Dataset

    target.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(
        lambda: iter_mint_ppi_rows(
            links_gz,
            seqs,
            args.split,
            args.max_protein_length,
            args.max_records,
        ),
        cache_dir=str(args.output_dir / ".cache"),
    )
    dataset.save_to_disk(str(target), max_shard_size="512MB")
    manifest = {
        "source": "mint_ppi",
        "split": args.split,
        "split_policy": MINT_STRING_PRETRAIN.policy_id,
        "rows": len(dataset),
        "path": str(target),
        "links_gz": str(links_gz),
    }
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
