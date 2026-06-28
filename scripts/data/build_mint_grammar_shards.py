#!/usr/bin/env python3
"""Build grammar-v1 Arrow shards from MINT STRING pretraining splits.

Prerequisites::

    bash scripts/data/download_stringdb_assets.sh
    # MMseqs2 clu50.tsv on protein.sequences.v12.0.fa
    python scripts/data/build_mint_string_splits.py

Then build shards (streaming, supports ~96M train pairs)::

    python scripts/data/build_mint_grammar_shards.py --source mint_ppi --split train
    python scripts/data/build_mint_grammar_shards.py --source mint_actions --split train

Outputs under ``data/bioseq_grammar_v1/{mint_ppi,mint_actions}/{train,valid}/``.
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
_ppi_relations = _load_module("ppi_relations", _data / "ppi_relations.py")
ppi_record = _grammar_builders.ppi_record
semantic_row = _grammar_builders.semantic_row
MINT_STRING_PRETRAIN = _ppi_splits.MINT_STRING_PRETRAIN
MINT_STRING_ACTIONS_V11 = _ppi_splits.MINT_STRING_ACTIONS_V11
validate_split = _ppi_splits.validate_split
normalize_relation = _ppi_relations.normalize_relation

DEFAULT_MINT_PPI_DIR = PROJECT_ROOT / "data/ppi_task_raw/processed/mint_string_pretrain_v1"
DEFAULT_MINT_ACTIONS_DIR = PROJECT_ROOT / "data/ppi_task_raw/processed/mint_string_actions_v11.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/bioseq_grammar_v1"

SOURCE_CONFIG = {
    "mint_ppi": {
        "source_id": "stringdb_mint",
        "policy": MINT_STRING_PRETRAIN,
        "default_mint_dir": DEFAULT_MINT_PPI_DIR,
        "shard_name": "mint_ppi",
        "record_source": "mint_string_ppi",
        "default_relation": "binding",
    },
    "mint_actions": {
        "source_id": "stringdb_actions",
        "policy": MINT_STRING_ACTIONS_V11,
        "default_mint_dir": DEFAULT_MINT_ACTIONS_DIR,
        "shard_name": "mint_actions",
        "record_source": "mint_string_actions",
        "default_relation": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=tuple(SOURCE_CONFIG),
        default="mint_ppi",
        help="mint_ppi=physical binding splits; mint_actions=protein.actions mode splits.",
    )
    parser.add_argument("--mint-dir", type=Path, default=None)
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


def iter_mint_rows(
    links_gz: Path,
    seqs: dict[str, str],
    split: str,
    *,
    source_name: str,
    record_source: str,
    default_relation: str | None,
    max_protein_length: int,
    max_records: int | None,
) -> Iterator[dict[str, Any]]:
    kept = 0
    with gzip.open(links_gz, "rt") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            target_name, actor_name = parts[0], parts[1]
            relation = default_relation
            if len(parts) >= 3:
                relation = normalize_relation(parts[2])
            elif default_relation is None:
                raise ValueError(f"{source_name} links must include relation column: {line[:120]}")

            seq_target = seqs.get(target_name, "")
            seq_actor = seqs.get(actor_name, "")
            record = ppi_record(
                seq_target,
                seq_actor,
                split=split,
                relation=relation,
                source=record_source,
                pair_key=(target_name, actor_name),
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
    cfg = SOURCE_CONFIG[args.source]
    mint_dir = args.mint_dir or cfg["default_mint_dir"]
    validate_split(cfg["source_id"], args.split, policy_id=cfg["policy"].policy_id)
    links_gz, seqs_gz = split_paths(mint_dir, args.split)
    for path in (links_gz, seqs_gz):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Build MINT/actions splits first for source={args.source!r}."
            )

    target = args.output_dir / cfg["shard_name"] / args.split
    if target.exists():
        if not args.force:
            from datasets import Dataset

            existing = Dataset.load_from_disk(str(target))
            print(
                json.dumps(
                    {
                        "source": cfg["shard_name"],
                        "split": args.split,
                        "rows": len(existing),
                        "path": str(target),
                    }
                )
            )
            return
        shutil.rmtree(target)

    print(f"Loading sequences from {seqs_gz}", flush=True)
    seqs = load_sequence_map(seqs_gz)
    print(f"Loaded {len(seqs):,} sequences; streaming links from {links_gz}", flush=True)

    from datasets import Dataset

    target.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(
        lambda: iter_mint_rows(
            links_gz,
            seqs,
            args.split,
            source_name=cfg["shard_name"],
            record_source=cfg["record_source"],
            default_relation=cfg["default_relation"],
            max_protein_length=args.max_protein_length,
            max_records=args.max_records,
        ),
        cache_dir=str(args.output_dir / ".cache"),
    )
    dataset.save_to_disk(str(target), max_shard_size="512MB")
    manifest = {
        "source": cfg["shard_name"],
        "split": args.split,
        "split_policy": cfg["policy"].policy_id,
        "rows": len(dataset),
        "path": str(target),
        "links_gz": str(links_gz),
    }
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
