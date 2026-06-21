#!/usr/bin/env python3
"""Build grammar-v1 TCR Arrow shards from PISTE official random split.

Uses split policy ``piste_tcr_random`` (see ppi_splits.py).

Example::

    python scripts/data/build_tcr_grammar_shards.py --split train
    python scripts/data/build_tcr_grammar_shards.py --splits train,valid
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_data = PROJECT_ROOT / "dllm/pipelines/qwen3_vl_arch/data"
_grammar_builders = _load_module("grammar_builders", _data / "grammar_builders.py")
_ppi_splits = _load_module("ppi_splits", _data / "ppi_splits.py")

semantic_row = _grammar_builders.semantic_row
BioSeqChain = _grammar_builders.BioSeqChain
BioSeqRecord = _grammar_builders.BioSeqRecord
is_valid_protein_sequence = _grammar_builders.is_valid_protein_sequence
normalize_sequence = _grammar_builders.normalize_sequence
PISTE_TCR_RANDOM = _ppi_splits.PISTE_TCR_RANDOM
validate_split = _ppi_splits.validate_split

PISTE_RANDOM = PROJECT_ROOT / "data/ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/random"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/bioseq_grammar_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piste-dir", type=Path, default=PISTE_RANDOM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-name", default="tcr_piste", help="Shard directory name under output-dir.")
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_csv_path(piste_dir: Path, split: str) -> Path:
    name = "val" if split in {"valid", "validation"} else split
    return piste_dir / f"{name}_data.csv"


def iter_piste_rows(csv_path: Path, split: str, limit: int | None) -> Iterator[dict[str, Any]]:
    kept = 0
    for chunk in pd.read_csv(csv_path, chunksize=50000, low_memory=False):
        for raw in chunk.to_dict(orient="records"):
            cdr3 = normalize_sequence(raw.get("CDR3"))
            peptide = normalize_sequence(raw.get("MT_pep"))
            hla_seq = normalize_sequence(raw.get("HLA_sequence"))
            label = str(raw.get("Label", "")).strip()
            if not is_valid_protein_sequence(peptide):
                continue
            chains: list[BioSeqChain] = []
            if is_valid_protein_sequence(hla_seq):
                chains.append(BioSeqChain(hla_seq, "hla"))
            chains.append(BioSeqChain(peptide, "epitope"))
            if is_valid_protein_sequence(cdr3):
                chains.append(BioSeqChain(cdr3, "tcr_beta"))
            relation = "binding" if label in {"1", "1.0", "True", "true"} else "nonbinding"
            record = BioSeqRecord(
                chains=chains,
                task_type="tcr_pmhc" if any(c.role == "hla" for c in chains) else "tcr",
                source="piste_tcr_epitope_hla",
                split=split,
                labels={"relation": relation, "hla_type": str(raw.get("HLA_type") or "")},
            )
            yield semantic_row(record, split)
            kept += 1
            if limit is not None and kept >= limit:
                return


def build_split(
    split: str,
    piste_dir: Path,
    output_dir: Path,
    source_name: str,
    limit: int | None,
    force: bool,
) -> dict[str, Any]:
    validate_split("piste_tcr_epitope_hla", split, policy_id=PISTE_TCR_RANDOM.policy_id)
    csv_path = split_csv_path(piste_dir, split)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing PISTE split file: {csv_path}")

    target = output_dir / source_name / split
    if target.exists():
        if not force:
            from datasets import Dataset

            dataset = Dataset.load_from_disk(str(target))
            return {
                "source": source_name,
                "split": split,
                "rows": len(dataset),
                "path": str(target),
                "split_policy": PISTE_TCR_RANDOM.policy_id,
            }
        shutil.rmtree(target)

    from datasets import Dataset

    target.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(
        lambda: iter_piste_rows(csv_path, split, limit),
        cache_dir=str(output_dir / ".cache"),
    )
    dataset.save_to_disk(str(target), max_shard_size="512MB")
    return {
        "source": source_name,
        "split": split,
        "rows": len(dataset),
        "path": str(target),
        "split_policy": PISTE_TCR_RANDOM.policy_id,
        "input_csv": str(csv_path),
    }


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    results = []
    for split in splits:
        result = build_split(
            split,
            args.piste_dir,
            args.output_dir,
            args.source_name,
            args.max_records,
            args.force,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    manifest = args.output_dir / f"{args.source_name}_manifest.json"
    manifest.write_text(json.dumps({"datasets": results}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
