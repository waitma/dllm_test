#!/usr/bin/env python3
"""Build grammar-v1 Arrow shards for supervised interaction sources.

Reads ``interaction_records_unified.csv`` (with ``grammar_relation``) and writes
per-source semantic Arrow caches for mixed training.

Supported sources today::

    covabdab_neutralization  -> data/bioseq_grammar_v1/neutralization/train
    (extend SUPERVISED_SOURCE_BUILDERS below)

Example::

    python scripts/data/build_ppi_unified_csv.py
    python scripts/data/build_supervised_grammar_shards.py --sources covabdab_neutralization
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

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
antibody_neutralization_record = _grammar_builders.antibody_neutralization_record
semantic_row = _grammar_builders.semantic_row

DEFAULT_UNIFIED = PROJECT_ROOT / "data/ppi_task_raw/processed/interaction_records_unified.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/bioseq_grammar_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-csv", type=Path, default=DEFAULT_UNIFIED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sources", default="covabdab_neutralization")
    parser.add_argument("--split", default="train", help="Output split name (CoV-AbDab has no official split).")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _clean(value: str) -> str:
    text = str(value or "").strip()
    return "" if text.upper() in {"", "ND", "NAN", "NONE"} else text


def iter_covabdab_rows(row: dict[str, str], split: str) -> Iterator[dict[str, Any]]:
    heavy = _clean(row.get("antibody_heavy") or row.get("sequence_a"))
    light = _clean(row.get("antibody_light") or row.get("sequence_b"))
    ab_type = _clean(row.get("entity_a_id") or row.get("dataset_name")).lower()
    is_nanobody = "nb" in ab_type or row.get("task_family") == "antibody_neutralization" and not light
    if not heavy:
        return
    record = antibody_neutralization_record(
        heavy,
        light or None,
        split=split,
        is_nanobody=is_nanobody,
    )
    if record is not None:
        yield semantic_row(record, split)


def iter_unified_source(
    unified_csv: Path,
    source_id: str,
    builder: Callable[[dict[str, str], str], Iterator[dict[str, Any]]],
    split: str,
) -> Iterator[dict[str, Any]]:
    with unified_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("source_id") != source_id:
                continue
            yield from builder(row, split)


SUPERVISED_SOURCE_BUILDERS: dict[str, tuple[str, Callable[..., Iterator[dict[str, Any]]]]] = {
    "covabdab_neutralization": ("neutralization", iter_covabdab_rows),
}


def build_source(
    source_id: str,
    unified_csv: Path,
    output_dir: Path,
    split: str,
    force: bool,
) -> dict[str, Any]:
    if source_id not in SUPERVISED_SOURCE_BUILDERS:
        raise ValueError(f"Unsupported supervised source: {source_id}. Known: {list(SUPERVISED_SOURCE_BUILDERS)}")

    shard_name, builder = SUPERVISED_SOURCE_BUILDERS[source_id]
    target = output_dir / shard_name / split
    if target.exists():
        if not force:
            from datasets import Dataset

            dataset = Dataset.load_from_disk(str(target))
            return {"source": shard_name, "split": split, "rows": len(dataset), "path": str(target)}
        shutil.rmtree(target)

    from datasets import Dataset

    target.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(
        lambda: iter_unified_source(unified_csv, source_id, builder, split),
        cache_dir=str(output_dir / ".cache"),
    )
    dataset.save_to_disk(str(target), max_shard_size="128MB")
    return {"source": shard_name, "split": split, "rows": len(dataset), "path": str(target)}


def main() -> None:
    args = parse_args()
    if not args.unified_csv.exists():
        raise FileNotFoundError(
            f"Missing {args.unified_csv}. Run scripts/data/build_ppi_unified_csv.py first."
        )
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    results = []
    for source_id in sources:
        result = build_source(source_id, args.unified_csv, args.output_dir, args.split, args.force)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    manifest_path = args.output_dir / "supervised_manifest.json"
    manifest_path.write_text(json.dumps({"datasets": results}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
