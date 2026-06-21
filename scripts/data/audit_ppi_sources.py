#!/usr/bin/env python3
"""Audit local interaction / PPI assets and write an inventory manifest.

Example::

    python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_ppi_sources.py
    python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_ppi_sources.py --sample-unified 200000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.data.grammar import GRAMMAR_RELATIONS  # noqa: E402
from dllm.pipelines.qwen3_vl_arch.data.ppi_relations import (  # noqa: E402
    DATA_TIER,
    TASK_FAMILY_TO_RELATION,
    infer_grammar_relation,
    infer_relation_from_unified_row,
)

DATA_ROOT = PROJECT_ROOT / "data"
PPI_TASK_RAW = DATA_ROOT / "ppi_task_raw"
PROCESSED = PPI_TASK_RAW / "processed"
OUTPUT_MANIFEST = PPI_TASK_RAW / "processed" / "ppi_inventory.json"


def file_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def audit_string_downloads() -> dict:
    string_root = PPI_TASK_RAW / "raw/stringdb_mint"
    files = {}
    for path in sorted(string_root.glob("*.gz")):
        if path.name.endswith(".aria2"):
            continue
        files[path.name] = {
            "bytes": path.stat().st_size,
            "human_gb": round(path.stat().st_size / (1024**3), 2),
        }
    return {
        "path": str(string_root),
        "tier": DATA_TIER.get("stringdb_mint", "unknown"),
        "grammar_relation_default": "binding",
        "string_channel": "physical",
        "notes": (
            "MINT uses STRING physical links (~96M curated pairs). "
            "Functional channels (activation/inhibition/catalysis/...) require "
            "separate STRING link files not yet downloaded."
        ),
        "files": files,
    }


def audit_grammar_v1_ppi() -> dict:
    manifest_path = DATA_ROOT / "bioseq_grammar_v1/manifest.json"
    if not manifest_path.exists():
        return {"status": "missing", "path": str(manifest_path)}
    manifest = json.loads(manifest_path.read_text())
    ppi_rows = [item for item in manifest.get("datasets", []) if item.get("source") == "ppi"]
    return {
        "path": str(DATA_ROOT / "bioseq_grammar_v1/ppi"),
        "tier": DATA_TIER.get("string_model_org_90_90_split", "grammar_v1_current"),
        "grammar_relation_used": "binding",
        "splits": ppi_rows,
        "notes": "Built from string_model_org_90_90_split; all rows currently use <binding>.",
    }


def audit_string_eval_split() -> dict:
    split_root = DATA_ROOT / "ppi/string_model_org_90_90_split"
    if not split_root.exists():
        return {"status": "missing", "path": str(split_root)}
    try:
        from datasets import load_from_disk

        dataset = load_from_disk(str(split_root))
        counts = {split: len(dataset[split]) for split in dataset}
    except Exception as exc:
        counts = {"error": str(exc)}
    return {
        "path": str(split_root),
        "tier": DATA_TIER.get("string_model_org_90_90_split", "grammar_v1_current"),
        "counts": counts,
        "bytes": file_size_bytes(split_root),
    }


def audit_sources_manifest() -> list[dict]:
    manifest_path = PROCESSED / "interaction_sources_manifest.csv"
    if not manifest_path.exists():
        return []
    rows = []
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            source_id = row.get("source_id", "")
            task_family = row.get("task_family", "")
            rows.append(
                {
                    "source_id": source_id,
                    "task_family": task_family,
                    "tier": DATA_TIER.get(source_id, "unclassified"),
                    "grammar_relation_default": infer_grammar_relation(
                        task_family=task_family,
                        source_id=source_id,
                    ),
                    "status": row.get("status", ""),
                    "record_count_manifest": row.get("record_count", ""),
                    "raw_path": row.get("raw_path", ""),
                    "notes": row.get("notes", ""),
                }
            )
    return rows


def audit_unified_sample(sample_limit: int) -> dict:
    unified_path = PROCESSED / "interaction_records_unified.csv"
    if not unified_path.exists():
        return {"status": "missing", "path": str(unified_path)}
    families: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    rows_seen = 0
    with unified_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_seen += 1
            families[row.get("task_family", "")] += 1
            sources[row.get("source_id", "")] += 1
            labels[row.get("label", "")] += 1
            relation = infer_relation_from_unified_row(row)
            relations[relation] += 1
            if sample_limit and rows_seen >= sample_limit:
                break
    return {
        "path": str(unified_path),
        "bytes": unified_path.stat().st_size,
        "rows_sampled": rows_seen,
        "sample_limit": sample_limit,
        "task_family": dict(families.most_common()),
        "source_id": dict(sources.most_common(20)),
        "grammar_relation_inferred": dict(relations.most_common()),
        "label": dict(labels.most_common(10)),
        "notes": (
            "Unified CSV lacks grammar_relation column until rebuild via "
            "scripts/build_ppi_interaction_csv.py; relations above are inferred."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-unified",
        type=int,
        default=500_000,
        help="Rows to scan from interaction_records_unified.csv (0 = full file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_MANIFEST,
        help="Where to write the JSON inventory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = {
        "grammar_relations": list(GRAMMAR_RELATIONS),
        "task_family_to_relation": TASK_FAMILY_TO_RELATION,
        "data_tiers": DATA_TIER,
        "string_downloads": audit_string_downloads(),
        "string_eval_split": audit_string_eval_split(),
        "grammar_v1_ppi_cache": audit_grammar_v1_ppi(),
        "interaction_sources": audit_sources_manifest(),
        "unified_csv_sample": audit_unified_sample(args.sample_unified),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps(inventory, indent=2, sort_keys=True))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
