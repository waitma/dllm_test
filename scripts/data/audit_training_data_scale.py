#!/usr/bin/env python3
"""Audit grammar_v1 training data scale vs available corpora and split policies.

Example::

    python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_training_data_scale.py
    python .../audit_training_data_scale.py --json-out data/ppi_task_raw/processed/training_data_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
GRAMMAR_DIR = PROJECT_ROOT / "data/bioseq_grammar_v1"
PROCESSED_V2_STATS = PROJECT_ROOT / "data/processed_v2/stats.json"
PISTE_RANDOM = PROJECT_ROOT / "data/ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/random"
STRING_RAW = PROJECT_ROOT / "data/ppi_task_raw/raw/stringdb_mint"
MINT_SPLIT_DIR = PROJECT_ROOT / "data/ppi_task_raw/processed/mint_string_pretrain_v1"
UNIFIED_SUMMARY = PROJECT_ROOT / "data/ppi_task_raw/processed/interaction_records_summary.csv"


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def grammar_rows(source: str, split: str) -> int | None:
    target = GRAMMAR_DIR / source / split
    if not target.exists():
        return None
    try:
        from datasets import load_from_disk

        return len(load_from_disk(str(target)))
    except Exception:
        return None


def file_gb(path: Path) -> float | None:
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024**3), 2)


def load_processed_v2_stats() -> dict[str, Any]:
    if not PROCESSED_V2_STATS.exists():
        return {}
    return json.loads(PROCESSED_V2_STATS.read_text())


def parse_unified_summary() -> dict[str, int]:
    if not UNIFIED_SUMMARY.exists():
        return {}
    import csv

    counts: dict[str, int] = {}
    with UNIFIED_SUMMARY.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                counts[row["source_id"]] = int(row["record_count"])
            except (KeyError, ValueError):
                continue
    return counts


def assess_scale(current: int | None, target: int, label: str) -> str:
    if current is None:
        return "missing"
    if current == 0:
        return "empty"
    ratio = current / target if target else 0
    if ratio >= 0.9:
        return "full"
    if ratio >= 0.1:
        return "partial"
    return "smoke_or_tiny"


def build_report() -> dict[str, Any]:
    v2 = load_processed_v2_stats()
    unified = parse_unified_summary()
    manifest_path = GRAMMAR_DIR / "manifest.json"
    grammar_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    oas_train = grammar_rows("oas", "train")
    ots_train = grammar_rows("ots", "train")
    tcr_train = grammar_rows("tcr", "train")
    ppi_train = grammar_rows("ppi", "train")

    piste_train = count_csv_rows(PISTE_RANDOM / "train_data.csv")
    piste_val = count_csv_rows(PISTE_RANDOM / "val_data.csv")
    piste_test = count_csv_rows(PISTE_RANDOM / "test_data.csv")

    string_files = {
        "sequences_gz_gb": file_gb(STRING_RAW / "protein.sequences.v12.0.fa.gz"),
        "physical_full_gz_gb": file_gb(STRING_RAW / "protein.physical.links.full.v12.0.txt.gz"),
        "clu50_tsv": (STRING_RAW / "clu50.tsv").exists(),
        "mint_splits_built": (MINT_SPLIT_DIR / "manifest.json").exists(),
    }

    report: dict[str, Any] = {
        "summary": {
            "grammar_v1_is_smoke_for_tcr": tcr_train is not None and tcr_train < 500_000 and piste_train > tcr_train,
            "grammar_v1_is_smoke_for_ppi": ppi_train is not None and ppi_train < 1_000_000,
            "ppi_wrong_tier_for_pretraining": True,
            "notes": (
                "OAS/OTS use project-owned splits and are full-scale. "
                "TCR grammar cache (~163k) is a small processed_v2 subset; PISTE alone has ~284k train. "
                "PPI grammar cache (~319k) uses Bernett 90/90 eval tier, not MINT 96M pretraining."
            ),
        },
        "grammar_v1_cache": {
            "path": str(GRAMMAR_DIR),
            "manifest": grammar_manifest,
            "rows": {
                "oas_train": oas_train,
                "ots_train": ots_train,
                "tcr_train": tcr_train,
                "tcr_valid": grammar_rows("tcr", "valid"),
                "ppi_train": ppi_train,
                "ppi_valid": grammar_rows("ppi", "valid"),
                "mint_ppi_train": grammar_rows("mint_ppi", "train"),
                "neutralization_train": grammar_rows("neutralization", "train"),
            },
        },
        "split_policies": {
            "oas": {"policy": "project_owned", "reference": "data/oas_previous_clean/splits"},
            "ots": {"policy": "project_owned", "reference": "data/ots_paired_clean/final"},
            "tcr_current": {
                "policy": "processed_v2_adhoc",
                "reference": "data/processed_v2 train/val.jsonl",
                "problem": "Not a published benchmark split; excludes PISTE/TDC/IEDB merged corpora",
            },
            "tcr_recommended": [
                {
                    "policy_id": "piste_random_split",
                    "reference": "PISTE github Armilius/PISTE data/random/{train,val,test}_data.csv",
                    "train": piste_train,
                    "valid": piste_val,
                    "test": piste_test,
                },
                {
                    "policy_id": "nat_methods_2025_tcr_benchmark",
                    "reference": "Nat Methods 2025 figshare 10.6084/m9.figshare.27020455",
                    "status": "not_downloaded",
                },
            ],
            "ppi_current": {
                "policy_id": "bernett_string_90_90_hf",
                "reference": "string_model_org_90_90_split",
                "train": ppi_train,
                "problem": "Eval-tier gold standard; ~0.33% of MINT STRING pretraining scale",
            },
            "ppi_recommended_pretrain": {
                "policy_id": "mint_string_pretrain_v1",
                "reference": "MINT Nat Commun 2026; ~95.8M train pairs",
                "raw_ready": string_files["physical_full_gz_gb"] is not None,
                "clu50_ready": string_files["clu50_tsv"],
                "splits_ready": string_files["mint_splits_built"],
            },
        },
        "available_not_in_grammar_v1": {
            "processed_v2_train_total": v2.get("total_train"),
            "processed_v2_tcr_sources": {
                k: v2.get("source_dist_train", {}).get(k)
                for k in ("vdjdb", "mcpas", "mira")
            },
            "processed_v2_ppi_excluded": v2.get("source_dist_train", {}).get("ppi"),
            "unified_csv_counts": unified,
            "piste_random_total": piste_train + piste_val + piste_test,
            "flab_unified": unified.get("flab"),
            "tdc_tcr_unified": unified.get("tdc_tcr_epitope"),
            "teim_unified": unified.get("teim_interface"),
        },
        "expansion_priority": [
            {
                "priority": 1,
                "action": "Build MINT STRING splits (mmseqs clu50) + mint_ppi grammar shards",
                "target_rows": 95_800_000,
                "current_rows": grammar_rows("mint_ppi", "train") or 0,
            },
            {
                "priority": 2,
                "action": "Replace grammar TCR with PISTE random official train/val (+ keep test for eval)",
                "target_rows": piste_train,
                "current_rows": tcr_train or 0,
            },
            {
                "priority": 3,
                "action": "Download Nat Methods 2025 TCR benchmark figshare for seen/unseen epitope eval splits",
                "url": "https://doi.org/10.6084/m9.figshare.27020455",
            },
            {
                "priority": 4,
                "action": "Add neutralization + FLAb supervised shards with official/author splits only",
                "flab_rows": unified.get("flab"),
                "covabdab_rows": unified.get("covabdab_neutralization"),
            },
            {
                "priority": 5,
                "action": "Merge VDJdb/IEDB/McPAS via published tpp or NetTCR-2 protocol (not random split)",
                "local_vdjdb": "data/tcr/vdjdb*.txt present",
            },
        ],
        "scale_assessment": {
            "oas": assess_scale(oas_train, 2_400_000, "oas"),
            "ots": assess_scale(ots_train, 2_100_000, "ots"),
            "tcr": assess_scale(tcr_train, piste_train or 280_000, "tcr"),
            "ppi_pretrain": assess_scale(ppi_train, 95_800_000, "ppi"),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = build_report()
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
        print(f"wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
