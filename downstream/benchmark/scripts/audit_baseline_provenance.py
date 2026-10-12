"""Audit active baseline report integrity without rewriting source results.

Run: python downstream/benchmark/scripts/audit_baseline_provenance.py --summary-dir <T1_REPORT> --strict
Missing published values are disclosed, not substituted. Frozen tasks are outside this audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from common.reporting import new_report_dir, write_report_manifest
from tcr_binding.original_protocol import ORIGINAL_BASELINES, PAPER_RESULT_LABEL

REQUIRED_PAPER_BENCHMARKS = {"t1_binding", "t2_clustering", "t3_representation", "t4_generation", "antibody_cdr", "light_chain_pairing"}
PAPER_CSV = BENCH / "outputs" / "external" / "paper_reported_baselines.csv"


def validate_t1(table: pd.DataFrame, paper: pd.DataFrame) -> list[str]:
    errors = []
    expected = {spec.tag for spec in ORIGINAL_BASELINES}
    required = {"tag", "selected_source", "result_label", "comparison_allowed"}
    required |= {f"{split}_{metric}" for split in ("seen", "unseen") for metric in ("auprc", "auroc")}
    if not required.issubset(table):
        return [f"T1 table missing columns: {sorted(required - set(table))}"]
    if set(table["tag"]) != expected or len(table) != len(expected):
        errors.append("T1 missing/duplicate/noncanonical baseline rows")
    for spec in ORIGINAL_BASELINES:
        rows = table[table["tag"] == spec.tag]
        if len(rows) != 1:
            continue
        row = rows.iloc[0]
        reference = paper[(paper["benchmark"] == "t1_binding") & (paper["task"] == "original_cdr3b_as") & (paper["method"] == spec.paper_method)]
        expected_source = "paper_reported" if not reference.empty else "unavailable"
        if row["selected_source"] != expected_source:
            errors.append(f"{spec.tag}: primary source must be {expected_source}")
        if expected_source == "paper_reported" and row["result_label"] != PAPER_RESULT_LABEL:
            errors.append(f"{spec.tag}: invalid paper label")
        if str(row["comparison_allowed"]).lower() != "false":
            errors.append(f"{spec.tag}: paper/local equivalence lacks evidence")
        for split in ("seen", "unseen"):
            for metric in ("auprc", "auroc"):
                hit = reference[(reference["split"] == split) & (reference["metric"] == metric)]
                expected_value = float(hit.iloc[0]["value"]) if len(hit) == 1 else float("nan")
                actual = row[f"{split}_{metric}"]
                if pd.isna(expected_value):
                    if not pd.isna(actual):
                        errors.append(f"{spec.tag}/{split}/{metric}: missing paper cell was filled")
                elif pd.isna(actual) or abs(float(actual) - expected_value) > 1e-12:
                    errors.append(f"{spec.tag}/{split}/{metric}: primary differs from paper")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--strict", action="store_true", help="Fail on integrity errors; disclosed missing evidence is pending")
    args = parser.parse_args(argv)
    destination = new_report_dir(args.output_dir, "baseline_audit")
    errors, pending = [], []
    inputs = [Path(__file__), PAPER_CSV]
    if not PAPER_CSV.is_file():
        errors.append("missing paper registry")
        paper = pd.DataFrame(columns=["benchmark", "task", "method", "split", "metric", "value"])
    else:
        paper = pd.read_csv(PAPER_CSV)
    missing = sorted(REQUIRED_PAPER_BENCHMARKS - set(paper["benchmark"]))
    pending += [f"paper reference unavailable: {task}" for task in missing]
    table_path = args.summary_dir / "summary_original_baselines.csv"
    inputs.append(table_path)
    if not table_path.is_file():
        errors.append("missing T1 canonical report")
    else:
        errors += validate_t1(pd.read_csv(table_path), paper)
    main_path = args.summary_dir / "summary_main.csv"
    inputs.append(main_path)
    if not main_path.is_file():
        errors.append("missing T1 main report")
    else:
        main = pd.read_csv(main_path)
        if not {"tag", "type"}.issubset(main):
            errors.append("invalid T1 main report columns")
        else:
            catalog = {spec.tag for spec in ORIGINAL_BASELINES}
            primary = main[main["tag"].isin(catalog)]
            errors += validate_t1(primary, paper)
            if not main.loc[~main["tag"].isin(catalog), "type"].astype(str).str.startswith(("ours", "training-free", "floor")).all():
                errors.append("noncanonical external rows entered primary table")
    report = {"schema_version": 2, "status": "fail" if errors else "pass",
              "policy": "paper_primary_local_separate; unknown_no_ranking",
              "active_paper_reference_tasks": sorted(REQUIRED_PAPER_BENCHMARKS),
              "errors": errors, "pending": pending,
              "scope": "report source selection, not statistical validity or pretraining cleanliness"}
    (destination / "baseline_provenance_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    write_report_manifest(destination, inputs, policy=report["policy"])
    print(json.dumps(report, indent=2))
    print(f"Report: {destination}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
