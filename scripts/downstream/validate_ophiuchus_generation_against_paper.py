"""Compare Ophiuchus-Ab generation metrics against paper reference values.

This script reads the local generation metrics summary and writes a concise
machine-readable and human-readable consistency report. It intentionally keeps
split provenance separate from numeric closeness: a metric can be numerically
close while still not being an exact reproduction if the paper split is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = Path("output/downstream_generation/ophiuchus_ab/generation_metrics_summary.json")
DEFAULT_JSON = Path("output/downstream_generation/ophiuchus_ab/paper_consistency_check.json")
DEFAULT_MD = Path("output/downstream_generation/ophiuchus_ab/paper_consistency_check.md")


def _metric_row(
    task: str,
    metric: str,
    paper: float,
    current: float,
    tolerance: float,
    unit: str,
    split_status: str,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    delta = current - paper
    abs_delta = abs(delta)
    within_tolerance = abs_delta <= tolerance
    if lower_is_better:
        direction = "better" if current < paper else "worse" if current > paper else "same"
    else:
        direction = "better" if current > paper else "worse" if current < paper else "same"
    return {
        "task": task,
        "metric": metric,
        "paper": paper,
        "current": current,
        "delta": delta,
        "abs_delta": abs_delta,
        "tolerance": tolerance,
        "unit": unit,
        "within_tolerance": within_tolerance,
        "direction": direction,
        "split_status": split_status,
    }


def _pass_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["within_tolerance"]) / len(rows)


def _verdict(rows: list[dict[str, Any]], exact_split: bool, critical_metrics: set[str] | None = None) -> str:
    critical_metrics = critical_metrics or set()
    critical_fail = any(
        row["metric"] in critical_metrics and not row["within_tolerance"]
        for row in rows
    )
    pass_rate = _pass_rate(rows)
    if exact_split and pass_rate == 1.0:
        return "consistent"
    if exact_split and pass_rate >= 0.75 and not critical_fail:
        return "mostly_consistent"
    if not exact_split and pass_rate >= 0.75 and not critical_fail:
        return "numerically_close_but_not_exact_split"
    if pass_rate >= 0.5 and not critical_fail:
        return "partially_consistent"
    return "not_reproduced"


def build_check(summary: dict[str, Any]) -> dict[str, Any]:
    paper = summary["paper_reference"]
    cdr_rows: list[dict[str, Any]] = []

    sab23 = summary["cdr_infill"]["sab23h2"]["metrics"]
    sab23_paper = paper["cdr_sab23h2_ophiuchus_ab_aar"]
    for region in ["L1", "L2", "L3", "H1", "H2", "H3"]:
        current = float(sab23[f"cdr{region.lower()}"])
        reference = float(sab23_paper[region]) * 100.0
        cdr_rows.append(
            _metric_row(
                "cdr_sab23h2",
                region,
                paper=reference,
                current=current,
                tolerance=3.0,
                unit="percentage_points",
                split_status="local_downstream_input",
            )
        )

    sabdab = summary["cdr_infill"]["sabdab_external_indices"]
    sabdab_paper = paper["cdr_sabdab_ophiuchus_ab_heavy_aar_percent"]
    for region in ["H1", "H2", "H3"]:
        current = float(sabdab[f"cdr{region.lower()}"]["aar_percent"])
        reference = float(sabdab_paper[region])
        cdr_rows.append(
            _metric_row(
                "cdr_sabdab_external_indices",
                region,
                paper=reference,
                current=current,
                tolerance=3.0,
                unit="percentage_points",
                split_status="local_downstream_input",
            )
        )

    light_rows: list[dict[str, Any]] = []
    light_tolerances = {
        "ImmunoMatch": (0.05, "score", False),
        "Better_percent": (5.0, "percentage_points", False),
        "Chain_percent": (5.0, "percentage_points", False),
        "V": (0.05, "rate", False),
        "J": (0.05, "rate", False),
        "V_fam": (0.05, "rate", False),
        "J_fam": (0.05, "rate", False),
        "Div": (0.05, "score", False),
        "Valid_percent": (1.0, "percentage_points", False),
        "W_property": (0.02, "wasserstein_distance", True),
    }
    light_map = {
        "ImmunoMatch": "gen_immunomatch_mean",
        "Better_percent": "gen_better_ratio",
        "Chain_percent": "overall_chain_match_rate",
        "V": "overall_v_gene_match_rate",
        "J": "overall_j_gene_match_rate",
        "V_fam": "overall_v_gene_family_match_rate",
        "J_fam": "overall_j_gene_family_match_rate",
        "Div": "diversity_mean",
        "Valid_percent": "gen_valid_rate",
        "W_property": "w_property_avg_wd",
    }

    split_notes = {
        "holdout500_airgen32_prompt3": "exact_oas_holdout_500_gumbel_argmax_32",
        "no_prompt_exact_holdout_500": "exact_oas_holdout_500_argmax_4_legacy",
    }
    for label, paper_key, summary_key in [
        (
            "light_chain_pairing_prompt3",
            "light_chain_pairing_ophiuchus_ab_with_initial_3_residue_prompting",
            "holdout500_airgen32_prompt3",
        ),
        (
            "light_chain_pairing_no_prompt",
            "light_chain_pairing_ophiuchus_ab_without_prompting",
            "no_prompt_exact_holdout_500",
        ),
    ]:
        paper_metrics = paper[paper_key]
        current_metrics = summary["light_chain_pairing"][summary_key]["metrics"]
        for metric, current_key in light_map.items():
            if current_key not in current_metrics:
                continue
            tolerance, unit, lower_is_better = light_tolerances[metric]
            reference = float(paper_metrics[metric])
            current = float(current_metrics[current_key])
            if metric in {"Better_percent", "Chain_percent", "Valid_percent"}:
                current *= 100.0
            light_rows.append(
                _metric_row(
                    label,
                    metric,
                    paper=reference,
                    current=current,
                    tolerance=tolerance,
                    unit=unit,
                    split_status=split_notes.get(summary_key, "reconstructed_oas_500_not_exact_paper_csv"),
                    lower_is_better=lower_is_better,
                )
            )

    human_rows: list[dict[str, Any]] = []
    human_paper = paper["humanization_ophiuchus_ab"]
    human_current = summary["humanization"]["exact27_airgen32"]["metrics"]
    human_oasis = human_current.get("biophi_oasis_relaxed") or {}
    if human_oasis:
        human_rows.append(
            _metric_row(
                "humanization_exact27_airgen32",
                "OASis_score_or_identity",
                paper=float(human_paper["OASis_score"]),
                current=float(human_oasis.get("oasis_identity_percent", float("nan"))),
                tolerance=5.0,
                unit="percentage_points",
                split_status="exact_uploaded_27_pair_split",
            )
        )
        human_rows.append(
            _metric_row(
                "humanization_exact27_airgen32",
                "VH_sequence_identity_percent",
                paper=float(human_paper["VH_sequence_identity_percent"]),
                current=float(human_oasis.get("heavy_germline_content_mean", float("nan"))) * 100.0,
                tolerance=5.0,
                unit="percentage_points",
                split_status="exact_uploaded_27_pair_split",
            )
        )
        human_rows.append(
            _metric_row(
                "humanization_exact27_airgen32",
                "VL_sequence_identity_percent",
                paper=float(human_paper["VL_sequence_identity_percent"]),
                current=float(human_oasis.get("light_germline_content_mean", float("nan"))) * 100.0,
                tolerance=5.0,
                unit="percentage_points",
                split_status="exact_uploaded_27_pair_split",
            )
        )

    grouped = {
        "cdr": cdr_rows,
        "light_chain_pairing": light_rows,
        "humanization": human_rows,
    }
    return {
        "criteria": {
            "cdr_aar_tolerance_percentage_points": 3.0,
            "light_chain_pairing_tolerances": {
                metric: tolerance for metric, (tolerance, _, _) in light_tolerances.items()
            },
            "humanization_identity_tolerance_percentage_points": 5.0,
            "note": "Tolerance checks measure numeric closeness only. Exact reproduction still requires the exact paper split.",
        },
        "split_caveats": {
            "cdr": "Uses local downstream CDR inputs; SAb23H2 and SAbDab metrics are directly comparable.",
        "light_chain_pairing": "Uses exact uploaded OAS holdout CSV (500 rows). Correct full-length generation config: gumbel_argmax, max_iter=32.",
        "humanization": "Uses exact uploaded 27-pair humanisation split with gumbel_argmax/max_iter=32; BioPhi OASis via public endpoint.",
        },
        "task_verdicts": {
            "cdr": {
                "verdict": _verdict(cdr_rows, exact_split=True),
                "within_tolerance": sum(row["within_tolerance"] for row in cdr_rows),
                "total": len(cdr_rows),
            },
            "light_chain_pairing_prompt3": {
                "verdict": _verdict(
                    [row for row in light_rows if row["task"] == "light_chain_pairing_prompt3"],
                    exact_split=True,
                    critical_metrics={"ImmunoMatch", "Div"},
                ),
                "within_tolerance": sum(
                    row["within_tolerance"]
                    for row in light_rows
                    if row["task"] == "light_chain_pairing_prompt3"
                ),
                "total": sum(1 for row in light_rows if row["task"] == "light_chain_pairing_prompt3"),
            },
            "light_chain_pairing_no_prompt": {
                "verdict": _verdict(
                    [row for row in light_rows if row["task"] == "light_chain_pairing_no_prompt"],
                    exact_split=False,
                    critical_metrics={"ImmunoMatch", "Div", "W_property"},
                ),
                "within_tolerance": sum(
                    row["within_tolerance"]
                    for row in light_rows
                    if row["task"] == "light_chain_pairing_no_prompt"
                ),
                "total": sum(1 for row in light_rows if row["task"] == "light_chain_pairing_no_prompt"),
            },
            "humanization": {
                "verdict": _verdict(human_rows, exact_split=True, critical_metrics={"OASis_score_or_identity"})
                if human_rows
                else "pending_biophi_oasis",
                "within_tolerance": sum(row["within_tolerance"] for row in human_rows),
                "total": len(human_rows),
            },
        },
        "rows": grouped,
    }


def _fmt(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def write_markdown(check: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Ophiuchus-Ab Paper Consistency Check",
        "",
        "This file compares the local generation metrics against the paper reference values.",
        "Numeric tolerance does not override split provenance: older argmax/max_iter=4 runs are kept in the summary for reference only.",
        "",
        "## Verdicts",
        "",
        "| Task | Verdict | Within tolerance | Split note |",
        "| --- | --- | ---: | --- |",
    ]
    caveats = check["split_caveats"]
    for task, verdict in check["task_verdicts"].items():
        if task.startswith("light_chain"):
            split_note = caveats["light_chain_pairing"]
        elif task == "humanization":
            split_note = caveats["humanization"]
        else:
            split_note = caveats["cdr"]
        lines.append(
            f"| {task} | {verdict['verdict']} | {verdict['within_tolerance']}/{verdict['total']} | {split_note} |"
        )

    for section, rows in check["rows"].items():
        lines.extend([
            "",
            f"## {section.replace('_', ' ').title()}",
            "",
            "| Task | Metric | Paper | Current | Delta | Tol. | Pass |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in rows:
            ok = "yes" if row["within_tolerance"] else "no"
            lines.append(
                f"| {row['task']} | {row['metric']} | {_fmt(row['paper'])} | {_fmt(row['current'])} | {_fmt(row['delta'])} | {_fmt(row['tolerance'])} | {ok} |"
            )
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local Ophiuchus-Ab generation metrics against paper values.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    check = build_check(summary)
    args.output_json.write_text(json.dumps(check, indent=2, sort_keys=True) + "\n")
    write_markdown(check, args.output_md)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    for task, verdict in check["task_verdicts"].items():
        print(f"{task}: {verdict['verdict']} ({verdict['within_tolerance']}/{verdict['total']})")


if __name__ == "__main__":
    main()
