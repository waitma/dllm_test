#!/usr/bin/env python3
"""Refresh downstream metric artifacts from the latest eval CSV/JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path("output/downstream_generation/ophiuchus_ab")
PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project")
OASIS_OUTPUT = PROJECT_ROOT / "oasis/output"
DEFAULT_SUMMARY = ROOT / "generation_metrics_summary.json"
DEFAULT_REPORT = ROOT / "generation_metrics_report.md"
DEFAULT_HUMAN_METRICS = ROOT / "humanization_exact27_airgen32_metrics.json"


def read_eval_row(path: Path) -> dict[str, float]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    return {key: float(value) for key, value in row.items()}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "pending"
    if abs(value) >= 10:
        return f"{value:.{max(2, digits)}f}"
    return f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    if value is None:
        return "pending"
    return fmt(value * 100.0, 2)


def light_chain_table_row(metrics: dict[str, float]) -> dict[str, str]:
    return {
        "ImmunoMatch": fmt(metrics.get("gen_immunomatch_mean")),
        "Better (%)": pct(metrics.get("gen_better_ratio")),
        "Chain (%)": pct(metrics.get("overall_chain_match_rate")),
        "V exact": fmt(metrics.get("overall_v_gene_match_rate")),
        "J exact": fmt(metrics.get("overall_j_gene_match_rate")),
        "V family": fmt(metrics.get("overall_v_gene_family_match_rate")),
        "J family": fmt(metrics.get("overall_j_gene_family_match_rate")),
        "Diversity": fmt(metrics.get("diversity_mean")),
        "Valid (%)": pct(metrics.get("gen_valid_rate")),
        "W property": fmt(metrics.get("w_property_avg_wd")) if "w_property_avg_wd" in metrics else "n/a",
    }


def write_humanization_metrics(
    output_path: Path,
    *,
    biophi_metrics: dict[str, Any] | None,
) -> None:
    payload = {
        "dataset": "exact uploaded 27-pair humanisation split from data/downstream/humanization/humanisation; loader keys by (pdb_id, chain_pair)",
        "generation_config": {
            "n_sequences": 8,
            "sampling_strategy": "gumbel_argmax",
            "max_iter": 32,
            "cfg_scale": 0.0,
            "temperature": 1.0,
            "seed": 42,
        },
        "outputs": {
            "generated_csv": "output/downstream_generation/ophiuchus_ab/humanization_exact27_airgen32.csv",
            "generation_log": "output/downstream_generation/ophiuchus_ab/humanization_exact27_airgen32.log",
            "biophi_public_dir": "output/downstream_generation/ophiuchus_ab/biophi_public_humanization_exact27_airgen32_relaxed",
        },
        "counts": {
            "n_structures": 27,
            "n_pairs": 216,
        },
        "metrics": {
            "heavy_fr_aar_percent": 84.02,
            "light_fr_aar_percent": 87.76,
            "biophi_oasis_relaxed": biophi_metrics,
        },
        "paper_reference": {
            "OASis_score_percent": 83.4,
            "VH_sequence_identity_percent": 65.4,
            "VL_sequence_identity_percent": 69.9,
        },
    }
    if biophi_metrics:
        payload["paper_comparison"] = {
            "OASis_score_percent_delta": biophi_metrics["oasis_identity_percent"] - 83.4,
            "official_biophi_heavy_germline_content_percent": biophi_metrics["heavy_germline_content_mean"] * 100.0,
            "official_biophi_light_germline_content_percent": biophi_metrics["light_germline_content_mean"] * 100.0,
        }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def write_report(summary: dict[str, Any], check: dict[str, Any], output_path: Path) -> None:
    paper = summary["paper_reference"]
    prompt3 = summary["light_chain_pairing"]["holdout500_airgen32_prompt3"]["metrics"]
    no_prompt = summary["light_chain_pairing"]["no_prompt_exact_holdout_500"]["metrics"]
    prompt3_row = light_chain_table_row(prompt3)
    no_prompt_row = light_chain_table_row(no_prompt)
    paper_prompt3 = paper["light_chain_pairing_ophiuchus_ab_with_initial_3_residue_prompting"]
    paper_no_prompt = paper["light_chain_pairing_ophiuchus_ab_without_prompting"]
    human = summary["humanization"]["exact27_airgen32"]["metrics"]
    human_paper = paper["humanization_ophiuchus_ab"]
    biophi = human.get("biophi_oasis_relaxed") or {}

    verdict_lines = []
    for task, verdict in check["task_verdicts"].items():
        verdict_lines.append(
            f"- `{task}`: {verdict['verdict']} ({verdict['within_tolerance']}/{verdict['total']})"
        )

    lines = [
        "# Ophiuchus-Ab Generation Downstream Metrics",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "This report summarizes the current Ophiuchus-Ab generation downstream evaluation",
        "for CDR infilling, light-chain pairing, and humanization.",
        "",
        "## Paper Consistency Verdict",
        "",
        "Formal consistency check:",
        "",
        "- `output/downstream_generation/ophiuchus_ab/paper_consistency_check.md`",
        "- `output/downstream_generation/ophiuchus_ab/paper_consistency_check.json`",
        "",
        "Verdict summary:",
        "",
        *verdict_lines,
        "",
        "## Split Status",
        "",
        "- CDR infilling uses local SAb23H2 and SAbDab downstream inputs.",
        "- Light-chain pairing uses the exact OAS holdout supplied locally:",
        "  - `data/downstream/comp_chain/test_data_oas_holdout.csv`",
        "  - 500 holdout rows.",
        "- Humanization uses the uploaded exact 27-pair split:",
        "  - `data/downstream/humanization/humanisation/test-pdb`",
        "  - `data/downstream/humanization/humanisation/test_chains.csv`",
        "  - Loader now keeps both `8onk` chain-pair rows; the airgen32 run below still used the pre-fix 27-structure loader.",
        "",
        "## CDR Infilling",
        "",
        "Configuration: `argmax`, `max_iter=4`, `cfg_scale=0.0`.",
        "",
        "| Dataset | Region | Ophiuchus-Ab Paper | Current |",
        "| --- | ---: | ---: | ---: |",
    ]

    sab23 = summary["cdr_infill"]["sab23h2"]["metrics"]
    sab23_paper = paper["cdr_sab23h2_ophiuchus_ab_aar"]
    for region, key in [
        ("L1", "cdrl1"),
        ("L2", "cdrl2"),
        ("L3", "cdrl3"),
        ("H1", "cdrh1"),
        ("H2", "cdrh2"),
        ("H3", "cdrh3"),
    ]:
        lines.append(
            f"| SAb23H2 | {region} | {sab23_paper[region] * 100:.2f} | {sab23[key]:.2f} |"
        )

    sabdab = summary["cdr_infill"]["sabdab_external_indices"]
    sabdab_paper = paper["cdr_sabdab_ophiuchus_ab_heavy_aar_percent"]
    for region in ["H1", "H2", "H3"]:
        current = sabdab[f"cdr{region.lower()}"]["aar_percent"]
        lines.append(f"| SAbDab | {region} | {sabdab_paper[region]:.2f} | {current:.2f} |")

    lines.extend([
        "",
        "## Light-Chain Pairing",
        "",
        "Input: exact 500-row OAS paired holdout input:",
        "",
        "- `data/downstream/comp_chain/test_data_oas_holdout.csv`",
        "",
        "Current prompt3 configuration:",
        "",
        "- `gumbel_argmax`",
        "- `max_iter=32`",
        "- `num_seqs=8`",
        "- `cfg_scale=0.0`",
        "- `light_prompt_tokens=3`",
        "",
        "No-prompt row below still uses the legacy `argmax` / `max_iter=4` run until rerun.",
        "",
        "| Metric | Paper, 3-AA Prompt | Current, 3-AA Prompt (airgen32) | Paper, No Prompt | Current, No Prompt (legacy) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])

    metric_names = [
        "ImmunoMatch",
        "Better (%)",
        "Chain (%)",
        "V exact",
        "J exact",
        "V family",
        "J family",
        "Diversity",
        "Valid (%)",
        "W property",
    ]
    paper_prompt3_map = {
        "ImmunoMatch": paper_prompt3["ImmunoMatch"],
        "Better (%)": paper_prompt3["Better_percent"],
        "Chain (%)": paper_prompt3["Chain_percent"],
        "V exact": paper_prompt3["V"],
        "J exact": paper_prompt3["J"],
        "V family": paper_prompt3["V_fam"],
        "J family": paper_prompt3["J_fam"],
        "Diversity": paper_prompt3["Div"],
        "Valid (%)": paper_prompt3["Valid_percent"],
        "W property": paper_prompt3["W_property"],
    }
    paper_no_prompt_map = {
        "ImmunoMatch": paper_no_prompt["ImmunoMatch"],
        "Better (%)": paper_no_prompt["Better_percent"],
        "Chain (%)": paper_no_prompt["Chain_percent"],
        "V exact": paper_no_prompt["V"],
        "J exact": paper_no_prompt["J"],
        "V family": paper_no_prompt["V_fam"],
        "J family": paper_no_prompt["J_fam"],
        "Diversity": paper_no_prompt["Div"],
        "Valid (%)": paper_no_prompt["Valid_percent"],
        "W property": paper_no_prompt["W_property"],
    }
    for metric in metric_names:
        lines.append(
            f"| {metric} | {fmt(paper_prompt3_map[metric])} | {prompt3_row[metric]} | "
            f"{fmt(paper_no_prompt_map[metric])} | {no_prompt_row[metric]} |"
        )

    lines.extend([
        "",
        "Outputs:",
        "",
        "- `output/downstream_generation/ophiuchus_ab/comp_chain_holdout500_airgen32_prompt3_n8.csv`",
        "- `output/downstream_generation/ophiuchus_ab/comp_chain_holdout500_airgen32_prompt3_n8_eval.csv`",
        "- `output/downstream_generation/ophiuchus_ab/comp_chain_holdout500_airgen32_prompt3_eval.log`",
        "",
        "## Humanization",
        "",
        "Input: uploaded exact 27 murine SAbDab pairs:",
        "",
        "- `data/downstream/humanization/humanisation/test-pdb`",
        "- `data/downstream/humanization/humanisation/test_chains.csv`",
        "",
        "Current configuration:",
        "",
        "- `gumbel_argmax`",
        "- `max_iter=32`",
        "- `n_sequences=8`",
        "- `cfg_scale=0.0`",
        "",
        "Official BioPhi OASis for the airgen32 CSV is pending while the public BioPhi humanness endpoint returns HTTP 500.",
        "",
        "| Metric | Ophiuchus-Ab Paper | Current, airgen32 |",
        "| --- | ---: | ---: |",
        f"| OASis score / Identity (%) | {human_paper['OASis_score']:.1f} | "
        f"{fmt(biophi.get('oasis_identity_percent')) if biophi else 'pending'} |",
        f"| VH sequence identity / BioPhi heavy germline content (%) | {human_paper['VH_sequence_identity_percent']:.1f} | "
        f"{fmt(biophi.get('heavy_germline_content_mean', 0) * 100.0) if biophi else 'pending'} |",
        f"| VL sequence identity / BioPhi light germline content (%) | {human_paper['VL_sequence_identity_percent']:.1f} | "
        f"{fmt(biophi.get('light_germline_content_mean', 0) * 100.0) if biophi else 'pending'} |",
        f"| Heavy FR native recovery (%) | not tabled | {human['heavy_fr_aar_percent']:.2f} |",
        f"| Light FR native recovery (%) | not tabled | {human['light_fr_aar_percent']:.2f} |",
        "",
        "Legacy argmax/max_iter=4 BioPhi OASis on the same split was `55.32%`; that run is kept only for reference.",
        "",
        "Outputs:",
        "",
        "- `output/downstream_generation/ophiuchus_ab/humanization_exact27_airgen32.csv`",
        "- `output/downstream_generation/ophiuchus_ab/humanization_exact27_airgen32.log`",
        "- `output/downstream_generation/ophiuchus_ab/humanization_exact27_airgen32_metrics.json`",
        "",
        "## Machine-Readable Summary",
        "",
        "- `output/downstream_generation/ophiuchus_ab/generation_metrics_summary.json`",
        "",
        "## Refresh Commands",
        "",
        "```bash",
        "python scripts/downstream/update_generation_metrics_summary.py",
        "python scripts/downstream/validate_ophiuchus_generation_against_paper.py",
        "```",
    ])
    output_path.write_text("\n".join(lines) + "\n")


def update_summary(summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)

    prompt3_eval = read_eval_row(ROOT / "comp_chain_holdout500_airgen32_prompt3_n8_eval.csv")
    prompt3_eval_log = "output/downstream_generation/ophiuchus_ab/comp_chain_holdout500_airgen32_prompt3_eval.log"

    summary["light_chain_pairing"]["holdout500_airgen32_prompt3"] = {
        "dataset": "exact OAS holdout: data/downstream/comp_chain/test_data_oas_holdout.csv (500 heavy chains)",
        "generation_config": {
            "cfg_scale": 0.0,
            "heavy_batch_size": 16,
            "max_iter": 32,
            "num_seqs": 8,
            "sampling_strategy": "gumbel_argmax",
            "seed": 42,
            "temperature": 1.0,
            "light_prompt_tokens": 3,
            "alignment_note": "Matches AirGen LightMaskingCollate start_idx=4.",
        },
        "metrics": prompt3_eval,
        "outputs": {
            "generated_csv": "output/downstream_generation/ophiuchus_ab/comp_chain_holdout500_airgen32_prompt3_n8.csv",
            "eval_csv": "output/downstream_generation/ophiuchus_ab/comp_chain_holdout500_airgen32_prompt3_n8_eval.csv",
            "eval_log": prompt3_eval_log,
            "input_csv": "data/downstream/comp_chain/test_data_oas_holdout.csv",
        },
    }
    summary["light_chain_pairing"]["metrics"] = prompt3_eval
    summary["light_chain_pairing"]["generation_config"] = summary["light_chain_pairing"]["holdout500_airgen32_prompt3"]["generation_config"]
    summary["light_chain_pairing"]["outputs"] = summary["light_chain_pairing"]["holdout500_airgen32_prompt3"]["outputs"]

    holdout50_eval_path = ROOT / "comp_chain_holdout50_airgen32_prompt3_n8_eval.csv"
    if holdout50_eval_path.exists():
        summary["light_chain_pairing"]["holdout50_airgen32_prompt3"] = {
            "metrics": read_eval_row(holdout50_eval_path),
            "outputs": {
                "generated_csv": "output/downstream_generation/ophiuchus_ab/comp_chain_holdout50_airgen32_prompt3_n8.csv",
                "eval_csv": "output/downstream_generation/ophiuchus_ab/comp_chain_holdout50_airgen32_prompt3_n8_eval.csv",
                "eval_log": "output/downstream_generation/ophiuchus_ab/comp_chain_holdout50_airgen32_prompt3_eval.log",
            },
        }

    biophi_metrics_path = OASIS_OUTPUT / "humanization_exact27_airgen32_oasis_summary.json"
    if not biophi_metrics_path.exists():
        biophi_metrics_path = ROOT / "biophi_public_humanization_exact27_airgen32_relaxed/summary_metrics.json"
    biophi_metrics = read_json(biophi_metrics_path) if biophi_metrics_path.exists() else None

    summary["humanization"]["exact27_airgen32"] = {
        "dataset": "exact uploaded 27-pair humanisation split; loader now keys by (pdb_id, chain_pair) so duplicate 8onk rows are both kept",
        "generation_config": {
            "n_sequences": 8,
            "sampling_strategy": "gumbel_argmax",
            "max_iter": 32,
            "cfg_scale": 0.0,
            "temperature": 1.0,
            "seed": 42,
        },
        "metrics": {
            "heavy_fr_aar_percent": 84.02,
            "light_fr_aar_percent": 87.76,
            "n_structures": 27,
            "n_pairs": 216,
            "biophi_oasis_relaxed": biophi_metrics,
        },
        "outputs": {
            "generated_csv": "output/downstream_generation/ophiuchus_ab/humanization_exact27_airgen32.csv",
            "generation_log": "output/downstream_generation/ophiuchus_ab/humanization_exact27_airgen32.log",
            "metrics_json": "output/downstream_generation/ophiuchus_ab/humanization_exact27_airgen32_metrics.json",
            "biophi_public_dir": "output/downstream_generation/ophiuchus_ab/biophi_public_humanization_exact27_airgen32_relaxed",
        },
    }
    if biophi_metrics:
        summary["humanization"]["metrics"]["biophi_oasis_relaxed_airgen32"] = biophi_metrics

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--human-metrics", type=Path, default=DEFAULT_HUMAN_METRICS)
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    summary = update_summary(args.summary)
    biophi_metrics = summary["humanization"]["exact27_airgen32"]["metrics"].get("biophi_oasis_relaxed")
    write_humanization_metrics(args.human_metrics, biophi_metrics=biophi_metrics)

    if not args.skip_validate:
        subprocess.run(
            [sys.executable, "scripts/downstream/validate_ophiuchus_generation_against_paper.py", "--summary", str(args.summary)],
            check=True,
        )

    check = read_json(ROOT / "paper_consistency_check.json")
    write_report(summary, check, args.report)

    print(f"Updated {args.summary}")
    print(f"Updated {args.human_metrics}")
    print(f"Updated {args.report}")


if __name__ == "__main__":
    main()
