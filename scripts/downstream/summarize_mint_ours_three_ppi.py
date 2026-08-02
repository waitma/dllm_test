"""Combine formal BioSeq results with the eight paper-reported MINT baselines."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from downstream.mint_tasks.extract_embeddings import RESULTS_ROOT, model_cache_name


ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
CHECKPOINT = ROOT / "output/grammar_v2_esmc300m_integrated_llada_7l_step121000/best.pt"
MODEL = f"grammar:{CHECKPOINT}"
PROTOCOL_TAG = "ours-step121000-paper3ppi-v1-l1024-h640-e100-r3-global"
OUT_DIR = RESULTS_ROOT / "_ours_step121000_official3ppi"
PAPER_TABLE = RESULTS_ROOT / "_selected_baseline_results.csv"
TASKS = {
    "HumanPPI": {
        "paper_task": "HumanPPI",
        "metric": "accuracy",
        "mean_key": "best_test_Accuracy_mean",
        "std_key": "best_test_Accuracy_std",
        "paper_comparable": True,
        "note": "exact public MINT fixed split; BioSeq native 1024 residues per chain",
    },
    "YeastPPI": {
        "paper_task": "YeastPPI",
        "metric": "accuracy",
        "mean_key": "best_test_Accuracy_mean",
        "std_key": "best_test_Accuracy_std",
        "paper_comparable": True,
        "note": "exact public MINT fixed split; BioSeq native 1024 residues per chain",
    },
    "Bernett": {
        "paper_task": "Gold-standard PPI",
        "metric": "auprc",
        "mean_key": "best_test_AUPRC_mean",
        "std_key": "best_test_AUPRC_std",
        "paper_comparable": False,
        "note": "public notebook has 163,192 train rows, +173 versus paper; not paper-exact",
    },
}


def _ours_row(task: str, spec: dict) -> dict:
    cache_name = model_cache_name(MODEL, False, None, PROTOCOL_TAG)
    result_dir = RESULTS_ROOT / task / cache_name
    manifest_path = result_dir / "formal_result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        raise RuntimeError(f"incomplete formal result: {manifest_path}")
    metrics_path = Path(manifest["metrics_path"])
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))[-1]
    return {
        "task": spec["paper_task"],
        "method": "BioSeq step121000",
        "metric": spec["metric"],
        "value": float(summary[spec["mean_key"]]),
        "std": float(summary[spec["std_key"]]),
        "source_label": "[C]",
        "claim_status": "local_formal_control",
        "paper_comparable": spec["paper_comparable"],
        "notes": spec["note"],
        "source_location": str(metrics_path),
    }


def main() -> None:
    paper = pd.read_csv(PAPER_TABLE)
    paper = paper[
        paper["task"].isin(spec["paper_task"] for spec in TASKS.values())
    ].copy()
    # The selected-baseline table predates this Ours comparison column.  Every
    # row selected here is a direct paper Source Data value, so make that
    # comparability explicit instead of assuming the optional column exists.
    if "paper_comparable" not in paper.columns:
        paper["paper_comparable"] = True
    paper = paper[[
        "task", "method", "metric", "value", "std", "source_label",
        "claim_status", "paper_comparable", "notes", "source_location",
    ]]
    ours = pd.DataFrame([_ours_row(task, spec) for task, spec in TASKS.items()])
    # Keep every paper baseline together and place Ours last in each task block.
    # Sorting all methods only by score would interleave the local control with
    # paper-reported values and obscure the evidence boundary.
    table = pd.concat([paper, ours], ignore_index=True)
    task_order = {name: idx for idx, name in enumerate(
        ["HumanPPI", "YeastPPI", "Gold-standard PPI"]
    )}
    table["_task_order"] = table["task"].map(task_order)
    table["_source_order"] = (table["source_label"] != "[P]").astype(int)
    table = table.sort_values(
        ["_task_order", "_source_order", "value"],
        ascending=[True, True, False],
    ).drop(columns=["_task_order", "_source_order"])

    comparisons = []
    for task, spec in TASKS.items():
        display = spec["paper_task"]
        paper_task = paper[paper["task"] == display]
        ours_task = ours[ours["task"] == display].iloc[0]
        best = paper_task.loc[paper_task["value"].idxmax()]
        comparisons.append(
            {
                "task": display,
                "metric": spec["metric"],
                "ours_value": float(ours_task["value"]),
                "ours_std": float(ours_task["std"]),
                "best_paper_method": best["method"],
                "best_paper_value": float(best["value"]),
                "best_paper_std": float(best["std"]),
                "ours_minus_best_paper": float(ours_task["value"] - best["value"]),
                "paper_comparable": bool(spec["paper_comparable"]),
                "notes": spec["note"],
            }
        )
    comparison = pd.DataFrame(comparisons)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "three_ppi_full_comparison.csv", index=False)
    comparison.to_csv(OUT_DIR / "ours_vs_best_paper.csv", index=False)
    payload = {
        "status": "success",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_step": 121000,
        "checkpoint_sha256": "51f0eee5fa7b127a0487a2ea8fdbc322bda6f6908ff2662087bd87d4980d4613",
        "protocol_tag": PROTOCOL_TAG,
        "source_labels": {
            "[P]": "paper-reported baseline; not rerun locally for these three tasks",
            "[C]": "our formal local BioSeq control run",
        },
        "comparisons": comparisons,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# BioSeq step121000 — MINT three-PPI comparison",
        "",
        "`[P]` is copied from the MINT paper Source Data; `[C]` is our formal local control run.",
        "",
    ]
    for display in ["HumanPPI", "YeastPPI", "Gold-standard PPI"]:
        task_table = table[table["task"] == display]
        lines.extend([
            f"## {display}",
            "",
            "| Source | Method | Metric | Mean | Std |",
            "|---|---|---:|---:|---:|",
        ])
        for row in task_table.itertuples(index=False):
            lines.append(
                f"| {row.source_label} | {row.method} | {row.metric} | "
                f"{row.value:.6f} | {row.std:.6f} |"
            )
        lines.append("")
    lines.extend([
        "Gold-standard `[C]` is not paper-exact because the public notebook has +173 training rows.",
        "",
    ])
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
