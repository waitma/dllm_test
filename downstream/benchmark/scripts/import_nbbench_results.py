#!/usr/bin/env python
"""Freeze the official NbBench leaderboard (Table 5) into a reproducible CSV.

Unlike the IRBench-TCR baselines, NbBench does not ship a machine-readable
results file in this repo, so the primary-metric numbers below are **transcribed
verbatim** from the published main table:

    NbBench: Benchmarking Language Models for Comprehensive Nanobody Tasks
    Zhang & Tsuda, Mach. Learn.: Sci. Technol. 6(4):040502, 2025,
    doi:10.1088/2632-2153/ae20ec (published 2025-11-27; preprint arXiv:2505.02022 v1, Table 5).

Each cell is the task's *primary* metric, mean over 3 seeds (std dropped here;
see the paper for dispersion). These are **reference numbers, never recomputed**
by us -- they let ``RESULTS.md`` cite an auditable ``official ESM2-150M`` and
``official best`` column instead of quoting the paper inline.

Frozen-backbone protocol note: NbBench uses a frozen LM + trainable MLP head,
whereas our ``nbbench/run.py`` uses a frozen LM + sklearn probe. Numbers are
therefore *comparable in spirit* (both head-only) but not identical setups; the
overlap on saturated/structural tasks (VRCls, NbType, Paratope-AUROC,
CDRInf-EM) is what validates our data loading and metric definitions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
OUT = BENCH / "outputs" / "external" / "nbbench_official.csv"

# Task -> primary metric name (as reported in the paper's main table).
TASK_METRIC = {
    "VRClassification": "accuracy",
    "CDRInfilling": "blosum62_recovery",   # BR (higher better; not in [0,1])
    "SARS-CoV-2": "auroc",
    "hIL6": "auroc",
    "Paratope": "auroc",
    "thermo-seq": "spearman",
    "thermo-tm": "spearman",
    "polyreaction": "auroc",
    "nanobody_type": "accuracy",
    "vhh_affinity-seq": "spearman",
    "vhh_affinity-score": "spearman",
}

# Column order matches TASK_METRIC keys.
_COLS = list(TASK_METRIC)

# Official primary-metric means, transcribed from NbBench (MLST 6(4):040502, 2025;
# DOI 10.1088/2632-2153/ae20ec) main table, = arXiv:2505.02022 v1 Table 5.
OFFICIAL = {
    "ProtBert":        [0.9983, 1.520, 0.852, 0.878, 0.916, 0.040, 0.128, 0.837, 0.957, 0.163, 0.084],
    "ESM-2 (150M)":    [0.9989, 1.499, 0.834, 0.850, 0.922, 0.389, 0.301, 0.833, 0.994, 0.170, 0.063],
    "ESM-2 (650M)":    [0.9984, 1.551, 0.850, 0.857, 0.928, 0.375, 0.315, 0.842, 0.995, 0.183, 0.089],
    "AbLang-H":        [0.9981, 1.452, 0.884, 0.925, 0.933, 0.566, 0.533, 0.831, 0.997, 0.151, 0.106],
    "AbLang-L":        [0.9984, 1.360, 0.811, 0.914, 0.933, 0.585, 0.502, 0.819, 0.988, 0.173, 0.128],
    "AntiBERTy":       [0.9983, 1.499, 0.810, 0.871, 0.932, 0.474, 0.385, 0.828, 0.999, 0.167, 0.119],
    "AntiBERTa2":      [0.9989, 1.489, 0.824, 0.887, 0.930, 0.587, 0.585, 0.833, 0.995, 0.162, 0.088],
    "AntiBERTa2-CSSP": [0.9983, 1.484, 0.845, 0.916, 0.939, 0.575, 0.593, 0.830, 0.998, 0.149, 0.085],
    "IgBert":          [0.9981, 1.311, 0.817, 0.821, 0.921, 0.351, 0.344, 0.829, 0.993, 0.184, 0.090],
    "NanoBERT":        [0.9947, 1.524, 0.876, 0.904, 0.928, 0.440, 0.298, 0.815, 0.988, 0.061, 0.118],
    "VHHBERT":         [0.9904, 1.244, 0.800, 0.873, 0.921, 0.484, 0.532, 0.818, 0.986, 0.145, 0.115],
}

MODEL_GROUP = {
    "ProtBert": "protein", "ESM-2 (150M)": "protein", "ESM-2 (650M)": "protein",
    "AbLang-H": "antibody", "AbLang-L": "antibody", "AntiBERTy": "antibody",
    "AntiBERTa2": "antibody", "AntiBERTa2-CSSP": "antibody", "IgBert": "antibody",
    "NanoBERT": "nanobody", "VHHBERT": "nanobody",
}


def main() -> None:
    rows = []
    for model, vals in OFFICIAL.items():
        assert len(vals) == len(_COLS), f"{model}: {len(vals)} != {len(_COLS)}"
        for task, val in zip(_COLS, vals):
            rows.append({
                "model": model,
                "group": MODEL_GROUP[model],
                "task": task,
                "metric": TASK_METRIC[task],
                "value": val,
                "evidence_type": "paper_reported",
                "protocol_alignment": "exact",
                "citation": "doi:10.1088/2632-2153/ae20ec; arXiv:2505.02022",
                "source_location": "Table 5, three-seed mean",
            })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # Console summary: ESM2-150M reference + per-task best model.
    print(f"wrote {len(df)} rows -> {OUT}")
    esm = df[df.model == "ESM-2 (150M)"].set_index("task")["value"]
    print("\ntask                 metric              ESM2-150M   best (model)")
    for task in _COLS:
        sub = df[df.task == task]
        # BR / accuracy / auroc / spearman are all higher-is-better here.
        top = sub.loc[sub["value"].idxmax()]
        print(f"{task:20s} {TASK_METRIC[task]:18s} {esm[task]:>9.4f}   "
              f"{top['value']:.4f} ({top['model']})")


if __name__ == "__main__":
    main()
