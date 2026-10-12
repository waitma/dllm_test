#!/usr/bin/env python
"""Report NbBench paper baselines separately from local probe diagnostics.

Scans ``outputs/nbbench/<task>/<model>/metrics.json`` (scalar via nbbench/run.py,
residue via nbbench/run_residue.py, generative via nbbench/run_generative.py) and
prints / writes a task x model table of primary metrics. Pure reader -- never
recomputes; anything missing is left blank so the table stays honest.  The two
protocols are deliberately kept in separate JSON objects and tables.

    python scripts/summarize_nbbench.py            # console + outputs/nbbench/_leaderboard.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
OUT = BENCH / "outputs" / "nbbench"
OFFICIAL = BENCH / "outputs" / "external" / "nbbench_official.csv"

# Canonical task order + primary metric per task (higher-is-better throughout).
TASKS = [
    ("VRClassification", "acc"),
    ("CDRInfilling", "br_masked"),
    ("Paratope", "auprc"),
    ("SARS-CoV-2", "probe_auroc"),
    ("hIL6", "probe_auroc"),
    ("hTNFa", "probe_auroc"),
    ("polyreaction", "probe_auroc"),
    ("nanobody_type", "probe_acc"),
    ("thermo-tm", "spearman"),
    ("thermo-seq", "spearman"),
    ("vhh_affinity-seq", "spearman"),
    ("vhh_affinity-score", "spearman"),
]

# Preferred model display order (spec -> label).
MODEL_ORDER = [
    ("esm2_150m", "ESM2-150M"),
    ("esm2_650m", "ESM2-650M"),
    ("protbert", "ProtBERT"),
    ("ablang_h", "AbLang-H"),
    ("antiberta2", "AntiBERTa2"),
    ("antiberta2_cssp", "AntiBERTa2-CSSP"),
    ("igbert", "IgBERT"),
    ("antiberty", "AntiBERTy"),
    ("nanobert", "NanoBERT"),
    ("vhhbert", "VHHBERT"),
    ("kmer", "k-mer(3)"),
    ("onehot", "one-hot(±3)"),
]

# Friendly display labels for our own model dirs (created by run*.py --tag).
# Kept separate from MODEL_ORDER so the discovery/ordering logic (which appends
# any ``bioseq*`` / ``gen_*`` dir at the end) is unaffected -- these only rename
# rows for the console + leaderboard readout.
OURS_LABELS = {
    # POST-LLaDA (grammar decoder) -- the headline Ours-BioSeq口径: scalar +
    # residue probe features have passed through the trained LLaDA decoder.
    "ours_esmc600m_llada": "Ours-BioSeq (post-LLaDA) esmc600m",
    "ours_esmc300m_llada": "Ours-BioSeq (post-LLaDA) esmc300m",
    "ours_esmc300m_mint_llada": "Ours-BioSeq (post-LLaDA) esmc300m-mint",
    # Generative-native CDRInfilling (already decodes through LLaDA diffusion).
    "gen_ours_esmc600m_llada": "Ours-BioSeq gen (post-LLaDA) esmc600m",
    "gen_ours_esmc300m_llada": "Ours-BioSeq gen (post-LLaDA) esmc300m",
    "gen_ours_esmc300m_mint_llada": "Ours-BioSeq gen (post-LLaDA) esmc300m-mint",
}


def _read(task: str, model: str):
    f = OUT / task / model / "metrics.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def collect() -> dict:
    # Discover every model dir present under any task.
    seen = {}
    for task, _ in TASKS:
        d = OUT / task
        if not d.exists():
            continue
        for md in sorted(p.name for p in d.iterdir() if p.is_dir()):
            seen.setdefault(md, True)
    def _is_ours(m):
        return m.startswith("ours_") or m.startswith("gen_ours_")

    ordered = [m for m, _ in MODEL_ORDER if m in seen]
    ordered += [m for m in seen if m not in dict(MODEL_ORDER) and not _is_ours(m)]
    # Our-model rows last: post-LLaDA probing (ours_*), then generative (gen_ours_*).
    ordered += [m for m in seen if m.startswith("ours_")]
    ordered += [m for m in seen if m.startswith("gen_ours_")]

    table = {}
    for model in ordered:
        row = {}
        for task, metric in TASKS:
            d = _read(task, model)
            if d is None:
                row[task] = None
            else:
                val = d.get("primary", d.get(metric))
                row[task] = val
        table[model] = row
    return table


def fmt(v):
    if v is None:
        return "   -  "
    return f"{v:.4f}"


def collect_paper() -> dict:
    if not OFFICIAL.is_file():
        return {}
    import pandas as pd

    df = pd.read_csv(OFFICIAL)
    table = {}
    for model, group in df.groupby("model", sort=False):
        table[model] = dict(zip(group["task"], group["value"]))
    return table


def main():
    table = collect()
    paper = collect_paper()
    if paper:
        print("=== PAPER-REPORTED BASELINES (NbBench Table 5; MLP head, 3 seeds) ===")
        paper_tasks = [task for task, _ in TASKS if task != "hTNFa"]
        header = f"{'model':18s} " + " ".join(f"{t[:9]:>9s}" for t in paper_tasks)
        print(header)
        print("-" * len(header))
        for model, row in paper.items():
            print(f"{model:18s} " + " ".join(f"{fmt(row.get(t)):>9s}" for t in paper_tasks))
        print()

    print("=== LOCAL DIAGNOSTIC ONLY (sklearn probe; not Table-5 comparable) ===")
    header = f"{'model':18s} " + " ".join(f"{t[:9]:>9s}" for t, _ in TASKS)
    print(header)
    print("-" * len(header))
    label = dict(MODEL_ORDER)
    label.update(OURS_LABELS)
    for model, row in table.items():
        name = label.get(model, model)
        print(f"{name:18s} " + " ".join(f"{fmt(row[t]):>9s}" for t, _ in TASKS))

    payload = {
        "tasks": {t: m for t, m in TASKS},
        "paper_reported": {
            "evidence_type": "paper_reported",
            "protocol": "frozen backbone + learned MLP head, three seeds",
            "citation": "doi:10.1088/2632-2153/ae20ec; arXiv:2505.02022",
            "table": paper,
        },
        "local_diagnostic": {
            "evidence_type": "local_reimplementation",
            "protocol_alignment": "not_comparable",
            "protocol": "sklearn LogReg/Ridge, single seed; validation ignored",
            "table": table,
        },
        # Backward-compatible alias; consumers must read local_diagnostic for
        # provenance and must not compare it numerically to paper_reported.
        "table": table,
    }
    out_f = OUT / "_leaderboard.json"
    out_f.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_f}")

    # Also report generative + developability if present.
    for gd in sorted(OUT.glob("CDRInfilling/gen_*/metrics.json")):
        d = json.loads(gd.read_text())
        print(f"\n[generative] {d.get('model')}: EM={d.get('em_masked')} "
              f"BR={d.get('br_masked')} (n={d.get('n_test')})")
        dev = d.get("developability_generated")
        if dev:
            print(f"  developability(gen): "
                  f"hallmark={dev.get('mean_vhh_hallmark'):.3f} "
                  f"liab_rate={dev.get('mean_liability_rate'):.3f} "
                  f"wh_sol={dev.get('mean_wh_solubility'):.3f}")


if __name__ == "__main__":
    sys.exit(main())
