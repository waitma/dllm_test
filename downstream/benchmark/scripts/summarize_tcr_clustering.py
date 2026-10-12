#!/usr/bin/env python
"""Consolidate the T2 clustering results into one comparison table.

Reads the nine imported baselines (``outputs/tcr_clustering/baseline_comparison.csv``)
and every embedding trade-off curve (``outputs/tcr_clustering/<tag>/curve.csv``),
then reports each embedder's purity interpolated at matched retention levels --
the operating points of the key baselines -- so embedding methods and the fixed
baselines can be compared apples-to-apples on the same Purity-Retention plane.

Writes ``outputs/tcr_clustering/embed_threshold_summary.csv`` and prints a table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
OUT = BENCH / "outputs" / "tcr_clustering"
PAPER_CSV = BENCH / "outputs" / "external" / "paper_reported_baselines.csv"

# Matched-retention anchors = the pair-universe retentions of the key baselines.
ANCHORS = {
    "local-pair ret~0.08 (clusTCR)": 0.08,
    "local-pair ret~0.19 (GLIPH2)": 0.19,
    "local-pair ret~0.25 (HD/LD/GIANA)": 0.25,
    "local-pair ret~0.39 (TCRdist3)": 0.39,
    "local-pair ret~0.82 (DeepTCR)": 0.82,
}
EMBED_TAGS = [
    ("kmer_thr", "k-mer(3)"),
    ("esm2_150m_thr", "ESM2-150M"),
    ("ophiuchus_thr", "Ophiuchus-Ab"),
    ("tcrvalid", "TCR-VALID (our protocol)"),
    # OUR model, post-LLaDA (single-chain LLaDA decoder final hidden states) — the
    # correct our-model口径.
    ("ours_postllada_esmc300m_cmp500k", "Ours-BioSeq esmc300m-cmp500k (post-LLaDA)"),
    ("ours_postllada_esmc600m_cmp500k", "Ours-BioSeq esmc600m-cmp500k (post-LLaDA)"),
    ("ours_postllada_esmc300m_mint", "Ours-BioSeq esmc300m-mint (post-LLaDA)"),
    # OUR model, ESMC-encoder-only (legacy, comparison only — NOT post-LLaDA).
    ("ours_esmc300m_cmp500k", "Ours-BioSeq esmc300m-cmp500k (enc-only)"),
    ("ours_esmc600m_cmp500k", "Ours-BioSeq esmc600m-cmp500k (enc-only)"),
    ("ours_esmc300m_mint", "Ours-BioSeq esmc300m-mint (enc-only)"),
]


def _interp_purity(curve: pd.DataFrame, target_ret: float) -> float:
    """Purity at a target retention, linearly interpolated along the curve."""
    c = curve.sort_values("retention")
    r = c["retention"].to_numpy()
    p = c["purity"].to_numpy()
    if target_ret < r.min() or target_ret > r.max():
        return float("nan")
    return float(np.interp(target_ret, r, p))


def main():
    rows = []
    for tag, label in EMBED_TAGS:
        cpath = OUT / tag / "curve.csv"
        if not cpath.exists():
            print(f"  [skip] {tag}: no curve.csv")
            continue
        curve = pd.read_csv(cpath)
        rec = {
            "method": label,
            "tag": tag,
            "evidence_type": "local_reimplementation",
            "protocol_alignment": "not_comparable",
            "citation": "doi:10.1093/nargab/lqaf150",
        }
        for name, r in ANCHORS.items():
            rec[name] = round(_interp_purity(curve, r), 3)
        # best purity in the "informative" mid band (retention 0.15-0.45).
        band = curve[(curve["retention"] >= 0.15) & (curve["retention"] <= 0.45)]
        rec["best_purity@ret0.15-0.45"] = round(band["purity"].max(), 3) if len(band) else float("nan")
        rec["auc"] = round(float(np.trapz(
            curve.sort_values("retention")["purity"],
            curve.sort_values("retention")["retention"])), 3)
        rows.append(rec)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "embed_threshold_summary.csv", index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(tbl.to_string(index=False))
    print(f"\n-> {OUT}/embed_threshold_summary.csv")

    # Paper-native operating points are the primary external baseline.  The
    # pair-universe rows below are a local hybrid and must not be called the
    # paper's nine-method table.
    if PAPER_CSV.is_file():
        paper = pd.read_csv(PAPER_CSV)
        paper = paper[paper["benchmark"] == "t2_clustering"].copy()
        paper.to_csv(OUT / "baseline_paper_reported.csv", index=False)
        pp = paper.pivot_table(index="method", columns="metric", values="value", aggfunc="first")
        print("\n=== paper-reported native-protocol baselines ===")
        print(pp[["retention", "purity"]].round(3).to_string())

    # Also echo the local fixed-baseline single points for context.
    base = OUT / "baseline_comparison.csv"
    if base.exists():
        b = pd.read_csv(base)[["model", "retention", "purity", "sensitivity_top6",
                               "pct_clusters_purity_gt90"]]
        print("\n=== local hybrid pair-universe rescoring (NOT paper-native) ===")
        with pd.option_context("display.width", 240):
            print(b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
