#!/usr/bin/env python
"""Build the T2 (clustering) dataset from the NAR-GAB 2025 curated database.

Anchored *only* on the i3-unit / TCR_Unsupervised_Benchmark repository
(``baselines/TCR_Unsupervised_Benchmark``). We reproduce the paper's
``Analysis/database_processing.R`` curation exactly (VS==2 & AIS>4.3 & CD8 with
V/J, CDR3 length 6-23, epitopes with >=2 unique alpha/beta pairs, minus the two
documented duplicate pairs) and emit a stable, benchmark-ready dataset:

- ``data/tcr_clustering/tcrs.csv`` -- one row per *unique paired TCR* with a
  stable ``pair_id`` and columns cdr3a/cdr3b/va/ja/vb/jb/peptide. ``peptide`` is
  the representative (majority) epitope of the pair; polyspecific pairs
  (~1.7 %) collapse to their dominant epitope. Epitope labels are for
  evaluation only -- clustering never sees them. ``seq_col`` defaults to cdr3b.
- ``data/tcr_clustering/meta.json`` -- reproduced sizes, the paper's target
  sizes, repo commit, filter description and the DB-drift caveat.

The committed database file drifted slightly from the 2023 version the paper ran
on (the repo README warns of this): we reproduce 5,368 pairs / 9,370 unique
single-chain sequences / 392 epitopes vs the paper's 4,779 / 8,395 / (2,313 in
the full pooled DB). The per-epitope counts of the six highlighted epitopes match
the paper exactly, confirming the curation logic is faithful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH / "baselines" / "TCR_Unsupervised_Benchmark"
DB_FILE = REPO / "Data" / "Database" / "database_pooled_human_2023_03_15.txt"
OUT_DIR = BENCH / "data" / "tcr_clustering"

# Paper reference figures (NAR-GAB 2025, Materials & Methods / Rmd denominators).
PAPER = {
    "pooled_db_tcrs": 190670,
    "pooled_db_epitopes": 2313,
    "curated_pairs": 4779,
    "curated_unique_cdr3a": 4103,
    "curated_unique_cdr3b": 4292,
    "curated_unique_sequences": 8395,
}
# The two duplicate pairs removed by the paper's curation script.
DROP_PAIRS = ["CASSDSRGTEAFF_CASSDSRGTEAFF", "CASSPITGTGAYGYTF_CASSSVNEQYF"]
# The 10X training-set publication excluded from the pooled-DB analysis subset.
TENX_PUBMED = ("https://pages.10xgenomics.com/rs/446-PBO-704/images/"
               "10x_AN047_IP_A_New_Way_of_Exploring_Immunity_Digital.pdf")
# Six highlighted epitopes the paper reports per-epitope counts for (sanity check).
SANITY = {
    "GILGFVFTL": (706, 484, 456), "NLVPMVATV": (397, 325, 349),
    "GLCTLVAML": (215, 145, 170), "FLCMKALLL": (136, 134, 136),
    "NYNYLYRLF": (56, 55, 46), "LLFGYPVYV": (10, 4, 7),
}


# The repo is vendored as an extracted source archive (not a git clone), so we
# record the archive commit explicitly rather than probing git (which would walk
# up to an unrelated parent repository).
REPO_COMMIT = "ac767882f6bb7a23ccbbe973c5487ff58add04cf"


def _commit() -> str:
    return REPO_COMMIT


def curate(db_file: Path) -> pd.DataFrame:
    """Reproduce Analysis/database_processing.R exactly."""
    db = pd.read_csv(db_file, sep="\t", low_memory=False, encoding="latin-1")
    d = db[(db["Verified_score"] == 2)
           & (db["Identification_score"] > 4.3)
           & (db["Epitope"].notna())].copy()
    # dplyr::filter drops rows whose predicate is NA, so the R original's
    # ``filter(PubMed_ID != <10x url>)`` also removes rows with a missing
    # PubMed_ID. pandas' ``!=`` returns True for NaN and would keep them. Without
    # the notna() term this single line let 634 extra rows through and inflated
    # the curated universe to 5,368 pairs / 9,370 sequences; with it the shipped
    # database reproduces the paper's 4,779 / 8,395 / 384 epitopes exactly.
    # See scripts/t2_curate_step_align.py for the per-step alignment.
    d = d[(d["PubMed_ID"] != TENX_PUBMED) & d["PubMed_ID"].notna()]
    d = d[d["Cell_subset"] == "CD8"]
    for c in ["V_alpha", "J_alpha", "V_beta", "J_beta"]:
        d = d[d[c].notna()]
    la = d["CDR3_alpha"].astype(str).str.len()
    lb = d["CDR3_beta"].astype(str).str.len()
    d = d[la.between(6, 23) & lb.between(6, 23)]
    d["pair"] = d["CDR3_beta"].astype(str) + "_" + d["CDR3_alpha"].astype(str)
    n_pair_per_epi = d.groupby("Epitope")["pair"].transform("nunique")
    d = d[n_pair_per_epi != 1]                      # keep epitopes with >=2 pairs
    d = d[~d["pair"].isin(DROP_PAIRS)]
    return d.drop_duplicates().reset_index(drop=True)


def _pair_id(cdr3b: str, cdr3a: str) -> str:
    h = hashlib.sha1(f"{cdr3b}|{cdr3a}".encode()).hexdigest()[:12]
    return f"pair_{h}"


def build_pairs(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse to one row per unique (cdr3b, cdr3a) pair with a majority epitope."""
    d = d.rename(columns={
        "CDR3_alpha": "cdr3a", "CDR3_beta": "cdr3b",
        "V_alpha": "va", "J_alpha": "ja", "V_beta": "vb", "J_beta": "jb",
        "Epitope": "peptide",
    })
    rows, poly = [], 0
    for (cdr3b, cdr3a), g in d.groupby(["cdr3b", "cdr3a"], sort=True):
        epi_counts = g["peptide"].value_counts()
        if len(epi_counts) > 1:
            poly += 1
        # majority epitope; deterministic tie-break (lexicographic).
        top = epi_counts.max()
        peptide = sorted(epi_counts[epi_counts == top].index)[0]
        first = g.iloc[0]
        rows.append({
            "pair_id": _pair_id(cdr3b, cdr3a),
            "cdr3a": cdr3a, "cdr3b": cdr3b,
            "va": first["va"], "ja": first["ja"],
            "vb": first["vb"], "jb": first["jb"],
            "peptide": peptide,
            "n_epitopes": int(len(epi_counts)),
        })
    pairs = pd.DataFrame(rows).sort_values(["cdr3b", "cdr3a"]).reset_index(drop=True)
    assert pairs["pair_id"].is_unique, "pair_id collision"
    stats = {
        "n_pairs": int(len(pairs)),
        "n_unique_cdr3a": int(pairs["cdr3a"].nunique()),
        "n_unique_cdr3b": int(pairs["cdr3b"].nunique()),
        "n_unique_sequences": int(len(set(pairs["cdr3a"]) | set(pairs["cdr3b"]))),
        "n_epitopes": int(pairs["peptide"].nunique()),
        "n_polyspecific_pairs": int(poly),
    }
    return pairs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-file", default=str(DB_FILE))
    args = ap.parse_args()

    db_file = Path(args.db_file)
    if not db_file.exists():
        raise SystemExit(
            f"pooled DB not found: {db_file}\n"
            "Clone baselines/TCR_Unsupervised_Benchmark first (gitignored).")

    d = curate(db_file)

    # Sanity: per-epitope counts of the six highlighted epitopes must match paper.
    print("Per-epitope sanity vs paper (pairs / cdr3a / cdr3b):")
    ok = True
    for epi, (pp, pa, pb) in SANITY.items():
        sub = d[d["Epitope"] == epi]
        mp = sub["pair"].nunique()
        ma = sub["CDR3_alpha"].nunique()
        mb = sub["CDR3_beta"].nunique()
        match = (mp, ma, mb) == (pp, pa, pb)
        ok &= match
        print(f"  {epi:10s} {mp:4d}/{ma:4d}/{mb:4d}  paper {pp:4d}/{pa:4d}/{pb:4d} "
              f"{'OK' if match else 'MISMATCH'}")
    if not ok:
        print("  WARNING: highlighted-epitope counts differ from the paper; "
              "the committed database may differ from the version used here.")

    pairs, stats = build_pairs(d)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["pair_id", "cdr3a", "cdr3b", "va", "ja", "vb", "jb", "peptide", "n_epitopes"]
    pairs[cols].to_csv(OUT_DIR / "tcrs.csv", index=False)

    meta = {
        "task": "T2 TCR clustering",
        "source": {
            "paper": "Benchmarking unsupervised methods for inferring TCR "
                     "specificity (NAR Genomics & Bioinformatics 2025, lqaf150)",
            "repo": "github.com/i3-unit/TCR_Unsupervised_Benchmark",
            "commit": _commit(),
            "database_file": db_file.name,
            "databases_pooled": ["IEDB", "VDJdb", "McPAS-TCR"],
        },
        "evaluation_unit": "unique paired TCR (cdr3b + cdr3a); seq_col default cdr3b",
        "filter": ("Verified_score==2 & Identification_score>4.3 & Cell_subset==CD8 "
                   "& has V/J(alpha,beta) & 6<=len(CDR3a,CDR3b)<=23 & epitope has "
                   ">=2 unique pairs; two duplicate pairs removed; 10X train "
                   "publication excluded (reproduces database_processing.R)."),
        "label_use": "epitope labels are held out for evaluation only",
        "reproduced": stats,
        "paper_reference": PAPER,
        "highlighted_epitopes_match_paper": bool(ok),
        "pair_count_gap_note": (
            f"We reproduce {stats['n_pairs']} pairs / "
            f"{stats['n_unique_sequences']} unique sequences / "
            f"{stats['n_epitopes']} epitopes against the paper's 4,779 / 8,395. "
            "The earlier +589 gap was NOT database drift, contrary to what this "
            "file used to claim: it was a single pandas-vs-dplyr semantics bug in "
            "curate(). dplyr::filter drops NA predicates, so the R original's "
            "filter(PubMed_ID != <10x url>) also removes rows with a missing "
            "PubMed_ID, while pandas' != keeps them. Those 634 rows contributed "
            "exactly the 589 extra pairs, and the pair sets were perfectly nested "
            "(589 pandas-only, 0 R-only). With the fix the shipped database "
            "reproduces the paper's constants exactly. Step-by-step evidence: "
            "scripts/t2_curate_step_align.py -> "
            "outputs/tcr_clustering/curate_step_align.json."),
        "metric_validation_note": (
            "The clustering-metric implementation is validated separately against "
            "the paper's own precomputed outputs on the paper's native protocol "
            "(outputs/tcr_clustering/calibration_report.csv): 7/9 methods "
            "reproduce Purity/Retention to |delta| <= 0.03."),
        "label_convention_note": (
            "build_pairs collapses polyspecific pairs to a single majority "
            "epitope (lexicographic tie-break), whereas the paper's "
            "purity_function keeps every observed epitope. 90 of "
            f"{stats['n_pairs']} pairs (1.68%) are polyspecific; the collapse "
            "depresses purity by at most 0.0023 across the nine methods "
            "(scripts/t2_label_collapse_effect.py). This affects the local hybrid "
            "tables RESULTS.md T2(b)/(c) only -- the T2(a) calibration path uses "
            "the paper's multi-label convention."),
    }
    with (OUT_DIR / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n[T2] wrote {stats['n_pairs']} paired TCRs over {stats['n_epitopes']} "
          f"epitopes -> {OUT_DIR}/tcrs.csv")
    print(f"     unique cdr3a={stats['n_unique_cdr3a']} cdr3b={stats['n_unique_cdr3b']} "
          f"union-seqs={stats['n_unique_sequences']} "
          f"polyspecific-pairs={stats['n_polyspecific_pairs']}")
    print(f"     meta -> {OUT_DIR}/meta.json")


if __name__ == "__main__":
    main()
