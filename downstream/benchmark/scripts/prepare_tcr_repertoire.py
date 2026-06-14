#!/usr/bin/env python
"""Build T2 (clustering) and T3 (representation) datasets from VDJdb positives.

Both tasks need TCRs with reliable epitope labels. We take the IMMREP23
paired-chain VDJdb positives, keep epitopes with >= ``min-count`` TCRs (so
clusters / classes are statistically meaningful), and drop exact-duplicate
clonotypes.

- T2 clustering : all kept TCRs -> ``data/tcr_clustering/tcrs.csv``. Epitope
  labels are written but are used ONLY for evaluation, never for clustering.
- T3 representation : clonotype-disjoint 80/20 split (a CDR3a|CDR3b clonotype
  never crosses train/test) -> ``data/tcr_representation/{train,test}.csv``.
  Asserts clonotype overlap == 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import schema, leakage  # noqa: E402

COLS = ["cdr3a", "cdr3b", "va", "ja", "vb", "jb", "tcra", "tcrb", "peptide", "mhc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pos = schema.load_immrep23_train()
    pos = pos.drop_duplicates(subset=["cdr3a", "cdr3b", "peptide"]).reset_index(drop=True)
    vc = pos["peptide"].value_counts()
    keep_eps = vc[vc >= args.min_count].index.tolist()
    kept = pos[pos["peptide"].isin(keep_eps)].reset_index(drop=True)
    print(f"epitopes>= {args.min_count} TCRs: {len(keep_eps)} | TCRs kept: {len(kept)}")

    # T2 clustering set
    clu_dir = BENCH / "data" / "tcr_clustering"
    clu_dir.mkdir(parents=True, exist_ok=True)
    kept[COLS].to_csv(clu_dir / "tcrs.csv", index=False)
    print(f"[T2] wrote {len(kept)} TCRs over {len(keep_eps)} epitopes -> {clu_dir}/tcrs.csv")

    # T3 representation: clonotype-disjoint split
    clono = schema.clonotype_key(kept, level="ab")
    train, test = leakage.seen_epitope_split(
        kept, test_frac=args.test_frac, clonotype=clono, seed=args.seed)
    # keep only epitopes present in BOTH splits (multiclass classifier needs them)
    common_eps = set(train["peptide"]) & set(test["peptide"])
    train = train[train["peptide"].isin(common_eps)].reset_index(drop=True)
    test = test[test["peptide"].isin(common_eps)].reset_index(drop=True)
    rep = leakage.leakage_report(train, test)
    assert rep["clonotype_overlap"] == 0, "representation split leaks clonotypes"

    rep_dir = BENCH / "data" / "tcr_representation"
    rep_dir.mkdir(parents=True, exist_ok=True)
    train[COLS].to_csv(rep_dir / "train.csv", index=False)
    test[COLS].to_csv(rep_dir / "test.csv", index=False)
    meta = {
        "n_train": len(train), "n_test": len(test),
        "n_classes": len(common_eps), "classes": sorted(common_eps),
        "clonotype_overlap": rep["clonotype_overlap"],
        "min_count": args.min_count, "test_frac": args.test_frac, "seed": args.seed,
    }
    with (rep_dir / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[T3] train={len(train)} test={len(test)} classes={len(common_eps)} "
          f"clono_overlap={rep['clonotype_overlap']} -> {rep_dir}/")


if __name__ == "__main__":
    main()
