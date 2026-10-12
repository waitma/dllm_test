#!/usr/bin/env python
"""Build the T5 epitope-conditioned TCR design dataset (TcrDesign / TCRT5 style).

Task: given a target epitope, *generate* CDR3-beta sequences that bind it. We
evaluate against held-out real binders, stratified by how much conditioning data
each epitope has (TcrDesign categories):

  common : >= --common-min binders     (many-shot design)
  rare   : --rare-min .. common-min     (few-shot design)
  novel  : epitope entirely absent from the conditioning corpus (zero-shot)

Pipeline
--------
1. Load IMMREP23 paired-chain VDJdb positives; keep valid CDR3b (5-30 aa).
2. Pick ``--n-novel`` epitopes (from those with >= --novel-min binders) to hold
   out ENTIRELY -> their binders are eval-only, 0 conditioning binders.
3. For every other epitope with >= --eval-min binders, split its unique CDR3b
   binders into conditioning (train) + held-out (ref) with NO CDR3b crossing the
   split (clonotype-disjoint). Tag common/rare by binder count.
4. Epitopes with too few binders stay wholly in the conditioning corpus
   (background; enrich training, not scored).
5. Emit train_binders.csv, eval.json, leakage_report.json.

Run:
    python scripts/prepare_tcr_design.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import schema  # noqa: E402

OUT_DIR = BENCH / "data" / "tcr_design"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _valid_beta(s: str) -> bool:
    s = str(s).strip().upper()
    return bool(s) and set(s) <= VALID_AA and 5 <= len(s) <= 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common-min", type=int, default=100)
    ap.add_argument("--rare-min", type=int, default=20)
    ap.add_argument("--eval-min", type=int, default=20,
                    help="min binders for an epitope to get a scored eval split")
    ap.add_argument("--novel-min", type=int, default=20,
                    help="min binders for an epitope eligible as held-out novel")
    ap.add_argument("--n-novel", type=int, default=7)
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("[1/5] loading IMMREP23 VDJdb paired-chain positives ...")
    pos = schema.load_immrep23_train()
    pos = pos[pos["label"] == 1].copy()
    pos = pos[pos["cdr3b"].map(_valid_beta)].copy()
    pos["cdr3b"] = pos["cdr3b"].str.strip().str.upper()
    # unique (epitope, cdr3b) binder rows; keep first alpha/mhc for reference
    pos = pos.drop_duplicates(subset=["peptide", "cdr3b"]).reset_index(drop=True)
    counts = pos["peptide"].value_counts()
    print(f"      {len(pos)} unique (epitope,CDR3b) binders over {len(counts)} epitopes")

    eligible_novel = sorted(counts[counts >= args.novel_min].index.tolist())
    rng.shuffle(eligible_novel)
    novel_epitopes = set(eligible_novel[: args.n_novel])
    print(f"[2/5] held-out NOVEL epitopes ({len(novel_epitopes)}): {sorted(novel_epitopes)}")

    train_rows = []
    eval_entry = {}

    print("[3/5] per-epitope conditioning/eval split (clonotype-disjoint on CDR3b) ...")
    for ep, grp in pos.groupby("peptide"):
        betas = grp["cdr3b"].tolist()
        n = len(betas)
        if ep in novel_epitopes:
            eval_entry[ep] = {"category": "novel", "train_binders": [],
                              "ref_binders": sorted(set(betas))}
            continue
        if n < args.eval_min:
            # background epitope: all binders enrich the conditioning corpus
            for _, r in grp.iterrows():
                train_rows.append(r)
            continue
        uniq = sorted(set(betas))
        rng.shuffle(uniq)
        n_eval = max(1, int(round(len(uniq) * args.eval_frac)))
        ref = set(uniq[:n_eval])
        cond = set(uniq[n_eval:])
        cat = "common" if n >= args.common_min else "rare"
        eval_entry[ep] = {"category": cat, "train_binders": sorted(cond),
                          "ref_binders": sorted(ref)}
        # conditioning rows for this epitope (only the cond CDR3b)
        for _, r in grp[grp["cdr3b"].isin(cond)].iterrows():
            train_rows.append(r)

    train_df = pd.DataFrame(train_rows)[
        ["peptide", "cdr3b", "cdr3a", "vb", "jb", "va", "ja", "mhc"]
    ].drop_duplicates().reset_index(drop=True)

    print("[4/5] leakage checks ...")
    report = {"dataset": "IRBench-T5 epitope-conditioned TCR design",
              "source": "IMMREP23 VDJdb paired-chain positives",
              "params": vars(args), "per_category": {}, "violations": []}
    # (a) novel epitopes must be absent from the conditioning corpus
    train_eps = set(train_df["peptide"])
    novel_leak = sorted(novel_epitopes & train_eps)
    if novel_leak:
        report["violations"].append({"novel_in_train": novel_leak})
    # (b) within scored epitopes, ref CDR3b must not appear in that epitope's cond
    per_cat = {"common": 0, "rare": 0, "novel": 0}
    cross_leak = 0
    for ep, e in eval_entry.items():
        per_cat[e["category"]] += 1
        if set(e["train_binders"]) & set(e["ref_binders"]):
            cross_leak += 1
    report["per_category"] = per_cat
    report["n_scored_epitopes"] = len(eval_entry)
    report["n_train_binders"] = int(len(train_df))
    report["cdr3b_cross_split_leak_epitopes"] = cross_leak
    report["novel_in_train_corpus"] = len(novel_leak)

    print("[5/5] writing outputs ...")
    train_df.to_csv(out_dir / "train_binders.csv", index=False)
    with (out_dir / "eval.json").open("w") as fh:
        json.dump(eval_entry, fh, indent=2)
    with (out_dir / "leakage_report.json").open("w") as fh:
        json.dump(report, fh, indent=2)

    print(f"      train binders: {len(train_df)} rows / {len(train_eps)} epitopes")
    print(f"      scored epitopes: {per_cat} (cross-split leak={cross_leak}, "
          f"novel-in-train={len(novel_leak)})")
    print(f"-> {out_dir}/")


if __name__ == "__main__":
    main()
