#!/usr/bin/env python
"""Build the leakage-controlled T1 TCR-epitope binding dataset.

Source: IMMREP23 paired-chain VDJdb (positives) + IMMREP23 labelled test
(``solutions.csv``, official swapped negatives).

Pipeline
--------
1. Load training positives (VDJdb paired chain, Target==1).
2. Drop any positive whose clonotype (CDR3a|CDR3b) also occurs in the test set
   -> removes train/test clonotype leakage.
3. Generate reference-TCR negatives for training (CDR3b Levenshtein > 3 from the
   epitope's positives), ratio configurable (default 5:1).
4. Test set = solutions.csv as-is; tag each test epitope ``seen`` / ``unseen``
   by membership in the (deduped) training epitope set.
5. Emit train.csv, test.csv, and leakage_report.json (overlaps must be 0).

Run:
    python scripts/prepare_tcr_binding.py --neg-ratio 5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import schema, leakage, negatives  # noqa: E402

OUT_DIR = BENCH / "data" / "tcr_binding"
KEEP_COLS = schema.TCR_COLUMNS + ["label", "split", "seen", "source", "id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-ratio", type=int, default=5)
    ap.add_argument("--dist-threshold", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading IMMREP23 positives + labelled test ...")
    pos = schema.load_immrep23_train()
    test = schema.load_immrep23_test()
    print(f"      positives={len(pos)}  test={len(test)} "
          f"(pos={int((test.label==1).sum())}, neg={int((test.label==0).sum())})")

    print("[2/5] de-duplicating train clonotypes that leak into test ...")
    raw_report = leakage.leakage_report(pos, test)
    pos_dedup = leakage.dedup_against(pos, test, keys=("cdr3a", "cdr3b"))
    print(f"      removed {len(pos)-len(pos_dedup)} leaking positives "
          f"(raw clonotype_overlap={raw_report['clonotype_overlap']})")

    print(f"[3/5] generating reference negatives (ratio {args.neg_ratio}:1, "
          f"Lev>{args.dist_threshold}) ...")
    train = negatives.build_binding_dataset(
        pos_dedup, ratio=args.neg_ratio,
        dist_threshold=args.dist_threshold, seed=args.seed,
    )
    train["split"] = "train"
    train["id"] = [f"tr{i}" for i in range(len(train))]
    print(f"      train rows={len(train)} "
          f"(pos={int((train.label==1).sum())}, neg={int((train.label==0).sum())})")

    print("[4/5] tagging seen / unseen test epitopes ...")
    train_eps = set(pos_dedup["peptide"].astype(str))
    test = test.copy()
    test["split"] = "test"
    test["seen"] = test["peptide"].astype(str).apply(
        lambda e: "seen" if e in train_eps else "unseen")
    n_seen = int((test["seen"] == "seen").sum())
    n_unseen = int((test["seen"] == "unseen").sum())
    seen_eps = sorted(set(test[test.seen == "seen"]["peptide"]))
    unseen_eps = sorted(set(test[test.seen == "unseen"]["peptide"]))
    print(f"      test rows: seen={n_seen} ({len(seen_eps)} epitopes), "
          f"unseen={n_unseen} ({len(unseen_eps)} epitopes)")

    train["seen"] = "train"
    for c in KEEP_COLS:
        if c not in train.columns:
            train[c] = ""
        if c not in test.columns:
            test[c] = ""

    print("[5/5] writing outputs + final leakage report ...")
    train[KEEP_COLS].to_csv(out_dir / "train.csv", index=False)
    test[KEEP_COLS].to_csv(out_dir / "test.csv", index=False)

    final_report = leakage.leakage_report(train, test)
    report = {
        "dataset": "IRBench-T1 TCR-epitope binding",
        "source": "IMMREP23 paired-chain VDJdb (train positives) + solutions.csv (test)",
        "params": vars(args),
        "raw_before_dedup": {k: raw_report[k] for k in
                             ["clonotype_overlap", "pair_overlap", "shared_epitopes"]},
        "final": {
            "n_train": len(train), "n_test": len(test),
            "train_pos": int((train.label == 1).sum()),
            "train_neg": int((train.label == 0).sum()),
            "test_pos": int((test.label == 1).sum()),
            "test_neg": int((test.label == 0).sum()),
            "clonotype_overlap": final_report["clonotype_overlap"],
            "pair_overlap": final_report["pair_overlap"],
            "n_seen_rows": n_seen, "n_unseen_rows": n_unseen,
            "seen_epitopes": seen_eps, "unseen_epitopes": unseen_eps,
        },
    }
    with (out_dir / "leakage_report.json").open("w") as fh:
        json.dump(report, fh, indent=2)

    assert final_report["clonotype_overlap"] == 0, "LEAKAGE: clonotype overlap != 0"
    assert final_report["pair_overlap"] == 0, "LEAKAGE: pair overlap != 0"
    print(f"      OK -> {out_dir}/  (clonotype/pair overlap = 0)")
    print(json.dumps(report["final"], indent=2)[:600])


if __name__ == "__main__":
    main()
