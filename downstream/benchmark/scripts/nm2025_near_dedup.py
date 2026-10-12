#!/usr/bin/env python
"""L1 -- near-duplicate dedup for the NM2025 TCR-binding cdr3b track.

Motivation (arXiv 2606.04994, 2026; Sci Rep 2025 s41598-025-26454-7)
--------------------------------------------------------------------
Exact-clonotype dedup misses *near-identical* CDR3beta sequences shared between
train and test. arXiv 2606.04994 shows that removing test sequences within a
Levenshtein distance <= 3 of the CDR3beta training data eliminates 40-70% of
test rows and that scores are inflated by such near-duplicates (memorization).
Sci Rep 2025 shows published models only beat random within edit distance <= 1
of the training epitope set.

What this does
--------------
For the cdr3b track (peptide + CDR3beta, no MHC), we define a *hardened* test
subset per (split, neg) by removing every test row whose CDR3beta is within
Levenshtein distance <= k (k = 1, 2, 3) of ANY CDR3beta in the official
``train.csv`` (the retraining corpus used by Ours-BioSeq / kNN / retrained
baselines). We report, per k:

  * removal fraction, n_before / n_after, n_pos / n_neg / n_epitopes after;
  * every method re-scored on the SAME hardened subset (overall AUPRC / AUROC /
    macro-AUPRC / AUC0.1), so the comparison stays apples-to-apples.

Methods scored via ``run_nm2025.py`` (random / knn / epitcr / ours_*) keep their
``id`` column, so we id-join their existing ``predictions.csv`` to the test rows
and filter -- no re-embedding needed. Official published baselines (ATM-TCR /
NetTCR / PanPep) shuffle/reorder their output, so we re-run the official wrapper
on the filtered test subset (its own conda env; seconds).

Outputs (original outputs are never modified):
    data/tcr_binding_nm2025/near_dedup_report.json
    outputs/tcr_binding_nm2025/near_dedup_summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import metrics  # noqa: E402
from tcr_binding.run_nm2025 import (  # noqa: E402
    _split_dirname, _metrics_block, join_predictions_to_test,
)

DATA = BENCH / "data" / "tcr_binding_nm2025"
OUT = BENCH / "outputs" / "tcr_binding_nm2025"
TRACK = "cdr3b"
KS = (1, 2, 3)
CAP = max(KS) + 1  # rapidfuzz score_cutoff; distances above become CAP

# Methods whose predictions.csv preserve the test `id` column (id-join path).
IDJOIN_METHODS = [
    "random", "knn", "epitcr_nomhc", "ours_biseq_mlp",
    "ours_biseq_mlp_esmc600m", "ours_biseq_mlp_esmc300m_mint",
]
# Official published baselines that reorder output -> re-run wrapper on subset.
WRAPPER_METHODS = {"atmtcr": "atmtcr", "nettcr": "nettcr", "panpep": "panpep",
                   "teim": "teim", "ergo2": "ergo2"}


def _min_lev_to_set(queries, refs, cap=CAP):
    """min Levenshtein distance from each query to any ref, capped at ``cap``."""
    from rapidfuzz.process import cdist
    from rapidfuzz.distance import Levenshtein
    q = list(queries)
    r = list(refs)
    # cdist returns distances; values above score_cutoff are set to score_cutoff+1.
    D = cdist(q, r, scorer=Levenshtein.distance, score_cutoff=cap, workers=-1)
    return D.min(axis=1)


def _block_from_pred(pred: pd.DataFrame) -> dict:
    """Overall/macro metrics from a (label, score, group) frame."""
    off = metrics.official_binding_report(pred["label"], pred["score"], pred["group"])
    leg = metrics.binding_metrics(pred["label"], pred["score"], pred["group"])
    return {
        "n": int(len(pred)),
        "n_pos": int((pred["label"] == 1).sum()),
        "n_epitopes": off["n_epitopes"],
        "auprc_overall": off["all_values"]["prc_auc"],
        "auroc_overall": off["all_values"]["roc_auc"],
        "auprc_macro": off["macro"]["prc_auc"],
        "auc01_macro": leg["macro_auc01"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["seen", "unseen"])
    ap.add_argument("--neg", default="AS")
    ap.add_argument("--dedup-ref", choices=["all", "pos"], default="all",
                    help="dedup test CDR3b against all train CDR3b or train positives only")
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train" / TRACK / "train.csv").fillna("")
    train_all = sorted(set(train["cdr3b"].astype(str)))
    train_pos = sorted(set(train.loc[train["label"] == 1, "cdr3b"].astype(str)))
    ref = train_all if args.dedup_ref == "all" else train_pos
    print(f"train cdr3b: all={len(train_all)} pos={len(train_pos)}; dedup-ref={args.dedup_ref} ({len(ref)})")

    report: dict = {"dedup_ref": args.dedup_ref, "n_train_ref": len(ref), "splits": {}}
    rows = []  # summary rows

    for split in args.splits:
        neg = args.neg
        test_path = DATA / split / TRACK / neg / "test.csv"
        if not test_path.exists():
            print(f"[skip] {test_path} missing")
            continue
        test = pd.read_csv(test_path).fillna("")
        test["id"] = test["id"].astype(str)

        # min-Lev per unique test CDR3b (dedup against BOTH all and pos for report).
        uniq = sorted(set(test["cdr3b"].astype(str)))
        mv_all = dict(zip(uniq, _min_lev_to_set(uniq, train_all)))
        mv_pos = dict(zip(uniq, _min_lev_to_set(uniq, train_pos)))
        ref_map = mv_all if args.dedup_ref == "all" else mv_pos
        test["min_lev"] = test["cdr3b"].astype(str).map(ref_map)
        test["min_lev_all"] = test["cdr3b"].astype(str).map(mv_all)
        test["min_lev_pos"] = test["cdr3b"].astype(str).map(mv_pos)

        split_rep = {"n_before": int(len(test)), "per_k": {}}
        keep_ids_by_k = {}
        # k=0 baseline is the full test.
        for k in (0,) + KS:
            keep = test["min_lev"] > k if k > 0 else pd.Series(True, index=test.index)
            kept = test[keep]
            keep_ids_by_k[k] = set(kept["id"])
            n_after = int(len(kept))
            split_rep["per_k"][k] = {
                "n_after": n_after,
                "removed": int(len(test) - n_after),
                "removed_frac": round(1 - n_after / max(1, len(test)), 4),
                "n_pos_after": int((kept["label"] == 1).sum()),
                "n_neg_after": int((kept["label"] == 0).sum()),
                "n_epitopes_after": int(kept["peptide"].nunique()),
            }
        report["splits"][f"{split}/{neg}"] = split_rep
        print(f"\n[{split}/{neg}] n={len(test)}  " +
              "  ".join(f"k<={k}:rm {split_rep['per_k'][k]['removed']}"
                        f"({split_rep['per_k'][k]['removed_frac']*100:.0f}%)->{split_rep['per_k'][k]['n_after']}"
                        for k in KS))

        test_id2cdr = dict(zip(test["id"], test["cdr3b"].astype(str)))

        # ---- id-join methods (reuse existing predictions) ----
        for m in IDJOIN_METHODS:
            pred_path = OUT / m / TRACK / _split_dirname(split, neg) / "predictions.csv"
            if not pred_path.exists():
                continue
            pred = join_predictions_to_test(pd.read_csv(pred_path), test, split, neg, m)
            if pred is None:
                continue
            pred = pred.merge(test[["id", "min_lev"]], on="id", how="left")
            for k in (0,) + KS:
                sub = pred if k == 0 else pred[pred["min_lev"] > k]
                if len(sub) == 0:
                    continue
                blk = _block_from_pred(sub.rename(columns={}))
                rows.append({"method": m, "split": split, "neg": neg, "k": k, **blk})

        # ---- official wrappers (re-run on filtered subset) ----
        for m, mod in WRAPPER_METHODS.items():
            base_pred = OUT / m / TRACK / _split_dirname(split, neg) / "predictions.csv"
            if not base_pred.exists():
                continue
            try:
                import importlib
                wrapper = importlib.import_module(f"baselines.wrappers.{mod}")
                if not wrapper.is_available():
                    print(f"  [skip] {m}: wrapper unavailable")
                    continue
            except Exception as e:
                print(f"  [skip] {m}: import failed {e}")
                continue
            for k in (0,) + KS:
                if k == 0:
                    # reuse the existing full-set metrics from predictions.csv
                    p = pd.read_csv(base_pred)
                    blk = _block_from_pred(p)
                    rows.append({"method": m, "split": split, "neg": neg, "k": k, **blk})
                    continue
                keep_ids = keep_ids_by_k[k]
                sub_test = test[test["id"].isin(keep_ids)]
                try:
                    preds = wrapper.predict_official(sub_test)
                    blk = _block_from_pred(preds.rename(columns={}))
                    rows.append({"method": m, "split": split, "neg": neg, "k": k, **blk})
                    print(f"  {m} k<={k}: n={len(preds)} AUPRC={blk['auprc_overall']:.3f}")
                except Exception as e:
                    print(f"  [warn] {m} k<={k}: rerun failed {e}")

    (DATA / "near_dedup_report.json").write_text(json.dumps(report, indent=2))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "near_dedup_summary.csv", index=False)
    print(f"\n-> {DATA/'near_dedup_report.json'}")
    print(f"-> {OUT/'near_dedup_summary.csv'}  ({len(df)} rows)")

    # Pretty print unseen AS main comparison.
    if len(df):
        piv = df[(df["neg"] == args.neg)].pivot_table(
            index=["method"], columns=["split", "k"], values="auprc_overall")
        pd.set_option("display.width", 200, "display.max_columns", 40)
        print("\n=== overall AUPRC by method x (split,k) ===")
        print(piv.round(3))


if __name__ == "__main__":
    main()
