#!/usr/bin/env python
"""T1 -- pair-level contamination of the NM2025 *original-protocol* test splits.

Why this is separate from ``nm2025_near_dedup.py``
--------------------------------------------------
``nm2025_near_dedup.py`` filters test rows whose CDR3beta is within Levenshtein
distance <= k of *any* CDR3beta in ``train.csv``, ignoring which epitope that
train row was paired with. That is a deliberately loose, epitope-agnostic
criterion (a CDR3beta seen against epitope X still counts as a near-duplicate
when tested against epitope Y).

This script measures the *strict* criterion instead: how many test rows are an
exact ``(CDR3beta, epitope)`` pair that already appears in ``train.csv``. For an
"unseen-epitope" split that number should be zero by construction, so any hit is
a labelling defect in the shipped split rather than a modelling choice.

Reported per (split, neg):
  * n exact (cdr3b, peptide) pairs shared with train.csv, and their label agreement;
  * which epitopes those hits concentrate on, and whether the epitope itself
    appears in train.csv at all (epitope-level contamination);
  * every method's metrics recomputed on three nested views -- full / drop the
    exact pairs / drop the whole contaminated epitope -- by id-joining the
    existing ``predictions.csv`` (no model is re-run).

Reading the view deltas
-----------------------
All contaminated rows are positives, so dropping them lowers the positive
prevalence and therefore lowers AUPRC *even for a random scorer*. Raw AUPRC
deltas across views are consequently not interpretable on their own; the table
reports prevalence per view and ``auprc_lift = auprc_overall - prevalence`` so
the prevalence artifact is removed. AUROC is prevalence-insensitive and needs no
correction.

Only methods trained on ``train.csv`` (Ours-* / kNN, plus the random control)
can memorise these pairs. Published-checkpoint baselines were trained on their
own corpora, so their view deltas reflect only the change of test composition.

Two id conventions coexist in the artifacts: ``run_nm2025.py`` methods keep the
test ``id`` (``nm2025_<split>_<track>_<neg>-<i>``), while official wrappers emit
positional ``<split>_<neg>_<i>``. The positional form is joined by row index and
accepted only when (label, group) agree on every row; wrappers that reorder
their output fail that check and are reported as unjoinable instead of guessed.

Outputs (existing artifacts are never modified):
    outputs/tcr_binding_nm2025/original_unseen_contamination.json
    outputs/tcr_binding_nm2025/original_unseen_contamination.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import metrics  # noqa: E402
from tcr_binding.run_nm2025 import (  # noqa: E402
    _split_dirname, join_predictions_to_test,
)

DATA = BENCH / "data" / "tcr_binding_nm2025"
OUT = BENCH / "outputs" / "tcr_binding_nm2025"
TRACK = "cdr3b"


def _block(pred: pd.DataFrame) -> dict:
    off = metrics.official_binding_report(pred["label"], pred["score"], pred["group"])
    leg = metrics.binding_metrics(pred["label"], pred["score"], pred["group"])
    n_pos = int((pred["label"] == 1).sum())
    prevalence = n_pos / max(1, len(pred))
    auprc = off["all_values"]["prc_auc"]
    return {
        "n": int(len(pred)),
        "n_pos": n_pos,
        "prevalence": round(prevalence, 6),
        "n_epitopes": off["n_epitopes"],
        "auprc_overall": auprc,
        "auprc_lift": auprc - prevalence,
        "auroc_overall": off["all_values"]["roc_auc"],
        "auprc_macro": off["macro"]["prc_auc"],
        "auc01_macro": leg["macro_auc01"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["seen", "unseen"])
    ap.add_argument("--neg", default="AS")
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train" / TRACK / "train.csv").fillna("")
    train_pairs: dict[tuple[str, str], set[int]] = {}
    for c, p, l in zip(train["cdr3b"].astype(str), train["peptide"].astype(str), train["label"]):
        train_pairs.setdefault((c, p), set()).add(int(l))
    train_epitopes = train["peptide"].astype(str).value_counts().to_dict()
    print(f"train.csv: {len(train)} rows, {len(train_pairs)} unique (cdr3b,peptide) pairs, "
          f"{len(train_epitopes)} epitopes")

    report: dict = {
        "train_csv": str(DATA / "train" / TRACK / "train.csv"),
        "n_train_rows": int(len(train)),
        "n_train_pairs": len(train_pairs),
        "n_train_epitopes": len(train_epitopes),
        "splits": {},
    }
    rows: list[dict] = []

    for split in args.splits:
        test_path = DATA / split / TRACK / args.neg / "test.csv"
        if not test_path.exists():
            print(f"[skip] {test_path} missing")
            continue
        test = pd.read_csv(test_path).fillna("")
        test["id"] = test["id"].astype(str)
        key = list(zip(test["cdr3b"].astype(str), test["peptide"].astype(str)))
        test["_hit"] = [k in train_pairs for k in key]
        test["_label_agrees"] = [
            (k in train_pairs) and (int(l) in train_pairs[k])
            for k, l in zip(key, test["label"])
        ]

        hits = test[test["_hit"]]
        hit_epitopes = sorted(hits["peptide"].astype(str).unique())
        # epitopes of this split that appear anywhere in train.csv
        split_epitopes = sorted(test["peptide"].astype(str).unique())
        contaminated_epitopes = [e for e in split_epitopes if e in train_epitopes]
        contam_rows = test[test["peptide"].astype(str).isin(contaminated_epitopes)]

        srep = {
            "n_test": int(len(test)),
            "n_epitopes": len(split_epitopes),
            "exact_pair_hits": int(len(hits)),
            "exact_pair_hit_frac": round(len(hits) / max(1, len(test)), 4),
            "label_agreement": int(test["_label_agrees"].sum()),
            "hit_label_counts": {str(k): int(v) for k, v in
                                 hits["label"].value_counts().to_dict().items()},
            "hit_epitopes": hit_epitopes,
            "epitopes_present_in_train": contaminated_epitopes,
            "n_rows_on_contaminated_epitopes": int(len(contam_rows)),
            "contaminated_epitope_train_rows": {e: int(train_epitopes[e])
                                                for e in contaminated_epitopes},
        }
        report["splits"][f"{split}/{args.neg}"] = srep
        print(f"\n[{split}/{args.neg}] n={len(test)} epitopes={len(split_epitopes)}")
        print(f"  exact (cdr3b,peptide) pairs in train.csv: {len(hits)} "
              f"({srep['exact_pair_hit_frac']*100:.2f}%), label agreement "
              f"{srep['label_agreement']}/{len(hits)}, labels={srep['hit_label_counts']}")
        print(f"  epitopes also in train.csv: {contaminated_epitopes} "
              f"-> {len(contam_rows)} test rows")

        drop_pair = set(hits["id"])
        drop_epi = set(contam_rows["id"])
        views = {
            "full": None,
            "drop_exact_pairs": drop_pair,
            "drop_contaminated_epitopes": drop_epi,
        }

        joined, unjoinable = [], []
        for mdir in sorted(OUT.glob(f"*/{TRACK}/{_split_dirname(split, args.neg)}/predictions.csv")):
            method = mdir.parents[2].name
            pred = join_predictions_to_test(pd.read_csv(mdir), test, split, args.neg, method)
            if pred is None:
                unjoinable.append(method)
                continue
            joined.append(method)
            for view, drop in views.items():
                sub = pred if drop is None else pred[~pred["id"].isin(drop)]
                if len(sub) == 0 or sub["label"].nunique() < 2:
                    continue
                rows.append({"method": method, "split": split, "neg": args.neg,
                             "view": view, **_block(sub)})
        srep["methods_joined"] = sorted(joined)
        srep["methods_unjoinable"] = sorted(unjoinable)
        print(f"  joined {len(joined)} methods, unjoinable {len(unjoinable)}"
              + (f": {sorted(unjoinable)}" if unjoinable else ""))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "original_unseen_contamination.json").write_text(json.dumps(report, indent=2))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "original_unseen_contamination.csv", index=False)
    print(f"\n-> {OUT/'original_unseen_contamination.json'}")
    print(f"-> {OUT/'original_unseen_contamination.csv'}  ({len(df)} rows)")

    if len(df):
        pd.set_option("display.width", 220, "display.max_columns", 40)
        for split in args.splits:
            d = df[df["split"] == split]
            if not len(d):
                continue
            print(f"\n=== {split}/{args.neg}: prevalence by view ===")
            print(d.pivot_table(index="view", values=["n", "n_pos", "prevalence"]).round(4))
            piv = d.pivot_table(index="method", columns="view",
                                values=["auprc_lift", "auroc_overall"])
            print(f"\n=== {split}/{args.neg}: prevalence-corrected AUPRC lift "
                  f"and AUROC by view ===")
            print(piv.round(4))


if __name__ == "__main__":
    main()
