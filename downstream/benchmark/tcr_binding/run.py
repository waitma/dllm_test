#!/usr/bin/env python
"""T1 -- TCR-epitope binding runner.

Scores the leakage-controlled test set with one method, writes a prediction
file, and reports AUROC / AUPRC / Macro-AUC0.1 split by seen / unseen epitope.

Methods
-------
random   : uniform random scores (sanity floor).
knn      : TCRdist/TCRbase-style CDR3 edit-distance kNN (CPU, training-free).
embed    : frozen sequence embedder + logistic-regression probe. The embedder
           is pluggable (``--embedder esm2_150m`` / ``ophiuchus`` /
           ``bioseq:/path/final.pt``); features = [emb(CDR3b) | emb(CDR3a) |
           emb(peptide)]. This is how the immune-receptor foundation model
           enters the leaderboard once trained.
epitcr   : official epiTCR Random Forest (pre-trained pickle, no re-training).
           ``--epitcr-mhc`` enables MHC-aware model.

Examples
--------
    python tcr_binding/run.py --method knn
    python tcr_binding/run.py --method embed --embedder esm2_150m
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import schema, metrics, featurizers  # noqa: E402

DATA = BENCH / "data" / "tcr_binding"
OUT = BENCH / "outputs" / "tcr_binding"


def _embed_columns(embedder, df_train, df_test, columns):
    """Embed unique values per column once, map back to rows, concatenate."""
    feats_tr, feats_te = [], []
    for col in columns:
        vals = pd.concat([df_train[col], df_test[col]]).astype(str)
        uniq = sorted(set(vals))
        emb = embedder.embed(uniq)
        lut = {u: emb[i] for i, u in enumerate(uniq)}
        feats_tr.append(np.stack([lut[str(v)] for v in df_train[col]]))
        feats_te.append(np.stack([lut[str(v)] for v in df_test[col]]))
        print(f"      embedded {len(uniq)} unique '{col}' -> dim {emb.shape[1]}")
    return np.concatenate(feats_tr, axis=1), np.concatenate(feats_te, axis=1)


def run_embed(train, test, embedder_spec, columns, seed):
    from common.model_api import build_embedder
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    print(f"[embed] building embedder '{embedder_spec}' ...")
    embedder = build_embedder(embedder_spec)
    print(f"[embed] embedding columns {columns} ...")
    xtr, xte = _embed_columns(embedder, train, test, columns)
    print(f"[embed] features train={xtr.shape} test={xte.shape}; fitting probe ...")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, random_state=seed),
    )
    clf.fit(xtr, train["label"].to_numpy())
    scores = clf.predict_proba(xte)[:, 1]
    return scores, embedder.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["random", "knn", "embed", "epitcr"], default="knn")
    ap.add_argument("--embedder", default="esm2_150m")
    ap.add_argument("--epitcr-mhc", action="store_true", help="use epiTCR MHC-aware model")
    ap.add_argument("--columns", nargs="+", default=["cdr3b", "cdr3a", "peptide"])
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="output subdir name override")
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv").fillna("")
    test = pd.read_csv(DATA / "test.csv").fillna("")
    print(f"loaded train={len(train)} test={len(test)}")

    t0 = time.time()
    if args.method == "random":
        rng = np.random.default_rng(args.seed)
        scores = rng.random(len(test))
        model_name = "Random"
    elif args.method == "knn":
        print(f"[knn] fitting CDR3 edit-distance kNN (k={args.knn_k}) ...")
        scorer = featurizers.DistanceKNNScorer(k=args.knn_k).fit(train)
        scores = scorer.score(test)
        model_name = f"CDR3-kNN(k={args.knn_k})"
    elif args.method == "epitcr":
        from baselines.wrappers.epitcr import predict_scores
        print(f"[epitcr] loading official RF (mhc={args.epitcr_mhc}) ...")
        scores = predict_scores(test, use_mhc=args.epitcr_mhc)
        model_name = f"epiTCR-{'MHC' if args.epitcr_mhc else 'noMHC'}"
    else:
        scores, model_name = run_embed(train, test, args.embedder, args.columns, args.seed)
    elapsed = time.time() - t0

    if args.tag:
        tag = args.tag
    elif args.method == "embed":
        tag = args.embedder
    elif args.method == "epitcr":
        tag = f"epitcr_{'mhc' if args.epitcr_mhc else 'nomhc'}"
    else:
        tag = args.method
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.DataFrame({
        "id": test["id"], "score": scores,
        "label": test["label"], "group": test["peptide"], "seen": test["seen"],
    })
    pred.to_csv(out_dir / "predictions.csv", index=False)

    def _block(mask):
        sub = pred[mask]
        return metrics.binding_metrics(sub["label"], sub["score"], sub["group"])

    result = {
        "method": args.method, "model": model_name,
        "embedder": args.embedder if args.method == "embed" else None,
        "columns": args.columns if args.method == "embed" else None,
        "elapsed_sec": round(elapsed, 1),
        "overall": _block(pred["seen"] != "___"),
        "seen": _block(pred["seen"] == "seen"),
        "unseen": _block(pred["seen"] == "unseen"),
    }
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(result, fh, indent=2)

    def _fmt(b):
        return (f"AUROC={b['macro_auroc']:.3f} AUPRC={b['macro_auprc']:.3f} "
                f"AUC0.1={b['macro_auc01']:.3f} (n_ep={b['n_epitopes_scored']})")
    print(f"\n=== {model_name} ({elapsed:.1f}s) ===")
    print(f"  UNSEEN (main): {_fmt(result['unseen'])}")
    print(f"  SEEN         : {_fmt(result['seen'])}")
    print(f"  OVERALL      : {_fmt(result['overall'])}")
    print(f"  -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
