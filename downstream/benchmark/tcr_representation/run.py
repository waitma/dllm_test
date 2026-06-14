#!/usr/bin/env python
"""T3 -- TCR representation quality (epitope-specificity probing).

Frozen backbone; only a linear probe (logistic regression) and a 1-NN classifier
are trained on top of embeddings to predict the epitope a TCR binds (multiclass).
The train/test split is clonotype-disjoint (built by prepare_tcr_repertoire.py),
so high scores reflect transferable representation, not memorized clonotypes.

Methods (embedders): esm2_*, ophiuchus, bioseq:/path/final.pt, or ``kmer`` for a
training-free composition baseline.

Example:
    python tcr_representation/run.py --embedder esm2_150m --columns cdr3b cdr3a
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

from common import metrics  # noqa: E402

DATA = BENCH / "data" / "tcr_representation"
OUT = BENCH / "outputs" / "tcr_representation"


def featurize(embedder_spec, train, test, columns):
    if embedder_spec == "kmer":
        from common.featurizers import KmerFeaturizer
        kf = KmerFeaturizer(k=3)
        xtr = np.concatenate([kf.transform(train[c].astype(str)) for c in columns], 1)
        xte = np.concatenate([kf.transform(test[c].astype(str)) for c in columns], 1)
        return xtr, xte, "kmer-3"

    from common.model_api import build_embedder
    emb = build_embedder(embedder_spec)
    feats_tr, feats_te = [], []
    for c in columns:
        vals = pd.concat([train[c], test[c]]).astype(str)
        uniq = sorted(set(vals))
        e = emb.embed(uniq)
        lut = {u: e[i] for i, u in enumerate(uniq)}
        feats_tr.append(np.stack([lut[str(v)] for v in train[c]]))
        feats_te.append(np.stack([lut[str(v)] for v in test[c]]))
    return (np.concatenate(feats_tr, 1), np.concatenate(feats_te, 1), emb.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="esm2_150m")
    ap.add_argument("--columns", nargs="+", default=["cdr3b", "cdr3a"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv").fillna("")
    test = pd.read_csv(DATA / "test.csv").fillna("")
    classes = sorted(set(train["peptide"]) | set(test["peptide"]))
    cls2id = {c: i for i, c in enumerate(classes)}
    ytr = train["peptide"].map(cls2id).to_numpy()
    yte = test["peptide"].map(cls2id).to_numpy()
    print(f"train={len(train)} test={len(test)} classes={len(classes)}")

    t0 = time.time()
    xtr, xte, model_name = featurize(args.embedder, train, test, args.columns)
    probe = metrics.linear_probe_metrics(xtr, ytr, xte, yte, multiclass=True, seed=args.seed)
    knn = metrics.knn_top1_metric(xtr, ytr, xte, yte, metric="cosine")
    elapsed = time.time() - t0

    out_dir = OUT / args.embedder
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"embedder": args.embedder, "model": model_name,
              "columns": args.columns, "n_classes": len(classes),
              "elapsed_sec": round(elapsed, 1), **probe, **knn}
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n=== {model_name} ({elapsed:.1f}s) ===")
    print(f"  probe-AUROC={probe.get('probe_auroc', float('nan')):.3f} "
          f"probe-Acc={probe['probe_acc']:.3f} kNN-top1={knn['knn_top1_acc']:.3f} "
          f"({len(classes)}-way)")
    print(f"  -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
