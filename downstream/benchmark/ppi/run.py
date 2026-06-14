#!/usr/bin/env python
"""P1 -- PPI binary interaction (STRING 90/90).

Features per protein, combined into a symmetric pair feature (Hadamard product,
since PPI is order-invariant), then a logistic-regression classifier. Report
AUROC / AUPRC on the held-out 90/90 test split (sequence-similarity controlled).

Methods (embedders):
  kmer        : k-mer composition (k=3), CPU, no length limit (default).
  esm2_*      : ESM2 mean-pool with truncation (GPU). Proteins are truncated to
                --max-length for tractability.
  bioseq:...  : foundation-model embedder (once trained).

Example:
    python ppi/run.py --embedder kmer --max-train 30000
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

DATA = BENCH / "data" / "ppi"
OUT = BENCH / "outputs" / "ppi"


def protein_features(embedder_spec, proteins, max_length):
    if embedder_spec == "kmer":
        from common.featurizers import KmerFeaturizer
        kf = KmerFeaturizer(k=3)
        return kf.transform(proteins), "kmer-3"
    from common.model_api import build_embedder
    emb = build_embedder(embedder_spec, max_length=max_length)
    return emb.embed(proteins), emb.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="kmer")
    ap.add_argument("--max-train", type=int, default=30000)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv").fillna("")
    test = pd.read_csv(DATA / "test.csv").fillna("")
    if args.max_train and len(train) > args.max_train:
        train = train.sample(n=args.max_train, random_state=args.seed).reset_index(drop=True)
    print(f"train={len(train)} test={len(test)}")

    # unique proteins across used rows
    prot_map = {}
    for df in (train, test):
        for col in ("idA", "idB"):
            for pid, seq in zip(df[col], df[col.replace("id", "seq")]):
                prot_map.setdefault(pid, str(seq))
    pids = sorted(prot_map.keys())
    print(f"embedding {len(pids)} unique proteins via '{args.embedder}' ...")

    t0 = time.time()
    feats, model_name = protein_features(args.embedder, [prot_map[p] for p in pids], args.max_length)
    lut = {p: feats[i] for i, p in enumerate(pids)}

    def pair_feats(df):
        a = np.stack([lut[p] for p in df["idA"]])
        b = np.stack([lut[p] for p in df["idB"]])
        return (a * b).astype(np.float32)  # symmetric interaction feature

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    xtr, xte = pair_feats(train), pair_feats(test)
    clf = make_pipeline(StandardScaler(with_mean=False),
                        LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed))
    clf.fit(xtr, train["label"].to_numpy())
    scores = clf.predict_proba(xte)[:, 1]
    elapsed = time.time() - t0

    y = test["label"].to_numpy()
    auroc = metrics._safe_auroc(y, scores)
    auprc = metrics._safe_auprc(y, scores)
    out_dir = OUT / args.embedder
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"idA": test["idA"], "idB": test["idB"], "score": scores, "label": y}).to_csv(
        out_dir / "predictions.csv", index=False)
    result = {"embedder": args.embedder, "model": model_name, "split": "90/90 test",
              "n_train": len(train), "n_test": len(test),
              "auroc": float(auroc), "auprc": float(auprc),
              "elapsed_sec": round(elapsed, 1)}
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n=== PPI {model_name} ({elapsed:.1f}s) ===")
    print(f"  AUROC={auroc:.3f} AUPRC={auprc:.3f} (test n={len(test)})")
    print(f"  -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
