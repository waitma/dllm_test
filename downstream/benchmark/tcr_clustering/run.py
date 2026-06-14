#!/usr/bin/env python
"""T2 -- TCR clustering.

Cluster CDR3 sequences without using epitope labels, then measure agreement of
clusters with the held-out epitope labels (Purity / Retention / NMI / ARI).

Methods
-------
editdist : single-linkage / connected-components clustering on CDR3b Levenshtein
           distance <= threshold (clusTCR/GIANA/tcrdist family, training-free).
embed    : agglomerative clustering on frozen embeddings (cosine), n_clusters =
           number of true epitopes (an upper-bound-aware reference).

Examples
--------
    python tcr_clustering/run.py --method editdist --threshold 1
    python tcr_clustering/run.py --method embed --embedder esm2_150m
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
from common.leakage import edit_distance  # noqa: E402

DATA = BENCH / "data" / "tcr_clustering"
OUT = BENCH / "outputs" / "tcr_clustering"


def cluster_editdist(seqs, threshold=1, min_size=2):
    """Connected components over CDR3b with edit distance <= threshold.

    Length bucketing + per-bucket pairwise distance keeps it tractable. Singleton
    components are marked unclustered (-1) so Retention is meaningful.
    """
    n = len(seqs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Only compare sequences whose length differs by <= threshold.
    order = sorted(range(n), key=lambda i: len(seqs[i]))
    for ii in range(n):
        i = order[ii]
        li = len(seqs[i])
        for jj in range(ii + 1, n):
            j = order[jj]
            if len(seqs[j]) - li > threshold:
                break
            if edit_distance(seqs[i], seqs[j]) <= threshold:
                union(i, j)

    comp = np.array([find(i) for i in range(n)])
    # relabel; singletons (size<min_size) -> -1
    labels = np.full(n, -1)
    cid = 0
    for c in pd.unique(comp):
        idx = np.where(comp == c)[0]
        if len(idx) >= min_size:
            labels[idx] = cid
            cid += 1
    return labels


def cluster_embed(seqs, n_clusters, embedder_spec):
    from common.model_api import build_embedder
    from sklearn.cluster import AgglomerativeClustering

    emb = build_embedder(embedder_spec)
    x = emb.embed(list(seqs))
    # cosine via L2-normalize + euclidean
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(x), emb.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["editdist", "embed"], default="editdist")
    ap.add_argument("--embedder", default="esm2_150m")
    ap.add_argument("--threshold", type=int, default=1)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--seq-col", default="cdr3b")
    args = ap.parse_args()

    df = pd.read_csv(DATA / "tcrs.csv").fillna("")
    seqs = df[args.seq_col].astype(str).tolist()
    true = df["peptide"].astype(str).tolist()
    n_ep = df["peptide"].nunique()
    print(f"loaded {len(df)} TCRs over {n_ep} epitopes (seq_col={args.seq_col})")

    t0 = time.time()
    if args.method == "editdist":
        labels = cluster_editdist(seqs, threshold=args.threshold, min_size=args.min_size)
        model_name = f"CDR3-editdist(t={args.threshold})"
    else:
        labels, model_name = cluster_embed(seqs, n_ep, args.embedder)
    elapsed = time.time() - t0

    m = metrics.clustering_metrics(labels, true, unclustered_value=-1)
    tag = (f"editdist_t{args.threshold}" if args.method == "editdist"
           else args.embedder)
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cdr3b": seqs, "cluster": labels, "epitope": true}).to_csv(
        out_dir / "assignments.csv", index=False)
    result = {"method": args.method, "model": model_name,
              "elapsed_sec": round(elapsed, 1), **m}
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n=== {model_name} ({elapsed:.1f}s) ===")
    print(f"  Purity={m['purity']:.3f} Retention={m['retention']:.3f} "
          f"NMI={m['nmi']:.3f} ARI={m['ari']:.3f} "
          f"(clusters={m['n_clusters']}, clustered={m['n_clustered']}/{m['n_total']})")
    print(f"  -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
