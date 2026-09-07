#!/usr/bin/env python
"""How much signal is actually inside the FINAL decoder feature?

Constraint for this study: the representation must be the decoder's last
hidden state (``hidden_states[-1]``, post-``ln_f``, mean-pooled over residues).
Nothing else -- no intermediate layers, no ESMC encoder features.

Given that constraint the open question is whether the BERT arm's final feature
is genuinely information-poor, or whether it holds comparable information in a
geometry that un-centered cosine similarity cannot exploit. So we hold the
feature fixed and vary only the *readout*, from the current headline
(L2-normalise and go) up to a supervised linear probe.

Reads the ``per_layer.npy`` dumps from ``diag_repr_layers.py --save-embeddings``
and scores layer -1 with the official 19-point K sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DIAG = ROOT / "output" / "repr_diagnostics"
T2_CSV = ROOT / "downstream" / "benchmark" / "data" / "tcr_clustering_embed" / "tcrs.csv"

# official basis-B sweep: K in {10, 15, ..., 100}
K_SWEEP = list(range(10, 101, 5))


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def readout(x: np.ndarray, mode: str) -> np.ndarray:
    """Post-process the final decoder feature. All variants end in L2-norm,
    matching what the benchmark does before cosine / K-means."""
    if mode == "raw":
        return l2(x)
    if mode == "center":
        return l2(x - x.mean(0, keepdims=True))
    if mode == "zscore":
        mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
        return l2((x - mu) / (sd + 1e-6))
    if mode.startswith("abtt"):
        # "all-but-the-top" (Mu & Viswanath 2018): centre, then project out the
        # top-d PCs that carry the shared anisotropic direction.
        d = int(mode.split("-")[1])
        xc = x - x.mean(0, keepdims=True)
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
        top = vt[:d]
        return l2(xc - (xc @ top.T) @ top)
    if mode.startswith("pcaw"):
        # PCA whitening: equalise variance across the retained directions.
        k = int(mode.split("-")[1])
        xc = x - x.mean(0, keepdims=True)
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
        k = min(k, len(s))
        return l2(u[:, :k] * np.sqrt(len(xc)))
    raise ValueError(mode)


def cluster_scores(x, true, ks, seed=0) -> dict:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    aris, nmis, purs = [], [], []
    for k in ks:
        lab = MiniBatchKMeans(
            n_clusters=k, random_state=seed, n_init=3, batch_size=2048
        ).fit_predict(x)
        aris.append(adjusted_rand_score(true, lab))
        nmis.append(normalized_mutual_info_score(true, lab))
        df = pd.DataFrame({"c": lab, "t": true})
        purs.append(
            df.groupby("c")["t"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df)
        )
    return {
        "ari_mean": round(float(np.mean(aris)), 4),
        "ari_best": round(float(np.max(aris)), 4),
        "nmi_mean": round(float(np.mean(nmis)), 4),
        "purity_mean": round(float(np.mean(purs)), 4),
    }


def knn_top1(x, y, seed=0) -> float:
    """Leave-one-out 1-NN accuracy under cosine -- unsupervised, matches the
    spirit of the T3 nearest-neighbour protocol."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(x)
    _, idx = nn.kneighbors(x)
    return float((y[idx[:, 1]] == y).mean())


def probe(x, y, seed=0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    yy = LabelEncoder().fit_transform(y)
    xtr, xte, ytr, yte = train_test_split(
        x, yy, test_size=0.3, random_state=seed, stratify=yy
    )
    clf = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1).fit(xtr, ytr)
    p = clf.predict_proba(xte)
    return {
        "probe_auroc": round(
            float(roc_auc_score(yte, p, multi_class="ovr", average="macro")), 4
        ),
        "probe_acc": round(float((clf.predict(xte) == yte).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument(
        "--modes",
        default="raw,center,zscore,abtt-7,pcaw-256",
        help="readouts applied to the final decoder feature",
    )
    args = ap.parse_args()

    df = pd.read_csv(T2_CSV).fillna("")
    true = df["epitope"].astype(str).to_numpy()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    rows = []
    for tag in args.tags:
        x = np.load(DIAG / tag / "per_layer.npy")[-1].astype(np.float64)
        print(f"\n=== {tag}  final decoder feature  {x.shape} ===")
        print(
            f"  {'readout':<12}{'ARI':>8}{'ARI_best':>10}{'NMI':>8}{'Purity':>8}"
            f"{'kNN@1':>8}{'probe':>8}"
        )
        for mode in modes:
            xp = readout(x, mode)
            rec = {"tag": tag, "readout": mode}
            rec.update(cluster_scores(xp, true, K_SWEEP))
            rec["knn_top1"] = round(knn_top1(xp, true), 4)
            rec.update(probe(xp, true))
            rows.append(rec)
            print(
                f"  {mode:<12}{rec['ari_mean']:>8.4f}{rec['ari_best']:>10.4f}"
                f"{rec['nmi_mean']:>8.4f}{rec['purity_mean']:>8.4f}"
                f"{rec['knn_top1']:>8.4f}{rec['probe_auroc']:>8.4f}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    dest = DIAG / "final_readout_ladder.csv"
    out.to_csv(dest, index=False)
    print(f"\n-> {dest}")

    if len(args.tags) == 2:
        a, b = args.tags
        print(f"\n=== gap ({b} minus {a}) on the same final-layer feature ===")
        print(f"  {'readout':<12}{'dARI':>10}{'dkNN@1':>10}{'dprobe':>10}")
        for mode in modes:
            ra = out[(out.tag == a) & (out.readout == mode)].iloc[0]
            rb = out[(out.tag == b) & (out.readout == mode)].iloc[0]
            print(
                f"  {mode:<12}{rb.ari_mean - ra.ari_mean:>+10.4f}"
                f"{rb.knn_top1 - ra.knn_top1:>+10.4f}"
                f"{rb.probe_auroc - ra.probe_auroc:>+10.4f}"
            )


if __name__ == "__main__":
    main()
