#!/usr/bin/env python
"""Is the BERT-vs-diffusion gap bigger than the measurement noise?

The layer/readout diagnostics reported single point estimates: one
``train_test_split(random_state=0)`` for the probe, one K-means seed for ARI.
The arm-vs-arm differences quoted from them are ~0.003-0.036, so without an
error bar it is unknown whether the reported *sign flip* under whitening
(probe delta +0.0363 raw -> -0.0055 whitened) is a real effect or resampling
noise. This script supplies the missing uncertainty.

Two things are measured:

1. **Paired uncertainty.** Both arms are scored on the *same* resample, so the
   arm difference is a paired statistic and split variance cancels out of it.
   Reported as mean +- std over repeats, plus the fraction of repeats where the
   sign agrees with the point estimate.
2. **Whether whitening's gain survives train/test discipline.** The original
   probe whitened using all 9,033 rows including the probe's own test rows --
   transductive. ``pcaw256-honest`` fits the whitening on the training split
   only, so any gain that was actually leakage disappears.

kNN@1 is deterministic given the features, so its uncertainty is sampling
noise over sequences and is estimated by a paired bootstrap instead.
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
K_SWEEP = list(range(10, 101, 5))
ARMS = {"bert": "bert_270m_49000", "diff": "diff_270m_42000"}


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def fit_pcaw(x: np.ndarray, k: int) -> dict:
    mean = x.mean(0, keepdims=True)
    _, sv, vt = np.linalg.svd(x - mean, full_matrices=False)
    k = min(k, len(sv))
    return {"mean": mean, "basis": vt[:k], "scale": np.sqrt(len(x)) / (sv[:k] + 1e-8)}


def apply_pcaw(x: np.ndarray, fit: dict) -> np.ndarray:
    return l2(((x - fit["mean"]) @ fit["basis"].T) * fit["scale"])


def readout(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return l2(x)
    if mode.startswith("pcaw"):
        return apply_pcaw(x, fit_pcaw(x, int(mode.split("-")[1])))
    raise ValueError(mode)


def probe_once(xtr, xte, ytr, yte) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1).fit(xtr, ytr)
    return float(
        roc_auc_score(yte, clf.predict_proba(xte), multi_class="ovr", average="macro")
    )


def probe_repeats(feats: dict[str, np.ndarray], y: np.ndarray, repeats: int) -> dict:
    """Paired probe over `repeats` stratified splits, for three readouts.

    ``pcaw256`` whitens on everything (the original, transductive protocol);
    ``pcaw256-honest`` whitens on the training rows only.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    modes = ["raw", "pcaw-256", "pcaw-256-honest"]
    pre = {
        (arm, m): readout(feats[arm], m)
        for arm in feats
        for m in ("raw", "pcaw-256")
    }
    out: dict[str, list[float]] = {f"{a}|{m}": [] for a in feats for m in modes}

    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.3, random_state=0)
    for i, (tr, te) in enumerate(sss.split(np.zeros(len(y)), y)):
        for arm, raw in feats.items():
            for m in modes:
                if m == "pcaw-256-honest":
                    fit = fit_pcaw(raw[tr], 256)
                    xtr, xte = apply_pcaw(raw[tr], fit), apply_pcaw(raw[te], fit)
                else:
                    x = pre[(arm, m)]
                    xtr, xte = x[tr], x[te]
                out[f"{arm}|{m}"].append(probe_once(xtr, xte, y[tr], y[te]))
        print(f"  probe split {i + 1}/{repeats} done", flush=True)
    return out


def knn_paired_bootstrap(feats: dict[str, np.ndarray], y: np.ndarray, boots: int) -> dict:
    """Per-sequence leave-one-out 1-NN correctness, then a paired bootstrap."""
    from sklearn.neighbors import NearestNeighbors

    hits: dict[str, np.ndarray] = {}
    for arm, raw in feats.items():
        for m in ("raw", "pcaw-256"):
            x = readout(raw, m)
            nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(x)
            _, idx = nn.kneighbors(x)
            hits[f"{arm}|{m}"] = (y[idx[:, 1]] == y).astype(np.float64)

    rng = np.random.default_rng(0)
    n = len(y)
    res = {"point": {k: float(v.mean()) for k, v in hits.items()}, "delta": {}}
    for m in ("raw", "pcaw-256"):
        d = hits[f"diff|{m}"] - hits[f"bert|{m}"]
        draws = np.array([d[rng.integers(0, n, n)].mean() for _ in range(boots)])
        res["delta"][m] = {
            "point": float(d.mean()),
            "std": float(draws.std()),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "p_sign_flip": float((draws <= 0).mean() if d.mean() > 0 else (draws >= 0).mean()),
        }
    return res


def ari_seeds(feats: dict[str, np.ndarray], y: np.ndarray, seeds: int) -> dict:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score

    out: dict[str, list[float]] = {}
    for arm, raw in feats.items():
        for m in ("raw", "pcaw-256"):
            x = readout(raw, m)
            vals = []
            for s in range(seeds):
                aris = [
                    adjusted_rand_score(
                        y,
                        MiniBatchKMeans(
                            n_clusters=k, random_state=s, n_init=3, batch_size=2048
                        ).fit_predict(x),
                    )
                    for k in K_SWEEP
                ]
                vals.append(float(np.mean(aris)))
            out[f"{arm}|{m}"] = vals
            print(f"  ari {arm}|{m}: {np.mean(vals):.4f} +- {np.std(vals):.4f}", flush=True)
    return out


def summarise(name: str, out: dict, modes: list[str]) -> None:
    print(f"\n### {name}: per-arm mean +- std over repeats")
    print(f"  {'readout':<20}{'BERT':>18}{'Diffusion':>18}")
    for m in modes:
        b = np.array(out[f"bert|{m}"])
        d = np.array(out[f"diff|{m}"])
        print(
            f"  {m:<20}{b.mean():>10.4f}+-{b.std():<6.4f}{d.mean():>10.4f}+-{d.std():<6.4f}"
        )

    print(f"\n### {name}: PAIRED delta (diffusion - BERT), same resample")
    print(f"  {'readout':<20}{'mean':>9}{'std':>9}{'|mean|/std':>11}  verdict")
    for m in modes:
        b = np.array(out[f"bert|{m}"])
        d = np.array(out[f"diff|{m}"])
        delta = d - b
        mu, sd = delta.mean(), delta.std(ddof=1)
        ratio = abs(mu) / sd if sd > 0 else float("inf")
        same = (np.sign(delta) == np.sign(mu)).mean()
        verdict = (
            f"consistent (sign holds {same:.0%} of repeats)"
            if ratio >= 2 and same >= 0.95
            else f"NOT resolvable (sign holds only {same:.0%})"
        )
        print(f"  {m:<20}{mu:>9.4f}{sd:>9.4f}{ratio:>11.2f}  {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--ari-seeds", type=int, default=10)
    ap.add_argument("--skip-ari", action="store_true")
    args = ap.parse_args()

    y = pd.read_csv(T2_CSV).fillna("")["epitope"].astype(str).to_numpy()
    feats = {
        arm: np.load(DIAG / tag / "per_layer.npy")[-1].astype(np.float64)
        for arm, tag in ARMS.items()
    }
    print(f"loaded {[(a, v.shape) for a, v in feats.items()]}, {len(set(y))} classes")

    result: dict = {}

    print("\n=== 1. kNN@1 paired bootstrap ===")
    knn = knn_paired_bootstrap(feats, y, args.boots)
    result["knn"] = knn
    print(json.dumps(knn, indent=2))

    print(f"\n=== 2. probe over {args.repeats} stratified splits ===")
    pr = probe_repeats(feats, y, args.repeats)
    result["probe"] = pr
    summarise("probe AUROC", pr, ["raw", "pcaw-256", "pcaw-256-honest"])

    if not args.skip_ari:
        print(f"\n=== 3. ARI over {args.ari_seeds} K-means seeds ===")
        ari = ari_seeds(feats, y, args.ari_seeds)
        result["ari"] = ari
        summarise("K-sweep ARI", ari, ["raw", "pcaw-256"])

    dest = DIAG / "noise_floor.json"
    dest.write_text(json.dumps(result, indent=2))
    print(f"\n-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
