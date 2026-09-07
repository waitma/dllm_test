#!/usr/bin/env python
"""Is the decoder transforming the representation, or just passing it through?

Consumes the ``per_layer.npy`` / ``esmc.npy`` dumps written by
``diag_repr_layers.py --save-embeddings`` and reports, per layer:

* linear CKA against layer 0 (the decoder input = token embedding + projected
  ESMC condition). CKA near 1.0 means that layer carries the same information
  as the decoder input, i.e. the block stack is近似 identity.
* linear CKA against the raw ESMC encoder feature.
* R^2 of a ridge regression predicting the layer from the ESMC feature -- how
  much of the layer is a linear function of what ESMC already knew.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DIAG = ROOT / "output" / "repr_diagnostics"


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two representation matrices (rows = samples)."""
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    # ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    xty = x.T @ y
    num = float((xty**2).sum())
    xtx = x.T @ x
    yty = y.T @ y
    den = float(np.sqrt((xtx**2).sum()) * np.sqrt((yty**2).sum()))
    return num / (den + 1e-12)


def ridge_r2(src: np.ndarray, dst: np.ndarray, alpha: float = 1.0, seed: int = 0) -> float:
    """Held-out R^2 of predicting ``dst`` from ``src`` with ridge regression."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    xtr, xte, ytr, yte = train_test_split(src, dst, test_size=0.3, random_state=seed)
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    model = Ridge(alpha=alpha).fit((xtr - mu) / sd, ytr)
    pred = model.predict((xte - mu) / sd)
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--subsample", type=int, default=4000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for tag in args.tags:
        d = DIAG / tag
        per_layer = np.load(d / "per_layer.npy")
        esmc = np.load(d / "esmc.npy")
        n = per_layer.shape[1]
        idx = rng.choice(n, size=min(args.subsample, n), replace=False)
        pl = per_layer[:, idx, :].astype(np.float64)
        es = esmc[idx].astype(np.float64)

        print(f"\n=== {tag} ===")
        print(f"{'layer':>7} {'CKA(l,layer0)':>14} {'CKA(l,ESMC)':>12} {'R2(ESMC->l)':>12}")
        for li in range(pl.shape[0]):
            rec = {
                "tag": tag,
                "layer": li,
                "cka_layer0": round(linear_cka(pl[li], pl[0]), 4),
                "cka_esmc": round(linear_cka(pl[li], es), 4),
                "ridge_r2_esmc": round(ridge_r2(es, pl[li]), 4),
            }
            rows.append(rec)
            print(
                f"{li:>7} {rec['cka_layer0']:>14.4f} "
                f"{rec['cka_esmc']:>12.4f} {rec['ridge_r2_esmc']:>12.4f}"
            )

    import pandas as pd

    dest = DIAG / "layer_cka.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
