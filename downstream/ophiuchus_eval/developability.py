"""Ophiuchus-Ab official ckpt: GDPa1 developability (Table 4).

Matches AirGen ``dev/finetune_dev.py``: frozen backbone, ``--use_multimer
--sep_chains``, PowerTransformer on labels, Ridge alpha grid, predefined
isotype-stratified folds. Headline paper columns are AC-SINS / HIC / PR_CHO /
Titer; Tm2 is in the official script and is reported as extra.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_predict
from sklearn.preprocessing import PowerTransformer

from downstream.ophiuchus_eval.probe_common import embed_pairs, load_embedder, resolve_checkpoint

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DEFAULT_CSV = ROOT / "data/downstream/dev/GDPa1_v1.2_20250814.csv"
DEFAULT_OUT = ROOT / "output/downstream_generation/ophiuchus_ab"
PAPER_COLS = ("AC-SINS_pH7.4", "HIC", "PR_CHO", "Titer")
ALL_COLS = (*PAPER_COLS, "Tm2")
FOLD_COL = "hierarchical_cluster_IgG_isotype_stratified_fold"
PAPER_OPHIUCHUS = {
    "AC-SINS_pH7.4": 0.511,
    "HIC": 0.550,
    "PR_CHO": 0.460,
    "Titer": 0.359,
}
LAMBDA_GRID = np.logspace(3, -6, num=10).tolist() + [0.0]


def spearman_corr(y_true, y_pred) -> float:
    return float(spearmanr(y_true, y_pred).correlation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-pt", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    ckpt = resolve_checkpoint(args.checkpoint_path)
    df = pd.read_csv(args.csv)
    if FOLD_COL not in df.columns:
        raise ValueError(f"{args.csv} missing {FOLD_COL}")
    heavy = df["vh_protein_sequence"].astype(str).str.replace("J", "L")
    light = df["vl_protein_sequence"].astype(str).str.replace("J", "L")
    print(f"gdp_a1 n={len(df)} ckpt={ckpt} device={device}")

    cache_pt = args.cache_pt or (args.out_dir / "gdp_a1_embeddings.pt")
    cache_pt.parent.mkdir(parents=True, exist_ok=True)
    if cache_pt.is_file():
        payload = torch.load(cache_pt, map_location="cpu")
        features = payload["features"]
        print(f"loaded cache {cache_pt} {tuple(features.shape)}")
    else:
        model = load_embedder(ckpt, device, sep_chains=True)
        features = embed_pairs(
            model,
            heavy.tolist(),
            light.tolist(),
            batch_size=args.batch_size,
            device=device,
            desc="gdp_a1",
        )
        torch.save({"features": features, "ckpt": str(ckpt)}, cache_pt)

    x_all = features.numpy()
    properties = {}
    for col in ALL_COLS:
        mask = df[col].notna().to_numpy()
        x = x_all[mask]
        y = df.loc[mask, col].to_numpy(dtype=np.float64)
        folds = df.loc[mask, FOLD_COL].to_numpy()
        y_trans = PowerTransformer().fit_transform(y.reshape(-1, 1)).ravel()
        cv = PredefinedSplit(folds)
        clf = GridSearchCV(
            Ridge(),
            {"alpha": LAMBDA_GRID},
            cv=cv,
            scoring=make_scorer(spearman_corr, greater_is_better=True),
        )
        clf.fit(x, y_trans)
        pred = cross_val_predict(clf.best_estimator_, x, y_trans, cv=cv)
        raw = spearman_corr(y, pred)
        properties[col] = {
            "n": int(mask.sum()),
            "best_alpha": float(clf.best_params_["alpha"]),
            "cv_spearman_transformed": float(clf.best_score_),
            "cv_spearman_raw_vs_pred": raw,
            "paper": PAPER_OPHIUCHUS.get(col),
        }
        print(
            f"{col}: n={properties[col]['n']} "
            f"cv={properties[col]['cv_spearman_transformed']:.3f} "
            f"paper={PAPER_OPHIUCHUS.get(col)}"
        )

    summary = {
        "task": "developability_gdp_a1",
        "protocol": (
            "frozen Ophiuchus-Ab --use_multimer --sep_chains; "
            "PowerTransformer on all labels; Ridge GridSearchCV; "
            "PredefinedSplit(hierarchical_cluster_IgG_isotype_stratified_fold)"
        ),
        "checkpoint": str(ckpt),
        "n": int(len(df)),
        "paper_ophiuchus_ab": dict(PAPER_OPHIUCHUS),
        "headline_mean_paper_cols": float(
            np.mean([properties[c]["cv_spearman_transformed"] for c in PAPER_COLS])
        ),
        "properties": properties,
    }
    out = args.out_dir / "developability_gdp_a1_metrics.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
