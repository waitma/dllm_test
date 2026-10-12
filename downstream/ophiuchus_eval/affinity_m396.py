"""Ophiuchus-Ab official ckpt: m396 in-silico affinity (Fig 4 / AbMAP low-N).

MINT protocol: embed WT (PDB 2G75) and each mutant with --use_multimer --sep_chains,
take WT − mutant, Ridge(alpha=0.01). AbMAP splits 0.5%–20%. Spearman averaged
over the five energy functions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from downstream.ophiuchus_eval.probe_common import embed_pairs, load_embedder, resolve_checkpoint

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DEFAULT_CSV = ROOT / "data/downstream/in_silico/Desautels_insilico_data.csv"
# Full-length RCSB 2G75 (245+213). Do not use rcsb_pdb_2G75_vr.fasta (Fv 118+107).
DEFAULT_FASTA = ROOT / "data/downstream/in_silico/rcsb_pdb_2G75.fasta"
DEFAULT_OUT = ROOT / "output/downstream_generation/ophiuchus_ab"
HEAVY_SPLIT = 245
SCORE_COLS = [
    "FoldX_Average_Whole_Model_DDG",
    "FoldX_Average_Interface_Only_DDG",
    "Statium",
    "Sum_of_Rosetta_Flex_single_point_mutations",
    "Sum_of_Rosetta_Total_Energy_single_point_mutations",
]
SPLITS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / np.clip(x.std(axis=0), 1e-8, None)


def read_wt_fasta(path: Path) -> tuple[str, str]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    heavy = light = None
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            seq = lines[i + 1] if i + 1 < len(lines) else ""
            header = lines[i].lower()
            if "light" in header or heavy is not None:
                light = seq
            else:
                heavy = seq
            i += 2
            continue
        i += 1
    if not heavy or not light:
        raise ValueError(f"need heavy+light in {path}")
    return heavy, light


def split_concat_sequence(sequence: str) -> tuple[str, str]:
    return sequence[:HEAVY_SPLIT], sequence[HEAVY_SPLIT:]


def load_desautels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    present = [c for c in SCORE_COLS if c in df.columns]
    if len(present) < 3:
        raise ValueError(f"{csv_path} missing energy columns; have {list(df.columns)}")
    if "Statium" in df.columns:
        df = df[~df["Statium"].isnull()].copy()
    if "h_sequence" in df.columns and "l_sequence" in df.columns:
        df["h_sequence"] = df["h_sequence"].astype(str)
        df["l_sequence"] = df["l_sequence"].astype(str)
    elif "Antibody_Sequence" in df.columns:
        pairs = df["Antibody_Sequence"].astype(str).map(split_concat_sequence)
        df["h_sequence"] = [p[0] for p in pairs]
        df["l_sequence"] = [p[1] for p in pairs]
    else:
        raise ValueError(f"{csv_path} needs Antibody_Sequence or h/l_sequence")
    df = df.reset_index(drop=True)
    return df, present


def score_multitarget(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    spearmans = []
    pearsons = []
    mses = []
    for i in range(pred.shape[1]):
        spearmans.append(float(spearmanr(target[:, i], pred[:, i]).correlation))
        pearsons.append(float(pearsonr(target[:, i], pred[:, i])[0]))
        mses.append(float(mean_squared_error(target[:, i], pred[:, i])))
    return {
        "spearman_mean": float(np.mean(spearmans)),
        "pearson_mean": float(np.mean(pearsons)),
        "mse_mean": float(np.mean(mses)),
        "spearman_per_target": spearmans,
        "pearson_per_target": pearsons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--wt-fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--split-seed", type=int, default=2023)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-pt", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    ckpt = resolve_checkpoint(args.checkpoint_path)
    df, score_cols = load_desautels(args.csv)
    wt_heavy, wt_light = read_wt_fasta(args.wt_fasta)
    print(f"m396 n={len(df)} scores={score_cols} ckpt={ckpt} device={device}")

    cache_pt = args.cache_pt or (args.out_dir / "m396_delta_embeddings.pt")
    cache_pt.parent.mkdir(parents=True, exist_ok=True)
    if cache_pt.is_file():
        payload = torch.load(cache_pt, map_location="cpu")
        delta = payload["delta"]
        print(f"loaded cache {cache_pt} {tuple(delta.shape)}")
    else:
        model = load_embedder(ckpt, device, sep_chains=True)
        wt_emb = embed_pairs(
            model, [wt_heavy], [wt_light], batch_size=1, device=device, desc="wt"
        )[0]
        mut_emb = embed_pairs(
            model,
            df["h_sequence"].tolist(),
            df["l_sequence"].tolist(),
            batch_size=args.batch_size,
            device=device,
            desc="m396",
        )
        delta = wt_emb.unsqueeze(0) - mut_emb
        torch.save(
            {"delta": delta, "mut": mut_emb, "wt": wt_emb, "ckpt": str(ckpt)},
            cache_pt,
        )

    x_all = delta.numpy()
    y_all = df[score_cols].to_numpy(dtype=np.float64)
    split_rows = []
    for frac in SPLITS:
        rng_df = df.copy()
        train_df = rng_df.sample(frac=frac, random_state=args.split_seed)
        test_df = rng_df.drop(train_df.index)
        train_idx = train_df.index.to_numpy()
        test_idx = test_df.index.to_numpy()
        y_train = standardize(y_all[train_idx])
        y_test = standardize(y_all[test_idx])
        ridge = Ridge(alpha=args.ridge_alpha)
        ridge.fit(x_all[train_idx], y_train)
        pred = ridge.predict(x_all[test_idx])
        metrics = score_multitarget(pred, y_test)
        row = {
            "train_frac": frac,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            **{k: v for k, v in metrics.items() if k.endswith("_mean")},
            "spearman_per_target": metrics["spearman_per_target"],
            "targets": score_cols,
        }
        split_rows.append(row)
        print(
            f"split {frac}: n_train={row['n_train']} spearman={row['spearman_mean']:.4f} "
            f"pearson={row['pearson_mean']:.4f}"
        )

    summary = {
        "task": "affinity_m396_desautels",
        "protocol": "frozen Ophiuchus-Ab --use_multimer --sep_chains; WT(2G75)−mut; Ridge(alpha=0.01); AbMAP low-N fracs; labels z-scored within split",
        "note": "labels are in-silico ΔΔG (FoldX/Rosetta/STATIUM), not wet-lab Kd",
        "checkpoint": str(ckpt),
        "n": int(len(df)),
        "paper": "Ophiuchus-Ab Fig 4: beats MINT/AbMAP on all splits; +6% vs MINT at 0.5% train",
        "splits": split_rows,
    }
    out = args.out_dir / "affinity_m396_metrics.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
