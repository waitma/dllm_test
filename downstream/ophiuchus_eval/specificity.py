"""Ophiuchus-Ab official ckpt: HD vs Flu vs CoV specificity (Table 5).

Uses CurrAb's released 5 TTE folds when present. Frozen backbone,
``--use_multimer --sep_chains`` embeddings, then the ESM classification
head (AirGen / CurrAb protocol: lr 5e-5, batch 8; default 5 epochs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers.optimization import get_linear_schedule_with_warmup

from downstream.ophiuchus_eval.probe_common import embed_pairs, load_embedder, resolve_checkpoint

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DEFAULT_CSV = ROOT / "data/downstream/specificity/hd-0_flu-1_cov-2.csv"
DEFAULT_TTE = ROOT / "data/downstream/specificity/TTE"
DEFAULT_OUT = ROOT / "output/downstream_generation/ophiuchus_ab"


class EsmClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int = 3):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


def _load_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"h_sequence", "heavy", "vh", "vh_protein_sequence"}:
            rename[col] = "h_sequence"
        elif key in {"l_sequence", "light", "vl", "vl_protein_sequence"}:
            rename[col] = "l_sequence"
        elif key in {"label", "y"}:
            rename[col] = "label"
        elif key in {"name", "id"}:
            rename[col] = "name"
    df = df.rename(columns=rename)
    missing = [c for c in ("h_sequence", "l_sequence", "label") if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing {missing}; have {list(df.columns)}")
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def _pair_key(row: pd.Series) -> tuple[str, str, int]:
    return str(row["h_sequence"]), str(row["l_sequence"]), int(row["label"])


def load_tte_folds(tte_dir: Path) -> list[tuple[np.ndarray, np.ndarray]] | None:
    splits = []
    for fold in range(5):
        train_p = tte_dir / f"hd-0_flu-1_cov-2_train{fold}.csv"
        test_p = tte_dir / f"hd-0_flu-1_cov-2_test{fold}.csv"
        if not train_p.is_file() or not test_p.is_file():
            return None
        splits.append((train_p, test_p))
    return splits


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def train_esm_head(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int = 5,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    head = EsmClassificationHead(hidden_dim=x_train.shape[-1], num_classes=int(y_train.max().item()) + 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    steps = epochs * (len(x_train) // batch_size + 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    crit = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    pred = np.zeros(len(y_test), dtype=int)
    for epoch in range(1, epochs + 1):
        head.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(head(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        head.eval()
        with torch.no_grad():
            logits = head(x_test.to(device))
            test_loss = float(crit(logits, y_test.to(device)).item())
            pred = logits.argmax(dim=-1).cpu().numpy()
        row = {
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            "test_loss": test_loss,
            **_metrics(y_test.cpu().numpy(), pred),
        }
        history.append(row)
        if epoch == 1 or epoch == epochs or epoch % max(log_every, 1) == 0:
            print(
                f"  epoch {epoch}/{epochs} train={row['train_loss']:.4f} "
                f"test={row['test_loss']:.4f} acc={row['accuracy']:.4f}",
                flush=True,
            )
    return pred, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--tte-dir", type=Path, default=DEFAULT_TTE)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head", choices=("esm", "logreg"), default="esm")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--cls-batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-pt", type=Path, default=None)
    parser.add_argument("--metrics-name", type=str, default="specificity_hd_flu_cov_metrics.json")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    ckpt = resolve_checkpoint(args.checkpoint_path)
    tte = load_tte_folds(args.tte_dir)
    if tte is not None:
        df = pd.concat([_load_frame(te) for _, te in tte], ignore_index=True)
        print(f"using CurrAb TTE folds under {args.tte_dir}")
    else:
        df = _load_frame(args.csv)
        print(f"TTE folds missing; StratifiedKFold seed={args.seed}")
    print(f"specificity n={len(df)} ckpt={ckpt} device={device} head={args.head}")

    cache_pt = args.cache_pt or (args.out_dir / "specificity_embeddings.pt")
    cache_pt.parent.mkdir(parents=True, exist_ok=True)
    key_to_idx: dict[tuple[str, str, int], int] = {}
    if cache_pt.is_file():
        payload = torch.load(cache_pt, map_location="cpu")
        features = payload["features"]
        labels = payload["labels"]
        keys = payload.get("keys")
        print(f"loaded cache {cache_pt} {tuple(features.shape)}")
        if keys is not None:
            key_to_idx = {tuple(k): i for i, k in enumerate(keys)}
    else:
        model = load_embedder(ckpt, device, sep_chains=True)
        features = embed_pairs(
            model,
            df["h_sequence"].tolist(),
            df["l_sequence"].tolist(),
            batch_size=args.batch_size,
            device=device,
            desc="specificity",
        )
        labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
        keys = [_pair_key(row) for _, row in df.iterrows()]
        key_to_idx = {k: i for i, k in enumerate(keys)}
        torch.save({"features": features, "labels": labels, "keys": keys, "ckpt": str(ckpt)}, cache_pt)

    def rows_to_index(frame: pd.DataFrame) -> np.ndarray:
        if not key_to_idx:
            raise RuntimeError("embedding cache has no sequence keys; delete cache and rerun")
        return np.array([key_to_idx[_pair_key(row)] for _, row in frame.iterrows()], dtype=int)

    if tte is not None:
        fold_splits = []
        for train_p, test_p in tte:
            fold_splits.append((rows_to_index(_load_frame(train_p)), rows_to_index(_load_frame(test_p))))
    else:
        xdummy = np.zeros(len(df))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        fold_splits = list(skf.split(xdummy, df["label"].to_numpy()))

    fold_rows = []
    x = features
    y = labels
    all_hist: list[list[dict[str, float]]] = []
    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        if args.head == "logreg":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=4000, C=1.0, random_state=args.seed),
            )
            clf.fit(x[train_idx].numpy(), y[train_idx].numpy())
            pred = clf.predict(x[test_idx].numpy())
            hist: list[dict[str, float]] = []
        else:
            torch.manual_seed(args.seed + fold)
            pred, hist = train_esm_head(
                x[train_idx],
                y[train_idx],
                x[test_idx],
                y[test_idx],
                epochs=args.epochs,
                batch_size=args.cls_batch_size,
                lr=args.lr,
                device=device,
                log_every=args.log_every,
            )
        all_hist.append(hist)
        row = {"fold": fold, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), **_metrics(y[test_idx].numpy(), pred)}
        fold_rows.append(row)
        print(f"fold {fold}: {row}", flush=True)

    epoch_mean = []
    if all_hist and all_hist[0]:
        n_ep = len(all_hist[0])
        for i in range(n_ep):
            epoch_mean.append({
                "epoch": all_hist[0][i]["epoch"],
                "train_loss": float(np.mean([h[i]["train_loss"] for h in all_hist])),
                "test_loss": float(np.mean([h[i]["test_loss"] for h in all_hist])),
                "accuracy": float(np.mean([h[i]["accuracy"] for h in all_hist])),
                "f1_macro": float(np.mean([h[i]["f1_macro"] for h in all_hist])),
                "mcc": float(np.mean([h[i]["mcc"] for h in all_hist])),
            })

    summary = {
        "task": "specificity_hd_flu_cov",
        "protocol": (
            "frozen Ophiuchus-Ab --use_multimer --sep_chains; "
            f"head={args.head}; epochs={args.epochs}; lr={args.lr}; CurrAb TTE 5-fold"
            if tte is not None else
            f"frozen Ophiuchus-Ab --use_multimer --sep_chains; head={args.head}; epochs={args.epochs}; StratifiedKFold seed={args.seed}"
        ),
        "checkpoint": str(ckpt),
        "n": int(len(df)),
        "epochs": args.epochs,
        "lr": args.lr,
        "paper_ophiuchus_ab": {"accuracy": 0.6796, "f1": 0.6790, "mcc": 0.5203},
        "paper_currab": {"accuracy": 0.6796, "f1": 0.6801, "mcc": 0.5200},
        "mean": {
            "accuracy": float(np.mean([r["accuracy"] for r in fold_rows])),
            "f1_macro": float(np.mean([r["f1_macro"] for r in fold_rows])),
            "mcc": float(np.mean([r["mcc"] for r in fold_rows])),
        },
        "std": {
            "accuracy": float(np.std([r["accuracy"] for r in fold_rows])),
            "f1_macro": float(np.std([r["f1_macro"] for r in fold_rows])),
            "mcc": float(np.std([r["mcc"] for r in fold_rows])),
        },
        "epoch_mean": epoch_mean,
        "fold_histories": all_hist,
        "folds": fold_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / args.metrics_name
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["mean"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
