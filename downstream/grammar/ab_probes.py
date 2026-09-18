"""Frozen AB probes on our native ESMC + LLaDA antibody representations.

Run: python -m downstream.grammar.ab_probes --task specificity --checkpoint /abs/ckpt --out-dir /abs/new_run
Preflight without loading weights: python -m downstream.grammar.ab_probes --task m396 --validate-only
Use separate non-preemptible one-GPU Volc jobs for specificity, gdp_a1, m396.
GDPa1 uses the Ophiuchus reference CV recipe only; nested CV is not run.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, PredefinedSplit, cross_val_predict
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import DataLoader, TensorDataset
from transformers.optimization import get_linear_schedule_with_warmup

from downstream.grammar.ab_features import ROOT, cached_features, clean_sequence, file_sha256, json_hash, write_json
from downstream.ophiuchus_eval.affinity_m396 import SCORE_COLS, SPLITS, load_desautels, read_wt_fasta, score_multitarget
from downstream.ophiuchus_eval.developability import ALL_COLS, FOLD_COL, LAMBDA_GRID
from downstream.ophiuchus_eval.specificity import EsmClassificationHead, _metrics

DATA = ROOT / "data/downstream"


def rho(y, pred):
    return float(spearmanr(y, pred).statistic)


def sequence_pairs(df, heavy="h_sequence", light="l_sequence"):
    return [(clean_sequence(h), clean_sequence(l)) for h, l in zip(df[heavy], df[light])]


def load_specificity():
    """Require the existing five folds, exact membership, and one OOF use per ID."""
    base = DATA / "specificity"
    paths = [base / "hd-0_flu-1_cov-2.csv"]
    df = pd.read_csv(paths[0])
    if df["name"].duplicated().any() or set(df.label) != {0, 1, 2}:
        raise ValueError("Specificity IDs/classes invalid")
    pairs = sequence_pairs(df)
    ids = {str(name): i for i, name in enumerate(df.name)}
    seen = np.zeros(len(df), dtype=int)
    folds = []
    for k in range(5):
        idx = []
        for side in ("train", "test"):
            path = base / "TTE" / f"hd-0_flu-1_cov-2_{side}{k}.csv"
            frame = pd.read_csv(path)  # Missing official fold is an error, not a fallback.
            paths.append(path)
            if frame.name.duplicated().any():
                raise ValueError(f"Repeated IDs: {path}")
            selected = np.array([ids[str(name)] for name in frame.name])
            if sequence_pairs(frame) != [pairs[i] for i in selected]:
                raise ValueError(f"Fold sequence mismatch: {path}")
            if not np.array_equal(frame.label.to_numpy(), df.label.to_numpy()[selected]):
                raise ValueError(f"Fold label mismatch: {path}")
            idx.append(selected)
        tr, te = idx
        if set(tr) & set(te) or set(tr) | set(te) != set(range(len(df))):
            raise ValueError("Invalid outer fold partition")
        if {pairs[i] for i in tr} & {pairs[i] for i in te}:
            raise ValueError("Identical sequence pair crosses specificity train/test")
        seen[te] += 1
        folds.append((tr, te))
    if not (seen == 1).all():
        raise ValueError("Every specificity sample must be tested exactly once")
    return df, pairs, paths, folds


def prepare_task(task):
    if task == "specificity":
        return load_specificity()
    if task == "gdp_a1":
        path = DATA / "dev/GDPa1_v1.2_20250814.csv"
        df = pd.read_csv(path)
        pairs = sequence_pairs(df, "vh_protein_sequence", "vl_protein_sequence")
        if df[FOLD_COL].isna().any() or set(df[FOLD_COL]) != set(range(5)):
            raise ValueError("GDPa1 requires its predefined five folds")
        assignments = {}
        for pair, fold in zip(pairs, df[FOLD_COL]):
            if pair in assignments and assignments[pair] != fold:
                raise ValueError("Identical GDPa1 sequence pair crosses folds")
            assignments[pair] = fold
        return df, pairs, [path], None
    path = DATA / "in_silico/Desautels_insilico_data.csv"
    wt_path = DATA / "in_silico/rcsb_pdb_2G75.fasta"
    df, cols = load_desautels(path)
    if cols != SCORE_COLS or not np.isfinite(df[cols].to_numpy(dtype=float)).all():
        raise ValueError("m396 requires all five finite target values")
    pairs = sequence_pairs(df)
    wt = read_wt_fasta(wt_path)
    if tuple(map(len, wt)) != (245, 213):
        raise ValueError("m396 WT must use full H245/L213")
    return df, [wt] + pairs, [path, wt_path], None


def fit_selected_head(x_train, y_train, *, epochs, batch_size, lr, seed, device, epoch_observer=None):
    """Fixed-last training; optional diagnostics never select a state or change LR."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head = EsmClassificationHead(x_train.shape[1], 3).to(device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    steps = epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        total = 0.0
        first_lr = optimizer.param_groups[0]["lr"]
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(xb.to(device)), yb.to(device))
            loss.backward()
            last_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(xb)
        history.append({"epoch": epoch, "train_loss": total / len(y_train)})
        if epoch_observer is not None:
            epoch_observer(head, {**history[-1], "lr_first_step": first_lr,
                "lr_last_step": last_lr, "lr_next_step": optimizer.param_groups[0]["lr"]})
    head.eval()
    return head, history


def specificity_epoch_observer(x_test, y_test, folder, *, final_epoch, save_every=20):
    """Record held-out curves only; no early stopping, tuning, or epoch selection."""
    rows = []

    def observe(head, train_row):
        was_training = head.training
        head.eval()
        device = next(head.parameters()).device
        with torch.no_grad():
            logits = head(x_test.to(device))
            test_loss = float(torch.nn.functional.cross_entropy(logits, y_test.to(device)))
        scores = _metrics(y_test.cpu().numpy(), logits.argmax(-1).cpu().numpy())
        row = {**train_row, "test_loss": test_loss, **scores, "diagnostic_only": True}
        rows.append(row)
        write_json(folder / "epoch_metrics.json", rows)
        epoch = train_row["epoch"]
        if epoch % save_every == 0 or epoch == final_epoch:
            torch.save({"state_dict": {n: v.detach().cpu() for n, v in head.state_dict().items()},
                        "hidden_dim": x_test.shape[1], "epoch": epoch, "diagnostic_only": True},
                       folder / f"diagnostic_head_epoch_{epoch:03d}.pt")
        print(f"specificity {folder.name} epoch {epoch}/{final_epoch}: {row}", flush=True)
        head.train(was_training)

    return observe


def run_specificity(df, x, folds, args):
    out = args.out_dir
    y = torch.tensor(df.label.to_numpy(), dtype=torch.long)
    predictions, metrics = [], []
    for k, (tr, te) in enumerate(folds):
        folder = out / f"fold_{k}"
        folder.mkdir(exist_ok=True)
        observer = (specificity_epoch_observer(x[te], y[te], folder, final_epoch=args.epochs)
                    if getattr(args, "log_epoch_metrics", False) else None)
        head, history = fit_selected_head(x[tr], y[tr], epochs=args.epochs,
            batch_size=args.head_batch_size, lr=args.lr, seed=args.seed + k, device=args.device,
            epoch_observer=observer)
        saved = {"state_dict": {n: v.detach().cpu() for n, v in head.state_dict().items()},
                 "hidden_dim": x.shape[1], "selected_epoch": args.epochs,
                 "selection_rule": "fixed_last", "seed": args.seed + k}
        torch.save(saved, folder / "selected_head.pt")
        # Final predictions are made from the persisted selected state.
        restored = EsmClassificationHead(x.shape[1], 3).to(args.device).eval()
        restored.load_state_dict(torch.load(folder / "selected_head.pt", map_location="cpu",
                                             weights_only=False)["state_dict"])
        with torch.no_grad():
            logits = restored(x[te].to(args.device)).cpu()
            original_logits = head(x[te].to(args.device)).cpu()
            if not torch.allclose(logits, original_logits, rtol=1e-6, atol=1e-6) or not torch.equal(logits.argmax(-1), original_logits.argmax(-1)):
                raise ValueError("Selected-head save/load changed predictions")
        pred = logits.argmax(-1).numpy()
        scores = _metrics(y[te].numpy(), pred)
        metrics.append({"fold": k, "n_train": len(tr), "n_test": len(te), **scores})
        write_json(folder / "selection.json", {"selected_epoch": args.epochs,
                   "rule": "fixed_last_no_test_selection", "seed": args.seed + k,
                   "train_ids": df.name.iloc[tr].tolist(), "test_ids": df.name.iloc[te].tolist()})
        write_json(folder / "train_history.json", history)
        frame = pd.DataFrame({"sample_id": df.name.iloc[te].to_numpy(), "fold": k,
                              "y_true": y[te].numpy(), "y_pred": pred})
        for c in range(3):
            frame[f"logit_{c}"] = logits[:, c].numpy()
            frame[f"prob_{c}"] = logits.softmax(-1)[:, c].numpy()
        frame.to_csv(folder / "predictions.csv", index=False)
        predictions.append(frame)
        print(f"specificity fold {k}: {scores}", flush=True)
    pd.concat(predictions).to_csv(out / "oof_predictions.csv", index=False)
    if getattr(args, "log_epoch_metrics", False):
        curves = [json.loads((out / f"fold_{k}" / "epoch_metrics.json").read_text()) for k in range(len(folds))]
        fields = ("train_loss", "test_loss", "accuracy", "f1_macro", "mcc", "lr_first_step", "lr_last_step", "lr_next_step")
        epoch_mean = [{"epoch": e + 1, "diagnostic_only": True,
                       **{key: float(np.mean([curve[e][key] for curve in curves])) for key in fields}}
                      for e in range(args.epochs)]
        write_json(out / "epoch_mean.json", epoch_mean)
    return {"protocol": "native_ab_fixed_last_head_5fold_v1", "folds": metrics,
            "mean": {key: float(np.mean([r[key] for r in metrics])) for key in scores},
            "sample_sd_ddof1": {key: float(np.std([r[key] for r in metrics], ddof=1)) for key in scores},
            "selected_epoch": args.epochs, "aggregation": "mean_of_selected_fold_metrics",
            "disclosure": "Historical baseline experiments inspected these outer test folds; not a new blind test."}


def run_gdp(df, x, args):
    """Run reference CV only; retain legacy artifact names for compatibility."""
    properties = {}
    for col in ALL_COLS:
        valid = df[col].notna().to_numpy()
        indices = np.flatnonzero(valid)
        xx, y = x[valid].numpy(), df.loc[valid, col].to_numpy(dtype=float)
        fold = df.loc[valid, FOLD_COL].to_numpy(dtype=int)
        folder = args.out_dir / col
        folder.mkdir(exist_ok=True)
        scorer = make_scorer(rho)
        # User-selected reference recipe: all-label transform and same-CV alpha
        # selection/reporting. This is not an independent outer-test estimate.
        transform = PowerTransformer().fit(y[:, None])
        yt = transform.transform(y[:, None]).ravel()
        cv = PredefinedSplit(fold)
        legacy = GridSearchCV(Ridge(), {"alpha": LAMBDA_GRID}, scoring=scorer, cv=cv)
        legacy.fit(xx, yt)
        legacy_pred = cross_val_predict(legacy.best_estimator_, xx, yt, cv=cv)
        joblib.dump({"ridge": legacy.best_estimator_, "transformer_all_labels": transform}, folder / "legacy_reference.joblib")
        np.savez_compressed(folder / "legacy_oof.npz", row_index=indices, fold=fold,
                            y_true=y, y_transformed=yt, prediction=legacy_pred)
        properties[col] = {"n": len(y),
            "legacy_reference": {"protocol": "all_label_transform_same_cv_alpha_selection",
                "cv_mean_spearman": float(legacy.best_score_), "pooled_oof_spearman": rho(y, legacy_pred),
                "alpha": float(legacy.best_params_["alpha"])}}
        print(f"GDPa1 {col}: {properties[col]}", flush=True)
    return {"protocol": "native_ab_gdp_reference_only_v2", "properties": properties}


def m396_splits(df, groups, seed):
    for mode in ("row_reference", "sequence_grouped"):
        for frac in SPLITS:
            if mode == "row_reference":
                tr = df.sample(frac=frac, random_state=seed).index.to_numpy()
                te = df.drop(tr).index.to_numpy()
            else:
                splitter = GroupShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
                tr, te = next(splitter.split(np.zeros(len(df)), groups=groups))
            yield mode, frac, tr, te


def run_m396(df, features, args):
    x = (features[0:1] - features[1:]).numpy()
    y = df[SCORE_COLS].to_numpy(dtype=float)
    groups = np.array([json_hash(pair) for pair in sequence_pairs(df)])
    rows = []
    for mode, frac, tr, te in m396_splits(df, groups, args.split_seed):
        overlap = int(np.isin(groups[te], groups[tr]).sum())
        if mode == "sequence_grouped" and overlap:
            raise ValueError("Sequence group crosses m396 split")
        mean, std = y[tr].mean(0), np.clip(y[tr].std(0), 1e-8, None)
        model = Ridge(alpha=args.ridge_alpha).fit(x[tr], (y[tr] - mean) / std)
        prediction = model.predict(x[te])
        if mode == "row_reference":
            target = (y[te] - y[te].mean(0)) / np.clip(y[te].std(0), 1e-8, None)
            scale = "train_and_test_separately_zscored_reference_recipe"
        else:
            prediction = prediction * std + mean
            target = y[te]
            scale = "train_only_transform_predictions_restored_to_raw_energy"
        folder = args.out_dir / f"{mode}_{frac:g}"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / "predictions.npz", train_index=tr, test_index=te,
                            y_true=target, prediction=prediction)
        joblib.dump({"ridge": model, "train_mean": mean, "train_std": std}, folder / "selected_ridge.joblib")
        row = {"split_protocol": mode, "train_fraction": frac, "n_train": len(tr), "n_test": len(te),
               "fraction_unit": "rows" if mode == "row_reference" else "unique_sequence_groups",
               "test_rows_sequence_seen_in_train": overlap, "target_scale": scale,
               "train_index_sha256": json_hash(tr.tolist()), "test_index_sha256": json_hash(te.tolist()),
               "targets": SCORE_COLS, **score_multitarget(prediction, target)}
        rows.append(row)
        print(f"m396 {mode} {frac}: {row['spearman_mean']}", flush=True)
    return {"protocol": "native_full_length_ab_wt_minus_mut_ridge_v1", "splits": rows,
            "n": len(df), "n_unique_sequences": len(set(groups)),
            "note": "Computational energy, not wet-lab Kd. Row and group budgets/protocols must not be mixed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("specificity", "gdp_a1", "m396"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log-epoch-metrics", action="store_true",
                        help="Specificity-only diagnostic held-out curves; fixed-last reporting remains unchanged")
    parser.add_argument("--head-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=2023)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.log_epoch_metrics and args.task != "specificity":
        parser.error("--log-epoch-metrics is only supported for specificity")
    df, pairs, paths, folds = prepare_task(args.task)
    if args.validate_only:
        print(json.dumps({"task": args.task, "n": len(df), "n_pair_inputs": len(pairs),
                          "max_H_L_lengths": [max(len(p[i]) for p in pairs) for i in (0, 1)],
                          "data_sha256": {str(p): file_sha256(p) for p in paths}}, indent=2))
        return
    if args.checkpoint is None or args.out_dir is None:
        parser.error("--checkpoint and --out-dir are required for evaluation")
    if min(args.epochs, args.head_batch_size, args.embedding_batch_size, args.max_length) < 1:
        parser.error("Epochs, batch sizes and max length must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.ridge_alpha) or args.ridge_alpha < 0:
        parser.error("Invalid learning rate or ridge alpha")
    if (args.out_dir / "metrics.json").exists():
        raise FileExistsError("Completed metrics exist; use a fresh output directory")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"task": args.task, "model": "ours_esmc_llada_fusion",
                "runtime_versions": {"torch": str(torch.__version__), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                    "joblib": joblib.__version__, "cuda": torch.version.cuda},
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "data_sha256": {str(p): file_sha256(p) for p in paths},
                "runner_sha256": file_sha256(Path(__file__)), "sample_ids": df.name.tolist()
                if args.task == "specificity" else df.index.tolist()}
    manifest_path = args.out_dir / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Run provenance changed; choose a new output directory")
    write_json(manifest_path, manifest)
    features, feature_manifest = cached_features(pairs, checkpoint=args.checkpoint, out_dir=args.out_dir,
        device=args.device, batch_size=args.embedding_batch_size, max_length=args.max_length)
    if args.task == "specificity":
        result = run_specificity(df, features, folds, args)
    elif args.task == "gdp_a1":
        result = run_gdp(df, features, args)
    else:
        result = run_m396(df, features, args)
    write_json(args.out_dir / "metrics.json", {"task": args.task, "model": "ours_esmc_llada_fusion",
        "feature_manifest": feature_manifest, "run_manifest_sha256": file_sha256(manifest_path), **result})


if __name__ == "__main__":
    main()
