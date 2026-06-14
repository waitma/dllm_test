#!/usr/bin/env python
"""A1 -- NbBench antibody/nanobody head-only probing.

Frozen sequence embedder (ESM2 / Ophiuchus / BioSeq) + sklearn probe on the
official NbBench train/test splits. No backbone fine-tuning.

Examples:
    python nbbench/run.py --task VRClassification --embedder esm2_150m
    python nbbench/run.py --task all --embedder esm2_150m
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import metrics  # noqa: E402
from nbbench.tasks import NB_TASKS, data_dir  # noqa: E402

OUT = BENCH / "outputs" / "nbbench"


def _concat_seq(df: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    if len(columns) == 1:
        return df[columns[0]].astype(str).tolist()
    return [
        "|".join(vals).replace("nan", "")
        for vals in zip(*[df[c].astype(str) for c in columns])
    ]


def featurize(embedder_spec: str, train: pd.DataFrame, test: pd.DataFrame,
              columns: tuple[str, ...]):
    if embedder_spec == "kmer":
        from common.featurizers import KmerFeaturizer
        kf = KmerFeaturizer(k=3)
        if len(columns) == 1:
            tr_seqs = train[columns[0]].astype(str).tolist()
            te_seqs = test[columns[0]].astype(str).tolist()
            return kf.transform(tr_seqs), kf.transform(te_seqs), "kmer-3"
        # multi-column: embed each column separately then concat
        parts_tr, parts_te = [], []
        for c in columns:
            vals = pd.concat([train[c], test[c]]).astype(str)
            uniq = sorted(set(vals))
            e = kf.transform(uniq)
            lut = {u: e[i] for i, u in enumerate(uniq)}
            parts_tr.append(np.stack([lut[str(v)] for v in train[c]]))
            parts_te.append(np.stack([lut[str(v)] for v in test[c]]))
        return np.concatenate(parts_tr, 1), np.concatenate(parts_te, 1), "kmer-3"

    from common.model_api import build_embedder
    emb = build_embedder(embedder_spec)
    if len(columns) == 1:
        tr_seqs = train[columns[0]].astype(str).tolist()
        te_seqs = test[columns[0]].astype(str).tolist()
        all_seqs = sorted(set(tr_seqs) | set(te_seqs))
        e = emb.embed(all_seqs)
        lut = {s: e[i] for i, s in enumerate(all_seqs)}
        xtr = np.stack([lut[s] for s in tr_seqs])
        xte = np.stack([lut[s] for s in te_seqs])
        return xtr, xte, emb.name

    # Multi-chain: embed each column's *unique* values once, then concat per row.
    # Avoids O(N) embed calls when VHH×Ag pairs are mostly unique (hIL6/SARS).
    feats_tr, feats_te = [], []
    for c in columns:
        vals = pd.concat([train[c], test[c]]).astype(str)
        uniq = sorted(set(vals))
        print(f"      embedding {len(uniq)} unique '{c}' ...")
        e = emb.embed(uniq)
        lut = {u: e[i] for i, u in enumerate(uniq)}
        feats_tr.append(np.stack([lut[str(v)] for v in train[c]]))
        feats_te.append(np.stack([lut[str(v)] for v in test[c]]))
    return np.concatenate(feats_tr, 1), np.concatenate(feats_te, 1), emb.name


def run_classification(xtr, ytr, xte, yte, multiclass: bool, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, random_state=seed),
    )
    clf.fit(xtr, ytr)
    if multiclass:
        pred = clf.predict(xte)
        probe = metrics.linear_probe_metrics(xtr, ytr, xte, yte,
                                             multiclass=True, seed=seed)
        return {"probe_acc": probe["probe_acc"],
                "probe_auroc": probe.get("probe_auroc", float("nan")),
                "predictions": pred}
    scores = np.asarray(clf.predict_proba(xte)[:, 1]).ravel()
    yte = np.asarray(yte).ravel()
    return {
        "probe_auroc": float(metrics._safe_auroc(yte, scores)),
        "probe_acc": float(np.mean((scores >= 0.5).astype(int) == yte)),
        "scores": scores,
    }


def run_regression(xtr, ytr, xte, yte, seed: int):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_squared_error

    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
    reg.fit(xtr, ytr)
    pred = reg.predict(xte)
    rho, _ = spearmanr(yte, pred)
    return {
        "spearman": float(rho) if not np.isnan(rho) else float("nan"),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "predictions": pred,
    }


def evaluate_task(task_name: str, embedder_spec: str, seed: int) -> dict:
    task = NB_TASKS[task_name]
    ddir = data_dir(task_name)
    train = pd.read_csv(ddir / "train.csv").fillna("")
    test = pd.read_csv(ddir / "test.csv").fillna("")
    ytr = train[task.label_column]
    yte = test[task.label_column]

    if task.task_type == "classification":
        ytr = ytr.astype(int).to_numpy()
        yte = yte.astype(int).to_numpy()
        multiclass = len(np.unique(ytr)) > 2
    else:
        ytr = ytr.astype(float).to_numpy()
        yte = yte.astype(float).to_numpy()
        multiclass = False

    print(f"[{task_name}] train={len(train)} test={len(test)} "
          f"type={task.task_type} embedder={embedder_spec}")
    t0 = time.time()
    xtr, xte, model_name = featurize(embedder_spec, train, test, task.seq_columns)

    if task.task_type == "classification":
        result = run_classification(xtr, ytr, xte, yte, multiclass, seed)
    else:
        result = run_regression(xtr, ytr, xte, yte, seed)

    elapsed = time.time() - t0
    out = {
        "task": task_name,
        "task_type": task.task_type,
        "embedder": embedder_spec,
        "model": model_name,
        "metric": task.metric,
        "primary": result.get(task.metric, float("nan")),
        "n_train": len(train),
        "n_test": len(test),
        "elapsed_sec": round(elapsed, 1),
        **{k: v for k, v in result.items() if k not in ("scores", "predictions")},
    }
    out_dir = OUT / task_name / embedder_spec
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  {task.metric}={out['primary']:.4f} ({elapsed:.1f}s) -> {out_dir}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="VRClassification",
                    help="task name or 'all'")
    ap.add_argument("--embedder", default="esm2_150m")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = list(NB_TASKS) if args.task == "all" else [args.task]
    summary = []
    for t in tasks:
        if t not in NB_TASKS:
            raise SystemExit(f"unknown task: {t}; available: {sorted(NB_TASKS)}")
        summary.append(evaluate_task(t, args.embedder, args.seed))

    if len(summary) > 1:
        out_all = OUT / f"_summary_{args.embedder}.json"
        out_all.parent.mkdir(parents=True, exist_ok=True)
        with out_all.open("w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\n=== summary ({len(summary)} tasks) -> {out_all} ===")
        for row in summary:
            print(f"  {row['task']:22s} {row['metric']}={row['primary']:.4f}")


if __name__ == "__main__":
    main()
