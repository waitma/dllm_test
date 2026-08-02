#!/usr/bin/env python
"""A1 -- local NbBench antibody/nanobody linear-probe diagnostic.

Frozen sequence embedder (ESM2 / Ophiuchus / BioSeq) + sklearn probe on the
released NbBench train/test splits. No backbone fine-tuning.  This is not the
paper's learned-MLP/three-seed protocol: paper Table 5 values are imported as
``paper_reported`` rows and remain the primary external baselines.

Examples:
    python nbbench/run.py --task VRClassification --embedder esm2_150m
    python nbbench/run.py --task all --embedder esm2_150m
"""

from __future__ import annotations

import argparse
import json
import re
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


def out_dirname(spec: str, tag: str = "") -> str:
    """Clean, filesystem-safe leaderboard dir name for an embedder spec.

    Baseline specs (``esm2_150m`` / ``kmer`` / ``onehot`` / ``protbert`` ...)
    have no special characters and are used verbatim, so existing dirs keep
    working. A path-carrying spec like ``bioseq:/abs/x/<run>/best.pt`` would
    otherwise create nested dirs from its slashes, so we derive
    ``<kind>_<run>`` (e.g. ``bioseq_grammar_v2_esmc600m_cmp500k_llada``). An
    explicit ``--tag`` always wins (recommended for concise names).
    """
    if tag:
        return tag
    if "/" in spec or ":" in spec:
        kind = spec.split(":", 1)[0]
        run = Path(spec.split(":", 1)[-1]).parent.name
        if run:
            return f"{kind}_{run}"
        return re.sub(r"[^0-9A-Za-z_.-]+", "_", spec)
    return spec


def _concat_seq(df: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    if len(columns) == 1:
        return df[columns[0]].astype(str).tolist()
    return [
        "|".join(vals).replace("nan", "")
        for vals in zip(*[df[c].astype(str) for c in columns])
    ]


# GrammarRenderer hard-limits single chains to 1024 aa (grammar.py ppi_max_protein_length).
# NbBench binding tasks (hTNFa/hIL6/SARS-CoV-2) can carry longer antigens; PALM baselines
# truncate via tokenizer. For post-LLaDA grammar embedders we take the N-terminal head-1024
# before embed — aligned with NBBENCH_DESIGN.md L3 long-antigen truncation caveat.
_GRAMMAR_MAX_CHAIN_LEN = 1024
_ANTIGEN_COLUMNS = frozenset({"Ag_sequence"})


def _is_post_llada_grammar_spec(embedder_spec: str) -> bool:
    if embedder_spec.startswith("bioseq-llada:"):
        return True
    if embedder_spec.startswith("grammar:"):
        rest = embedder_spec.split(":", 1)[1]
        source = rest.split(":", 1)[0]
        return source not in ("encoder",)
    return False


def _maybe_truncate_antigen(seqs: list[str], column: str, embedder_spec: str) -> list[str]:
    if column not in _ANTIGEN_COLUMNS or not _is_post_llada_grammar_spec(embedder_spec):
        return seqs
    cap = _GRAMMAR_MAX_CHAIN_LEN
    out, n_trunc = [], 0
    for s in seqs:
        if len(s) > cap:
            out.append(s[:cap])
            n_trunc += 1
        else:
            out.append(s)
    if n_trunc:
        print(f"      head-{cap} truncate: {n_trunc}/{len(seqs)} unique '{column}' "
              f"(NbBench L3 long-antigen caveat, grammar renderer limit)")
    return out


def _embed_key(seq: str, column: str, embedder_spec: str) -> str:
    """Key for embed lookup — must match sequences passed to ``embed()``."""
    s = str(seq)
    if column in _ANTIGEN_COLUMNS and _is_post_llada_grammar_spec(embedder_spec):
        cap = _GRAMMAR_MAX_CHAIN_LEN
        if len(s) > cap:
            return s[:cap]
    return s


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
        uniq = _maybe_truncate_antigen(uniq, c, embedder_spec)
        print(f"      embedding {len(uniq)} unique '{c}' ...")
        e = emb.embed(uniq)
        lut = {u: e[i] for i, u in enumerate(uniq)}
        feats_tr.append(np.stack([lut[_embed_key(v, c, embedder_spec)] for v in train[c]]))
        feats_te.append(np.stack([lut[_embed_key(v, c, embedder_spec)] for v in test[c]]))
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


def evaluate_task(task_name: str, embedder_spec: str, seed: int, tag: str = "") -> dict:
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
        "protocol": "local_sklearn_logreg_or_ridge_single_seed_train_test_only",
        "split_usage": {"train": True, "validation": False, "test": True},
        "baseline_provenance": {
            "evidence_type": "local_reimplementation",
            "protocol_alignment": "not_comparable",
            "paper_comparable": False,
            "citation": "doi:10.1088/2632-2153/ae20ec; arXiv:2505.02022",
            "source_location": "local sklearn probe on released NbBench CSV splits",
            "notes": "paper uses a learned MLP head, validation split, three seeds, and fixed ESM2-650M antigen features",
        },
        **{k: v for k, v in result.items() if k not in ("scores", "predictions")},
    }
    out_dir = OUT / task_name / out_dirname(embedder_spec, tag)
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
    ap.add_argument("--tag", default="",
                    help="clean leaderboard dir name (recommended for path-"
                         "carrying specs like bioseq:/abs/best.pt)")
    args = ap.parse_args()

    tasks = list(NB_TASKS) if args.task == "all" else [args.task]
    summary = []
    for t in tasks:
        if t not in NB_TASKS:
            raise SystemExit(f"unknown task: {t}; available: {sorted(NB_TASKS)}")
        summary.append(evaluate_task(t, args.embedder, args.seed, args.tag))

    if len(summary) > 1:
        out_all = OUT / f"_summary_{out_dirname(args.embedder, args.tag)}.json"
        out_all.parent.mkdir(parents=True, exist_ok=True)
        with out_all.open("w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\n=== summary ({len(summary)} tasks) -> {out_all} ===")
        for row in summary:
            print(f"  {row['task']:22s} {row['metric']}={row['primary']:.4f}")


if __name__ == "__main__":
    main()
