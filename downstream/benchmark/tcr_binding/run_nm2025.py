#!/usr/bin/env python
"""T1 local probes/controls on the Nature Methods 2025 splits.

Scores the official seen / unseen test sets (produced by
``scripts/prepare_nm2025_binding.py``) with one method and writes predictions +
metrics under ``outputs/tcr_binding_nm2025/<tag>/<track>/<split>/``. Metrics use
the official-parity report (per-epitope + overall AUPRC/AUROC; headline AUPRC).

Methods
-------
random   : uniform random scores (sanity floor).
knn      : CDR3 edit-distance kNN vs the epitope's training binders (training-free).
epitcr   : legacy compatibility entry for the official pretrained epiTCR RF.
embed    : frozen sequence embedder + head. ``--head {logreg,mlp}``. Features =
           concat of pooled per-column embeddings (e.g. emb(cdr3b) | emb(peptide)).
           Our model enters via ``--embedder grammar:decoder:/abs/best.pt`` (or
           plain ``grammar:/abs/best.pt``, which also defaults to the decoder):
           each column is rendered as a single-chain grammar record and pooled
           from the **LLaDA decoder** final hidden states (post-LLaDA, the correct
           our-model口径). ``grammar:encoder:/abs/best.pt`` forces the ESMC-encoder
           feature source (contextless baseline, kept only for comparison). This is
           the retrained-probe protocol: train the head on the official train.csv
           frozen features, evaluate on official seen/unseen.

This file is not the published-baseline runner.  Published original-model
baselines must use ``run_official_baseline.py``: official checkpoint + official
inference code + tests derived from ``original.zip``, with no ``train.csv``.

Examples
--------
    python tcr_binding/run_nm2025.py --method knn --track cdr3b
    python tcr_binding/run_nm2025.py --method embed --track cdr3b \
        --embedder grammar:decoder:/abs/best.pt --head mlp --tag ours_postllada_mlp
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

from common import schema, metrics, featurizers  # noqa: E402

DATA = BENCH / "data" / "tcr_binding_nm2025"
OUT = BENCH / "outputs" / "tcr_binding_nm2025"

# (split, neg-source) test files per track.
TRACK_TESTS = {
    "cdr3b": [("seen", "AS"), ("seen", "PS"), ("seen", "HS"),
              ("unseen", "AS"), ("unseen", "PS"), ("unseen", "HS")],
    "others": [("seen", "AS"), ("unseen", "AS")],
}
DEFAULT_COLUMNS = {"cdr3b": ["cdr3b", "peptide"],
                   "others": ["cdr3b", "cdr3a", "peptide"]}


def _load_tests(track: str):
    """Load only the author test artifacts; never touch ``train.csv``."""

    tests = {}
    for split, neg in TRACK_TESTS[track]:
        p = DATA / split / track / neg / "test.csv"
        if p.exists():
            tests[(split, neg)] = pd.read_csv(p).fillna("")
    return tests


def _load_track(track: str):
    """Load train + test for local kNN/frozen-head probes only."""

    train = pd.read_csv(DATA / "train" / track / "train.csv").fillna("")
    tests = _load_tests(track)
    return train, tests


def _split_dirname(split: str, neg: str) -> str:
    return split if neg == "AS" else f"{split}_{neg}"


def join_predictions_to_test(pred, test, split: str, neg: str, method: str = ""):
    """Attach the test ``id`` to a ``predictions.csv`` frame, or return ``None``.

    Two id conventions coexist in ``outputs/tcr_binding_nm2025``: methods run
    through this module keep the native test id (``nm2025_<split>_<track>_<neg>-<i>``),
    while the official wrappers emit a positional ``<split>_<neg>_<i>``. Some
    wrappers also reorder their output rows, so the positional form is joined by
    row index and accepted **only** when (label, group) agree on every row; a
    wrapper that reordered is reported as unjoinable rather than silently
    mis-joined.

    Any offline analysis that needs to subset a stored prediction file by test-row
    identity should go through this helper instead of assuming an id convention.
    """
    pred = pred.copy()
    pred["id"] = pred["id"].astype(str)
    if set(pred["id"]) & set(test["id"].astype(str)):
        return pred

    prefix = f"{split}_{neg}_"
    if not pred["id"].str.startswith(prefix).all():
        print(f"  [warn] {method}: unrecognised id convention "
              f"(e.g. {pred['id'].iloc[0]!r}), skip")
        return None
    idx = pred["id"].str.slice(len(prefix)).astype(int)
    if idx.min() < 0 or idx.max() >= len(test):
        print(f"  [warn] {method}: positional index out of range, skip")
        return None
    mapped = test.iloc[idx.to_numpy()].reset_index(drop=True)
    pv = pred.reset_index(drop=True)
    agree = (mapped["label"].to_numpy() == pv["label"].to_numpy()) & (
        mapped["group"].astype(str).to_numpy() == pv["group"].astype(str).to_numpy())
    if not agree.all():
        print(f"  [warn] {method}: positional join rejected "
              f"({int((~agree).sum())}/{len(agree)} rows disagree on label/group; "
              f"wrapper reorders its output), skip")
        return None
    pv["id"] = mapped["id"].to_numpy()
    return pv


# ---------------------------------------------------------------------------
# Embedding + heads
# ---------------------------------------------------------------------------

def _embed_columns(embedder, frames: list[pd.DataFrame], columns):
    """Embed unique values per column once across all frames; return per-frame X.

    This is the CONTEXTLESS per-chain featurizer: every column (CDR3β / peptide /
    …) is embedded on its own and the pooled vectors are concatenated. For our
    model this is the ESMC-encoder ablation口径 (``grammar:encoder:`` /
    ``bioseq:``): each chain is encoded in isolation, so the TCR never "sees" its
    epitope. The joint post-LLaDA口径 lives in :func:`_build_joint_decoder`.
    """
    lut_per_col = {}
    for col in columns:
        vals = pd.concat([f[col] for f in frames]).astype(str)
        uniq = sorted(set(vals))
        emb = embedder.embed(uniq)
        lut_per_col[col] = ({u: emb[i] for i, u in enumerate(uniq)}, emb.shape[1])
        print(f"      embedded {len(uniq)} unique '{col}' -> dim {emb.shape[1]}")

    def build(df):
        feats = []
        for col in columns:
            lut, dim = lut_per_col[col]
            feats.append(np.stack([lut[str(v)] for v in df[col]]))
        return np.concatenate(feats, axis=1)

    return build


def _build_joint_decoder(embedder, columns):
    """JOINT post-LLaDA featurizer for our grammar model (``feature_source=decoder``).

    The whole TCR-epitope record (CDR3β [+ CDR3α] + peptide) is rendered as ONE
    ``task_type=tcr`` grammar record, pushed through the clean ESMC→LLaDA pipeline,
    and **globally mean-pooled** over all residue positions → one ``[hidden_dim]``
    vector per sample (``pool_mode="global"``, the our-model headline口径).
    """
    beta_col = "cdr3b" if "cdr3b" in columns else columns[0]
    alpha_col = "cdr3a" if "cdr3a" in columns else None
    pep_col = "peptide" if "peptide" in columns else None
    order = tuple(s for s, present in (
        ("beta", True), ("alpha", alpha_col is not None),
        ("peptide", pep_col is not None)) if present)
    print(f"      JOINT post-LLaDA embed_pairs: beta='{beta_col}' "
          f"alpha={alpha_col!r} peptide={pep_col!r} segments={order}")

    def build(df):
        beta = df[beta_col].astype(str).tolist()
        alpha = df[alpha_col].astype(str).tolist() if alpha_col else None
        pep = df[pep_col].astype(str).tolist() if pep_col else None
        return embedder.embed_pairs(beta=beta, alpha=alpha, peptide=pep, order=order)

    return build


def _train_mlp(xtr, ytr, seed, epochs=200, hidden=256, patience=15, val_frac=0.1):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n = len(xtr)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    mu = xtr[tr_idx].mean(0)
    sd = xtr[tr_idx].std(0) + 1e-6
    xn = (xtr - mu) / sd

    Xtr = torch.tensor(xn[tr_idx], dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr[tr_idx], dtype=torch.float32, device=device)
    Xval = torch.tensor(xn[val_idx], dtype=torch.float32, device=device)
    yval_t = torch.tensor(ytr[val_idx], dtype=torch.float32, device=device)

    model = nn.Sequential(
        nn.Linear(xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden // 2, 1),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()

    bs = 512
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xtr[b]).squeeze(-1)
            loss = lossf(out, ytr_t[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xval).squeeze(-1), yval_t).item()
        if vloss < best_val - 1e-4:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(x):
        xt = torch.tensor((x - mu) / sd, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.sigmoid(model(xt).squeeze(-1)).cpu().numpy()

    return predict, {"best_val_loss": round(best_val, 4), "epochs_run": ep + 1,
                     "n_train": int(len(tr_idx)), "n_val": int(len(val_idx))}


def run_embed(train, tests, embedder_spec, columns, head, seed):
    from common.model_api import build_embedder
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    print(f"[embed] building embedder '{embedder_spec}' ...")
    embedder = build_embedder(embedder_spec)
    frames = [train] + list(tests.values())
    # Our model's correct口径 = JOINT post-LLaDA: a grammar embedder whose feature
    # source is the LLaDA decoder is featurized with embed_pairs (TCR + epitope in
    # one record, cross-chain contextual), NOT per-column in isolation. All other
    # embedders (encoder ablation, ESM2, …) keep the contextless per-column path.
    joint_decoder = getattr(embedder, "feature_source", None) == "decoder"
    if joint_decoder:
        print(f"[embed] JOINT post-LLaDA (embed_pairs) featurizer over columns {columns}")
        build = _build_joint_decoder(embedder, columns)
    else:
        print(f"[embed] per-column (contextless) featurizer over columns {columns} "
              f"across {len(frames)} frames ...")
        build = _embed_columns(embedder, frames, columns)
    xtr = build(train)
    ytr = train["label"].astype(int).to_numpy()
    print(f"[embed] train features {xtr.shape}; fitting head='{head}' ...")

    info = {}
    if head == "mlp":
        predict, info = _train_mlp(xtr, ytr, seed)
    else:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=1.0, random_state=seed))
        clf.fit(xtr, ytr)
        predict = lambda x: clf.predict_proba(x)[:, 1]

    scores = {key: predict(build(df)) for key, df in tests.items()}
    return scores, embedder.name, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _metrics_block(df_pred):
    """Flat metric block for one (split, neg) run.

    ``auprc_*`` keep sklearn ``average_precision_score`` for backward
    compatibility with every existing summary; ``auprc_*_precrec`` carry the
    estimator the Nature Methods 2025 paper actually used
    (``precrec::evalmod``). Quote the ``_precrec`` variants whenever the value
    lands in a table beside a paper-reported number -- sklearn AP is one-sided
    high by roughly +0.001 (seen) / +0.002 (unseen). See RESULTS.md section L4.
    """
    off = metrics.official_binding_report(df_pred["label"], df_pred["score"], df_pred["group"])
    leg = metrics.binding_metrics(df_pred["label"], df_pred["score"], df_pred["group"])
    return {
        "auprc_macro": off["macro"]["prc_auc"],
        "auprc_macro_precrec": off["macro"]["prc_auc_precrec"],
        "auroc_macro": off["macro"]["roc_auc"],
        "auprc_overall": off["all_values"]["prc_auc"],
        "auprc_overall_precrec": off["all_values"]["prc_auc_precrec"],
        "auroc_overall": off["all_values"]["roc_auc"],
        "auc01_macro": leg["macro_auc01"],
        "accuracy_overall": off["all_values"]["accuracy"],
        "mcc_overall": off["all_values"]["mcc"],
        "n_epitopes": off["n_epitopes"],
        "official_report": off,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["random", "knn", "embed", "epitcr"], default="knn")
    ap.add_argument("--track", choices=["cdr3b", "others"], default="cdr3b")
    ap.add_argument("--embedder", default="grammar:decoder:"
                    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/"
                    "grammar_v2_esmc300m_cmp500k_llada/best.pt")
    ap.add_argument("--head", choices=["logreg", "mlp"], default="mlp")
    ap.add_argument("--epitcr-mhc", action="store_true")
    ap.add_argument("--columns", nargs="+", default=None)
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    columns = args.columns or DEFAULT_COLUMNS[args.track]
    if args.method in {"knn", "embed"}:
        train, tests = _load_track(args.track)
        train_info = str(len(train))
    else:
        train = None
        tests = _load_tests(args.track)
        train_info = "not loaded"
    print(
        f"track={args.track} train={train_info} "
        f"tests={ {f'{s}/{n}': len(d) for (s,n),d in tests.items()} }"
    )

    t0 = time.time()
    scores_by_test: dict[tuple, np.ndarray] = {}
    info = {}
    if args.method == "random":
        rng = np.random.default_rng(args.seed)
        for key, df in tests.items():
            scores_by_test[key] = rng.random(len(df))
        model_name = "Random"
    elif args.method == "knn":
        assert train is not None
        print(f"[knn] fitting CDR3 edit-distance kNN (k={args.knn_k}) ...")
        scorer = featurizers.DistanceKNNScorer(k=args.knn_k, use_alpha=(args.track == "others")).fit(train)
        for key, df in tests.items():
            scores_by_test[key] = scorer.score(df)
        model_name = f"CDR3-kNN(k={args.knn_k})"
    elif args.method == "epitcr":
        from baselines.wrappers.epitcr import predict_scores
        print("[deprecated] external epiTCR reporting should use run_official_baseline.py")
        print(f"[epitcr] loading official RF (mhc={args.epitcr_mhc}) ...")
        for key, df in tests.items():
            scores_by_test[key] = predict_scores(df, use_mhc=args.epitcr_mhc)
        model_name = f"epiTCR-{'MHC' if args.epitcr_mhc else 'noMHC'}"
    else:
        assert train is not None
        scores_by_test, model_name, info = run_embed(
            train, tests, args.embedder, columns, args.head, args.seed)
    elapsed = time.time() - t0

    if args.tag:
        tag = args.tag
    elif args.method == "embed":
        spec = args.embedder or ""
        if spec.startswith("grammar:"):
            # grammar[:decoder|encoder]:/abs/best.pt -> ours_{postllada|enc}_{run}_{head}
            rest = spec.split(":", 1)[1]
            if rest.split(":", 1)[0] in ("decoder", "encoder"):
                source, path = rest.split(":", 1)
            else:
                source, path = "decoder", rest
            run = (Path(path).parent.name or Path(path).stem).replace("grammar_v2_", "")
            tag = f"ours_{'postllada' if source == 'decoder' else 'enc'}_{run}_{args.head}"
        else:
            tag = f"{spec.split(':')[0]}_{args.head}"
    elif args.method == "epitcr":
        tag = f"epitcr_{'mhc' if args.epitcr_mhc else 'nomhc'}"
    else:
        tag = args.method

    summary = {}
    for (split, neg), df in tests.items():
        out_dir = OUT / tag / args.track / _split_dirname(split, neg)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred = pd.DataFrame({
            "id": df["id"], "score": scores_by_test[(split, neg)],
            "label": df["label"].astype(int), "group": df["peptide"], "seen": split,
        })
        pred.to_csv(out_dir / "predictions.csv", index=False)
        block = _metrics_block(pred)
        result = {
            "method": args.method, "model": model_name, "track": args.track,
            "split": split, "neg_source": neg, "columns": columns if args.method == "embed" else None,
            "embedder": args.embedder if args.method == "embed" else None,
            "head": args.head if args.method == "embed" else None,
            "elapsed_sec": round(elapsed, 1), "head_info": info,
            **block,
        }
        with (out_dir / "metrics.json").open("w") as fh:
            json.dump(result, fh, indent=2)
        summary[f"{split}/{neg}"] = block

    print(f"\n=== {model_name} | track={args.track} ({elapsed:.1f}s) ===")
    for key in sorted(summary):
        b = summary[key]
        print(f"  {key:12s}: AUPRC={b['auprc_macro']:.3f}(macro)/{b['auprc_overall']:.3f}(all) "
              f"AUROC={b['auroc_macro']:.3f}/{b['auroc_overall']:.3f} n_ep={b['n_epitopes']}")
    print(f"  -> {OUT / tag / args.track}")


if __name__ == "__main__":
    main()
