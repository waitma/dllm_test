#!/usr/bin/env python
"""T3 -- TCR representation quality (epitope specificity).

Two protocols on a **clonotype-disjoint** train/test split (overlap=0):

- ``fewshot`` (**main**, SCEPTR / Cell Systems 2024 protocol): for each target
  epitope and each support-set size ``k``, sample ``k`` reference binders from
  the train pool and rank the whole test set by nearest-neighbour distance to
  that support set; report per-epitope AUROC macro-averaged over epitopes,
  vs. shots. This is the regime where SCEPTR showed generic PLMs (ESM2/ProtBert)
  lose to sequence-comparison methods (TCRdist / CDR3 Levenshtein).
- ``probe`` (auxiliary, backward-compatible): the original 24-way linear probe
  (logistic regression) + 1-NN top-1 over frozen embeddings.

Methods:
- ``levenshtein`` : CDR3b(+CDR3a) Levenshtein NN (zero extra deps, main control).
- ``tcrdist``     : official tcrdist3 distances (main control).
- ``sceptr``      : official SCEPTR contrastive model (native cdist + 64-d probe).
- ``embed``       : an embedder via ``--embedder`` (esm2_*, kmer, ophiuchus,
                    bioseq:/abs/best.pt), cosine-distance few-shot + probe.

Few-shot results -> ``outputs/tcr_representation/<tag>/fewshot.json``;
probe results    -> ``outputs/tcr_representation/<tag>/metrics.json``.

Examples:
    python tcr_representation/run.py --method levenshtein
    python tcr_representation/run.py --method sceptr
    python tcr_representation/run.py --method embed --embedder esm2_150m
    python tcr_representation/run.py --method embed --embedder bioseq:/abs/best.pt
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

from common import metrics  # noqa: E402
from common import fewshot as fs  # noqa: E402
from common.provenance import checkpoint_provenance  # noqa: E402

DATA = BENCH / "data" / "tcr_representation"
OUT = BENCH / "outputs" / "tcr_representation"


class _KmerEmbedder:
    """Adapter so the k-mer composition featurizer plugs into the embed backend."""

    def __init__(self, k: int = 3):
        from common.featurizers import KmerFeaturizer
        self._kf = KmerFeaturizer(k=k)
        self.name = f"kmer-{k}"
        self.dim = self._kf.dim

    def embed(self, seqs):
        return self._kf.transform([str(s) for s in seqs])


def _tag_for(method: str, embedder: str | None) -> str:
    if method != "embed":
        return method
    spec = embedder or ""
    if spec.startswith("grammar:"):
        # grammar:decoder:/abs/best.pt (LLaDA decoder multi-chain, our correct口径)
        rest = spec.split(":", 1)[1]
        if rest.split(":", 1)[0] in ("decoder", "encoder"):
            source, path = rest.split(":", 1)
        else:
            source, path = "decoder", rest
        ckpt = Path(path)
        run = (ckpt.parent.name or ckpt.stem).replace("grammar_v2_", "")
        # decoder = post-LLaDA joint peptide-free paired record (grammar_name
        # "tcr_pair"); encoder = ESMC condition mean-pool. Tag reflects口径.
        return f"grammar_{'tcrpair' if source == 'decoder' else 'enc'}_{run}"
    if spec.startswith("bioseq:"):
        ckpt = Path(spec.split(":", 1)[1])
        # name after the checkpoint's parent dir, e.g. grammar_v2_esmc300m_cmp500k_llada
        return "bioseq_" + (ckpt.parent.name or ckpt.stem)
    return spec.replace("/", "_").replace(":", "_")


def _build_embed_backend(embedder_spec: str):
    if embedder_spec == "kmer":
        return _KmerEmbedder(k=3)
    from common.model_api import build_embedder
    return build_embedder(embedder_spec)


def featurize_probe(embedder, train, test, columns):
    """Build probe features; grammar decoder uses one joint global-pooled vector."""
    if getattr(embedder, "feature_source", None) == "decoder":
        from common.fewshot import grammar_pair_embed
        return (grammar_pair_embed(embedder, train, columns),
                grammar_pair_embed(embedder, test, columns))
    feats_tr, feats_te = [], []
    for c in columns:
        vals = pd.concat([train[c], test[c]]).astype(str)
        uniq = sorted(set(vals))
        e = embedder.embed(uniq)
        lut = {u: e[i] for i, u in enumerate(uniq)}
        feats_tr.append(np.stack([lut[str(v)] for v in train[c]]))
        feats_te.append(np.stack([lut[str(v)] for v in test[c]]))
    return np.concatenate(feats_tr, 1), np.concatenate(feats_te, 1)


def run_probe(method, embedder_spec, train, test, columns, seed):
    """24-way linear probe + kNN top-1 (auxiliary protocol)."""
    classes = sorted(set(train["peptide"]) | set(test["peptide"]))
    cls2id = {c: i for i, c in enumerate(classes)}
    ytr = train["peptide"].map(cls2id).to_numpy()
    yte = test["peptide"].map(cls2id).to_numpy()

    if method == "sceptr":
        from common.model_api import build_distance_source
        src = build_distance_source("sceptr")
        # SCEPTR embeds the whole paired TCR at once (not per-column).
        xtr = src.embed(train)
        xte = src.embed(test)
        model_name = src.name
    elif method == "embed":
        emb = _build_embed_backend(embedder_spec)
        xtr, xte = featurize_probe(emb, train, test, columns)
        model_name = emb.name
    else:
        return None  # distance-only methods have no probe features

    probe = metrics.linear_probe_metrics(xtr, ytr, xte, yte, multiclass=True, seed=seed)
    knn = metrics.knn_top1_metric(xtr, ytr, xte, yte, metric="cosine")
    return {"model": model_name, "n_classes": len(classes), **probe, **knn}


def run_fewshot(method, embedder_spec, train, test, columns, shots, seeds, agg):
    """Few-shot per-epitope NN AUROC (main protocol)."""
    kwargs = dict(columns=columns, shots=shots, seeds=seeds, agg=agg)
    if method == "levenshtein":
        res = fs.evaluate_fewshot(train, test, method="editdist", **kwargs)
        res["model"] = "CDR3-Levenshtein-NN"
    elif method == "sceptr":
        from common.model_api import build_distance_source
        src = build_distance_source("sceptr")
        res = fs.evaluate_fewshot(train, test, method="native", cdist_fn=src.cdist, **kwargs)
        res["model"] = src.name
    elif method == "tcrdist":
        from common.model_api import build_distance_source
        src = build_distance_source("tcrdist")
        res = fs.evaluate_fewshot(train, test, method="native", cdist_fn=src.cdist, **kwargs)
        res["model"] = src.name
    elif method == "embed":
        emb = _build_embed_backend(embedder_spec)
        res = fs.evaluate_fewshot(train, test, method="embed", embedder=emb, **kwargs)
        res["model"] = emb.name
    else:
        raise ValueError(f"unknown method: {method}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["fewshot", "probe", "both"], default="both")
    ap.add_argument("--method", choices=["levenshtein", "tcrdist", "sceptr", "embed"],
                    default="levenshtein")
    ap.add_argument("--embedder", default=None,
                    help="for --method embed: esm2_150m / kmer / ophiuchus / bioseq:/abs/best.pt")
    ap.add_argument("--columns", nargs="+", default=["cdr3b", "cdr3a"])
    ap.add_argument("--shots", nargs="+", type=int, default=list(fs.DEFAULT_SHOTS))
    ap.add_argument("--seeds", type=int, default=fs.DEFAULT_SEEDS)
    ap.add_argument("--agg", choices=["min", "mean"], default="min")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.method == "embed" and not args.embedder:
        ap.error("--method embed requires --embedder")

    train = pd.read_csv(DATA / "train.csv").fillna("")
    test = pd.read_csv(DATA / "test.csv").fillna("")
    tag = args.tag or _tag_for(args.method, args.embedder)
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"method={args.method} tag={tag} protocol={args.protocol} "
          f"train={len(train)} test={len(test)} columns={args.columns}")

    if args.protocol in ("fewshot", "both"):
        t0 = time.time()
        res = run_fewshot(args.method, args.embedder, train, test,
                          args.columns, args.shots, args.seeds, args.agg)
        res["elapsed_sec"] = round(time.time() - t0, 1)
        res["tag"] = tag
        if args.method == "embed" and args.embedder:
            prov = checkpoint_provenance(args.embedder)
            if prov is not None:
                res["checkpoint"] = prov
        with (out_dir / "fewshot.json").open("w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\n=== few-shot per-epitope NN AUROC :: {res['model']} "
              f"({res['elapsed_sec']}s) ===")
        for k in res["shots"]:
            s = res["auroc_by_shot"][k]
            n = res["n_epitopes_by_shot"][k]
            print(f"  k={k:<4d} AUROC={s['mean']:.4f}±{s['std']:.4f} (n_epitopes={n})")
        print(f"  -> {out_dir}/fewshot.json")

    if args.protocol in ("probe", "both"):
        t0 = time.time()
        probe = run_probe(args.method, args.embedder, train, test, args.columns, args.seed)
        if probe is None:
            print(f"\n[probe] skipped: method '{args.method}' is distance-only "
                  f"(no probe features); few-shot is the main protocol.")
        else:
            probe["method"] = args.method
            probe["embedder"] = args.embedder
            probe["columns"] = args.columns
            probe["elapsed_sec"] = round(time.time() - t0, 1)
            with (out_dir / "metrics.json").open("w") as fh:
                json.dump(probe, fh, indent=2)
            print(f"\n=== 24-way probe (aux) :: {probe['model']} "
                  f"({probe['elapsed_sec']}s) ===")
            print(f"  probe-AUROC={probe.get('probe_auroc', float('nan')):.3f} "
                  f"probe-Acc={probe['probe_acc']:.3f} "
                  f"kNN-top1={probe['knn_top1_acc']:.3f} ({probe['n_classes']}-way)")
            print(f"  -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
