#!/usr/bin/env python
"""T2 second basis -- embedding -> clustering (ARI / NMI / Purity, K sweep).

Reproduces the *A comprehensive benchmarking for evaluating TCR embeddings*
(Brief Bioinformatics 2025, ``deepomicslab/TCREmbedding``) clustering protocol on
its fixed labelled dataset (``data/tcr_clustering_embed/tcrs.csv``, 9,033 unique
CDR3beta / 25 epitopes):

    embed CDR3beta -> L2-normalise -> cluster with a K sweep -> ARI / NMI / Purity

The paper sweeps the target cluster count K over {10, 15, ..., 100} (step 5) "to
alleviate the influence of the target number of clusters" and reports ARI / NMI /
Purity. We do the same, with K-means as the primary algorithm (fast, matches the
paper's k-means row) and agglomerative (hierarchical) as a secondary check, and
report the mean and best over the K sweep.

This is OUR unified evaluation shell for models/embeddings that have no official
clustering output (our BioSeq model + generic reference embeddings ESM2 / k-mer).
It is reported SEPARATELY from the nine official NAR-GAB baselines (different
basis, single-chain CDR3beta caliber) -- same principle as ``embed-threshold``.

Examples
--------
    python tcr_clustering/run_embed_bench.py --embedder esm2_150m --tag esm2_150m
    python tcr_clustering/run_embed_bench.py --embedder kmer --tag kmer
    python tcr_clustering/run_embed_bench.py \
        --embedder bioseq:/abs/grammar_v2_esmc300m_cmp500k_llada/best.pt \
        --tag ours_esmc300m_cmp500k
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

DATA = BENCH / "data" / "tcr_clustering_embed"
OUT = BENCH / "outputs" / "tcr_clustering_embed"

# Paper's continuous-K strategy: K in {10, 15, ..., 100}.
K_SWEEP = list(range(10, 101, 5))


class _KmerEmbedder:
    """k-mer composition featurizer adapter (training-free reference embedding).

    Default ``k=2`` (400-dim) keeps the reference low-dimensional and clustering
    tractable; the amino-acid k-mer composition is the classic training-free TCR
    encoding (no learned parameters).
    """

    def __init__(self, k: int = 2):
        from common.featurizers import KmerFeaturizer
        self._kf = KmerFeaturizer(k=k)
        self.name = f"kmer-{k}"
        self.dim = self._kf.dim

    def embed(self, seqs):
        return self._kf.transform([str(s) for s in seqs])


class _FeaturizerEmbedder:
    """Adapter for the handcrafted per-residue featurizers (training-free).

    Wraps ``AtchleyFeaturizer`` / ``Blosum62Featurizer`` / ``OneHotFeaturizer``
    behind the same ``embed()`` interface as the other reference embedders.
    """

    def __init__(self, featurizer):
        self._f = featurizer
        self.name = featurizer.name
        self.dim = featurizer.dim

    def embed(self, seqs):
        return self._f.transform([str(s) for s in seqs])


def _build_backend(spec):
    if spec == "kmer" or spec.startswith("kmer:"):
        k = int(spec.split(":", 1)[1]) if ":" in spec else 2
        return _KmerEmbedder(k=k)
    if spec in ("atchley", "blosum62", "onehot"):
        from common import featurizers as fz
        featurizer = {"atchley": fz.AtchleyFeaturizer,
                      "blosum62": fz.Blosum62Featurizer,
                      "onehot": fz.OneHotFeaturizer}[spec]()
        return _FeaturizerEmbedder(featurizer)
    from common.model_api import build_embedder
    return build_embedder(spec)


def _embed(df, spec):
    """Embed unique CDR3beta then map back to row order; L2-normalise rows."""
    if spec.startswith("emb:"):
        rest = spec[len("emb:"):]
        name, path = ("precomputed-emb", rest)
        if ":" in rest and not rest.startswith("/"):
            name, path = rest.split(":", 1)
        x = np.load(path).astype(np.float32)
        if x.shape[0] != len(df):
            raise ValueError(
                f"precomputed embedding rows ({x.shape[0]}) != tcrs.csv rows ({len(df)})")
        info = {"source": "precomputed", "path": path}
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return x, name, info

    if spec == "sceptr" or spec.startswith("sceptr:"):
        # SCEPTR is a native TCR representation model (not a plain SequenceEmbedder):
        # it needs structured columns (CDR3 junction + V gene) rather than a raw
        # CDR3beta string list. basis-B is CDR3beta-only + TRBV, so we feed the beta
        # chain as SCEPTR partial input (alpha left empty -> treated as missing),
        # matching SCEPTR's documented beta-only usage. tcrs.csv is already unique on
        # cdr3b, so embed every row directly (SCEPTR's 153k-param model is fast).
        from common.model_api import build_distance_source
        src = build_distance_source(spec)
        sdf = pd.DataFrame({
            "cdr3b": df["cdr3b"].astype(str),
            "vb": df["vgene"].astype(str) if "vgene" in df.columns else "",
            "cdr3a": "",
            "va": "",
        })
        vec = np.asarray(src.embed(sdf), dtype=np.float32)
        x = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
        info = {"source": "sceptr", "variant": getattr(src, "variant", "default"),
                "input": "beta-only (cdr3b+TRBV); alpha missing (SCEPTR partial input)"}
        return x, src.name, info

    emb = _build_backend(spec)
    seqs = df["cdr3b"].astype(str).tolist()
    uniq = list(dict.fromkeys(seqs))
    vec = emb.embed(uniq)
    vec = np.asarray(vec, dtype=np.float32)
    table = {s: vec[i] for i, s in enumerate(uniq)}
    x = np.stack([table[s] for s in seqs])
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    info = getattr(emb, "_load_info", {}) or {}
    return x, emb.name, info


def _cluster_all(x, algo, ks, seed=0):
    """Return {k: labels} for the whole K sweep, computing shared work once."""
    if algo == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        return {k: MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                                   batch_size=2048).fit_predict(x) for k in ks}
    if algo == "hierarchical":
        # Ward linkage is O(n^2 * d); project to <=50 PCA components first so it
        # stays tractable across high-dim encoders (documented in metrics.json).
        from scipy.cluster.hierarchy import linkage, fcluster
        xin = x
        if x.shape[1] > 50:
            from sklearn.decomposition import PCA
            xin = PCA(n_components=50, random_state=seed).fit_transform(x)
        z = linkage(xin, method="ward")
        return {k: fcluster(z, t=k, criterion="maxclust") for k in ks}
    raise ValueError(algo)


def _score(labels, true):
    # All points are assigned (partition clustering, retention == 1); use the
    # NAR-GAB metric bundle but only keep purity / nmi / ari for this basis.
    m = metrics.clustering_metrics(labels, true, unclustered_value=-999)
    return {"purity": m["purity"], "nmi": m["nmi"], "ari": m["ari"],
            "n_clusters": m["n_clusters"]}


def run(spec, tag, algos, ks, seed):
    df = pd.read_csv(DATA / "tcrs.csv").fillna("")
    true = df["epitope"].astype(str).to_numpy()
    n_ep = df["epitope"].nunique()
    print(f"loaded {len(df)} unique CDR3beta over {n_ep} epitopes")

    t0 = time.time()
    x, model_name, load_info = _embed(df, spec)
    embed_sec = time.time() - t0
    print(f"embedded with {model_name} ({embed_sec:.1f}s, dim={x.shape[1]})"
          + (f"  load_info={load_info}" if load_info else ""))

    rows = []
    for algo in algos:
        t1 = time.time()
        labels_by_k = _cluster_all(x, algo, ks, seed)
        for k in ks:
            s = _score(labels_by_k[k], true)
            rows.append({"algo": algo, "k": k, **s})
            print(f"  {algo:12s} K={k:3d}  ARI={s['ari']:.3f} "
                  f"NMI={s['nmi']:.3f} Purity={s['purity']:.3f}")
        print(f"  [{algo} sweep {time.time() - t1:.1f}s]")
    curve = pd.DataFrame(rows)

    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out_dir / "curve.csv", index=False)

    summary = {"method": "embed-cluster-Ksweep", "model": model_name, "tag": tag,
               "embedder": spec, "embed_sec": round(embed_sec, 1),
               "n_units": int(len(df)), "n_epitopes": int(n_ep),
               "k_sweep": ks, "algos": algos, "seed": seed,
               "load_info": load_info}
    for algo in algos:
        sub = curve[curve["algo"] == algo]
        for met in ("ari", "nmi", "purity"):
            summary[f"{algo}_{met}_mean"] = round(float(sub[met].mean()), 4)
            summary[f"{algo}_{met}_best"] = round(float(sub[met].max()), 4)
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n=== {model_name} (basis-B embedding->clustering) ===")
    for algo in algos:
        print(f"  {algo:12s} mean-over-K  ARI={summary[f'{algo}_ari_mean']:.3f} "
              f"NMI={summary[f'{algo}_nmi_mean']:.3f} "
              f"Purity={summary[f'{algo}_purity_mean']:.3f}  |  best "
              f"ARI={summary[f'{algo}_ari_best']:.3f} "
              f"NMI={summary[f'{algo}_nmi_best']:.3f} "
              f"Purity={summary[f'{algo}_purity_best']:.3f}")
    print(f"  -> {out_dir}/curve.csv (+ metrics.json)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", required=True,
                    help="kmer / atchley / blosum62 / onehot / esm2_150m / "
                         "bioseq:/abs/best.pt / emb:NAME:/abs/x.npy")
    ap.add_argument("--tag", default="", help="output subfolder / label")
    ap.add_argument("--algos", default="kmeans",
                    help="comma list of {kmeans,hierarchical}; hierarchical uses PCA-50")
    ap.add_argument("--ks", default="", help="comma list of K; default 10..100 step 5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    ks = [int(k) for k in args.ks.split(",")] if args.ks else K_SWEEP
    tag = args.tag or args.embedder.replace(":", "_").replace("/", "_")
    run(args.embedder, tag, algos, ks, args.seed)


if __name__ == "__main__":
    main()
