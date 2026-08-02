#!/usr/bin/env python
"""T2 -- TCR clustering (anchored on NAR-GAB 2025 / i3-unit benchmark).

Cluster paired TCRs without using epitope labels, then measure agreement of the
clusters with the held-out epitope labels using the NAR-GAB 2025 protocol
(Purity / Retention / Sensitivity / %high-purity clusters, plus NMI / ARI). The
evaluation unit is the unique paired TCR (``pair_id``); ``seq_col`` defaults to
CDR3beta.

Methods
-------
precomputed     : read an ``assignments.csv`` (pair_id, cdr3b, cluster, epitope)
                  and score it directly. Most assignments originate from the
                  authors' artifacts, but HD/LD are locally reconstructed and all
                  nine are projected onto our drifted paired universe. Therefore
                  these rows are local diagnostics, not the paper-native table.
embed-threshold : embed CDR3beta (+ optional CDR3alpha) -> cosine similarity
                  graph -> threshold tau connected components (singletons = -1) ->
                  sweep tau to trace the Purity-Retention trade-off curve. This is
                  OUR unified evaluation shell for models/embeddings that have no
                  official clustering output (our BioSeq model, generic protein-LM
                  references, TCR-VALID latents) -- reported separately from the
                  official-method baselines, not mixed into that column.
embed           : agglomerative clustering with n_clusters = number of epitopes
                  (legacy fixed-cluster-count reference; retention == 1).

Examples
--------
    # score one official baseline (author-precomputed assignments)
    python tcr_clustering/run.py --method precomputed \
        --assignments outputs/tcr_clustering/DeepTCR/assignments.csv --tag DeepTCR
    # score all nine official baselines at once + write the comparison table
    python tcr_clustering/run.py --method precomputed --all
    # our unified embedding trade-off curves (our model / reference embeddings)
    python tcr_clustering/run.py --method embed-threshold --embedder esm2_150m
    python tcr_clustering/run.py --method embed-threshold \
        --embedder bioseq:/abs/best.pt --use-alpha --tag Ours-esmc300m
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

DATA = BENCH / "data" / "tcr_clustering"
OUT = BENCH / "outputs" / "tcr_clustering"

# The six epitopes the NAR-GAB 2025 paper reports per-epitope sensitivity for.
PAPER_SENS_EPITOPES = ["GILGFVFTL", "NLVPMVATV", "GLCTLVAML",
                       "FLCMKALLL", "NYNYLYRLF", "LLFGYPVYV"]
BASELINE_METHODS = ["HD", "LD", "TCRMatch", "iSMART", "GIANA",
                    "clusTCR", "GLIPH2", "DeepTCR", "TCRdist3"]
# Cluster-size filters reported by the paper (size > 3 / 5 / 10 -> min size 4/6/11).
SIZE_FILTERS = {"gt3": 4, "gt5": 6, "gt10": 11}


# ---------------------------------------------------------------------------
# Metric assembly (main + size-filtered + paper-restricted sensitivity)
# ---------------------------------------------------------------------------
def score(labels, true) -> dict:
    """Full NAR-GAB metric bundle for one clustering of the pair universe."""
    m = metrics.clustering_metrics(labels, true, unclustered_value=-1,
                                   sensitivity_epitopes=None)
    # sensitivity over the paper's six highlighted epitopes (direct comparison).
    m["sensitivity_top6"] = metrics._clustering_sensitivity(
        np.asarray(labels), np.asarray(true), -1,
        epitopes=PAPER_SENS_EPITOPES)["sensitivity"]
    # size-filtered purity family (cluster-size > 3 / 5 / 10).
    for tag, k in SIZE_FILTERS.items():
        mk = metrics.clustering_metrics(labels, true, unclustered_value=-1,
                                        min_cluster_size=k)
        m[f"purity_{tag}"] = mk["purity"]
        m[f"retention_{tag}"] = mk["retention"]
        m[f"pct_clusters_purity_gt90_{tag}"] = mk["pct_clusters_purity_gt90"]
    return m


def _print_metrics(name, m):
    print(f"\n=== {name} ===")
    print(f"  Purity={m['purity']:.3f} Retention={m['retention']:.3f} "
          f"Sensitivity={m['sensitivity']:.3f} (top6={m['sensitivity_top6']:.3f}) "
          f"NMI={m['nmi']:.3f} ARI={m['ari']:.3f}")
    print(f"  %clusters purity>0.9={m['pct_clusters_purity_gt90']:.3f} "
          f"%seqs in high-purity={m['pct_seqs_in_high_purity_clusters']:.3f} "
          f"(clusters={m['n_clusters']}, clustered={m['n_clustered']}/{m['n_total']})")


# ---------------------------------------------------------------------------
# precomputed
# ---------------------------------------------------------------------------
def run_precomputed(assignments_csv: Path, tag: str):
    df = pd.read_csv(assignments_csv).fillna({"cluster": -1})
    labels = df["cluster"].astype(int).to_numpy()
    true = df["epitope"].astype(str).to_numpy()
    m = score(labels, true)
    m["method"] = "precomputed"
    m["model"] = tag
    m["source"] = str(assignments_csv.relative_to(BENCH))
    m["baseline_provenance"] = {
        "evidence_type": "local_reimplementation" if tag in {"HD", "LD"}
        else "official_artifact_rescored",
        "protocol_alignment": "mismatched",
        "paper_comparable": False,
        "citation": "doi:10.1093/nargab/lqaf150",
        "source_location": (
            "locally reconstructed distance graph on drifted DB" if tag in {"HD", "LD"}
            else "authors' assignment artifact projected to local pair universe"
        ),
        "notes": "paper-reported native-protocol retention/purity rows are primary",
    }
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(m, fh, indent=2)
    _print_metrics(tag, m)
    print(f"  -> {out_dir}/metrics.json")
    return m


def run_all_precomputed():
    rows = []
    for method in BASELINE_METHODS:
        path = OUT / method / "assignments.csv"
        if not path.exists():
            print(f"  [skip] {method}: {path} not found "
                  "(run scripts/import_clustering_baselines.py first)")
            continue
        rows.append(run_precomputed(path, method))
    if rows:
        _write_comparison(rows)
    return rows


def _write_comparison(rows):
    keep = ["model", "purity", "retention", "sensitivity", "sensitivity_top6",
            "nmi", "ari", "pct_clusters_purity_gt90",
            "pct_seqs_in_high_purity_clusters", "n_clusters", "n_clustered",
            "n_total", "purity_gt3", "retention_gt3", "purity_gt5", "purity_gt10"]
    tbl = pd.DataFrame(rows)[keep]
    tbl["evidence_type"] = [
        row["baseline_provenance"]["evidence_type"] for row in rows
    ]
    tbl["protocol_alignment"] = "mismatched"
    tbl["paper_comparable"] = False
    tbl["citation"] = "doi:10.1093/nargab/lqaf150"
    tbl.to_csv(OUT / "baseline_comparison.csv", index=False)
    print(f"\n[comparison] {len(rows)} methods -> {OUT}/baseline_comparison.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tbl[["model", "purity", "retention", "sensitivity_top6",
                   "pct_clusters_purity_gt90"]].to_string(index=False))


# ---------------------------------------------------------------------------
# embedding-based threshold clustering (Purity-Retention curve)
# ---------------------------------------------------------------------------
class _KmerEmbedder:
    """Adapter so the k-mer composition featurizer plugs into the embed backend."""

    def __init__(self, k: int = 3):
        from common.featurizers import KmerFeaturizer
        self._kf = KmerFeaturizer(k=k)
        self.name = f"kmer-{k}"
        self.dim = self._kf.dim

    def embed(self, seqs):
        return self._kf.transform([str(s) for s in seqs])


def _build_embed_backend(embedder_spec):
    if embedder_spec == "kmer" or embedder_spec.startswith("kmer:"):
        k = int(embedder_spec.split(":", 1)[1]) if ":" in embedder_spec else 3
        return _KmerEmbedder(k=k)
    from common.model_api import build_embedder
    return build_embedder(embedder_spec)


def _embed_features(df, embedder_spec, use_alpha):
    # Precomputed per-row embedding matrix (e.g. TCR-VALID latents produced in a
    # separate TF env), aligned to the tcrs.csv row order. ``emb:/abs/x.npy`` or
    # ``emb:NAME:/abs/x.npy`` to set the display name.
    if embedder_spec.startswith("emb:"):
        rest = embedder_spec[len("emb:"):]
        name = "precomputed-emb"
        if ":" in rest and not rest.startswith("/"):
            name, path = rest.split(":", 1)
        else:
            path = rest
        x = np.load(path).astype(np.float32)
        if x.shape[0] != len(df):
            raise ValueError(
                f"precomputed embedding rows ({x.shape[0]}) != tcrs.csv rows ({len(df)})")
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return x, name

    emb = _build_embed_backend(embedder_spec)

    if getattr(emb, "feature_source", None) == "decoder" and use_alpha:
        beta = df["cdr3b"].astype(str).tolist()
        alpha = df["cdr3a"].astype(str).tolist()
        vec = emb.embed_pairs(beta=beta, alpha=alpha, peptide=None)
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
        return vec.astype(np.float32), emb.name

    def _unit_embed(seqs):
        uniq = list(dict.fromkeys(seqs))
        vec = emb.embed(uniq)
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
        table = {s: vec[i] for i, s in enumerate(uniq)}
        return np.stack([table[s] for s in seqs])

    xb = _unit_embed(df["cdr3b"].astype(str).tolist())
    if use_alpha:
        xa = _unit_embed(df["cdr3a"].astype(str).tolist())
        x = np.concatenate([xb, xa], axis=1)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    else:
        x = xb
    return x, emb.name


def _components_at_threshold(sim, tau, min_size=2):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = sim.shape[0]
    adj = sim >= tau
    np.fill_diagonal(adj, False)
    ncomp, comp = connected_components(csr_matrix(adj), directed=False)
    counts = np.bincount(comp, minlength=ncomp)
    labels = np.full(n, -1)
    cid = 0
    remap = {}
    for i in range(n):
        c = comp[i]
        if counts[c] >= min_size:
            if c not in remap:
                remap[c] = cid
                cid += 1
            labels[i] = remap[c]
    return labels


# Target retention levels for the sweep. Include the baselines' own retentions so
# every embedder is scored at matched retention (purity at ret ~ clusTCR/GLIPH2/…).
TARGET_RETENTIONS = sorted(set(
    [0.05, 0.08, 0.09, 0.11, 0.15, 0.18, 0.19, 0.22, 0.25, 0.30, 0.39, 0.44,
     0.50, 0.60, 0.70, 0.82, 0.90]), reverse=True)


def _taus_from_targets(sim, targets, min_size=2, edge_floor=0.30):
    """Pick cosine thresholds that hit each target retention, in one pass.

    Retention(tau) is monotone in tau, so we sort all candidate edges by
    similarity once and add them (union-find) in descending order, tracking the
    fraction of nodes that sit in a non-singleton component. For each target
    retention we record the similarity at which the curve first reaches it -- that
    edge weight is the threshold. Far faster and more robust than re-running
    connected-components per candidate tau, and it works for any embedder whether
    its similarity cone is wide (k-mer) or extremely tight (protein LMs).
    """
    n = sim.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    w = sim[iu, ju]
    # For a curve down to the largest target retention we never need very weak
    # edges; keep an adaptive floor so tight-cone embedders still get enough.
    max_t = max(targets)
    floor = edge_floor
    while True:
        keep = w >= floor
        # rough upper bound on reachable retention: nodes touched by kept edges.
        if keep.sum() == 0 or floor <= -1.0:
            floor = -1.0
            keep = np.ones_like(w, dtype=bool)
            break
        touched = len(set(iu[keep]) | set(ju[keep])) / n
        if touched >= min(0.999, max_t + 0.05):
            break
        floor -= 0.1
    order = np.argsort(-w[keep])
    ei, ej, ew = iu[keep][order], ju[keep][order], w[keep][order]

    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    clustered = 0                       # nodes in components of size >= min_size
    targets_sorted = sorted(targets)    # ascending
    ti = 0
    hits = {}
    for k in range(len(ew)):
        ra, rb = find(ei[k]), find(ej[k])
        if ra != rb:
            sa, sb = size[ra], size[rb]
            before = (sa if sa >= min_size else 0) + (sb if sb >= min_size else 0)
            parent[ra] = rb
            size[rb] = sa + sb
            after = sa + sb if sa + sb >= min_size else 0
            clustered += after - before
        ret = clustered / n
        while ti < len(targets_sorted) and ret >= targets_sorted[ti]:
            hits[targets_sorted[ti]] = float(ew[k])
            ti += 1
        if ti >= len(targets_sorted):
            break
    taus = sorted({round(v, 5) for v in hits.values()})
    return np.array(taus) if taus else np.array([round(float(ew[-1]), 5)])


def run_embed_threshold(df, embedder_spec, tag, use_alpha, taus):
    true = df["peptide"].astype(str).to_numpy()
    t0 = time.time()
    x, model_name = _embed_features(df, embedder_spec, use_alpha)
    sim = (x @ x.T).astype(np.float32)
    embed_sec = time.time() - t0
    if taus is None:
        taus = _taus_from_targets(sim, TARGET_RETENTIONS)

    curve = []
    for tau in taus:
        labels = _components_at_threshold(sim, float(tau))
        m = score(labels, true)
        curve.append({"tau": round(float(tau), 4), **{
            k: m[k] for k in ["purity", "retention", "sensitivity",
                              "sensitivity_top6", "nmi", "ari",
                              "pct_clusters_purity_gt90",
                              "pct_seqs_in_high_purity_clusters",
                              "n_clusters", "n_clustered"]}})
    curve_df = pd.DataFrame(curve)

    out_tag = tag or (f"{embedder_spec.replace(':', '_').replace('/', '_')}"
                      f"{'_ab' if use_alpha else ''}_thr")
    out_dir = OUT / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(out_dir / "curve.csv", index=False)

    # Alignment points: purity at retention closest to each baseline's retention.
    ref_ret = {"clusTCR": 0.08, "TCRMatch": 0.11, "GLIPH2": 0.19,
               "TCRdist3": 0.39, "GIANA": 0.18, "DeepTCR": 0.82}
    aligned = {}
    for name, r in ref_ret.items():
        idx = (curve_df["retention"] - r).abs().idxmin()
        row = curve_df.loc[idx]
        aligned[f"purity@ret~{name}({r})"] = {
            "tau": row["tau"], "retention": round(row["retention"], 3),
            "purity": round(row["purity"], 3)}

    summary = {
        "method": "embed-threshold", "model": model_name, "tag": out_tag,
        "embedder": embedder_spec, "use_alpha": use_alpha,
        "embed_sec": round(embed_sec, 1),
        "n_units": int(len(df)), "taus": [round(float(t), 4) for t in taus],
        "aligned_points": aligned,
        "auc_purity_retention": float(np.trapz(
            curve_df.sort_values("retention")["purity"],
            curve_df.sort_values("retention")["retention"])),
    }
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n=== {model_name} embed-threshold ({embed_sec:.1f}s embed) ===")
    with pd.option_context("display.width", 200):
        print(curve_df[["tau", "retention", "purity", "sensitivity_top6",
                        "n_clusters", "n_clustered"]].to_string(index=False))
    print(f"  -> {out_dir}/curve.csv  (+ metrics.json)")
    return summary


# ---------------------------------------------------------------------------
# embed (legacy fixed-n_clusters agglomerative)
# ---------------------------------------------------------------------------
def cluster_embed(seqs, n_clusters, embedder_spec):
    from common.model_api import build_embedder
    from sklearn.cluster import AgglomerativeClustering

    emb = build_embedder(embedder_spec)
    x = emb.embed(list(seqs))
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(x), emb.name


def _parse_taus(spec: str):
    """Explicit comma list, or ``None`` to auto-pick from the similarity spectrum."""
    if spec:
        return np.array([float(t) for t in spec.split(",")])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="precomputed",
                    choices=["precomputed", "embed-threshold", "embed"])
    ap.add_argument("--assignments", help="assignments.csv for --method precomputed")
    ap.add_argument("--all", action="store_true",
                    help="score all nine official baselines (precomputed)")
    ap.add_argument("--embedder", default="esm2_150m")
    ap.add_argument("--use-alpha", action="store_true",
                    help="concatenate CDR3alpha embedding (embed-threshold)")
    ap.add_argument("--taus", default="", help="comma list of tau; default sweep")
    ap.add_argument("--seq-col", default="cdr3b")
    ap.add_argument("--tag", default="", help="output subfolder / model label")
    args = ap.parse_args()

    if args.method == "precomputed":
        if args.all:
            run_all_precomputed()
            return
        if not args.assignments:
            raise SystemExit("--method precomputed needs --assignments <csv> or --all")
        tag = args.tag or Path(args.assignments).parent.name
        run_precomputed(Path(args.assignments), tag)
        return

    df = pd.read_csv(DATA / "tcrs.csv").fillna("")
    n_ep = df["peptide"].nunique()
    print(f"loaded {len(df)} paired TCRs over {n_ep} epitopes (seq_col={args.seq_col})")

    if args.method == "embed-threshold":
        run_embed_threshold(df, args.embedder, args.tag, args.use_alpha,
                            _parse_taus(args.taus))
        return

    seqs = df[args.seq_col].astype(str).tolist()
    true = df["peptide"].astype(str).to_numpy()
    t0 = time.time()
    # embed (legacy fixed-n_clusters agglomerative reference)
    labels, model_name = cluster_embed(seqs, n_ep, args.embedder)
    tag = args.tag or args.embedder
    elapsed = time.time() - t0

    m = score(labels, true)
    m["method"] = args.method
    m["model"] = model_name
    m["elapsed_sec"] = round(elapsed, 1)
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pair_id": df["pair_id"], "cdr3b": seqs,
                  "cluster": labels, "epitope": true}).to_csv(
        out_dir / "assignments.csv", index=False)
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(m, fh, indent=2)
    _print_metrics(f"{model_name} ({elapsed:.1f}s)", m)
    print(f"  -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
