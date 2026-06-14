"""IRBench metrics for all task families.

Binding (T1):        per-epitope AUROC / AUPRC + Macro-AUC0.1 (IMMREP official).
Clustering (T2):     purity, retention, NMI, ARI (cluster-vs-epitope agreement).
Representation (T3):  linear-probe AUROC/accuracy + kNN top-1 (handled here).
Generation (T4):     amino-acid recovery, novelty, NN distance, k-mer JSD.

All functions are pure (numpy / sklearn / scipy) and degrade gracefully on
degenerate inputs (single-class groups, empty sets) so a full leaderboard run
never crashes on one bad epitope.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)


# ---------------------------------------------------------------------------
# Binding metrics
# ---------------------------------------------------------------------------

def _safe_auroc(y: np.ndarray, s: np.ndarray, max_fpr: float | None = None) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s, max_fpr=max_fpr))
    except ValueError:
        return float("nan")


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def binding_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str],
    max_fpr: float = 0.1,
) -> dict:
    """Aggregate + per-epitope binding metrics.

    Returns global AUROC/AUPRC (pooled) and macro averages over per-epitope
    AUROC, AUPRC, and AUC0.1 (the IMMREP Macro-AUC0.1 ranking metric).
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)

    per_ep = {}
    for ep in sorted(set(groups.tolist())):
        m = groups == ep
        per_ep[ep] = {
            "n": int(m.sum()),
            "n_pos": int(labels[m].sum()),
            "auroc": _safe_auroc(labels[m], scores[m]),
            "auprc": _safe_auprc(labels[m], scores[m]),
            "auc01": _safe_auroc(labels[m], scores[m], max_fpr=max_fpr),
        }

    def _macro(key):
        vals = [v[key] for v in per_ep.values() if not np.isnan(v[key])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "global_auroc": _safe_auroc(labels, scores),
        "global_auprc": _safe_auprc(labels, scores),
        "macro_auroc": _macro("auroc"),
        "macro_auprc": _macro("auprc"),
        "macro_auc01": _macro("auc01"),
        "n_epitopes_scored": int(sum(1 for v in per_ep.values() if not np.isnan(v["auroc"]))),
        "per_epitope": per_ep,
    }


# ---------------------------------------------------------------------------
# Clustering metrics (cluster labels vs. true epitope labels)
# ---------------------------------------------------------------------------

def clustering_metrics(
    cluster_ids: Sequence,
    true_labels: Sequence,
    unclustered_value=-1,
) -> dict:
    """Standard repertoire-clustering quality metrics.

    purity     : weighted fraction of each cluster that is its majority epitope
                 (clustered points only).
    retention  : fraction of points assigned to a real (non-singleton) cluster.
    nmi / ari  : information / pair agreement vs. true epitope labels.
    """
    cluster_ids = np.asarray(cluster_ids)
    true_labels = np.asarray(true_labels)
    n = len(true_labels)

    clustered = cluster_ids != unclustered_value
    retention = float(clustered.mean()) if n else float("nan")

    cl = cluster_ids[clustered]
    tr = true_labels[clustered]
    if len(cl) == 0:
        return {"purity": float("nan"), "retention": retention,
                "nmi": float("nan"), "ari": float("nan"),
                "n_clusters": 0, "n_clustered": 0, "n_total": int(n)}

    purity_num = 0
    for c in set(cl.tolist()):
        members = tr[cl == c]
        if len(members):
            purity_num += Counter(members.tolist()).most_common(1)[0][1]
    purity = purity_num / len(cl)

    return {
        "purity": float(purity),
        "retention": retention,
        "nmi": float(normalized_mutual_info_score(tr, cl)),
        "ari": float(adjusted_rand_score(tr, cl)),
        "n_clusters": int(len(set(cl.tolist()))),
        "n_clustered": int(len(cl)),
        "n_total": int(n),
    }


# ---------------------------------------------------------------------------
# Representation metrics
# ---------------------------------------------------------------------------

def linear_probe_metrics(
    train_x: np.ndarray, train_y: Sequence[int],
    test_x: np.ndarray, test_y: Sequence[int],
    multiclass: bool = False,
    seed: int = 0,
) -> dict:
    """Frozen-feature linear probe (logistic regression). Backbone is not tuned."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
    )
    clf.fit(train_x, train_y)
    acc = float((clf.predict(test_x) == test_y).mean())
    out = {"probe_acc": acc}
    try:
        if multiclass:
            proba = clf.predict_proba(test_x)
            out["probe_auroc"] = float(
                roc_auc_score(test_y, proba, multi_class="ovr", average="macro")
            )
        else:
            proba = clf.predict_proba(test_x)[:, 1]
            out["probe_auroc"] = _safe_auroc(test_y, proba)
            out["probe_auprc"] = _safe_auprc(test_y, proba)
    except Exception:
        out["probe_auroc"] = float("nan")
    return out


def knn_top1_metric(
    train_x: np.ndarray, train_y: Sequence,
    test_x: np.ndarray, test_y: Sequence,
    metric: str = "cosine",
) -> dict:
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
    knn.fit(train_x, np.asarray(train_y))
    pred = knn.predict(test_x)
    return {"knn_top1_acc": float((pred == np.asarray(test_y)).mean())}


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def amino_acid_recovery(generated: Sequence[str], reference: Sequence[str]) -> float:
    """Mean per-position AA identity over aligned (same-length) positions."""
    assert len(generated) == len(reference)
    tot, hit = 0, 0
    for g, r in zip(generated, reference):
        L = min(len(g), len(r))
        for i in range(L):
            tot += 1
            hit += int(g[i] == r[i])
        tot += abs(len(g) - len(r))  # length mismatch counts as misses
    return float(hit / tot) if tot else float("nan")


def novelty(generated: Sequence[str], train_set: Sequence[str]) -> float:
    """Fraction of generated sequences not present verbatim in the train set."""
    train = set(train_set)
    if not len(generated):
        return float("nan")
    return float(np.mean([g not in train for g in generated]))


def mean_nn_distance(generated: Sequence[str], train_set: Sequence[str]) -> float:
    """Mean Levenshtein distance from each generated seq to nearest train seq."""
    from .leakage import min_edit_distance
    refs = list(train_set)
    if not refs or not len(generated):
        return float("nan")
    return float(np.mean([min_edit_distance(g, refs) for g in generated]))


def _kmer_freq(seqs: Sequence[str], k: int = 3) -> Counter:
    c = Counter()
    for s in seqs:
        for i in range(len(s) - k + 1):
            c[s[i:i + k]] += 1
    return c


def kmer_jsd(generated: Sequence[str], reference: Sequence[str], k: int = 3) -> float:
    """Jensen-Shannon divergence between k-mer distributions (0 = identical)."""
    from scipy.spatial.distance import jensenshannon

    fg, fr = _kmer_freq(generated, k), _kmer_freq(reference, k)
    keys = sorted(set(fg) | set(fr))
    if not keys:
        return float("nan")
    pg = np.array([fg.get(x, 0) for x in keys], dtype=float)
    pr = np.array([fr.get(x, 0) for x in keys], dtype=float)
    if pg.sum() == 0 or pr.sum() == 0:
        return float("nan")
    pg /= pg.sum()
    pr /= pr.sum()
    d = jensenshannon(pg, pr, base=2)
    return float(d ** 2)  # JS divergence = (JS distance)^2


def generation_metrics(
    generated: Sequence[str],
    paired_reference: Sequence[str] | None = None,
    dist_reference: Sequence[str] | None = None,
    train_set: Sequence[str] | None = None,
    k: int = 3,
) -> dict:
    """Generation/infilling metrics.

    ``paired_reference``: one ground-truth per generated seq (infilling) -> AAR.
    ``dist_reference``   : a held-out distribution (unconditional gen) -> k-mer JSD.
    ``train_set``        : training corpus -> novelty + nearest-neighbour distance.
    """
    out = {}
    if paired_reference is not None:
        out["aar"] = amino_acid_recovery(generated, paired_reference)
        out["kmer_jsd"] = kmer_jsd(generated, paired_reference, k=k)
    elif dist_reference is not None:
        out["kmer_jsd"] = kmer_jsd(generated, dist_reference, k=k)
    if train_set is not None:
        out["novelty"] = novelty(generated, train_set)
        out["mean_nn_distance"] = mean_nn_distance(generated, train_set)
    out["n_generated"] = int(len(generated))
    out["n_unique"] = int(len(set(generated)))
    return out


__all__ = [
    "binding_metrics", "clustering_metrics",
    "linear_probe_metrics", "knn_top1_metric",
    "amino_acid_recovery", "novelty", "mean_nn_distance",
    "kmer_jsd", "generation_metrics",
]
