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
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
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


def precrec_pr_auc(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    x_bins: int = 1000,
) -> float:
    """Return the PRC-AUC produced by R ``precrec::evalmod`` defaults.

    The Nature Methods 2025 TCR benchmark computes its reported ``AUPRC`` with
    ``precrec::evalmod(scores, labels)`` and then ``precrec::auc``.  That value
    is *not* sklearn average precision.  In particular, saturated/tied model
    scores can make the two differ materially (ATM-TCR is one such case).

    This is a direct NumPy port of precrec 0.14.x's default ROC/PRC pipeline:
    equivalent-score tie handling, nonlinear PR interpolation on 1/``x_bins``
    recall intervals, and trapezoidal integration.  It intentionally lives
    beside :func:`_safe_auprc` so callers can report both metrics without
    silently changing legacy benchmark outputs.

    Source implementation:
    https://github.com/evalclass/precrec/blob/master/src/precrec_plx.cpp
    """

    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise ValueError("labels and scores must be one-dimensional and equal length")
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    if not np.isfinite(s).all():
        raise ValueError("precrec PRC-AUC requires finite scores")
    if x_bins <= 0:
        raise ValueError("x_bins must be positive")

    # precrec ranks scores descending. With ties_method='equiv', every member
    # of a tied block gets the block's first rank; create_confusion_matrices()
    # then replaces internal block points by equal fractional TP/FP steps.
    order = np.argsort(-s, kind="stable")
    sorted_scores = s[order]
    sorted_labels = y[order]
    n = len(y)
    tps = np.empty(n + 1, dtype=float)
    fps = np.empty(n + 1, dtype=float)
    tps[0] = 0.0
    fps[0] = 0.0
    tp = 0.0
    fp = 0.0
    out_idx = 1
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        block_n = stop - start
        block_tp = float(sorted_labels[start:stop].sum())
        block_fp = float(block_n) - block_tp
        fractions = np.arange(1, block_n + 1, dtype=float) / float(block_n)
        tps[out_idx:out_idx + block_n] = tp + block_tp * fractions
        fps[out_idx:out_idx + block_n] = fp + block_fp * fractions
        tp += block_tp
        fp += block_fp
        out_idx += block_n
        start = stop

    n_positive = float(y.sum())
    recall = tps / n_positive
    precision = np.divide(
        tps,
        tps + fps,
        out=np.zeros_like(tps),
        where=(tps + fps) != 0,
    )
    # precrec defines the highest-threshold precision as the next point rather
    # than sklearn's conventional 1.0 endpoint.
    precision[0] = precision[1]

    # create_prc_curve(..., x_bins=1000) inserts nonlinear Davis-Goadrich
    # interpolation points at a regular recall grid before trapezoidal AUC.
    step = 1.0 / float(x_bins)
    curve_x: list[float] = []
    curve_y: list[float] = []
    for idx in range(n + 1):
        if (
            idx != 0
            and recall[idx] == recall[idx - 1]
            and precision[idx] == precision[idx - 1]
        ):
            continue
        if idx > 0:
            tmp_recall = step * int(recall[idx - 1] / step)
            while tmp_recall < 1.0:
                tmp_recall += step
                if tmp_recall >= recall[idx]:
                    break
                if precision[idx] == precision[idx - 1]:
                    tmp_precision = precision[idx]
                else:
                    extra_tp = (
                        (tmp_recall - recall[idx - 1])
                        * tps[idx]
                        / recall[idx]
                    )
                    extra_fp = (
                        (fps[idx] - fps[idx - 1])
                        * extra_tp
                        / (tps[idx] - tps[idx - 1])
                    )
                    tmp_precision = (
                        (tps[idx - 1] + extra_tp)
                        / (tps[idx - 1] + extra_tp + fps[idx - 1] + extra_fp)
                    )
                if (
                    curve_x
                    and curve_x[-1] == tmp_recall
                    and curve_y[-1] == tmp_precision
                ):
                    continue
                curve_x.append(float(tmp_recall))
                curve_y.append(float(tmp_precision))
        curve_x.append(float(recall[idx]))
        curve_y.append(float(precision[idx]))

    xs = np.asarray(curve_x, dtype=float)
    ys = np.asarray(curve_y, dtype=float)
    return float(np.sum(0.5 * (ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1])))


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


def _confusion_row(y: np.ndarray, s: np.ndarray, threshold: float = 0.5) -> dict:
    """0.5-threshold confusion metrics matching the official ``calculate`` row.

    Mirrors ``Evaluation_metrics_calculation.ipynb``: ROC/PRC AUC from the
    continuous score plus accuracy / precision / recall / specificity / mcc / f1
    from a 0.5-threshold confusion matrix. PRC-AUC is reported as sklearn
    ``average_precision_score`` (the official precrec uses trapezoidal PRC
    interpolation, which differs slightly; see RESULTS.md).
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pred = (s >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    spec = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return {
        "roc_auc": _safe_auroc(y, s),
        "prc_auc": _safe_auprc(y, s),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": spec,
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "n": int(len(y)), "n_pos": int(y.sum()),
    }


def official_binding_report(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str],
    threshold: float = 0.5,
) -> dict:
    """Per-epitope + overall report mirroring the official R ``calculate``.

    Splits predictions by epitope and, per epitope AND overall ("all_values"),
    computes ROC-AUC and PRC-AUC (sklearn AP) plus the 0.5-threshold confusion
    metrics. Also returns macro-averages over epitopes (the paper's headline is
    AUPRC), so both the paper's macro-across-epitopes and its overall pooled
    numbers are available.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)

    per_epitope = {}
    for ep in sorted(set(groups.tolist())):
        m = groups == ep
        per_epitope[ep] = _confusion_row(labels[m], scores[m], threshold)

    def _macro(key):
        vals = [row[key] for row in per_epitope.values()
                if not (isinstance(row[key], float) and np.isnan(row[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    macro = {k: _macro(k) for k in
             ("roc_auc", "prc_auc", "accuracy", "precision", "recall",
              "specificity", "mcc", "f1")}
    return {
        "all_values": _confusion_row(labels, scores, threshold),
        "macro": macro,
        "n_epitopes": int(len(per_epitope)),
        "n_epitopes_scored": int(sum(
            1 for r in per_epitope.values() if not np.isnan(r["roc_auc"]))),
        "per_epitope": per_epitope,
    }


# ---------------------------------------------------------------------------
# Clustering metrics (cluster labels vs. true epitope labels)
# ---------------------------------------------------------------------------

def clustering_metrics(
    cluster_ids: Sequence,
    true_labels: Sequence,
    unclustered_value=-1,
    min_cluster_size: int | None = None,
    high_purity_threshold: float = 0.9,
    sensitivity_min_size: int = 3,
    sensitivity_dominance: float = 2.0,
    sensitivity_epitopes: Sequence | None = None,
) -> dict:
    """Repertoire-clustering quality metrics (NAR-GAB 2025 Purity/Retention/…).

    Backward-compatible core keys (unchanged names and semantics):

    purity     : weighted fraction of each cluster that is its majority epitope
                 (clustered points only).
    retention  : fraction of points assigned to a real (non-singleton) cluster.
    nmi / ari  : information / pair agreement vs. true epitope labels.
    n_clusters / n_clustered / n_total.

    Added keys (incremental, aligned with the NAR-GAB 2025 protocol):

    pct_clusters_purity_gt90        : fraction of clusters with purity >=
                                      ``high_purity_threshold``.
    pct_seqs_in_high_purity_clusters: fraction of *clustered* points that live in
                                      such high-purity clusters.
    sensitivity                     : macro-mean over epitopes of the fraction of
                                      an epitope's points captured by its
                                      "epitope-specific" clusters -- clusters that
                                      are larger than ``sensitivity_min_size`` and
                                      in which the epitope's points outnumber all
                                      others combined by >= ``sensitivity_dominance``
                                      (paper: size>3 and epitope share >= 2/3).
    n_epitopes_sensitivity          : number of epitopes averaged for sensitivity.

    ``min_cluster_size`` (optional): points in clusters smaller than this are
    treated as unclustered *before* any metric is computed, giving the paper's
    cluster-size>3/>5/>10 re-computations (pass 4 / 6 / 11). ``None`` keeps the
    labels as-is (singletons already marked ``unclustered_value``).
    """
    cluster_ids = np.asarray(cluster_ids)
    true_labels = np.asarray(true_labels)
    n = len(true_labels)

    clustered = cluster_ids != unclustered_value
    if min_cluster_size is not None:
        # Demote points in too-small clusters to "unclustered".
        sizes = Counter(cluster_ids[clustered].tolist())
        keep = np.array([
            clustered[i] and sizes[cluster_ids[i]] >= min_cluster_size
            for i in range(n)
        ])
        clustered = keep
    retention = float(clustered.mean()) if n else float("nan")

    cl = cluster_ids[clustered]
    tr = true_labels[clustered]
    empty = {
        "purity": float("nan"), "retention": retention,
        "nmi": float("nan"), "ari": float("nan"),
        "n_clusters": 0, "n_clustered": 0, "n_total": int(n),
        "pct_clusters_purity_gt90": float("nan"),
        "pct_seqs_in_high_purity_clusters": float("nan"),
        "sensitivity": float("nan"), "n_epitopes_sensitivity": 0,
    }
    if len(cl) == 0:
        return empty

    # Per-cluster majority tallies (single pass).
    purity_num = 0
    hp_clusters = 0
    hp_size = 0
    n_clusters = 0
    for c in set(cl.tolist()):
        members = tr[cl == c]
        size = len(members)
        if not size:
            continue
        n_clusters += 1
        maj = Counter(members.tolist()).most_common(1)[0][1]
        purity_num += maj
        if maj / size >= high_purity_threshold:
            hp_clusters += 1
            hp_size += size
    purity = purity_num / len(cl)

    out = {
        "purity": float(purity),
        "retention": retention,
        "nmi": float(normalized_mutual_info_score(tr, cl)),
        "ari": float(adjusted_rand_score(tr, cl)),
        "n_clusters": int(n_clusters),
        "n_clustered": int(len(cl)),
        "n_total": int(n),
        "pct_clusters_purity_gt90": float(hp_clusters / n_clusters) if n_clusters else float("nan"),
        "pct_seqs_in_high_purity_clusters": float(hp_size / len(cl)),
    }
    out.update(_clustering_sensitivity(
        cluster_ids, true_labels, unclustered_value,
        min_size=sensitivity_min_size, dominance=sensitivity_dominance,
        epitopes=sensitivity_epitopes,
    ))
    return out


def _clustering_sensitivity(
    cluster_ids: np.ndarray,
    true_labels: np.ndarray,
    unclustered_value,
    min_size: int = 3,
    dominance: float = 2.0,
    epitopes: Sequence | None = None,
) -> dict:
    """Macro sensitivity over "epitope-specific" clusters (NAR-GAB 2025).

    For each epitope e: find clusters where (a) size > ``min_size``, and (b) the
    e-points outnumber all non-e points by a factor >= ``dominance`` (the paper's
    "size>3 and epitope share >= 2/3", i.e. dominance == 2). Sensitivity_e is the
    fraction of all e-points that fall in those clusters; the reported value is
    the macro-mean over epitopes.
    """
    cluster_ids = np.asarray(cluster_ids)
    true_labels = np.asarray(true_labels)
    clustered = cluster_ids != unclustered_value
    cl = cluster_ids[clustered]
    tr = true_labels[clustered]
    if len(cl) == 0:
        return {"sensitivity": float("nan"), "n_epitopes_sensitivity": 0}

    # cluster -> Counter(epitope -> count) and total size.
    per_cluster: dict = {}
    for c, e in zip(cl.tolist(), tr.tolist()):
        per_cluster.setdefault(c, Counter())[e] += 1

    all_eps = list(epitopes) if epitopes is not None else sorted(set(true_labels.tolist()))
    total_per_epi = Counter(true_labels.tolist())

    sens_vals = []
    for e in all_eps:
        total_e = total_per_epi.get(e, 0)
        if total_e == 0:
            continue
        covered = 0
        for c, cnt in per_cluster.items():
            size = sum(cnt.values())
            n_e = cnt.get(e, 0)
            if n_e == 0 or size <= min_size:
                continue
            others = size - n_e
            if n_e >= dominance * others:      # e dominates the cluster
                covered += n_e
        sens_vals.append(covered / total_e)

    if not sens_vals:
        return {"sensitivity": float("nan"), "n_epitopes_sensitivity": 0}
    return {"sensitivity": float(np.mean(sens_vals)),
            "n_epitopes_sensitivity": int(len(sens_vals))}


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


# ---------------------------------------------------------------------------
# Conditional TCR design metrics (epitope-conditioned generation, T5)
# ---------------------------------------------------------------------------

def _mean_pairwise_edit(seqs: Sequence[str], cap: int = 500) -> float:
    """Mean pairwise Levenshtein distance within a set (diversity, higher=more diverse)."""
    from .leakage import edit_distance
    uniq = list(dict.fromkeys(seqs))
    if len(uniq) < 2:
        return float("nan")
    if len(uniq) > cap:  # subsample for tractability (O(n^2))
        rng = np.random.default_rng(0)
        uniq = list(rng.choice(uniq, size=cap, replace=False))
    dsum, cnt = 0, 0
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            dsum += edit_distance(uniq[i], uniq[j])
            cnt += 1
    return float(dsum / cnt) if cnt else float("nan")


def design_metrics(
    generated: Sequence[str],
    ref_binders: Sequence[str],
    train_binders: Sequence[str] | None = None,
    k: int = 3,
) -> dict:
    """Epitope-conditioned TCR design quality for one target epitope.

    Aligned with TCRT5 (Nat Mach Intell 2025) / TcrDesign: reward generating
    sequences that (a) exactly match held-out real binders, (b) recover a large
    fraction of the held-out binder set, (c) sit close to real binders in
    sequence space, (d) match the epitope-specific k-mer distribution, while
    (e) remaining novel vs the conditioning/train binders and (f) diverse.

    ``generated``     : sequences sampled for this epitope.
    ``ref_binders``   : held-out real binders for this epitope (evaluation truth).
    ``train_binders`` : binders seen during conditioning/training (for novelty).
    """
    gen = list(generated)
    ref_set = set(ref_binders)
    out = {
        "n_generated": int(len(gen)),
        "n_unique": int(len(set(gen))),
        "n_ref_binders": int(len(ref_set)),
    }
    if not gen or not ref_set:
        out.update({"exact_match_rate": float("nan"), "recovery": float("nan"),
                    "kmer_jsd": float("nan"), "mean_nn_to_ref": float("nan"),
                    "diversity": float("nan"), "novelty": float("nan")})
        return out
    gen_set = set(gen)
    # (a) precision: fraction of generated that are real binders
    out["exact_match_rate"] = float(np.mean([g in ref_set for g in gen]))
    # (b) recall / coverage: fraction of real binders recovered
    out["recovery"] = float(len(gen_set & ref_set) / len(ref_set))
    # (c) proximity to real binders (nearest-neighbour edit distance, lower=better)
    out["mean_nn_to_ref"] = mean_nn_distance(gen, list(ref_set))
    # (d) epitope-specific distribution match
    out["kmer_jsd"] = kmer_jsd(gen, list(ref_set), k=k)
    # (e) novelty vs conditioning binders (avoid trivially copying train)
    if train_binders is not None:
        out["novelty"] = novelty(gen, train_binders)
    # (f) internal diversity
    out["diversity"] = _mean_pairwise_edit(gen)
    return out


# ---------------------------------------------------------------------------
# Per-residue (token-level) classification metrics (NbBench Paratope / VR)
# ---------------------------------------------------------------------------

def token_binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
) -> dict:
    """Per-residue binary metrics over all positions pooled (Paratope).

    ``y_score`` is the probability of the positive (binding) class. Reports the
    metrics named in the NbBench Paratope card: accuracy / precision / recall /
    F1 / AUROC / AUPRC.
    """
    from sklearn.metrics import precision_recall_fscore_support

    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    pred = (s >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return {
        "acc": float((pred == y).mean()) if len(y) else float("nan"),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auroc": _safe_auroc(y, s),
        "auprc": _safe_auprc(y, s),
        "n_positions": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
    }


def token_multiclass_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict:
    """Per-residue multiclass metrics over all positions pooled (VR regions)."""
    from sklearn.metrics import precision_recall_fscore_support

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    macro = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {
        "acc": float((p == y).mean()) if len(y) else float("nan"),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "n_positions": int(len(y)),
        "n_classes": int(len(np.unique(y))),
    }


# BLOSUM62 substitution matrix for CDR-infilling soft recovery (NbBench BR).
_BLOSUM62_CACHE = None


def _blosum62():
    """Lazily load the standard BLOSUM62 matrix (Biopython). Returns ``None`` if
    Biopython is unavailable so BR degrades to NaN without crashing EM."""
    global _BLOSUM62_CACHE
    if _BLOSUM62_CACHE is None:
        try:
            from Bio.Align import substitution_matrices
            _BLOSUM62_CACHE = substitution_matrices.load("BLOSUM62")
        except Exception:
            _BLOSUM62_CACHE = False
    return None if _BLOSUM62_CACHE is False else _BLOSUM62_CACHE


def blosum62_score(a: str, b: str):
    """BLOSUM62 score for one residue pair, or ``None`` if either residue is not
    a standard amino acid (excluded from BR, matching NbBench 'valid positions')."""
    m = _blosum62()
    if m is None:
        return None
    try:
        return float(m[a, b])
    except (KeyError, IndexError):
        return None


def infilling_recovery(
    generated: Sequence[str],
    reference: Sequence[str],
    masks: Sequence[Sequence[bool]],
) -> dict:
    """Amino-acid recovery restricted to masked positions (CDR infilling).

    ``masks[i][j]`` is True where the original sequence was masked. Only those
    positions count toward recovery, matching the CDRInfilling objective (fill
    the blanked CDR residues); non-masked positions are copied verbatim.

    Reports both **hard** recovery -- exact-match amino-acid accuracy over masked
    positions (``aar_masked``, == NbBench Exact-Match) -- and **soft** recovery
    -- mean BLOSUM62 substitution score over masked positions (``br_masked``, ==
    NbBench BLOSUM62 Recovery), which gives partial credit for biochemically
    similar substitutions.
    """
    assert len(generated) == len(reference) == len(masks)
    tot, hit, exact_ok, exact_tot = 0, 0, 0, 0
    br_sum, br_n = 0.0, 0
    for g, r, m in zip(generated, reference, masks):
        L = min(len(g), len(r), len(m))
        seq_masked = 0
        seq_hit = 0
        for i in range(L):
            if not m[i]:
                continue
            seq_masked += 1
            tot += 1
            if g[i] == r[i]:
                hit += 1
                seq_hit += 1
            s = blosum62_score(r[i], g[i])
            if s is not None:
                br_sum += s
                br_n += 1
        if seq_masked:
            exact_tot += 1
            exact_ok += int(seq_hit == seq_masked)
    return {
        "aar_masked": float(hit / tot) if tot else float("nan"),
        "br_masked": float(br_sum / br_n) if br_n else float("nan"),
        "exact_seq_rate": float(exact_ok / exact_tot) if exact_tot else float("nan"),
        "n_masked_positions": int(tot),
        "n_br_positions": int(br_n),
        "n_sequences": int(exact_tot),
    }


# ---------------------------------------------------------------------------
# TCRT5 metric suite (Nat Mach Intell 2025) -- exact port of the official
# ``src/evaluation.py::ModelEvaluator`` static methods so the leaderboard uses
# the paper's own scoring code. Every function here is verified byte-for-byte
# against the upstream implementation in ``tests/test_tcrt5_metrics.py`` and
# cross-validated on ``benchmark_data_w_preds.csv`` (14 benchmark pMHC).
#
# Key semantics preserved from upstream (do NOT "fix"):
#   * precision@k divides correct hits by len(translations) and counts every
#     translation (duplicates included).
#   * recall@k uses set(correct) / min(k, n_references).
#   * mean_sequence_recovery restricts to same-length references and falls back
#     to normalised edit distance only when no same-length reference exists.
#   * char-BLEU picks the ``max_references`` closest references per hypothesis
#     and treats each residue as a token (uniform 1/max_ngram weights).
# ---------------------------------------------------------------------------

try:  # official code uses the C ``Levenshtein`` package
    import Levenshtein as _LEV

    def _tcrt5_lev(a: str, b: str) -> int:
        return _LEV.distance(a, b)
except Exception:  # pragma: no cover - fall back to leakage's implementation
    def _tcrt5_lev(a: str, b: str) -> int:
        from .leakage import edit_distance
        return edit_distance(a, b)


def find_n_closest_matches(query: str, references: Sequence[str], n: int) -> list:
    """Return the ``n`` references closest to ``query`` by Levenshtein distance.

    Mirrors ``ModelEvaluator.find_n_closest_matches``: a *stable* ascending sort
    on distance, so ties keep their original reference order.
    """
    distances = [(ref, _tcrt5_lev(query, ref)) for ref in references]
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:n]]


def tcrt5_precision_at_k(translations: Sequence[str], reference_translations: Sequence[str]) -> float:
    correct = [t for t in translations if t in reference_translations]
    return len(correct) / len(translations) if translations else float("nan")


def tcrt5_recall_at_k(translations: Sequence[str], reference_translations: Sequence[str], k: int) -> float:
    correct = [t for t in translations if t in reference_translations]
    denom = min(k, len(reference_translations))
    return len(set(correct)) / denom if denom else float("nan")


def tcrt5_f1_at_k(translations: Sequence[str], reference_translations: Sequence[str], k: int) -> float:
    precision = tcrt5_precision_at_k(translations, reference_translations)
    recall = tcrt5_recall_at_k(translations, reference_translations, k)
    if precision != precision or recall != recall:  # NaN guard
        return float("nan")
    return 0.0 if precision + recall == 0 else (2 * precision * recall / (precision + recall))


def tcrt5_mean_edit_distance(translations: Sequence[str], reference_translations: Sequence[str]) -> float:
    if not translations or not reference_translations:
        return float("nan")
    edit_distances = []
    for t in translations:
        closest = find_n_closest_matches(t, reference_translations, 1)[0]
        edit_distances.append(_tcrt5_lev(t, closest))
    return sum(edit_distances) / len(translations)


def tcrt5_mean_sequence_recovery(translations: Sequence[str], reference_translations: Sequence[str]) -> float:
    if not translations or not reference_translations:
        return float("nan")
    per_sequence_percents = []
    for t in translations:
        same_len = [r for r in reference_translations if len(r) == len(t)]
        if len(same_len) == 0:
            closest = find_n_closest_matches(t, reference_translations, 1)[0]
            per_sequence_percents.append(1 - _tcrt5_lev(t, closest) / len(closest))
            continue
        closest = find_n_closest_matches(t, same_len, 1)[0]
        idx_recovery = [1 if ch == closest[i] else 0 for i, ch in enumerate(t)]
        per_sequence_percents.append(sum(idx_recovery) / len(t))
    return float(np.mean(per_sequence_percents))


def tcrt5_sequence_bleu(
    translation: str,
    references: Sequence[str],
    max_references: int = 20,
    max_ngram: int = 4,
) -> float:
    """Sentence-level char-BLEU for one hypothesis (port of ``_sequence_bleu``)."""
    from nltk.translate.bleu_score import sentence_bleu

    refs = [list(x) for x in find_n_closest_matches(translation, references, n=max_references)]
    hyp = list(translation)
    if not refs or not hyp:
        return float("nan")
    return float(
        sentence_bleu(
            refs,
            hyp,
            weights=tuple([1 / max_ngram] * max_ngram),
            smoothing_function=None,
        )
    )


def tcrt5_design_metrics(
    generated: Sequence[str],
    references: Sequence[str],
    k: int,
    greedy: str | None = None,
    max_ngram: int = 4,
    max_references: int = 20,
) -> dict:
    """All per-epitope TCRT5 metrics for one target from pre-generated samples.

    ``generated``  : the K sampled CDR3b sequences (top-k / beam list).
    ``references`` : held-out real binders for this epitope (evaluation truth).
    ``greedy``     : optional single greedy decode used for char-BLEU (matches
                     the paper, which computes BLEU from a greedy hypothesis);
                     when absent we fall back to the first sampled sequence.
    """
    gen = [str(s) for s in generated]
    ref = [str(s) for s in references]
    out = {
        "n_generated": int(len(gen)),
        "n_unique": int(len(set(gen))),
        "n_references": int(len(ref)),
    }
    if not gen or not ref:
        out.update({m: float("nan") for m in
                    ("precision", "recall", "f1", "d_edit", "seq_recovery", "char_bleu", "diversity")})
        return out
    out["precision"] = tcrt5_precision_at_k(gen, ref)
    out["recall"] = tcrt5_recall_at_k(gen, ref, k=k)
    out["f1"] = tcrt5_f1_at_k(gen, ref, k=k)
    out["d_edit"] = tcrt5_mean_edit_distance(gen, ref)
    out["seq_recovery"] = tcrt5_mean_sequence_recovery(gen, ref)
    bleu_hyp = greedy if greedy else gen[0]
    out["char_bleu"] = tcrt5_sequence_bleu(bleu_hyp, ref, max_references=max_references, max_ngram=max_ngram)
    out["diversity"] = float(len(set(gen)) / len(gen))
    return out


def tcrt5_dataset_metrics(
    per_epitope_generated: dict[str, Sequence[str]],
    per_epitope_references: dict[str, Sequence[str]],
    k: int,
    per_epitope_greedy: dict[str, str] | None = None,
) -> dict:
    """Dataset-level TCRT5 metrics (port of ``dataset_metrics_at_k``).

    precision / recall / f1 / d_edit / seq_recovery are means over the per-source
    metrics; diversity is unique/total pooled over *all* generated sequences;
    char-BLEU is the mean sentence char-BLEU over sources (greedy hypothesis).
    """
    per_epitope_greedy = per_epitope_greedy or {}
    prec, rec, f1, dedit, srec, cbleu = [], [], [], [], [], []
    all_gen: list[str] = []
    per_ep: dict[str, dict] = {}
    for ep, gen in per_epitope_generated.items():
        ref = per_epitope_references.get(ep, [])
        m = tcrt5_design_metrics(gen, ref, k=k, greedy=per_epitope_greedy.get(ep))
        per_ep[ep] = m
        all_gen.extend([str(s) for s in gen])
        for bucket, key in ((prec, "precision"), (rec, "recall"), (f1, "f1"),
                            (dedit, "d_edit"), (srec, "seq_recovery"), (cbleu, "char_bleu")):
            v = m.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                bucket.append(v)

    def _mean(vals):
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "char_bleu": _mean(cbleu),
        "precision": _mean(prec),
        "recall": _mean(rec),
        "f1": _mean(f1),
        "d_edit": _mean(dedit),
        "seq_recovery": _mean(srec),
        "diversity": float(len(set(all_gen)) / len(all_gen)) if all_gen else float("nan"),
        "n_epitopes": int(len(per_ep)),
        "per_epitope": per_ep,
    }


def aggregate_design(per_epitope: dict, categories: dict | None = None) -> dict:
    """Macro-average per-epitope design metrics, optionally split by category.

    ``per_epitope`` : {epitope: design_metrics dict}.
    ``categories``  : {epitope: 'common'|'rare'|'novel'} for stratified macro means.
    """
    keys = ["exact_match_rate", "recovery", "mean_nn_to_ref", "kmer_jsd",
            "novelty", "diversity"]

    def _macro(eps):
        res = {}
        for key in keys:
            vals = [per_epitope[e][key] for e in eps
                    if e in per_epitope and key in per_epitope[e]
                    and not (isinstance(per_epitope[e][key], float)
                             and np.isnan(per_epitope[e][key]))]
            res[key] = float(np.mean(vals)) if vals else float("nan")
        res["n_epitopes"] = int(len(eps))
        return res

    summary = {"overall": _macro(list(per_epitope.keys()))}
    if categories:
        for cat in ("common", "rare", "novel"):
            eps = [e for e in per_epitope if categories.get(e) == cat]
            if eps:
                summary[cat] = _macro(eps)
    return summary


__all__ = [
    "binding_metrics", "official_binding_report", "clustering_metrics",
    "linear_probe_metrics", "knn_top1_metric",
    "amino_acid_recovery", "novelty", "mean_nn_distance",
    "kmer_jsd", "generation_metrics",
    "design_metrics", "aggregate_design",
    "token_binary_metrics", "token_multiclass_metrics", "infilling_recovery",
    # TCRT5 (Nat Mach Intell 2025) metric suite
    "find_n_closest_matches",
    "tcrt5_precision_at_k", "tcrt5_recall_at_k", "tcrt5_f1_at_k",
    "tcrt5_mean_edit_distance", "tcrt5_mean_sequence_recovery", "tcrt5_sequence_bleu",
    "tcrt5_design_metrics", "tcrt5_dataset_metrics",
]
