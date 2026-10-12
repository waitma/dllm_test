"""Few-shot per-epitope nearest-neighbour AUROC (SCEPTR protocol).

This implements the T3 *main* protocol from **SCEPTR (Cell Systems 2024)**: for a
target epitope ``e`` we are given only ``k`` reference TCRs known to bind ``e``
(the *support set*), and must rank every query TCR by how likely it is to bind
``e``. A TCR is scored by its nearest-neighbour distance to the support set
(negated so higher = more similar = more likely a binder). The per-epitope AUROC
over the query set (positives = query TCRs binding ``e``) measures how well a
representation/distance separates specific binders from the rest with almost no
supervision -- exactly the regime where SCEPTR showed generic PLMs (ESM2/ProtBert)
lose to sequence-comparison methods (TCRdist / CDR3 Levenshtein).

Backends (a "distance/score provider"), all reduced to a single
``[N_query, N_ref]`` distance matrix so few-shot sampling is just column
indexing:

1. **native cdist** -- ``cdist_fn(query_df, ref_df) -> ndarray[Nq, Nr]``
   (SCEPTR ``calc_cdist_matrix`` / TCRdist real distance matrices).
2. **embedding**    -- embed concatenated columns, then cosine distance
   (ESM2 / k-mer / Ophiuchus / BioSeq).
3. **editdist**     -- CDR3b (+ optional CDR3a) Levenshtein distance, reusing
   :func:`common.leakage.edit_distance` (rapidfuzz-backed).

Everything is pure numpy/sklearn and degrades gracefully: an epitope with fewer
than ``k`` reference binders is skipped for that shot, a single-class query
returns NaN AUROC (dropped from the macro mean) instead of crashing.

Run the self-check (zero extra deps, editdist backend) with::

    python -m common.fewshot
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from common import metrics  # noqa: E402

DEFAULT_SHOTS = (1, 2, 5, 10, 20, 50, 100)
DEFAULT_SEEDS = 5


# ---------------------------------------------------------------------------
# Distance-matrix builders (the three backends)
# ---------------------------------------------------------------------------

def _col_join(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Concatenate the requested sequence columns row-wise (e.g. CDR3b+CDR3a)."""
    parts = [df[c].fillna("").astype(str) for c in columns]
    return parts[0].str.cat(others=parts[1:], sep="") if len(parts) > 1 else list(parts[0])


def cosine_distance_matrix(query_emb: np.ndarray, ref_emb: np.ndarray) -> np.ndarray:
    """Cosine distance ``1 - cos`` between every query and reference embedding."""
    q = np.asarray(query_emb, dtype=np.float64)
    r = np.asarray(ref_emb, dtype=np.float64)
    qn = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
    rn = r / np.clip(np.linalg.norm(r, axis=1, keepdims=True), 1e-8, None)
    sim = qn @ rn.T
    return (1.0 - sim).astype(np.float32)


def edit_distance_matrix(
    query_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    columns: Sequence[str] = ("cdr3b",),
) -> np.ndarray:
    """Summed per-column Levenshtein distance matrix ``[Nq, Nr]``.

    Uses ``rapidfuzz.process.cdist`` (same Levenshtein metric as
    :func:`common.leakage.edit_distance`) for a fast C matrix when available,
    and otherwise falls back to a pure-Python loop over ``edit_distance``.
    """
    total = np.zeros((len(query_df), len(ref_df)), dtype=np.float32)
    for c in columns:
        q = query_df[c].fillna("").astype(str).tolist()
        r = ref_df[c].fillna("").astype(str).tolist()
        try:
            from rapidfuzz.distance import Levenshtein as _Lev
            from rapidfuzz import process as _rf_process

            d = _rf_process.cdist(q, r, scorer=_Lev.distance, dtype=np.float32)
        except Exception:  # pragma: no cover - fallback path
            from common.leakage import edit_distance

            d = np.empty((len(q), len(r)), dtype=np.float32)
            for i, qi in enumerate(q):
                for j, rj in enumerate(r):
                    d[i, j] = edit_distance(qi, rj)
        total += d
    return total


def grammar_pair_embed(
    embedder,
    df: pd.DataFrame,
    columns: Sequence[str] = ("cdr3b", "cdr3a"),
) -> np.ndarray:
    """Joint multi-chain LLaDA-decoder representation for the grammar model.

    Renders each TCR as a joint ``task_type=tcr`` record (β + optional α, no
    epitope for T3) and runs the clean LLaDA decoder forward. Returns **one
    globally mean-pooled** vector per row (``pool_mode="global"``). The peptide
    slot is omitted (``peptide=None``) so there is no epitope leakage in T3.
    """
    beta = df["cdr3b"].fillna("").astype(str).tolist()
    has_alpha = ("cdr3a" in columns) and ("cdr3a" in df.columns)
    alpha = df["cdr3a"].fillna("").astype(str).tolist() if has_alpha else None
    order = ("beta", "alpha") if has_alpha else ("beta",)
    vec = embedder.embed_pairs(beta=beta, alpha=alpha, peptide=None, order=order)
    return np.asarray(vec, dtype=np.float32)


def embedding_distance_matrix(
    embedder,
    query_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    columns: Sequence[str] = ("cdr3b", "cdr3a"),
) -> np.ndarray:
    """Embed concatenated ``columns`` for both pools, return cosine distance.

    Embeddings are computed per column over the *unique* strings (deduplicated)
    and concatenated, matching how ``tcr_representation`` featurizes columns.

    Exception: a **grammar decoder** embedder (``feature_source == "decoder"``,
    i.e. our LLaDA multi-chain representation) is not a per-column encoder — it
    needs the joint β(+α) record — so it is routed through
    :func:`grammar_pair_embed`, which omits the peptide segment entirely (not a
    constant placeholder) and therefore cannot leak the epitope. All other
    embedders keep the per-column path unchanged.
    """
    if getattr(embedder, "feature_source", None) == "decoder":
        return cosine_distance_matrix(
            grammar_pair_embed(embedder, query_df, columns),
            grammar_pair_embed(embedder, ref_df, columns),
        )

    def _feats(df: pd.DataFrame) -> np.ndarray:
        blocks = []
        for c in columns:
            vals = df[c].fillna("").astype(str)
            uniq = sorted(set(vals))
            emb = embedder.embed(uniq)
            lut = {u: emb[i] for i, u in enumerate(uniq)}
            blocks.append(np.stack([lut[v] for v in vals]))
        return np.concatenate(blocks, axis=1)

    return cosine_distance_matrix(_feats(query_df), _feats(ref_df))


def build_distance_matrix(
    method: str,
    query_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    columns: Sequence[str] = ("cdr3b", "cdr3a"),
    cdist_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray] | None = None,
    embedder=None,
) -> np.ndarray:
    """Dispatch to one of the three backends → ``[N_query, N_ref]`` distances."""
    if method == "native":
        if cdist_fn is None:
            raise ValueError("method='native' requires a cdist_fn")
        d = np.asarray(cdist_fn(query_df, ref_df), dtype=np.float32)
        if d.shape != (len(query_df), len(ref_df)):
            raise ValueError(
                f"cdist_fn returned {d.shape}, expected {(len(query_df), len(ref_df))}"
            )
        return d
    if method == "editdist":
        return edit_distance_matrix(query_df, ref_df, columns=columns)
    if method == "embed":
        if embedder is None:
            raise ValueError("method='embed' requires an embedder")
        return embedding_distance_matrix(embedder, query_df, ref_df, columns=columns)
    raise ValueError(f"unknown few-shot method: {method}")


# ---------------------------------------------------------------------------
# Core few-shot evaluator
# ---------------------------------------------------------------------------

def few_shot_nn_auroc(
    dist: np.ndarray,
    query_labels: Sequence[str],
    ref_labels: Sequence[str],
    epitopes: Sequence[str] | None = None,
    shots: Sequence[int] = DEFAULT_SHOTS,
    seeds: int = DEFAULT_SEEDS,
    agg: str = "min",
    base_seed: int = 0,
) -> dict:
    """SCEPTR-style few-shot per-epitope NN AUROC over a precomputed matrix.

    Parameters
    ----------
    dist : ndarray ``[N_query, N_ref]``
        Distance from every query TCR (rows) to every reference/pool TCR (cols);
        smaller = more similar.
    query_labels : the epitope each query TCR binds (evaluation truth).
    ref_labels   : the epitope each reference-pool TCR binds (support pool).
    epitopes     : which target epitopes to evaluate (default: all present in
                   both query and reference pools).
    shots        : list of support-set sizes ``k``.
    seeds        : number of independent support-set resamples ``R`` per (k, e).
    agg          : ``"min"`` (1-NN over support) or ``"mean"`` (mean-NN).

    Returns
    -------
    dict with ``auroc_by_shot`` (macro mean±std over epitopes), full
    ``per_epitope_by_shot``, plus ``shots``, ``seeds``, ``agg``,
    ``n_epitopes_by_shot``.
    """
    dist = np.asarray(dist, dtype=np.float32)
    query_labels = np.asarray(query_labels)
    ref_labels = np.asarray(ref_labels)

    if epitopes is None:
        epitopes = sorted(set(query_labels.tolist()) & set(ref_labels.tolist()))
    epitopes = list(epitopes)

    # Precompute reference indices per epitope.
    ref_idx_by_ep = {e: np.where(ref_labels == e)[0] for e in epitopes}

    auroc_by_shot: dict[int, dict] = {}
    per_epitope_by_shot: dict[int, dict] = {}
    n_ep_by_shot: dict[int, int] = {}

    for k in shots:
        per_ep: dict[str, float] = {}
        for e in epitopes:
            pool = ref_idx_by_ep[e]
            if len(pool) < k:
                continue  # not enough reference binders → skip this (k, e)
            labels = (query_labels == e).astype(int)
            if labels.sum() == 0 or labels.sum() == len(labels):
                continue  # single-class query → AUROC undefined
            seed_aurocs = []
            for s in range(seeds):
                rng = np.random.default_rng(base_seed + s + 1000 * int(k))
                ref_sel = rng.choice(pool, size=k, replace=False)
                sub = dist[:, ref_sel]  # [N_query, k]
                if agg == "mean":
                    nn = sub.mean(axis=1)
                else:
                    nn = sub.min(axis=1)
                score = -nn  # closer = higher score
                au = metrics._safe_auroc(labels, score)
                if not np.isnan(au):
                    seed_aurocs.append(au)
            if seed_aurocs:
                per_ep[e] = float(np.mean(seed_aurocs))
        vals = np.array(list(per_ep.values()), dtype=float)
        auroc_by_shot[int(k)] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std()) if len(vals) else float("nan"),
        }
        per_epitope_by_shot[int(k)] = {e: float(v) for e, v in per_ep.items()}
        n_ep_by_shot[int(k)] = int(len(vals))

    return {
        "auroc_by_shot": auroc_by_shot,
        "per_epitope_by_shot": per_epitope_by_shot,
        "n_epitopes_by_shot": n_ep_by_shot,
        "shots": [int(k) for k in shots],
        "seeds": int(seeds),
        "agg": agg,
    }


def evaluate_fewshot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str,
    columns: Sequence[str] = ("cdr3b", "cdr3a"),
    cdist_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray] | None = None,
    embedder=None,
    epitope_col: str = "peptide",
    shots: Sequence[int] = DEFAULT_SHOTS,
    seeds: int = DEFAULT_SEEDS,
    agg: str = "min",
) -> dict:
    """End-to-end: build the query(test)×ref(train) distance matrix, then score.

    ``train_df`` is the reference/support pool, ``test_df`` is the query set —
    they are clonotype-disjoint (overlap=0) so no leakage between support and
    query.
    """
    dist = build_distance_matrix(
        method, test_df, train_df, columns=columns,
        cdist_fn=cdist_fn, embedder=embedder,
    )
    result = few_shot_nn_auroc(
        dist,
        query_labels=test_df[epitope_col].astype(str).to_numpy(),
        ref_labels=train_df[epitope_col].astype(str).to_numpy(),
        shots=shots, seeds=seeds, agg=agg,
    )
    result["method"] = method
    result["columns"] = list(columns)
    result["n_train"] = int(len(train_df))
    result["n_test"] = int(len(test_df))
    return result


# ---------------------------------------------------------------------------
# Oracle self-check: references drawn from the query positives themselves.
# With perfect information the min-distance score isolates each positive
# (distance 0 to itself), so per-epitope AUROC must be ~1.0. This validates
# the scoring direction and AUROC plumbing independently of any model.
# ---------------------------------------------------------------------------

def _closeness_matrix_vec(
    universe_vec: np.ndarray,
    binder_idx: np.ndarray,
    metric: str,
    univ_norm: np.ndarray | None = None,
) -> np.ndarray:
    """Per-epitope closeness matrix ``[N_universe, n_binders]`` (higher=closer).

    ``metric="cosine"``    → cosine similarity (universe vs binder columns).
    ``metric="euclidean"`` → ``-squared_euclidean`` (rank-equivalent to
    ``-euclidean``; monotone so AUROC is identical, avoids a sqrt).

    All matmuls go through BLAS so the big epitopes (NLV ~10k binders ×70k
    universe) build in seconds. Reference sub-sampling is then just column
    indexing + a max reduction on this matrix.
    """
    B = universe_vec[binder_idx]
    if metric == "cosine":
        return universe_vec @ B.T  # inputs pre-normalised → cosine similarity
    if metric == "euclidean":
        G = universe_vec @ B.T
        un = univ_norm if univ_norm is not None else (universe_vec ** 2).sum(1)
        bn = un[binder_idx]
        # squared euclidean = ||u||^2 + ||b||^2 - 2 u.b ; negate → closeness
        return (2.0 * G) - un[:, None] - bn[None, :]
    raise ValueError(f"unknown vector metric: {metric}")


def few_shot_paper_auroc(
    peptides: Sequence[str],
    targets: Sequence[str],
    shots: Sequence[int] = (1, 2, 5, 10, 20, 50, 100, 200),
    seeds: int = 100,
    base_seed: int = 0,
    universe_vec: np.ndarray | None = None,
    metric: str = "cosine",
    pairwise_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    background_label: str = "__background__",
    max_ref_pool: int | None = 2000,
) -> dict:
    """SCEPTR III.1 / Table SI **deep** few-shot: per-pMHC NN AUROC vs full VDJdb.

    Unlike :func:`few_shot_nn_auroc` (broad 24-epitope T3, where negatives are the
    *other target epitopes*), here the negative background is the **entire
    filtered VDJdb repertoire**: for target ``e`` the query positives are ``e``'s
    held-out binders and the query negatives are *all* other universe rows
    (background + binders of the other 5 targets). This is the paper's口径.

    The universe rows are ordered ``[target_binders..., background...]`` and
    ``peptides[i]`` is the epitope label of row ``i`` (a target epitope or
    ``background_label``). Two backends produce the per-epitope closeness matrix
    ``C_e`` (``[N_universe, n_binders_e]``, higher = closer):

    - **vector** : pass ``universe_vec`` ``[N_universe, d]`` + ``metric``
      (``cosine`` for embeddings, ``euclidean`` for SCEPTR L2). Cosine assumes
      ``universe_vec`` is already L2-normalised.
    - **pairwise**: pass ``pairwise_fn(binder_idx) -> distance[N_universe,
      n_binders_e]`` (lower = closer; e.g. editdist / TCRdist). Converted to
      closeness via negation.

    For each ``(e, k)`` we resample ``k`` reference binders ``seeds`` times (same
    RNG stream per ``(e,k,seed)`` → identical splits across models), score every
    query by max closeness to the reference (1-NN), exclude the reference rows,
    and take the AUROC of held-out ``e`` positives vs everything else. Returns
    per-epitope AUROC per shot + the macro mean over the 6 targets.
    """
    peptides = np.asarray(peptides)
    targets = list(targets)
    nu = len(peptides)

    univ_norm = None
    if universe_vec is not None:
        universe_vec = np.asarray(universe_vec, dtype=np.float32)
        if metric == "cosine":
            universe_vec = universe_vec / np.clip(
                np.linalg.norm(universe_vec, axis=1, keepdims=True), 1e-8, None)
        elif metric == "euclidean":
            univ_norm = (universe_vec ** 2).sum(1).astype(np.float32)

    per_epitope_by_shot: dict[int, dict] = {int(k): {} for k in shots}
    n_ref_by_shot: dict[int, dict] = {int(k): {} for k in shots}

    for e_idx, e in enumerate(targets):
        binder_idx = np.where(peptides == e)[0]     # ALL binders (positive labels)
        nb = len(binder_idx)
        pos_all = (peptides == e).astype(int)

        # Reference *candidates* = the binder columns of the closeness matrix.
        # Capping only bounds which binders can be sampled as references (and the
        # matrix width for editdist/TCRdist); ALL binders remain query positives,
        # so held-out-positive counts are unchanged. The subsample is a fixed,
        # model-independent draw so every model shares the same candidate pool.
        if max_ref_pool is not None and nb > max_ref_pool:
            crng = np.random.default_rng(7_000_000 + e_idx)
            cand_idx = np.sort(crng.choice(binder_idx, size=max_ref_pool, replace=False))
        else:
            cand_idx = binder_idx
        ncand = len(cand_idx)

        # Build the per-epitope closeness matrix once, reuse for every (k, seed).
        if universe_vec is not None:
            C = _closeness_matrix_vec(universe_vec, cand_idx, metric, univ_norm)
        else:
            if pairwise_fn is None:
                raise ValueError("pairwise_fn required when universe_vec is None")
            D = np.asarray(pairwise_fn(cand_idx), dtype=np.float32)
            C = -D
        C = np.asarray(C, dtype=np.float32)

        for k in shots:
            k = int(k)
            if ncand <= k or nb <= k:
                continue  # need >=1 held-out positive and k candidate references
            seed_aurocs = []
            for s in range(seeds):
                # Deterministic across processes (no salted hash) so every model
                # sees identical reference/query splits per (epitope, k, seed).
                seed_val = (
                    base_seed
                    + s
                    + 1_000 * k
                    + 1_000_000 * e_idx
                )
                rng = np.random.default_rng(seed_val)
                ref_cols = rng.choice(ncand, size=k, replace=False)
                score = C[:, ref_cols].max(axis=1)          # 1-NN closeness
                ref_global = cand_idx[ref_cols]
                keep = np.ones(nu, dtype=bool)
                keep[ref_global] = False                    # drop reference rows
                y = pos_all[keep]
                sc = score[keep]
                au = metrics._safe_auroc(y, sc)
                if not np.isnan(au):
                    seed_aurocs.append(au)
            if seed_aurocs:
                per_epitope_by_shot[k][e] = {
                    "mean": float(np.mean(seed_aurocs)),
                    "std": float(np.std(seed_aurocs)),
                    "n_seeds": len(seed_aurocs),
                    "n_pos_heldout": int(nb - k),
                }
                n_ref_by_shot[k][e] = int(nb)
        del C

    # Macro mean over the six targets, per shot.
    auroc_by_shot: dict[int, dict] = {}
    for k in shots:
        k = int(k)
        vals = [d["mean"] for d in per_epitope_by_shot[k].values()]
        auroc_by_shot[k] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if vals else float("nan"),
            "n_epitopes": len(vals),
        }

    return {
        "auroc_by_shot": auroc_by_shot,
        "per_epitope_by_shot": per_epitope_by_shot,
        "n_binders_by_shot": n_ref_by_shot,
        "shots": [int(k) for k in shots],
        "seeds": int(seeds),
        "targets": targets,
        "protocol": "sceptr_paper6_deep",
    }


def oracle_check(
    test_df: pd.DataFrame,
    method: str = "editdist",
    columns: Sequence[str] = ("cdr3b", "cdr3a"),
    cdist_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray] | None = None,
    embedder=None,
    epitope_col: str = "peptide",
) -> dict:
    """Sanity oracle: the support set *contains the query's own positives*.

    We deliberately break the few-shot regime and use **all** query positives of
    an epitope as its reference set. Every positive query then sits at distance 0
    from itself in the reference set, so its NN score is maximal → per-epitope
    AUROC must be ~1.0 for a correct scoring direction + AUROC computation. This
    is independent of the backend / model quality; a failure means the plumbing
    (sign of the score, label alignment, metric) is wrong.
    """
    dist = build_distance_matrix(
        method, test_df, test_df, columns=columns,
        cdist_fn=cdist_fn, embedder=embedder,
    )
    labels = test_df[epitope_col].astype(str).to_numpy()
    epitopes = sorted(set(labels.tolist()))
    per_ep: dict[str, float] = {}
    for e in epitopes:
        ref_idx = np.where(labels == e)[0]
        y = (labels == e).astype(int)
        if y.sum() == 0 or y.sum() == len(y) or len(ref_idx) == 0:
            continue
        score = -dist[:, ref_idx].min(axis=1)  # positives include themselves → 0
        au = metrics._safe_auroc(y, score)
        if not np.isnan(au):
            per_ep[e] = float(au)
    vals = np.array(list(per_ep.values()), dtype=float)
    return {
        "method": f"oracle:{method}",
        "mean_auroc": float(vals.mean()) if len(vals) else float("nan"),
        "min_auroc": float(vals.min()) if len(vals) else float("nan"),
        "n_epitopes": int(len(vals)),
        "per_epitope": per_ep,
    }


def _self_test() -> int:
    """Run the editdist oracle on the T3 dataset and assert AUROC≈1.0."""
    data = _BENCH / "data" / "tcr_representation"
    test = pd.read_csv(data / "test.csv").fillna("")
    train = pd.read_csv(data / "train.csv").fillna("")
    print(f"[self-test] loaded train={len(train)} test={len(test)}")

    oracle = oracle_check(test, method="editdist", columns=("cdr3b", "cdr3a"))
    print("[oracle] editdist (support ⊇ query positives):")
    print(f"  mean AUROC={oracle['mean_auroc']:.4f}  "
          f"min AUROC={oracle['min_auroc']:.4f}  (n_epitopes={oracle['n_epitopes']})")
    assert oracle["mean_auroc"] > 0.999 and oracle["min_auroc"] > 0.99, (
        "ORACLE FAILED: few-shot AUROC should be ~1.0 with perfect refs"
    )
    print("[oracle] PASS (AUROC ~ 1.0 as expected)\n")

    # Quick real few-shot with the zero-dependency editdist backend.
    res = evaluate_fewshot(
        train, test, method="editdist", columns=("cdr3b", "cdr3a"),
        shots=(1, 5, 20, 100), seeds=3,
    )
    print("[editdist few-shot] CDR3b+CDR3a Levenshtein NN AUROC:")
    for k in res["shots"]:
        s = res["auroc_by_shot"][k]
        print(f"  k={k:<3d} AUROC={s['mean']:.4f}±{s['std']:.4f} "
              f"(n_epitopes={res['n_epitopes_by_shot'][k]})")
    return 0


__all__ = [
    "DEFAULT_SHOTS", "DEFAULT_SEEDS",
    "cosine_distance_matrix", "edit_distance_matrix", "embedding_distance_matrix",
    "grammar_pair_embed",
    "build_distance_matrix", "few_shot_nn_auroc", "evaluate_fewshot",
    "few_shot_paper_auroc", "oracle_check",
]


if __name__ == "__main__":
    raise SystemExit(_self_test())
