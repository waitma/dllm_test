"""Leakage-controlled splitting for IRBench.

Data leakage is the single biggest threat to TCR / PPI benchmarks: the same
clonotype (or a near-identical one) appearing in both train and test inflates
every metric. This module centralizes the controls so each task gets the same
guarantees:

1. ``epitope_holdout_split``  -- unseen-epitope generalization (test epitopes
   are absent from train). This is the *main* TCR-binding ranking.
2. ``seen_epitope_split``     -- epitopes shared, clonotypes disjoint.
3. ``dedup_against``          -- drop train rows whose clonotype occurs in test.
4. ``leakage_report``         -- assert/measure cross-split overlap (must be 0).
5. ``min_edit_distance`` / ``passes_distance`` -- edit-distance guards used by
   negative sampling (a negative TCR must be far from the epitope's positives).

Edit distance uses ``rapidfuzz`` when available and falls back to a length-
prefiltered pure-Python Levenshtein otherwise.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:  # fast C implementation
    from rapidfuzz.distance import Levenshtein as _RF_LEV

    def _lev(a: str, b: str) -> int:
        return _RF_LEV.distance(a, b)

    def _lev_min_to_set(query: str, refs: Sequence[str], cutoff: int) -> int:
        """Minimum Levenshtein distance from ``query`` to any ref, capped logic.

        Returns the true min distance; uses rapidfuzz score_cutoff for speed.
        """
        best = len(query) + 1
        for r in refs:
            d = _RF_LEV.distance(query, r, score_cutoff=best)
            if d < best:
                best = d
                if best == 0:
                    break
        return best

    _HAVE_RAPIDFUZZ = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_RAPIDFUZZ = False

    def _lev(a: str, b: str) -> int:
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            cur = [i] + [0] * lb
            ca = a[i - 1]
            for j in range(1, lb + 1):
                cost = 0 if ca == b[j - 1] else 1
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            prev = cur
        return prev[lb]

    def _lev_min_to_set(query: str, refs: Sequence[str], cutoff: int) -> int:
        best = len(query) + 1
        lq = len(query)
        for r in refs:
            if abs(len(r) - lq) >= best:  # length lower-bounds edit distance
                continue
            d = _lev(query, r)
            if d < best:
                best = d
                if best == 0:
                    break
        return best


def edit_distance(a: str, b: str) -> int:
    return _lev(a or "", b or "")


def min_edit_distance(query: str, refs: Iterable[str]) -> int:
    refs = [r for r in refs if r]
    if not refs:
        return len(query or "")
    return _lev_min_to_set(query or "", refs, cutoff=10**9)


def passes_distance(query: str, refs: Iterable[str], threshold: int) -> bool:
    """True iff ``query`` is > ``threshold`` edits from every ref (a valid negative)."""
    return min_edit_distance(query, refs) > threshold


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def epitope_holdout_split(
    df: pd.DataFrame,
    test_epitopes: Sequence[str],
    epitope_col: str = "peptide",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unseen-epitope split: rows of ``test_epitopes`` go to test, rest to train.

    Guarantees no epitope appears in both splits.
    """
    test_set = set(test_epitopes)
    is_test = df[epitope_col].isin(test_set)
    return df[~is_test].copy(), df[is_test].copy()


def seen_epitope_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    clonotype: pd.Series | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seen-epitope split with clonotype-disjoint train/test.

    Clonotypes (not rows) are partitioned so a TCR clonotype never crosses the
    split, while every epitope can appear in both.
    """
    if clonotype is None:
        from .schema import clonotype_key
        clonotype = clonotype_key(df)
    clono = clonotype.to_numpy()
    uniq = np.array(sorted(set(clono)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = int(round(len(uniq) * test_frac))
    test_clonos = set(uniq[:n_test])
    is_test = np.array([c in test_clonos for c in clono])
    return df[~is_test].copy(), df[is_test].copy()


def dedup_against(
    train: pd.DataFrame,
    test: pd.DataFrame,
    keys: Sequence[str] = ("cdr3a", "cdr3b"),
) -> pd.DataFrame:
    """Drop train rows whose key-tuple also appears in test (clonotype leakage)."""
    def _key(frame):
        return frame[list(keys)].astype(str).agg("|".join, axis=1)

    test_keys = set(_key(test))
    train_keys = _key(train)
    mask = ~train_keys.isin(test_keys)
    return train[mask.to_numpy()].copy()


def leakage_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    clono_keys: Sequence[str] = ("cdr3a", "cdr3b"),
    epitope_col: str = "peptide",
) -> dict:
    """Quantify cross-split overlap. All clonotype overlaps should be 0."""
    def _key(frame, cols):
        return set(frame[list(cols)].astype(str).agg("|".join, axis=1))

    tr_clono = _key(train, clono_keys)
    te_clono = _key(test, clono_keys)
    tr_pair = _key(train, list(clono_keys) + [epitope_col])
    te_pair = _key(test, list(clono_keys) + [epitope_col])
    tr_ep = set(train[epitope_col].astype(str))
    te_ep = set(test[epitope_col].astype(str))
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "clonotype_overlap": int(len(tr_clono & te_clono)),
        "pair_overlap": int(len(tr_pair & te_pair)),
        "shared_epitopes": int(len(tr_ep & te_ep)),
        "test_only_epitopes": sorted(te_ep - tr_ep),
        "n_test_only_epitopes": int(len(te_ep - tr_ep)),
    }


__all__ = [
    "edit_distance", "min_edit_distance", "passes_distance",
    "epitope_holdout_split", "seen_epitope_split",
    "dedup_against", "leakage_report",
]
