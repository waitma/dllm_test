"""Dependency-light, CPU-only baselines for IRBench.

These are *not* re-implementations of published deep models (those are run from
their official repos under ``baselines/``). They are the classic, training-free
reference points every TCR benchmark reports:

- ``DistanceKNNScorer``  -- TCRdist/TCRbase-style nearest-neighbour scoring: a
  test pair (TCR, epitope) is scored by similarity (negative CDR3 edit distance)
  to the epitope's known binders. Honest on unseen epitopes: with no reference
  binders it returns the neutral prior (=> ~0.5 AUROC), which is the correct
  expectation for a pure distance method facing a novel epitope.
- ``KmerFeaturizer``     -- fixed k-mer composition vectors for a cheap linear
  baseline / sanity featurization.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd

from .leakage import edit_distance

AA = "ACDEFGHIKLMNPQRSTVWY"


class KmerFeaturizer:
    def __init__(self, k: int = 3, alphabet: str = AA):
        self.k = k
        self.alphabet = alphabet
        self.vocab = self._build_vocab()
        self.dim = len(self.vocab)

    def _build_vocab(self) -> dict:
        import itertools
        return {"".join(p): i for i, p in
                enumerate(itertools.product(self.alphabet, repeat=self.k))}

    def transform(self, seqs: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(seqs), self.dim), dtype=np.float32)
        for i, s in enumerate(seqs):
            s = str(s)
            if len(s) < self.k:
                continue
            c = Counter(s[j:j + self.k] for j in range(len(s) - self.k + 1))
            tot = sum(c.values())
            for kmer, cnt in c.items():
                idx = self.vocab.get(kmer)
                if idx is not None:
                    out[i, idx] = cnt / tot
        return out


class DistanceKNNScorer:
    """TCRdist/TCRbase-style scorer over CDR3 edit distance.

    fit(): index training positives by epitope.
    score(): for each test row, score = exp(-min_dist / scale) over the K
             nearest training binders of the *same* epitope (CDR3b, optionally
             combined with CDR3a). Unknown epitope -> neutral 0.0.
    """

    def __init__(self, k: int = 5, scale: float = 2.0, use_alpha: bool = True,
                 epitope_col: str = "peptide"):
        self.k = k
        self.scale = scale
        self.use_alpha = use_alpha
        self.epitope_col = epitope_col
        self._index: dict[str, list[tuple[str, str]]] = {}

    def fit(self, df: pd.DataFrame, label_col: str = "label") -> "DistanceKNNScorer":
        pos = df[df[label_col] == 1]
        self._index = {}
        for ep, g in pos.groupby(self.epitope_col):
            self._index[str(ep)] = list(zip(
                g["cdr3b"].astype(str).tolist(),
                g["cdr3a"].astype(str).tolist(),
            ))
        return self

    def _pair_distance(self, qb: str, qa: str, rb: str, ra: str) -> int:
        d = edit_distance(qb, rb)
        if self.use_alpha and qa and ra:
            d += edit_distance(qa, ra)
        return d

    def score(self, df: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(df), dtype=float)
        qb = df["cdr3b"].astype(str).tolist()
        qa = df["cdr3a"].astype(str).tolist()
        eps = df[self.epitope_col].astype(str).tolist()
        for i in range(len(df)):
            refs = self._index.get(eps[i])
            if not refs:
                scores[i] = 0.0
                continue
            dists = [self._pair_distance(qb[i], qa[i], rb, ra) for rb, ra in refs]
            dists.sort()
            knn = dists[: self.k] if len(dists) >= self.k else dists
            scores[i] = float(np.mean(np.exp(-np.asarray(knn) / self.scale)))
        return scores


__all__ = ["KmerFeaturizer", "DistanceKNNScorer", "AA"]
