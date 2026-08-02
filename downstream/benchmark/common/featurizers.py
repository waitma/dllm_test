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
- ``AtchleyFeaturizer`` / ``Blosum62Featurizer`` / ``OneHotFeaturizer`` -- the
  three canonical *handcrafted* per-residue encodings that TCR-embedding
  benchmarks (e.g. Feng et al., Brief. Bioinform. 2025) use as training-free
  references. Each maps a sequence to a fixed vector by pooling per-residue
  descriptors, so they slot into the same embed->cluster pipeline as the PLMs.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd

from .leakage import edit_distance

AA = "ACDEFGHIKLMNPQRSTVWY"

# Atchley et al. (2005) PNAS 102(18):6395-6400, Table 2 -- the five factor
# scores per amino acid (F1 polarity, F2 secondary-structure propensity, F3
# molecular size, F4 codon composition, F5 electrostatic charge). Transcribed
# verbatim from the published table.
ATCHLEY_FACTORS = {
    "A": (-0.591, -1.302, -0.733,  1.570, -0.146),
    "C": (-1.343,  0.465, -0.862, -1.020, -0.255),
    "D": ( 1.050,  0.302, -3.656, -0.259, -3.242),
    "E": ( 1.357, -1.453,  1.477,  0.113, -0.837),
    "F": (-1.006, -0.590,  1.891, -0.397,  0.412),
    "G": (-0.384,  1.652,  1.330,  1.045,  2.064),
    "H": ( 0.336, -0.417, -1.673, -1.474, -0.078),
    "I": (-1.239, -0.547,  2.131,  0.393,  0.816),
    "K": ( 1.831, -0.561,  0.533, -0.277,  1.648),
    "L": (-1.019, -0.987, -1.505,  1.266, -0.912),
    "M": (-0.663, -1.524,  2.219, -1.005,  1.212),
    "N": ( 0.945,  0.828,  1.299, -0.169,  0.933),
    "P": ( 0.189,  2.081, -1.628,  0.421, -1.392),
    "Q": ( 0.931, -0.179, -3.005, -0.503, -1.853),
    "R": ( 1.538, -0.055,  1.502,  0.440,  2.897),
    "S": (-0.228,  1.399, -4.760,  0.670, -2.647),
    "T": (-0.032,  0.326,  2.213,  0.908,  1.313),
    "V": (-1.337, -0.279, -0.544,  1.242, -1.262),
    "W": (-0.595,  0.009,  0.672, -2.128, -0.184),
    "Y": ( 0.260,  0.830,  3.097, -0.838,  1.512),
}


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


class AtchleyFeaturizer:
    """Atchley five-factor per-residue encoding, mean+std pooled (10-dim).

    Each residue maps to the five Atchley factor scores; a variable-length CDR3
    is summarised by the per-position mean and standard deviation of its factor
    vectors, giving a fixed 10-dim descriptor (5 mean + 5 std). Unknown residues
    are skipped; an empty/all-unknown sequence maps to the zero vector.
    """

    def __init__(self):
        self._tab = {aa: np.asarray(v, dtype=np.float32)
                     for aa, v in ATCHLEY_FACTORS.items()}
        self.dim = 10
        self.name = "atchley"

    def transform(self, seqs: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(seqs), self.dim), dtype=np.float32)
        for i, s in enumerate(seqs):
            rows = [self._tab[c] for c in str(s) if c in self._tab]
            if not rows:
                continue
            m = np.stack(rows)
            out[i, :5] = m.mean(axis=0)
            out[i, 5:] = m.std(axis=0)
        return out


class Blosum62Featurizer:
    """BLOSUM62 substitution-profile encoding, mean pooled (20-dim).

    Each residue maps to its BLOSUM62 row over the 20 standard amino acids; the
    sequence descriptor is the per-position mean of those rows. This is the
    classic substitution-matrix embedding used as a handcrafted reference in TCR
    embedding benchmarks. BLOSUM62 is loaded from Biopython's bundled matrix.
    """

    def __init__(self, alphabet: str = AA):
        from Bio.Align import substitution_matrices
        mat = substitution_matrices.load("BLOSUM62")
        self.alphabet = alphabet
        self.dim = len(alphabet)
        self._row = {}
        for aa in alphabet:
            self._row[aa] = np.asarray(
                [float(mat[aa, b]) for b in alphabet], dtype=np.float32)
        self.name = "blosum62"

    def transform(self, seqs: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(seqs), self.dim), dtype=np.float32)
        for i, s in enumerate(seqs):
            rows = [self._row[c] for c in str(s) if c in self._row]
            if not rows:
                continue
            out[i] = np.stack(rows).mean(axis=0)
        return out


class OneHotFeaturizer:
    """Mean-pooled one-hot encoding == amino-acid composition (20-dim).

    Each residue is a 20-dim one-hot vector; the sequence descriptor is their
    per-position mean, i.e. the amino-acid frequency profile. It is the simplest
    handcrafted reference (identity substitution profile) and complements the
    Atchley/BLOSUM encodings.
    """

    def __init__(self, alphabet: str = AA):
        self.alphabet = alphabet
        self._idx = {aa: i for i, aa in enumerate(alphabet)}
        self.dim = len(alphabet)
        self.name = "onehot"

    def transform(self, seqs: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(seqs), self.dim), dtype=np.float32)
        for i, s in enumerate(seqs):
            idxs = [self._idx[c] for c in str(s) if c in self._idx]
            if not idxs:
                continue
            counts = np.bincount(idxs, minlength=self.dim).astype(np.float32)
            out[i] = counts / counts.sum()
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


__all__ = ["KmerFeaturizer", "AtchleyFeaturizer", "Blosum62Featurizer",
           "OneHotFeaturizer", "DistanceKNNScorer", "AA", "ATCHLEY_FACTORS"]
