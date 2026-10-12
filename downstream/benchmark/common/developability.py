"""Sequence-only nanobody (VHH) developability scorers.

Implements the six sequence-based developability signals used as rewards by
**Aiki-GeNano** (Sharma Meda et al., bioRxiv 2026.04.28.721526) so that
generated / infilled VHH sequences can be scored *without any wet-lab or
structural input*:

  1. FR2 hydrophobicity          -- Kyte-Doolittle mean over the FR2 interface.
  2. Hydrophobic-patch coverage  -- fraction of residues inside a hydrophobic run.
  3. Chemical-liability motifs    -- deamidation / isomerization / glycosylation /
                                     oxidation / unpaired-Cys / fragmentation.
  4. Wilkinson-Harrison P(soluble)-- recombinant E. coli soluble-expression proxy.
  5. VHH hallmark residues        -- the FR2 tetrad (Kabat 37/44/45/47) VHH signature.
  6. Scaffold integrity           -- conserved di-Cys + plausible length + FR4 motif.

All scorers are pure functions of the amino-acid string. Region boundaries (FR2 /
CDRs) are found by anchoring on the invariant FR2 tryptophan (Kabat 36, the
``W`` in the ``W.RQ`` / ``W..RQ`` motif); when that anchor is absent we fall
back to composition-only signals and flag ``fr2_found = False`` so downstream
aggregation stays honest.

References
----------
- Aiki-GeNano: Multi-Stage Preference Optimization for Generative Design of
  Developable Nanobodies. bioRxiv 2026.04.28.721526.
- Wilkinson & Harrison (1991) Bio/Technology 9:443-448; Davis et al. (1999)
  modification -- the exact constants (L1=15.43, L2=-29.56, CV'=1.71,
  A/B/C = 0.4934/0.276/-0.0392) match the canonical reference implementation.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

AA = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydropathy (higher = more hydrophobic).
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Chemical-liability motifs (regex over the raw sequence). Deamidation (N-x) and
# isomerization (D-x) hot-spots follow the standard therapeutic-antibody CQA
# lists; N-glyc is the canonical sequon; oxidation flags Met/Trp; DP fragments.
LIABILITY_MOTIFS = {
    "deamidation": r"N[GSTNH]",        # NG/NS most labile, plus NT/NN/NH
    "isomerization": r"D[GSTDH]",       # DG/DS/DD/DT/DH aspartate isomerization
    "fragmentation": r"DP",             # acid-labile Asp-Pro bond
    "nglyc_sequon": r"N[^P][ST]",       # N-X-S/T (X != P) N-glycosylation
    "met_oxidation": r"M",              # methionine oxidation
    "trp_oxidation": r"W",              # tryptophan oxidation
}

# VHH FR2 hallmark tetrad (Kabat 37/44/45/47). Camelid VHH replaces the VH-VL
# interface residues (V37/G44/L45/W47 in conventional VH) with hydrophilic ones.
VHH_HALLMARK = {
    37: set("FYL"),      # F37 (sometimes Y/L)
    44: set("EQ"),       # E44 (sometimes Q)
    45: set("RCLP"),     # R45 (sometimes C/L/P)
    47: set("GLFS"),     # G/L/F/S47
}

_FR2_ANCHOR = re.compile(r"W.?.?RQ")  # invariant Trp then ...RQ (W36..Q39)


def _clean(seq: str) -> str:
    return "".join(c for c in str(seq).upper() if c.isalpha())


def find_fr2_start(seq: str) -> int:
    """Index of the invariant FR2 tryptophan (Kabat 36), or -1 if not found.

    Anchors on the ``W.?.?RQ`` motif (``WVRQ`` / ``WFRQ`` / ``WLRQ`` ...). Falls
    back to the first ``W`` occurring in the 30-45 residue window if the motif is
    absent (heavily mutated CDR1 can shift it)."""
    m = _FR2_ANCHOR.search(seq)
    if m:
        return m.start()
    for i in range(30, min(len(seq), 46)):
        if seq[i] == "W":
            return i
    return -1


def fr2_hydrophobicity(seq: str, fr2_start: int | None = None) -> float:
    """Mean Kyte-Doolittle hydropathy over the FR2 window (Kabat 36-49).

    Lower = more hydrophilic = better VHH developability (no VL to shield the
    former interface). Returns NaN if FR2 cannot be located."""
    fr2 = find_fr2_start(seq) if fr2_start is None else fr2_start
    if fr2 < 0:
        return float("nan")
    window = seq[fr2:fr2 + 14]
    vals = [KD[c] for c in window if c in KD]
    return float(np.mean(vals)) if vals else float("nan")


def hydrophobic_patch_coverage(seq: str, window: int = 5, thresh: float = 1.0) -> float:
    """Fraction of residues that lie inside a hydrophobic patch.

    A position is 'in a patch' if the mean KD hydropathy of the ``window``
    centred on it exceeds ``thresh``. Higher = more exposed hydrophobicity =
    higher aggregation risk (worse)."""
    L = len(seq)
    if L == 0:
        return float("nan")
    kd = np.array([KD.get(c, 0.0) for c in seq], dtype=float)
    half = window // 2
    covered = np.zeros(L, dtype=bool)
    for i in range(L):
        lo, hi = max(0, i - half), min(L, i + half + 1)
        if kd[lo:hi].mean() > thresh:
            covered[lo:hi] = True
    return float(covered.mean())


def liability_motifs(seq: str) -> dict:
    """Count each chemical-liability motif + total, and the unpaired-Cys flag."""
    counts = {name: len(re.findall(pat, seq)) for name, pat in LIABILITY_MOTIFS.items()}
    n_cys = seq.count("C")
    counts["unpaired_cys"] = int(n_cys % 2 == 1)  # canonical VHH has an even (2) Cys
    counts["total"] = int(sum(v for k, v in counts.items() if k != "total"))
    return counts


def wilkinson_harrison_solubility(seq: str) -> float:
    """Wilkinson-Harrison (Davis 1999 modification) P(soluble) in [0, 1].

    CV = 15.43*(N+G+P+S)/n - 29.56*|(R+K-D-E)/n - 0.03|;  the probability is
    ``0.4934 + 0.276*|CV-1.71| - 0.0392*|CV-1.71|^2`` (clamped to [0,1]),
    matching the canonical reference implementation."""
    n = len(seq)
    if n == 0:
        return float("nan")
    c = {a: seq.count(a) for a in "NGPSRKDE"}
    cv = (15.43 * ((c["N"] + c["G"] + c["P"] + c["S"]) / n)
          - 29.56 * abs((c["R"] + c["K"] - c["D"] - c["E"]) / n - 0.03))
    cvn = abs(cv - 1.71)
    prob = 0.4934 + 0.276 * cvn - 0.0392 * (cvn ** 2)
    return float(min(1.0, max(0.0, prob)))


def vhh_hallmark_score(seq: str, fr2_start: int | None = None) -> float:
    """Fraction of the FR2 hallmark tetrad (Kabat 37/44/45/47) matching VHH.

    Positions are taken relative to the invariant FR2 Trp (Kabat 36). Returns
    NaN if FR2 cannot be located."""
    fr2 = find_fr2_start(seq) if fr2_start is None else fr2_start
    if fr2 < 0:
        return float("nan")
    offsets = {37: 1, 44: 8, 45: 9, 47: 11}
    hits, tot = 0, 0
    for kabat, off in offsets.items():
        idx = fr2 + off
        if idx < len(seq):
            tot += 1
            if seq[idx] in VHH_HALLMARK[kabat]:
                hits += 1
    return float(hits / tot) if tot else float("nan")


def scaffold_integrity(seq: str) -> float:
    """0-1 mean of three scaffold sanity checks.

    (a) exactly two cysteines (the canonical Cys22-Cys92 disulfide),
    (b) length in the plausible VHH range [100, 140],
    (c) an FR4 ``WG.G`` J-motif in the C-terminal 20 residues."""
    checks = [
        seq.count("C") == 2,
        100 <= len(seq) <= 140,
        bool(re.search(r"WG.G", seq[-20:])) if len(seq) >= 20 else False,
    ]
    return float(np.mean([1.0 if c else 0.0 for c in checks]))


def developability_metrics(seq: str) -> dict:
    """Full per-sequence developability profile (the six Aiki-GeNano signals)."""
    seq = _clean(seq)
    L = max(1, len(seq))
    fr2 = find_fr2_start(seq)
    lm = liability_motifs(seq)
    return {
        "length": len(seq),
        "fr2_found": fr2 >= 0,
        "fr2_hydrophobicity": fr2_hydrophobicity(seq, fr2),
        "hydrophobic_patch_coverage": hydrophobic_patch_coverage(seq),
        "liability_total": int(lm["total"]),
        "liability_rate": float(lm["total"] / L),
        "n_deamidation": int(lm["deamidation"]),
        "n_isomerization": int(lm["isomerization"]),
        "n_nglyc_sequon": int(lm["nglyc_sequon"]),
        "n_met_oxidation": int(lm["met_oxidation"]),
        "unpaired_cys": int(lm["unpaired_cys"]),
        "wh_solubility": wilkinson_harrison_solubility(seq),
        "vhh_hallmark": vhh_hallmark_score(seq, fr2),
        "scaffold_integrity": scaffold_integrity(seq),
    }


def _nanmean(vals):
    arr = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))],
                   dtype=float)
    return float(arr.mean()) if len(arr) else float("nan")


def score_sequences(seqs: Sequence[str]) -> dict:
    """Aggregate developability over a set of VHH sequences (macro means).

    Returns population means for each signal plus ``n`` and ``fr2_found_rate``
    so a caller can judge how many sequences had a locatable FR2 anchor.
    """
    profiles = [developability_metrics(s) for s in seqs]
    if not profiles:
        return {"n": 0}
    keys = ["fr2_hydrophobicity", "hydrophobic_patch_coverage", "liability_total",
            "liability_rate", "n_deamidation", "n_isomerization", "n_nglyc_sequon",
            "n_met_oxidation", "unpaired_cys", "wh_solubility", "vhh_hallmark",
            "scaffold_integrity", "length"]
    out = {f"mean_{k}": _nanmean([p[k] for p in profiles]) for k in keys}
    out["n"] = len(profiles)
    out["fr2_found_rate"] = float(np.mean([1.0 if p["fr2_found"] else 0.0 for p in profiles]))
    return out


__all__ = [
    "developability_metrics", "score_sequences", "find_fr2_start",
    "fr2_hydrophobicity", "hydrophobic_patch_coverage", "liability_motifs",
    "wilkinson_harrison_solubility", "vhh_hallmark_score", "scaffold_integrity",
    "KD", "LIABILITY_MOTIFS", "VHH_HALLMARK",
]
