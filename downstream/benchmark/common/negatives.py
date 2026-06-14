"""Reference-TCR negative sampling for TCR-epitope binding.

Random shuffled negatives overstate generalization, so IRBench follows the
IMMREP convention: for each epitope, build negatives by pairing it with TCRs
drawn from a reference pool (TCRs binding *other* epitopes), keeping only TCRs
whose CDR3 is far (Levenshtein > ``dist_threshold``) from that epitope's own
positives. This avoids labelling a near-identical binder as a negative.

The function is deterministic given ``seed`` and never reuses a positive
clonotype as a negative for the same epitope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .leakage import passes_distance


def generate_reference_negatives(
    positives: pd.DataFrame,
    ratio: int = 5,
    dist_threshold: int = 3,
    dist_col: str = "cdr3b",
    epitope_col: str = "peptide",
    seed: int = 0,
    max_tries_factor: int = 50,
) -> pd.DataFrame:
    """Return a negatives frame (label=0) with ~``ratio`` negatives per positive.

    Each negative copies a reference TCR row (all chain columns) but swaps in a
    foreign epitope. Distance filtering uses ``dist_col`` (CDR3 beta by default).
    """
    rng = np.random.default_rng(seed)
    pos = positives.reset_index(drop=True)
    all_eps = pos[epitope_col].unique().tolist()

    # Per-epitope positive CDR3 sets for the distance guard.
    pos_cdr3_by_ep = {
        ep: set(g[dist_col].astype(str)) for ep, g in pos.groupby(epitope_col)
    }
    pos_clono_by_ep = {
        ep: set((g["cdr3a"].astype(str) + "|" + g["cdr3b"].astype(str)))
        for ep, g in pos.groupby(epitope_col)
    }

    tcr_rows = pos.to_dict("records")
    n_rows = len(tcr_rows)
    neg_records = []

    for ep in all_eps:
        n_pos = int((pos[epitope_col] == ep).sum())
        n_target = n_pos * ratio
        ep_pos_cdr3 = pos_cdr3_by_ep[ep]
        ep_pos_clono = pos_clono_by_ep[ep]
        made = 0
        tries = 0
        max_tries = n_target * max_tries_factor + 100
        seen_clono = set()
        while made < n_target and tries < max_tries:
            tries += 1
            idx = int(rng.integers(0, n_rows))
            row = tcr_rows[idx]
            if str(row[epitope_col]) == str(ep):
                continue
            clono = f"{row['cdr3a']}|{row['cdr3b']}"
            if clono in ep_pos_clono or clono in seen_clono:
                continue
            if not passes_distance(str(row[dist_col]), ep_pos_cdr3, dist_threshold):
                continue
            seen_clono.add(clono)
            neg = dict(row)
            neg[epitope_col] = ep
            neg["label"] = 0
            neg["neg_source_epitope"] = row[epitope_col]
            neg_records.append(neg)
            made += 1

    neg_df = pd.DataFrame(neg_records)
    return neg_df


def build_binding_dataset(
    positives: pd.DataFrame,
    ratio: int = 5,
    dist_threshold: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Concatenate positives (label=1) and generated negatives (label=0)."""
    pos = positives.copy()
    pos["label"] = 1
    neg = generate_reference_negatives(
        positives, ratio=ratio, dist_threshold=dist_threshold, seed=seed
    )
    out = pd.concat([pos, neg], ignore_index=True, sort=False)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


__all__ = ["generate_reference_negatives", "build_binding_dataset"]
