"""Near-duplicate audit for the T1 retrained protocol (audit 2026-08-30).

Exact (CDR3B, Epitope) overlap between the official AS train folds and their
test sets is already known to be zero. That alone does not establish the splits
are clean: a test pair whose CDR3B differs from a training pair by a single
residue is, for a binding model, effectively the same example.

This script measures that. A pair counts as a near-duplicate when the epitope
matches exactly and the CDR3B is within Levenshtein distance ``--max-dist`` of
some training CDR3B for that same epitope. Restricting to exact-epitope groups
is what makes the O(n*m) edit-distance sweep tractable and is also the only
form that matters here: a different epitope is a different task, so a similar
CDR3B against another epitope is not leakage.

Run: python scripts/t1_retrained_near_duplicate.py
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from tcr_binding.retrained_protocol import RETRAIN_DATA_ZIP  # noqa: E402

# The archive stores these directory names in a legacy encoding; matching on the
# mojibake literal is what ``resolve_test_member`` already does.
SEEN = "retrain/CDR3ª┬_only_seen/AS/"
UNSEEN = "retrain/CDR3ª┬_only_unseen/AS/"


def _near_dup_fraction(test: pd.DataFrame, train: pd.DataFrame,
                       max_dist: int) -> dict:
    """Fraction of test pairs within ``max_dist`` of a same-epitope train pair."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    by_ep = defaultdict(list)
    for cdr, ep in zip(train["CDR3B"], train["Epitope"]):
        by_ep[ep].append(cdr)
    for ep in by_ep:
        by_ep[ep] = sorted(set(by_ep[ep]))

    test_pairs = sorted(set(zip(test["CDR3B"], test["Epitope"])))
    hit = 0
    no_epitope = 0
    per_ep_hits = defaultdict(int)
    grouped = defaultdict(list)
    for cdr, ep in test_pairs:
        grouped[ep].append(cdr)

    for ep, cdrs in grouped.items():
        refs = by_ep.get(ep)
        if not refs:
            no_epitope += len(cdrs)
            continue
        # score_cutoff lets rapidfuzz abandon a comparison early; anything above
        # the cutoff comes back as max_dist+1, so a row min <= max_dist is a hit.
        d = cdist(cdrs, refs, scorer=Levenshtein.distance,
                  score_cutoff=max_dist, workers=-1)
        n = int((d.min(axis=1) <= max_dist).sum())
        hit += n
        if n:
            per_ep_hits[ep] = n

    return {
        "n_test_unique_pairs": len(test_pairs),
        "n_near_duplicate": hit,
        "frac_near_duplicate": hit / len(test_pairs) if test_pairs else 0.0,
        "n_test_pairs_with_unseen_epitope": no_epitope,
        "top_epitopes_by_hits": dict(sorted(per_ep_hits.items(),
                                            key=lambda kv: -kv[1])[:5]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dist", type=int, default=1)
    ap.add_argument("--out", default=str(BENCH / "outputs" / "t1_retrained_near_duplicate.json"))
    args = ap.parse_args()

    z = ZipFile(RETRAIN_DATA_ZIP)

    def rd(member: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(z.read(member)))

    out = {
        "purpose": ("near-duplicate (same epitope, CDR3B within Levenshtein "
                    f"<= {args.max_dist}) overlap between official AS train folds "
                    "and their test sets"),
        "max_dist": args.max_dist,
        "archive": str(RETRAIN_DATA_ZIP),
        "seen_test_by_fold": {},
    }

    train_union = []
    for fold in range(1, 6):
        train = rd(SEEN + f"{fold}_1_1train.csv")
        test = rd(SEEN + f"{fold}_1_1test.csv")
        train_union.append(train)
        r = _near_dup_fraction(test, train, args.max_dist)
        out["seen_test_by_fold"][fold] = r
        print(f"fold{fold} seen_test: {r['n_near_duplicate']}/{r['n_test_unique_pairs']} "
              f"= {r['frac_near_duplicate']:.2%} near-duplicate")

    union = pd.concat(train_union, ignore_index=True).drop_duplicates()
    for tag, member in (("seen_independent", SEEN + "1_1_1independent_test.csv"),
                        ("unseen_independent", UNSEEN + "1_1_1independent_test.csv")):
        r = _near_dup_fraction(rd(member), union, args.max_dist)
        out[tag] = r
        print(f"{tag}: {r['n_near_duplicate']}/{r['n_test_unique_pairs']} "
              f"= {r['frac_near_duplicate']:.2%} near-duplicate "
              f"(pairs whose epitope is absent from train: "
              f"{r['n_test_pairs_with_unseen_epitope']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
