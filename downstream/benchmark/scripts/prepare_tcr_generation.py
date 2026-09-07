#!/usr/bin/env python
"""Extract CDR3-beta repertoires for the T4 generation task (OTS paired TCR).

Pulls CDR3-beta (the ANARCI 'B' chain) from the OTS paired-clean final split:
- train (capped) -> the corpus a generator is trained on + the novelty reference.
- holdout        -> the held-out distribution to match (never seen at train).

Outputs: ``data/tcr_generation/{train_cdr3b.txt, holdout_cdr3b.txt}``.
Holdout CDR3b that also appear in the (capped) train set are dropped so novelty
and distribution metrics are not contaminated.

CAUTION -- ``--train-cap`` is a correctness knob, not a speed knob. It bounds
BOTH the novelty reference and the set the holdout is deduplicated against. At
the historical default of 200,000 it covered only 9.5% of the 2,102,715-row OTS
train split, which left 1,893 of 9,767 holdout CDR3b (19.4%) sitting inside the
training data, and let any generated sequence from train rows 200k-2.1M count as
"novel". Pass ``--train-cap 0`` for the full split. The 2026-08-28 audit
(``scripts/data/dedup/audit_downstream_leakage.py``) is what surfaced this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
OTS = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final")
OUT = BENCH / "data" / "tcr_generation"

USECOLS = ["chain1_cdr3", "chain2_cdr3", "chain1_anarci_type", "chain2_anarci_type"]
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def extract_beta(df: pd.DataFrame) -> list[str]:
    out = []
    for c1, c2, t1, t2 in zip(df["chain1_cdr3"], df["chain2_cdr3"],
                              df["chain1_anarci_type"], df["chain2_anarci_type"]):
        seq = c1 if str(t1).upper() == "B" else (c2 if str(t2).upper() == "B" else None)
        if not seq:
            continue
        seq = str(seq).strip().upper()
        if seq and set(seq) <= VALID_AA and 5 <= len(seq) <= 30:
            out.append(seq)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-cap", type=int, default=200000,
                    help="0 = use the whole train split (correct); the 200000 "
                         "default is kept only so re-running does not silently "
                         "change the reference the current numbers were scored on")
    ap.add_argument("--out-dir", default=str(OUT),
                    help="write elsewhere to measure the impact of a cap change "
                         "without overwriting the live benchmark reference")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    nrows = args.train_cap if args.train_cap > 0 else None
    print(f"reading OTS train ({'ALL rows' if nrows is None else f'first {nrows} rows'}) ...")
    train = pd.read_csv(OTS / "train.csv", usecols=USECOLS, nrows=nrows)
    train_beta = extract_beta(train)
    train_set = set(train_beta)

    print("reading OTS holdout ...")
    hold = pd.read_csv(OTS / "holdout.csv", usecols=USECOLS)
    hold_all = extract_beta(hold)
    hold_beta = [s for s in hold_all if s not in train_set]

    (out / "train_cdr3b.txt").write_text("\n".join(train_beta) + "\n")
    (out / "holdout_cdr3b.txt").write_text("\n".join(hold_beta) + "\n")
    print(f"train CDR3b: {len(train_beta)} ({len(train_set)} unique)")
    print(f"holdout CDR3b: {len(hold_all)} extracted -> {len(hold_beta)} kept "
          f"({len(hold_all) - len(hold_beta)} dropped as seen in train)")
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
