#!/usr/bin/env python
"""Extract CDR3-beta repertoires for the T4 generation task (OTS paired TCR).

Pulls CDR3-beta (the ANARCI 'B' chain) from the OTS paired-clean final split:
- train (capped) -> the corpus a generator is trained on + the novelty reference.
- holdout        -> the held-out distribution to match (never seen at train).

Outputs: ``data/tcr_generation/{train_cdr3b.txt, holdout_cdr3b.txt}``.
Holdout CDR3b that also appear in the (capped) train set are dropped so novelty
and distribution metrics are not contaminated.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cap", type=int, default=200000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"reading OTS train (first {args.train_cap} rows) ...")
    train = pd.read_csv(OTS / "train.csv", usecols=USECOLS, nrows=args.train_cap)
    train_beta = extract_beta(train)
    train_set = set(train_beta)

    print("reading OTS holdout ...")
    hold = pd.read_csv(OTS / "holdout.csv", usecols=USECOLS)
    hold_beta = [s for s in extract_beta(hold) if s not in train_set]

    (OUT / "train_cdr3b.txt").write_text("\n".join(train_beta) + "\n")
    (OUT / "holdout_cdr3b.txt").write_text("\n".join(hold_beta) + "\n")
    print(f"train CDR3b: {len(train_beta)} ({len(train_set)} unique)")
    print(f"holdout CDR3b (novel vs train): {len(hold_beta)}")
    print(f"-> {OUT}/")


if __name__ == "__main__":
    main()
