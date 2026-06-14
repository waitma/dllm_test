#!/usr/bin/env python
"""Build the P1 PPI binary-interaction dataset from STRING 90/90 positives.

The local ``string_model_org_90_90_split`` (Bernett-style gold standard) provides
only high-confidence positive pairs (STRING combined score >= 901), already
partitioned so that no test protein is >90% similar to a train protein (the key
PPI leakage control). We add balanced random negatives **within each split**:
random protein pairs from the same split's protein pool that are not known
positives. This keeps the 90/90 sequence-similarity partition intact.

Outputs: ``data/ppi/{train,valid,test}.csv`` (idA,idB,seqA,seqB,label) +
``leakage_report.json`` (cross-split protein-pair overlap, must be 0).

Note: STRING absence != true non-interaction, so random negatives are an
approximation; the 90/90 partition is what prevents the usual inflated PPI
numbers from sequence-similarity leakage.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

STRING_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi/"
                   "string_model_org_90_90_split")
OUT_DIR = BENCH / "data" / "ppi"


def read_arrow_split(split: str) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    tabs = []
    for fp in sorted(glob.glob(str(STRING_ROOT / split / "*.arrow"))):
        with pa.memory_map(fp) as src:
            try:
                t = ipc.open_stream(src).read_all()
            except Exception:
                src.seek(0)
                t = ipc.open_file(src).read_all()
        tabs.append(t)
    return pa.concat_tables(tabs).to_pandas()


def make_negatives(pos: pd.DataFrame, seqmap: dict, n_neg: int, seed: int) -> pd.DataFrame:
    """Random within-split protein pairs that are not known positives."""
    rng = np.random.default_rng(seed)
    proteins = np.array(sorted(seqmap.keys()))
    pos_set = set(zip(pos["idA"], pos["idB"])) | set(zip(pos["idB"], pos["idA"]))
    negs = []
    seen = set()
    tries = 0
    max_tries = n_neg * 50 + 1000
    while len(negs) < n_neg and tries < max_tries:
        tries += 1
        a, b = proteins[rng.integers(0, len(proteins))], proteins[rng.integers(0, len(proteins))]
        if a == b:
            continue
        if (a, b) in pos_set or (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        negs.append({"idA": a, "idB": b, "seqA": seqmap[a], "seqB": seqmap[b], "label": 0})
    return pd.DataFrame(negs)


def process_split(split: str, max_pos: int, neg_ratio: float, seed: int) -> pd.DataFrame:
    raw = read_arrow_split(split)
    raw = raw.rename(columns={"SeqA": "seqA", "SeqB": "seqB"})
    ids = raw["IDs"].str.split("|", expand=True)
    raw["idA"], raw["idB"] = ids[0], ids[1]
    if max_pos and len(raw) > max_pos:
        raw = raw.sample(n=max_pos, random_state=seed).reset_index(drop=True)
    raw["label"] = 1

    seqmap = {}
    for _, r in raw.iterrows():
        seqmap[r["idA"]] = r["seqA"]
        seqmap[r["idB"]] = r["seqB"]

    n_neg = int(round(len(raw) * neg_ratio))
    neg = make_negatives(raw[["idA", "idB"]].assign(), seqmap, n_neg, seed)
    pos = raw[["idA", "idB", "seqA", "seqB", "label"]]
    out = pd.concat([pos, neg], ignore_index=True)
    out["split"] = split
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True), set(seqmap.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train-pos", type=int, default=40000,
                    help="cap positive train pairs (full set is ~645k)")
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proteins_by_split = {}
    frames = {}
    for split, cap in [("train", args.max_train_pos), ("valid", 0), ("test", 0)]:
        print(f"[{split}] reading STRING + generating negatives ...")
        df, prots = process_split(split, cap, args.neg_ratio, args.seed)
        df = df.drop(columns=[c for c in ["proteins"] if c in df.columns])
        df.to_csv(OUT_DIR / f"{split}.csv", index=False)
        frames[split] = df
        proteins_by_split[split] = prots
        print(f"      {split}: {len(df)} pairs "
              f"(pos={int((df.label==1).sum())}, neg={int((df.label==0).sum())}), "
              f"{len(prots)} proteins")

    # leakage: protein-level overlap across splits (the 90/90 partition).
    def overlap(a, b):
        return len(proteins_by_split[a] & proteins_by_split[b])

    report = {
        "source": "STRING model_org 90/90 split (positives) + balanced random negatives",
        "params": vars(args),
        "protein_overlap_train_test": overlap("train", "test"),
        "protein_overlap_train_valid": overlap("train", "valid"),
        "protein_overlap_valid_test": overlap("valid", "test"),
        "n": {k: len(v) for k, v in frames.items()},
    }
    with (OUT_DIR / "leakage_report.json").open("w") as fh:
        json.dump(report, fh, indent=2)
    print("leakage report:", json.dumps({k: report[k] for k in report if "overlap" in k}))


if __name__ == "__main__":
    main()
