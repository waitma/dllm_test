"""Does the decoding rule, not the weights, destroy the germline J motif?

``_sample_tokens`` routes the default ``gumbel_argmax`` strategy to
``stochastic_sample_from_categorical(logits, temperature=0.0, noise_scale=1.0)``,
and that helper never reads its ``temperature`` argument -- so every T4 run
sampled each position from the full per-position softmax with full-scale Gumbel
noise, and ``--temperature`` was a no-op.  A 5-residue germline motif needs five
consecutive high-probability picks, which that rule will rarely deliver even
when the model puts most of its mass there.

This script generates unconditional CDR3b from one checkpoint under several
decoding rules and reports how often the C-terminus lands on a real J motif.
If the mode-seeking rules recover the motif, the knowledge is in the weights and
the gap is a decoding choice; if none do, it is the weights.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.grammar.tcr_generation import BioSeqTcrSampler  # noqa: E402

PISTE = (PROJECT_ROOT / "data" / "ppi_task_raw" / "raw" / "piste_tcr_epitope_hla"
         / "PISTE" / "data" / "random" / "train_data.csv")


def j_motifs(top: int = 30) -> list[str]:
    cd = [s for s in pd.read_csv(PISTE)["CDR3"].dropna().astype(str) if len(s) > 6]
    return [m for m, _ in collections.Counter(s[-6:] for s in cd).most_common(top)]


def summarize(seqs: list[str], motifs: list[str]) -> dict:
    seqs = [s for s in seqs if len(s) > 6]
    if not seqs:
        return {"n": 0}
    return {
        "n": len(seqs),
        "j_motif_rate": float(np.mean([s[-6:] in motifs for s in seqs])),
        "ends_F_or_W": float(np.mean([s[-1] in "FW" for s in seqs])),
        "starts_C": float(np.mean([s.startswith("C") for s in seqs])),
        "unique_frac": float(len(set(seqs)) / len(seqs)),
        "mean_len": float(np.mean([len(s) for s in seqs])),
        "examples": seqs[:8],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-iter", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    motifs = j_motifs()
    print("reference J motifs (top 8):", motifs[:8])

    sampler = BioSeqTcrSampler(args.checkpoint, device=args.device, seed=42)
    trials = [
        ("gumbel_argmax (current default)", "gumbel_argmax", 1.0),
        ("argmax (mode-seeking)", "argmax", 1.0),
        ("vanilla T=1.0", "vanilla", 1.0),
        ("vanilla T=0.5", "vanilla", 0.5),
        ("vanilla T=0.2", "vanilla", 0.2),
    ]
    results = {"checkpoint": args.checkpoint, "n": args.n,
               "max_iter": args.max_iter, "trials": {}}
    for label, strategy, temp in trials:
        seqs = sampler.unconditional_cdr3b(
            args.n, batch_size=args.batch_size, max_iter=args.max_iter,
            temperature=temp, sampling_strategy=strategy)
        row = summarize(seqs, motifs)
        results["trials"][label] = {"strategy": strategy, "temperature": temp, **row}
        print(f"\n=== {label} ===")
        print(f"  J_motif_rate={row['j_motif_rate']*100:5.1f}%  ends_F/W={row['ends_F_or_W']*100:5.1f}%"
              f"  unique={row['unique_frac']*100:5.1f}%  mean_len={row['mean_len']:.1f}  n={row['n']}")
        print(f"  e.g. {row['examples'][:6]}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
