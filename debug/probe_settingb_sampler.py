"""End-to-end: how much T4 Setting-B ``d_edit`` is the decoding rule worth?

``probe_sampling_strategy.py`` showed the default ``gumbel_argmax`` rule emits a
real germline J motif 1.6% of the time while plain categorical sampling at the
same nominal temperature 1.0 -- and the same 100% uniqueness -- reaches 15.6%.
This script carries that through to the benchmark number: it regenerates
epitope-conditioned designs for the cross-model-comparable slice and scores
``d_edit`` exactly as ``tcr_generation_bench`` does.

``gumbel_argmax`` is included as a control: without reproducing the stored
result the comparison would not be attributable to the decoding rule.

``d_edit`` is a mean over generated sequences, so it is k-independent and a
smaller ``-k`` stays comparable to the recorded runs (recall/precision are not,
and are therefore not reported here).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
BENCH = PROJECT_ROOT / "downstream" / "benchmark"
for p in (str(PROJECT_ROOT), str(BENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.metrics import tcrt5_mean_edit_distance  # noqa: E402
from downstream.grammar.tcr_generation import BioSeqTcrSampler  # noqa: E402

DATA = BENCH / "data" / "tcr_generation_bench"
RESERVED = "RVRAYTYSK_HLA-A*03:01"


def common_unseen(eval_all: dict) -> list[str]:
    unseen = set(json.loads((DATA / "bioseq_unseen_pmhc.json").read_text()))
    return sorted(p for p, e in eval_all.items()
                  if e.get("category") == "benchmark14" and p in unseen and p != RESERVED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("-k", "--k", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", default="gumbel_argmax:1.0,vanilla:1.0,vanilla:0.5",
                    help="comma list of strategy:temperature")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eval_all = json.loads((DATA / "eval_conditional.json").read_text())
    slice_pmhc = common_unseen(eval_all)
    print(f"slice = bioseq_unseen_common ({len(slice_pmhc)} pMHC), k={args.k}, "
          f"batch_size={args.batch_size}")

    sampler = BioSeqTcrSampler(args.checkpoint, device=args.device, seed=args.seed)
    trials = []
    for spec in args.trials.split(","):
        strategy, _, temp = spec.partition(":")
        temp = float(temp or 1.0)
        trials.append((f"{strategy} T={temp}", strategy, temp))
    results = {"checkpoint": args.checkpoint, "k": args.k,
               "slice": slice_pmhc, "trials": {}}

    for label, strategy, temp in trials:
        per_pmhc, designs = {}, {}
        for pmhc in slice_pmhc:
            seqs = sampler.conditional_cdr3b(
                eval_all[pmhc]["epitope"], args.k, batch_size=args.batch_size,
                max_iter=args.max_iter, temperature=temp, sampling_strategy=strategy)
            designs[pmhc] = seqs
            refs = eval_all[pmhc]["ref_binders"]
            per_pmhc[pmhc] = tcrt5_mean_edit_distance([s.upper() for s in seqs], refs)
            print(f"  [{label}] {pmhc:32s} d_edit={per_pmhc[pmhc]:.3f} n={len(seqs)}",
                  flush=True)
        pooled = [s for v in designs.values() for s in v]
        row = {"strategy": strategy, "temperature": temp,
               "d_edit_macro": float(np.mean(list(per_pmhc.values()))),
               "per_pmhc": per_pmhc,
               "unique_frac": float(len(set(pooled)) / max(1, len(pooled))),
               "examples": pooled[:8]}
        results["trials"][label] = row
        print(f"=== {label}: d_edit={row['d_edit_macro']:.3f} "
              f"unique={row['unique_frac']*100:.1f}% ===", flush=True)

    print("\n--- summary (bioseq_unseen_common d_edit, lower is better) ---")
    for label, row in results["trials"].items():
        print(f"  {label:42s} {row['d_edit_macro']:.3f}  (unique {row['unique_frac']*100:.1f}%)")
    print("  reference points: OLGA 6.343 | TCRdiff 5.157 | TCRdesign 4.857 | "
          "TCRT5 4.696 | GraTCR 4.605")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
