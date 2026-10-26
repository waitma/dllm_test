#!/usr/bin/env python
"""Run all three BioSeq TCR sampling settings in one process (single model load).

Run, for example::

    python scripts/run_bioseq_all.py --checkpoint /abs/path/to/best.pt \
        --tag ours_bioseq_step101300

Writes unified samples into the harness output tree:
  Setting B  outputs/tcr_generation_bench/setting_B/<tag>/designs.jsonl
  Setting A  outputs/tcr_generation_bench/setting_A/<tag>/samples.txt
  Setting C  outputs/tcr_generation_bench/setting_C/<tag>/full.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/project_c20250601/251105016/project/dllm_test")
BENCH = PROJECT_ROOT / "downstream" / "benchmark"
for p in (str(PROJECT_ROOT), str(BENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from downstream.grammar.tcr_generation import BioSeqTcrSampler, DEFAULT_CKPT  # noqa: E402

OUT = BENCH / "outputs" / "tcr_generation_bench"
EVAL_JSON = BENCH / "data" / "tcr_generation_bench" / "eval_conditional.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--n-uncond", type=int, default=5000)
    ap.add_argument("--n-full", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modes", default="conditional,uncond,fulllength")
    ap.add_argument("--tag", default="ours_bioseq",
                    help="checkpoint-specific output tag (default preserves legacy paths)")
    args = ap.parse_args()

    modes = set(args.modes.split(","))
    t0 = time.time()
    sampler = BioSeqTcrSampler(args.checkpoint, device=args.device, seed=args.seed)
    print(f"[load] model ready in {time.time() - t0:.1f}s", flush=True)

    if "conditional" in modes:
        eval_entries = json.loads(EVAL_JSON.read_text())
        out = OUT / "setting_B" / args.tag
        out.mkdir(parents=True, exist_ok=True)
        items = list(eval_entries.items())
        t = time.time()
        with (out / "designs.jsonl").open("w") as fh:
            for i, (pmhc, e) in enumerate(items):
                seqs = sampler.conditional_cdr3b(e["epitope"], args.k, batch_size=100, max_iter=32)
                fh.write(json.dumps({"pmhc": pmhc, "epitope": e["epitope"], "sequences": seqs}) + "\n")
                fh.flush()
                print(f"  [cond {i + 1}/{len(items)}] {pmhc}: {len(seqs)} seqs", flush=True)
        print(f"[conditional] done in {time.time() - t:.1f}s -> {out}/designs.jsonl", flush=True)

    if "uncond" in modes:
        out = OUT / "setting_A" / args.tag
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        seqs = sampler.unconditional_cdr3b(args.n_uncond, batch_size=256, max_iter=32)
        (out / "samples.txt").write_text("\n".join(seqs) + "\n")
        print(f"[uncond] {len(seqs)} CDR3b in {time.time() - t:.1f}s -> {out}/samples.txt", flush=True)

    if "fulllength" in modes:
        out = OUT / "setting_C" / args.tag
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        pairs = sampler.fulllength_pairs(args.n_full, batch_size=16, max_iter=64)
        with (out / "full.jsonl").open("w") as fh:
            for p in pairs:
                fh.write(json.dumps(p) + "\n")
        n_beta = sum(1 for p in pairs if p["tcr_beta"])
        print(f"[fulllength] {len(pairs)} pairs ({n_beta} with beta) in {time.time() - t:.1f}s "
              f"-> {out}/full.jsonl", flush=True)
        if pairs:
            print("  ex beta:", pairs[0]["tcr_beta"][:80], flush=True)

    print(f"[all] total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
