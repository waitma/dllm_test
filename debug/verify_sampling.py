#!/usr/bin/env python
"""Is the missing J motif a DECODING artifact rather than a model deficiency?

verify_jmotif.py found a 10x gap on the same checkpoint and the same masked
positions:

  teacher-forced per-position argmax -> valid top-30 J motif 34.4% of the time
  the real sampler (gumbel_argmax)   -> valid top-30 J motif  3.1% of the time

``gumbel_argmax`` calls ``stochastic_sample_from_categorical(logits,
temperature=0.0, noise_scale=1.0)``, and that helper *ignores* its temperature
argument -- it only ever adds full Gumbel noise, which is identical to sampling
at temperature 1.0. So every T4 run decoded germline-constrained positions at
full entropy, and the ``--temperature`` flag was dead.

This script re-decodes with several strategies, changing nothing else:

  S1  tail-window infill   (real sequence, only the J motif masked)
  S2  Setting-B generation (the actual benchmark: whole CDR3b from a placeholder)
      scored with the same d_edit the leaderboard uses.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _cap_threads() -> int:
    """Size the torch thread pool to the container's CPU quota, not the host's.

    lxcfs shows this container 13 CPUs in ``/proc/cpuinfo`` while
    ``sched_getaffinity`` reports all 128 host CPUs. Torch sizes its pool from
    the latter (64 intra-op threads), oversubscribing the cgroup CFS quota ~5x;
    once the quota is spent inside a period the entire cgroup is throttled,
    stalling every process in the container. 64 threads measured 8.10
    ms/matmul vs 3.04 ms at 13, so capping is also a 2.7x speedup.
    """
    try:
        with open("/proc/cpuinfo") as fh:
            quota = sum(1 for line in fh if line.startswith("processor"))
    except OSError:
        quota = 0
    n = max(1, quota or len(os.sched_getaffinity(0)))
    torch.set_num_threads(n)
    return n


_THREADS = _cap_threads()

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
BENCH = PROJECT_ROOT / "downstream" / "benchmark"
for p in (str(PROJECT_ROOT), str(BENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.metrics import tcrt5_mean_edit_distance  # noqa: E402
from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, BioSeqRecord  # noqa: E402
from downstream.grammar.common import (  # noqa: E402
    _inverse_remap_llada_tokens,
    build_eval_collator,
    load_grammar_checkpoint,
    run_grammar_generate,
)
from downstream.grammar.masks import (  # noqa: E402
    cdr3b_span_partial_mask,
    residue_positions_by_chain,
    tcr_generation_partial_mask,
)
from downstream.grammar.metrics import decode_residue_span  # noqa: E402

# The "vanilla" (temperature-controlled) strategy crashes in generate_bioseq:
# sample_from_categorical returns bf16 log-probs while output_scores is fp32, so
# masked_scatter raises. Cast here rather than editing the library mid-diagnosis
# -- it means the temperature path has never actually been run.
import dllm.pipelines.qwen3_vl_arch.sampling_bioseq as _sb  # noqa: E402

_orig_sample = _sb.sample_from_categorical


def _sample_fp32(logits=None, temperature: float = 1.0):
    tokens, scores = _orig_sample(logits=logits, temperature=temperature)
    return tokens, scores.to(torch.float32)


_sb.sample_from_categorical = _sample_fp32

PAPERS = PROJECT_ROOT / "data" / "tcr_papers_v2" / "dataset"
GB = BENCH / "data" / "tcr_generation_bench"
WIN = 5
RESERVED = "RVRAYTYSK_HLA-A*03:01"

# (label, strategy, temperature)
STRATEGIES = [
    ("gumbel_argmax (current default)", "gumbel_argmax", 1.0),
    ("argmax (greedy)", "argmax", 1.0),
    ("vanilla T=1.0", "vanilla", 1.0),
    ("vanilla T=0.5", "vanilla", 0.5),
    ("vanilla T=0.2", "vanilla", 0.2),
]

# Empirical FULL-length C..F distribution used by the shipped sampler. Kept
# as-is so S2 reproduces the benchmark run rather than a fixed version of it.
_LEN_DIST = {9: .0004, 10: .0032, 11: .0291, 12: .0727, 13: .1603, 14: .2255,
             15: .2453, 16: .1463, 17: .0671, 18: .0304, 19: .0121, 20: .0048,
             21: .0016, 22: .0007, 23: .0002}
_LV = np.array(sorted(_LEN_DIST))
_LP = np.array([_LEN_DIST[k] for k in _LV], float)
_LP /= _LP.sum()


def log(m: str = "") -> None:
    print(m, flush=True)


def head_(t: str) -> None:
    log("\n" + "=" * 78)
    log(t)
    log("=" * 78)


def train_top_motifs() -> tuple[set[str], list[str]]:
    df = pd.read_csv(PAPERS / "train.csv", low_memory=False)
    cores = [c for c in df["cdr3b"].dropna().astype(str).tolist() if len(c) >= WIN + 6]
    top = [m for m, _ in collections.Counter(c[-WIN:] for c in cores).most_common(30)]
    return set(top), top


def load_rows(n: int, seed: int = 0) -> list[dict]:
    df = pd.read_csv(PAPERS / "holdout.csv", low_memory=False)
    df = df[df["cdr3b"].notna() & df["epitope_seq"].notna()
            & df["cdr3a"].isna() & df["mhc_seq"].isna()]
    df = df[df["cdr3b"].astype(str).str.len() >= WIN + 6]
    df = df.sample(n=min(n, len(df)), random_state=seed)
    return [{"cdr3b": str(r.cdr3b).strip().upper(),
             "epitope": str(r.epitope_seq).strip().upper()} for r in df.itertuples()]


def build(rows, seqs) -> list[BioSeqRecord]:
    return [
        BioSeqRecord(
            chains=[BioSeqChain(r["epitope"], "peptide"), BioSeqChain(s, "tcr_beta")],
            task_type="tcr_epitope", source="verify_sampling")
        for r, s in zip(rows, seqs)
    ]


def collate(collator, recs, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in collator(recs).items()}


def decode_chain(model, tokenizer, token_ids, cols) -> str:
    idx = torch.tensor(cols, device=token_ids.device, dtype=torch.long)
    ids = _inverse_remap_llada_tokens(model, token_ids.index_select(0, idx).unsqueeze(0))
    return decode_residue_span(ids.squeeze(0), tokenizer, 0, ids.numel())


# --------------------------------------------------------------------------- #
# S1  tail-window infill under each decoding strategy
# --------------------------------------------------------------------------- #

def s1(model, collator, tokenizer, rows, device, top, strategy, temp,
       max_iter=32, bs=16) -> dict:
    hit = tot = exact = in_top = n = 0
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        batch = collate(collator, build(chunk, [r["cdr3b"] for r in chunk]), device)
        spans = [(len(r["cdr3b"]) - WIN, len(r["cdr3b"])) for r in chunk]
        partial = cdr3b_span_partial_mask(batch, chain_index=1, span=spans)
        tokens, _ = run_grammar_generate(model, batch, partial_mask=partial,
                                         max_iter=max_iter, sampling_strategy=strategy,
                                         temperature=temp)
        by_chain = residue_positions_by_chain(batch)
        for i, r in enumerate(chunk):
            pred = decode_chain(model, tokenizer, tokens[i], by_chain[i][1])
            if len(pred) != len(r["cdr3b"]):
                continue
            tw, pw = r["cdr3b"][-WIN:], pred[-WIN:]
            hit += sum(a == b for a, b in zip(tw, pw))
            tot += len(tw)
            exact += tw == pw
            in_top += pw in top
            n += 1
    return {"aar": hit / max(1, tot), "exact": exact / max(1, n),
            "in_top30": in_top / max(1, n), "n": n}


# --------------------------------------------------------------------------- #
# S2  the actual Setting-B benchmark under each decoding strategy
# --------------------------------------------------------------------------- #

def s2(model, collator, tokenizer, device, top, strategy, temp, eval_entries,
       pmhcs, k, rng, max_iter=32, bs=20) -> dict:
    per, all_seqs = [], []
    for pmhc in pmhcs:
        entry = eval_entries[pmhc]
        lens = [int(x) for x in rng.choice(_LV, size=k, p=_LP)]
        seqs: list[str] = []
        for s in range(0, k, bs):
            chunk = lens[s:s + bs]
            rows = [{"epitope": entry["epitope"]} for _ in chunk]
            ph = ["C" + "A" * (L - 2) + "F" for L in chunk]
            batch = collate(collator, build(rows, ph), device)
            partial = tcr_generation_partial_mask(batch, target_chain_indices=1)
            tokens, _ = run_grammar_generate(model, batch, partial_mask=partial,
                                             max_iter=max_iter,
                                             sampling_strategy=strategy,
                                             temperature=temp)
            by_chain = residue_positions_by_chain(batch)
            for i in range(len(chunk)):
                seq = decode_chain(model, tokenizer, tokens[i], by_chain[i][1])
                if seq:
                    seqs.append(seq)
        all_seqs += seqs
        # score exactly like the leaderboard: raw, and with anchors canonicalised
        raw = [s.upper() for s in seqs]
        canon = []
        for s in raw:
            x = s if s.startswith("C") else "C" + s
            x = x if x[-1] in "FW" else x + "F"
            canon.append(x)
        per.append({
            "raw": tcrt5_mean_edit_distance(raw, entry["ref_binders"]),
            "canon": tcrt5_mean_edit_distance(canon, entry["ref_binders"]),
        })
    def core(s):
        x = s[1:] if s.startswith("C") else s
        return x[:-1] if x and x[-1] in "FW" else x
    cores = [core(s) for s in all_seqs if core(s)]
    return {
        "d_edit_raw": float(np.mean([p["raw"] for p in per])),
        "d_edit_canon": float(np.mean([p["canon"] for p in per])),
        "jmotif": float(np.mean([c[-WIN:] in top for c in cores])),
        # Greedy decoding can buy d_edit by collapsing onto one consensus
        # sequence, which is worthless as a design set -- so report diversity
        # alongside it rather than letting d_edit stand alone.
        "diversity": len(set(all_seqs)) / max(1, len(all_seqs)),
        "n": len(all_seqs),
        "seqs": all_seqs,
        "ex": all_seqs[:3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-infill", type=int, default=24)
    ap.add_argument("-k", "--k", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-iter", type=int, default=32)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--skip-s1", action="store_true")
    ap.add_argument("--skip-s2", action="store_true")
    ap.add_argument("--only", default=None,
                    help="comma list of strategy labels to run, e.g. 'argmax,vanilla T=0.5'")
    ap.add_argument("--dump", default=None, help="where to write the generated designs")
    # torch defaults to 64 intra-op threads on this 128-core host, which is ~2.3x
    # slower than 16 for this shape (batch ~20, seq ~40, d=768) because the
    # per-op sync cost dominates. Two such processes reach 128 threads on 128
    # shared cores and fall off a ~16x oversubscription cliff.
    ap.add_argument("--threads", type=int, default=16, help="torch CPU threads (0 = leave default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    strategies = STRATEGIES
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        strategies = [s for s in STRATEGIES
                      if any(w in s[0] or w == s[1] for w in want)]

    log(f"threads: {_THREADS} (capped to the container CPU quota)")
    top, top_list = train_top_motifs()
    log(f"top-30 training J motifs: {top_list[:10]}")
    rows = load_rows(args.n_infill)
    base = np.mean([r["cdr3b"][-WIN:] in top for r in rows])
    log(f"held-out rows={len(rows)}   true tail in top-30 = {base:.3f} (the ceiling to beat)")

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available())
                          else "cpu")
    if device.type == "cpu":
        log("!! CUDA unavailable (driver 12.2 vs torch cu130) -> CPU")
    model, tokenizer = load_grammar_checkpoint(args.checkpoint, device=device)
    collator = build_eval_collator(model, tokenizer)
    log(f"model: {type(model).__name__}")

    if not args.skip_s1:
        head_("S1  tail-window infill: only the decoding strategy changes")
        log(f"  {'strategy':34s} {'AAR':>7s} {'exact':>7s} {'in_top30':>9s}")
        for label, strat, temp in strategies:
            r = s1(model, collator, tokenizer, rows, device, top, strat, temp,
                   args.max_iter, args.bs)
            log(f"  {label:34s} {r['aar']:7.3f} {r['exact']:7.3f} {r['in_top30']:9.3f}")

    if args.skip_s2:
        return

    head_("S2  the real Setting-B benchmark under each decoding strategy")
    eval_entries = json.loads((GB / "eval_conditional.json").read_text())
    unseen = set(json.loads((GB / "bioseq_unseen_pmhc.json").read_text()))
    pmhcs = sorted(p for p, e in eval_entries.items()
                   if e["category"] == "benchmark14" and p in unseen and p != RESERVED)
    log(f"  bioseq_unseen_common: {len(pmhcs)} pMHC, k={args.k}")
    log(f"  reference d_edit on this slice: OLGA 6.34, TCRT5 4.70, GraTCR 4.60")
    log(f"\n  {'strategy':34s} {'d_edit':>8s} {'d_edit*':>8s} {'J motif':>8s} "
        f"{'divers':>7s} {'n':>5s}")
    dump = {}
    for label, strat, temp in strategies:
        r = s2(model, collator, tokenizer, device, top, strat, temp,
               eval_entries, pmhcs, args.k, np.random.default_rng(42),
               args.max_iter, bs=min(args.bs, args.k))
        log(f"  {label:34s} {r['d_edit_raw']:8.3f} {r['d_edit_canon']:8.3f} "
            f"{r['jmotif']:8.3f} {r['diversity']:7.3f} {r['n']:5d}")
        log(f"      e.g. {r['ex']}")
        dump[label] = {k: v for k, v in r.items() if k != "seqs"}
        dump[label]["seqs"] = r["seqs"][:60]
    out = Path(args.dump or "/tmp/vs_s2_designs.json")
    out.write_text(json.dumps(dump, indent=2))
    log(f"\n  d_edit  = scored exactly as the leaderboard does (raw output)")
    log("  d_edit* = same designs with the C../F anchors canonicalised")
    log("  divers  = unique/total; greedy can win d_edit by collapsing, so read both")
    log(f"  designs -> {out}")


if __name__ == "__main__":
    main()
