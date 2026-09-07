"""Setting-B generation grid: decoding strategy x iterations x length prior.

Three things are varied, nothing else:

  strategy   gumbel_argmax (shipped default) / argmax / vanilla at a temperature.
             ``gumbel_argmax`` ignores its temperature argument and always adds
             full Gumbel noise, so it is temperature-1.0 sampling with a
             noise-driven commit order.
  iterations the denoiser budget. The repo's existing sweep only moved this under
             gumbel_argmax, where it was flat (held20 6.7620/6.7435/6.7785 at
             8/32/64); whether a mode-seeking rule responds differently is
             untested.
  length     ``_CDR3B_LEN_DIST`` in tcr_generation.py is commented as the FULL
             C..F length distribution (mean 14.56) but the v3 models emit
             anchor-free cores, so it is applied as a CORE length -- about 2
             residues longer than the 12.52 the training corpus actually has,
             and longer than the 12.42 of the eval references. ``--length core``
             swaps in the empirical core distribution measured from the training
             data so that bug can be priced.

Scored with the leaderboard's own d_edit on bioseq_unseen_common, plus the
composition/diversity columns d_edit cannot see.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _cap_threads() -> int:
    try:
        with open("/proc/cpuinfo") as fh:
            quota = sum(1 for line in fh if line.startswith("processor"))
    except OSError:
        quota = 0
    n = max(1, quota or len(os.sched_getaffinity(0)))
    torch.set_num_threads(n)
    return n


_THREADS = _cap_threads()

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
BENCH = ROOT / "downstream" / "benchmark"
for p in (str(ROOT), str(BENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.metrics import tcrt5_mean_edit_distance  # noqa: E402
import dllm.pipelines.qwen3_vl_arch.sampling_bioseq as _sb  # noqa: E402
from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, BioSeqRecord  # noqa: E402
from downstream.grammar.common import (  # noqa: E402
    _inverse_remap_llada_tokens,
    build_eval_collator,
    load_grammar_checkpoint,
    run_grammar_generate,
)
from downstream.grammar.masks import (  # noqa: E402
    residue_positions_by_chain,
    tcr_generation_partial_mask,
)
from downstream.grammar.metrics import decode_residue_span  # noqa: E402

# "vanilla" crashes in generate_bioseq: sample_from_categorical returns bf16
# log-probs into an fp32 buffer. Cast so the temperature path is runnable.
_orig = _sb.sample_from_categorical
_sb.sample_from_categorical = lambda logits=None, temperature=1.0: (
    lambda t, s: (t, s.to(torch.float32)))(*_orig(logits=logits, temperature=temperature))

GB = BENCH / "data" / "tcr_generation_bench"
PAPERS = ROOT / "data" / "tcr_papers_v2" / "dataset"
AA = "ACDEFGHIKLMNPQRSTVWY"
WIN = 5
RESERVED = "RVRAYTYSK_HLA-A*03:01"

# the shipped prior: commented as full C..F lengths, applied as core lengths
FULL_LEN_DIST = {9: .0004, 10: .0032, 11: .0291, 12: .0727, 13: .1603, 14: .2255,
                 15: .2453, 16: .1463, 17: .0671, 18: .0304, 19: .0121, 20: .0048,
                 21: .0016, 22: .0007, 23: .0002}


def core(s: str) -> str:
    s = "".join(str(s or "").split()).upper()
    if len(s) >= 5:
        if s[0] == "C":
            s = s[1:]
        if s and s[-1] in "FW":
            s = s[:-1]
    return s


def dist_from(counter: dict) -> tuple[np.ndarray, np.ndarray]:
    ks = np.array(sorted(counter))
    ps = np.array([counter[k] for k in ks], dtype=float)
    return ks, ps / ps.sum()


def core_len_dist() -> tuple[np.ndarray, np.ndarray]:
    """Empirical CORE length distribution from the training corpus."""
    d = pd.read_csv(PAPERS / "train.csv", low_memory=False, usecols=["cdr3b"])
    lens = [len(str(x)) for x in d["cdr3b"].dropna() if 5 <= len(str(x)) <= 26]
    return dist_from(collections.Counter(lens))


def top_motifs(n: int = 30) -> set[str]:
    d = pd.read_csv(PAPERS / "train.csv", low_memory=False, usecols=["cdr3b"])
    cs = [str(x) for x in d["cdr3b"].dropna() if len(str(x)) >= WIN + 6]
    return {m for m, _ in collections.Counter(c[-WIN:] for c in cs).most_common(n)}


def placeholder(L: int) -> str:
    L = max(3, int(L))
    return "C" + "A" * (L - 2) + "F"


def decode_chain(model, tok, ids, cols):
    idx = torch.tensor(cols, device=ids.device, dtype=torch.long)
    out = _inverse_remap_llada_tokens(model, ids.index_select(0, idx).unsqueeze(0))
    return decode_residue_span(out.squeeze(0), tok, 0, out.numel())


def comp(seqs, ref_comp=None):
    j = "".join(core(s) for s in seqs if core(s))
    n = len(j) or 1
    v = np.array([j.count(a) / n for a in AA])
    if ref_comp is None:
        return v
    p, q = v + 1e-12, ref_comp + 1e-12
    m = (p + q) / 2
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@torch.no_grad()
def run_cell(model, tok, col, ev, pmhcs, k, strategy, temp, max_iter,
             lv, lp, rng, bs, top, ref_comp):
    per, allseq, per_seqs = [], [], []
    for pmhc in pmhcs:
        ep = ev[pmhc]["epitope"]
        lens = [int(x) for x in rng.choice(lv, size=k, p=lp)]
        seqs = []
        for s in range(0, k, bs):
            chunk = lens[s:s + bs]
            recs = [BioSeqRecord(
                chains=[BioSeqChain(ep, "peptide"), BioSeqChain(placeholder(L), "tcr_beta")],
                task_type="tcr_epitope", source="grid") for L in chunk]
            batch = col(recs)
            partial = tcr_generation_partial_mask(batch, target_chain_indices=1)
            toks, _ = run_grammar_generate(model, batch, partial_mask=partial,
                                           max_iter=max_iter, sampling_strategy=strategy,
                                           temperature=temp)
            bc = residue_positions_by_chain(batch)
            for i in range(len(chunk)):
                q = decode_chain(model, tok, toks[i], bc[i][1])
                if q:
                    seqs.append(q)
        allseq += seqs
        per_seqs.append(seqs)
        per.append(tcrt5_mean_edit_distance([x.upper() for x in seqs], ev[pmhc]["ref_binders"]))
    cs = [core(s) for s in allseq if core(s)]
    return {
        "d_edit": float(np.mean(per)),
        "jmotif": float(np.mean([c[-WIN:] in top for c in cs])) if cs else float("nan"),
        "divers": len(set(allseq)) / max(1, len(allseq)),
        "corelen": float(np.mean([len(c) for c in cs])) if cs else float("nan"),
        "compJSD": comp(allseq, ref_comp),
        "gggg": float(np.mean([bool(re.search(r"G{4,}", c)) for c in cs])) if cs else float("nan"),
        "n": len(allseq),
        "ex": allseq[:2],
        "seqs": allseq,
        "by_pmhc": {p: s for p, s in zip(pmhcs, per_seqs)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("-k", "--k", type=int, default=20)
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--cells", required=True,
                    help="semicolon list of strategy,temp,iter,length e.g. "
                         "'gumbel_argmax,1.0,32,full;vanilla,0.5,32,core'")
    ap.add_argument("--out", default="/tmp/grid_sampling.json")
    args = ap.parse_args()

    ev = json.loads((GB / "eval_conditional.json").read_text())
    unseen = set(json.loads((GB / "bioseq_unseen_pmhc.json").read_text()))
    pmhcs = sorted(p for p, e in ev.items()
                   if e["category"] == "benchmark14" and p in unseen and p != RESERVED)
    ref_comp = comp([r for p in pmhcs for r in ev[p]["ref_binders"]])
    top = top_motifs()

    full_lv, full_lp = dist_from(FULL_LEN_DIST)
    core_lv, core_lp = core_len_dist()
    print(f"threads={_THREADS}  k={args.k}  slice={len(pmhcs)} pMHC")
    print(f"长度先验  full(现行, 注释说是C..F全长): 均值 {float((full_lv*full_lp).sum()):.2f}")
    print(f"长度先验  core(训练语料实测)          : 均值 {float((core_lv*core_lp).sum()):.2f}")
    print(f"参考序列 core 长度均值                : "
          f"{np.mean([len(core(r)) for p in pmhcs for r in ev[p]['ref_binders']]):.2f}")
    print("\n参考: OLGA 6.343(完全不看表位) | TCRdiff 5.157 | TCRT5 4.696 | GraTCR 4.605")
    print("注意: 这 6 个表位在训练语料中 0/6 出现过，是零样本新表位\n")

    model, tok = load_grammar_checkpoint(args.checkpoint, device="cpu")
    col = build_eval_collator(model, tok)
    print(f"model={type(model).__name__}\n")
    print(f"  {'strategy':16s} {'T':>4s} {'iter':>5s} {'len':>5s} | {'d_edit':>7s} "
          f"{'J基序':>6s} {'多样性':>7s} {'core长':>6s} {'compJSD':>8s} {'GGGG':>6s}")

    dump = {}
    for cell in args.cells.split(";"):
        cell = cell.strip()
        if not cell:
            continue
        strat, temp, it, lmode = cell.split(",")
        temp, it = float(temp), int(it)
        lv, lp = (core_lv, core_lp) if lmode == "core" else (full_lv, full_lp)
        r = run_cell(model, tok, col, ev, pmhcs, args.k, strat, temp, it, lv, lp,
                     np.random.default_rng(42), args.bs, top, ref_comp)
        print(f"  {strat:16s} {temp:4.1f} {it:5d} {lmode:>5s} | {r['d_edit']:7.3f} "
              f"{r['jmotif']:6.3f} {r['divers']:7.3f} {r['corelen']:6.2f} "
              f"{r['compJSD']:8.4f} {r['gggg']:6.3f}")
        print(f"      e.g. {r['ex']}")
        dump[cell] = r
        Path(args.out).write_text(json.dumps(dump, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
