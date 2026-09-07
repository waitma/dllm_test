#!/usr/bin/env python
"""Multi-angle verification that the missing CDR3b J-gene motif is a model
property and not an artifact of the eval / decode / mask / sampling code.

Each test isolates one link in the chain, so a failure localises the cause:

  T1 render+decode identity  the rendered training input really contains the
                             J motif at the end of the beta chain, and decode
                             returns it byte-for-byte.
  T2 mask audit              generation masks exactly the beta residues; the
                             epitope stays visible and no placeholder anchor
                             leaks into the output.
  T3 window infill           real held-out sequence, everything visible except
                             a 5-residue window. No length sampling, no anchor
                             convention, no d_edit.
  T4 single forward pass     p(true residue) at each masked position from ONE
                             denoiser call, which removes the iterative sampler
                             entirely. Run at head / middle / tail windows so
                             the model is compared against ITSELF, and against
                             a context-free positional-frequency (PWM) table.
  T5 layout parity           T4 repeated in tcr_pmhc (80% of training rows) so
                             a bad number cannot be blamed on an unseen layout.

The decisive contrast is T4 head/middle vs tail: same model, same forward pass,
same sequences, same code path. Only the masked window moves.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _cap_threads() -> int:
    """Size the torch thread pool to the container's CPU quota, not the host's.

    ``sched_getaffinity`` reports all 128 host CPUs, but lxcfs shows this
    container 13 in ``/proc/cpuinfo`` and the cgroup CFS quota enforces that.
    Torch sizes its pool from the host count (64 intra-op threads), so the
    default oversubscribes ~5x: once the quota is spent inside a period the
    whole cgroup is throttled, which freezes every process in the container --
    including unrelated shells. Measured 8.10 ms/matmul at 64 threads vs
    3.04 ms at 13, so this is a 2.7x speedup, not just a politeness fix.
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, BioSeqRecord
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import (
    _model_logits,
    build_generation_mask,
    initialize_output_tokens,
    resolve_partial_mask,
)
from downstream.grammar.common import (
    _inverse_remap_llada_tokens,
    build_eval_collator,
    load_grammar_checkpoint,
    run_grammar_generate,
)
from downstream.grammar.masks import (
    cdr3b_span_partial_mask,
    residue_positions_by_chain,
    tcr_generation_partial_mask,
)
from downstream.grammar.metrics import decode_residue_span

PAPERS = PROJECT_ROOT / "data" / "tcr_papers_v2" / "dataset"
WIN = 5
WINDOWS = ("head", "middle", "tail")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def head_(title: str) -> None:
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_rows(n: int, with_mhc: bool, seed: int = 0) -> list[dict]:
    """Held-out rows in the model's own training convention (anchor-free core)."""
    df = pd.read_csv(PAPERS / "holdout.csv", low_memory=False)
    df = df[df["cdr3b"].notna() & df["epitope_seq"].notna() & df["cdr3a"].isna()]
    df = df[df["cdr3b"].astype(str).str.len() >= WIN + 6]
    df = df[df["mhc_seq"].notna()] if with_mhc else df[df["mhc_seq"].isna()]
    df = df.sample(n=min(n, len(df)), random_state=seed)
    return [
        {
            "cdr3b": str(r.cdr3b).strip().upper(),
            "epitope": str(r.epitope_seq).strip().upper(),
            "mhc": (str(r.mhc_seq).strip().upper() if with_mhc else None),
        }
        for r in df.itertuples()
    ]


def train_stats() -> dict:
    df = pd.read_csv(PAPERS / "train.csv", low_memory=False)
    cores = [c for c in df["cdr3b"].dropna().astype(str).tolist() if len(c) >= WIN + 6]
    top = [m for m, _ in collections.Counter(c[-WIN:] for c in cores).most_common(30)]
    # Context-free positional frequency tables, one per window, built ONLY from
    # training data. This is the "no model at all" reference for T4.
    pwm: dict[str, list[collections.Counter]] = {}
    for where in WINDOWS:
        cnt = [collections.Counter() for _ in range(WIN)]
        for c in cores:
            lo, hi = window_of(len(c), where)
            for i, ch in enumerate(c[lo:hi]):
                cnt[i][ch] += 1
        pwm[where] = cnt
    return {"cores": cores, "top": set(top), "top_list": top, "pwm": pwm,
            "tail_cov": float(np.mean([c[-WIN:] in set(top) for c in cores]))}


def window_of(length: int, where: str) -> tuple[int, int]:
    if where == "head":
        return 0, WIN
    if where == "tail":
        return length - WIN, length
    lo = (length - WIN) // 2
    return lo, lo + WIN


def pwm_reference(rows: list[dict], pwm: dict, where: str) -> dict:
    """Accuracy / p(true) of a context-free positional frequency table."""
    cnt = pwm[where]
    tot = [sum(c.values()) for c in cnt]
    argmax_ch = [c.most_common(1)[0][0] for c in cnt]
    acc, ptrue = [], []
    for r in rows:
        lo, hi = window_of(len(r["cdr3b"]), where)
        for i, ch in enumerate(r["cdr3b"][lo:hi]):
            acc.append(ch == argmax_ch[i])
            ptrue.append(cnt[i].get(ch, 0) / max(1, tot[i]))
    return {"argmax_acc": float(np.mean(acc)), "p_true": float(np.mean(ptrue))}


def build_records(rows: list[dict], seqs: list[str]) -> tuple[list[BioSeqRecord], int]:
    """Records in the training layout. Returns (records, beta_chain_index)."""
    recs, beta_idx = [], 1
    for row, seq in zip(rows, seqs):
        chains = []
        if row["mhc"]:
            chains.append(BioSeqChain(row["mhc"], "mhc"))
        chains.append(BioSeqChain(row["epitope"], "peptide"))
        chains.append(BioSeqChain(seq, "tcr_beta"))
        beta_idx = 2 if row["mhc"] else 1
        recs.append(BioSeqRecord(
            chains=chains,
            task_type="tcr_pmhc" if row["mhc"] else "tcr_epitope",
            source="verify_jmotif",
        ))
    return recs, beta_idx


def collate(collator, recs, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in collator(recs).items()}


def decode_chain(model, tokenizer, token_ids, cols, device) -> str:
    """Decode one chain's residues, undoing the fusion LLaDA id remap first."""
    idx = torch.tensor(cols, device=token_ids.device, dtype=torch.long)
    ids = token_ids.index_select(0, idx)
    ids = _inverse_remap_llada_tokens(model, ids.unsqueeze(0)).squeeze(0)
    return decode_residue_span(ids, tokenizer, 0, ids.numel())


# --------------------------------------------------------------------------- #
# T1 render + decode identity
# --------------------------------------------------------------------------- #

def t1_identity(model, collator, tokenizer, rows, device) -> bool:
    head_("T1  render + decode identity  (does the model's input contain the J motif?)")
    recs, beta_idx = build_records(rows, [r["cdr3b"] for r in rows])
    batch = collate(collator, recs, device)
    by_chain = residue_positions_by_chain(batch)

    bad, len_ok, tail_ok = 0, 0, 0
    for i, row in enumerate(rows):
        cols = by_chain[i][beta_idx]
        got = decode_chain(model, tokenizer, batch["input_ids"][i], cols, device)
        len_ok += len(cols) == len(row["cdr3b"])
        tail_ok += got[-WIN:] == row["cdr3b"][-WIN:]
        if got != row["cdr3b"]:
            bad += 1
            if bad <= 3:
                log(f"  MISMATCH row {i}: want {row['cdr3b']!r} got {got!r}")
    n = len(rows)
    log(f"  residue columns == len(cdr3b)       : {len_ok}/{n}")
    log(f"  beta chain decoded byte-identical   : {n - bad}/{n}")
    log(f"  C-terminal {WIN}-mer present in input : {tail_ok}/{n}")
    ok = bad == 0
    log(f"  => {'PASS' if ok else 'FAIL'}: the rendered input "
        f"{'contains' if ok else 'does NOT contain'} the full core incl. its J motif")
    return ok


# --------------------------------------------------------------------------- #
# T2 mask audit
# --------------------------------------------------------------------------- #

def t2_mask_audit(collator, rows, device) -> bool:
    head_("T2  mask audit  (does the placeholder anchor leak? is the epitope kept?)")
    L = 15
    placeholder = "C" + "A" * (L - 2) + "F"  # exactly what conditional_cdr3b builds
    recs, beta_idx = build_records(rows[:8], [placeholder] * 8)
    batch = collate(collator, recs, device)
    partial = resolve_partial_mask(
        batch, tcr_generation_partial_mask(batch, target_chain_indices=beta_idx))
    gen_mask = build_generation_mask(batch, partial)
    by_chain = residue_positions_by_chain(batch)

    ok = True
    for i in range(len(recs)):
        beta_cols = set(by_chain[i][beta_idx])
        ep_cols = set(by_chain[i][beta_idx - 1])
        masked = {c for c in range(gen_mask.size(1)) if bool(gen_mask[i, c])}
        if masked != beta_cols:
            ok = False
            log(f"  row {i}: masked != beta residues "
                f"(missing {sorted(beta_cols - masked)}, extra {sorted(masked - beta_cols)})")
        if masked & ep_cols:
            ok = False
            log(f"  row {i}: epitope residues masked -> conditioning destroyed")
    log(f"  every beta residue masked and nothing else : {'yes' if ok else 'NO'}")
    log(f"  placeholder C/F anchors are masked as well, so a C-start in the output is")
    log(f"  model-generated rather than template leakage")
    log(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# T3 sampler infill
# --------------------------------------------------------------------------- #

def t3_infill(model, collator, tokenizer, rows, device, where, top,
              max_iter=32, bs=16) -> dict:
    hit = tot = exact = in_top = n = keep_ok = 0
    ex: list[str] = []
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        recs, beta_idx = build_records(chunk, [r["cdr3b"] for r in chunk])
        batch = collate(collator, recs, device)
        spans = [window_of(len(r["cdr3b"]), where) for r in chunk]
        partial = cdr3b_span_partial_mask(batch, chain_index=beta_idx, span=spans)
        tokens, _ = run_grammar_generate(model, batch, partial_mask=partial,
                                         max_iter=max_iter,
                                         sampling_strategy="gumbel_argmax")
        by_chain = residue_positions_by_chain(batch)
        for i, r in enumerate(chunk):
            pred = decode_chain(model, tokenizer, tokens[i], by_chain[i][beta_idx], device)
            if len(pred) != len(r["cdr3b"]):
                continue
            lo, hi = spans[i]
            tw, pw = r["cdr3b"][lo:hi], pred[lo:hi]
            # everything OUTSIDE the window must be preserved, else the mask is wrong
            keep_ok += (pred[:lo] == r["cdr3b"][:lo] and pred[hi:] == r["cdr3b"][hi:])
            hit += sum(a == b for a, b in zip(tw, pw))
            tot += len(tw)
            exact += tw == pw
            in_top += pred[-WIN:] in top
            n += 1
            if len(ex) < 3:
                ex.append(f"{r['cdr3b']} -> {pred}   [{tw} -> {pw}]")
    return {"aar": hit / max(1, tot), "exact": exact / max(1, n),
            "in_top30": in_top / max(1, n), "n": n,
            "context_preserved": keep_ok / max(1, n), "ex": ex}


# --------------------------------------------------------------------------- #
# T4 single forward pass
# --------------------------------------------------------------------------- #

@torch.no_grad()
def t4_forward(model, collator, tokenizer, rows, device, where, top,
               timestep=None, bs=16) -> dict:
    p_true, nll, correct, npos, motif_hit, nrec = [], [], 0, 0, 0, 0
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        recs, beta_idx = build_records(chunk, [r["cdr3b"] for r in chunk])
        batch = collate(collator, recs, device)
        spans = [window_of(len(r["cdr3b"]), where) for r in chunk]
        partial = resolve_partial_mask(
            batch, cdr3b_span_partial_mask(batch, chain_index=beta_idx, span=spans))
        gen_mask = build_generation_mask(batch, partial)
        mask_id = int(model.config.mask_token_id)
        out_tokens, _ = initialize_output_tokens(batch["input_ids"], gen_mask, mask_id)
        t = 1.0 if timestep is None else float(timestep)
        logits = _model_logits(
            model=model, batch=batch, output_tokens=out_tokens,
            generation_mask=gen_mask, mask_token_id=mask_id,
            timesteps=torch.tensor([t], device=device),
            cfg_scale=0.0, partial_mask=partial,
        )
        probs = torch.softmax(logits.float(), dim=-1)
        truth = batch["input_ids"]
        by_chain = residue_positions_by_chain(batch)
        for i in range(len(recs)):
            cols = by_chain[i][beta_idx]
            lo, hi = spans[i]
            am_ids = []
            for j in range(lo, hi):
                col = cols[j]
                tid = int(truth[i, col].item())
                p = float(probs[i, col, tid].item())
                p_true.append(p)
                nll.append(-np.log(max(p, 1e-12)))
                am = int(probs[i, col].argmax().item())
                am_ids.append(am)
                correct += int(am == tid)
                npos += 1
            pred_w = decode_chain(model, tokenizer,
                                  torch.tensor(am_ids, device=device),
                                  list(range(len(am_ids))), device)
            if where == "tail":
                motif_hit += pred_w in top
            nrec += 1
    return {"p_true": float(np.mean(p_true)), "p_true_med": float(np.median(p_true)),
            "argmax_acc": correct / max(1, npos), "nll": float(np.mean(nll)),
            "n_pos": npos, "argmax_in_top30": motif_hit / max(1, nrec)}


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("-n", "--n", type=int, default=200, help="rows for T4/T5")
    ap.add_argument("--n-sampler", type=int, default=48, help="rows for T3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-iter", type=int, default=32)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--skip-t3", action="store_true")
    # See verify_sampling.py: 64 (the default here) is ~2.3x slower than 16 for
    # this tensor shape, and concurrent runs hit an oversubscription cliff.
    ap.add_argument("--threads", type=int, default=16, help="torch CPU threads (0 = leave default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    label = args.label or Path(args.checkpoint).name
    log(f"checkpoint : {args.checkpoint}")
    log(f"label      : {label}")
    log(f"threads    : {_THREADS} (capped to the container CPU quota)")

    st = train_stats()
    log(f"\ntrain cores n={len(st['cores'])}  mean len={np.mean([len(c) for c in st['cores']]):.2f}")
    log(f"top-30 C-terminal {WIN}-mers cover {st['tail_cov'] * 100:.1f}% of training cores")
    log(f"  e.g. {st['top_list'][:10]}")

    rows = load_rows(args.n, with_mhc=False)
    rows_mhc = load_rows(args.n, with_mhc=True)
    rows_s = rows[:args.n_sampler]
    log(f"\nheld-out rows: tcr_epitope={len(rows)}  tcr_pmhc={len(rows_mhc)}")
    log(f"true C-terminal {WIN}-mer in top-30 on these rows: "
        f"{np.mean([r['cdr3b'][-WIN:] in st['top'] for r in rows]):.3f}")

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available())
                          else "cpu")
    if device.type == "cpu":
        log("\n!! CUDA unavailable in this shell (driver 12.2 vs torch cu130) -> CPU")
    model, tokenizer = load_grammar_checkpoint(args.checkpoint, device=device)
    collator = build_eval_collator(model, tokenizer)
    log(f"\nmodel: {type(model).__name__}  mask_token_id={model.config.mask_token_id}")

    ok1 = t1_identity(model, collator, tokenizer, rows, device)
    ok2 = t2_mask_audit(collator, rows, device)

    if not args.skip_t3:
        head_("T3  infill with the REAL sampler  (no length sampling, no anchors, no d_edit)")
        log(f"  {'window':14s} {'AAR':>7s} {'exact':>7s} {'in_top30':>9s} {'ctx_kept':>9s} {'n':>4s}")
        for where in WINDOWS:
            r = t3_infill(model, collator, tokenizer, rows_s, device, where,
                          st["top"], args.max_iter, args.bs)
            log(f"  {where:14s} {r['aar']:7.3f} {r['exact']:7.3f} "
                f"{r['in_top30']:9.3f} {r['context_preserved']:9.3f} {r['n']:4d}")
            for e in r["ex"]:
                log(f"      {e}")

    head_("T4  ONE forward pass, teacher-forced  (iterative sampler removed)")
    log("  Same model, same sequences, same code path -- only the masked window moves.")
    log(f"\n  {'window':14s} {'p(true)':>9s} {'median':>8s} {'argmax':>8s} {'NLL':>7s} "
        f"{'PWM p':>8s} {'PWM acc':>8s} {'am_top30':>9s}")
    for where in WINDOWS:
        r = t4_forward(model, collator, tokenizer, rows, device, where, st["top"], bs=args.bs)
        ref = pwm_reference(rows, st["pwm"], where)
        log(f"  {where:14s} {r['p_true']:9.4f} {r['p_true_med']:8.4f} "
            f"{r['argmax_acc']:8.3f} {r['nll']:7.3f} {ref['p_true']:8.4f} "
            f"{ref['argmax_acc']:8.3f} {r['argmax_in_top30']:9.3f}")
    log("  uniform-over-20AA reference: p(true)=0.0500  NLL=3.00  acc=0.050")

    head_("T4b timestep sensitivity  (is the tail result an artifact of the noise level?)")
    log(f"  {'t':>6s} {'head p(true)':>13s} {'middle p(true)':>15s} {'tail p(true)':>13s}")
    for t in (1.0, 0.5, 0.1):
        vals = [t4_forward(model, collator, tokenizer, rows[:64], device, w,
                           st["top"], timestep=t, bs=args.bs)["p_true"]
                for w in WINDOWS]
        log(f"  {t:6.2f} {vals[0]:13.4f} {vals[1]:15.4f} {vals[2]:13.4f}")

    if rows_mhc:
        head_("T5  layout parity: same test in tcr_pmhc (80% of training rows)")
        log(f"  {'window':14s} {'p(true)':>9s} {'argmax':>8s} {'NLL':>7s} {'am_top30':>9s}")
        for where in WINDOWS:
            r = t4_forward(model, collator, tokenizer, rows_mhc, device, where,
                           st["top"], bs=args.bs)
            log(f"  {where:14s} {r['p_true']:9.4f} {r['argmax_acc']:8.3f} "
                f"{r['nll']:7.3f} {r['argmax_in_top30']:9.3f}")

    head_("VERDICT INPUTS")
    log(f"  T1 render/decode identity : {'PASS' if ok1 else 'FAIL'}")
    log(f"  T2 mask audit             : {'PASS' if ok2 else 'FAIL'}")
    log("  T3/T4/T4b/T5 numbers above localise the J-motif result.")


if __name__ == "__main__":
    main()
