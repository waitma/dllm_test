"""Round 2: separate the three surviving explanations for the T4 J-motif failure.

Round 1 (debug/verify_jmotif.py) established that the chain-end machinery is
fine (masking a full-length chain's last 5 residues recovers them at AAR 0.98)
and that the same 5 CDR3b residues are recovered at AAR 0.77 *inside* a
full-length beta chain but only ~0.3 in the bare CDR3b layout. Three
explanations remain:

  H1  decoding schedule -- the model knows the joint J motif but parallel
      unmasking commits to marginals and cannot coordinate 5 positions.
      => E2 sweeps max_iter; E3 reads the marginals with zero decoding.
  H2  right context     -- the model only places J residues when residues
      *after* them are visible, so a mask that runs to the chain end is
      unanswerable for it. => E4 masks a mid-sequence window of identical size
      in the bare layout, and removes right context inside the full-length one.
  H3  model/objective   -- the J motif genuinely was not learned in the CDR3
      layout. Survives only if E2/E3/E4 all come back negative.

Usage: python -m debug.verify_jmotif2 --checkpoint <dir> --n 100 --out r2.json
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

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, BioSeqRecord
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import (
    _inference_timesteps,
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
from downstream.grammar.masks import residue_positions_by_chain
from downstream.grammar.metrics import decode_residue_span

AA = "ACDEFGHIKLMNPQRSTVWY"


class Harness:
    def __init__(self, ckpt: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, self.tok = load_grammar_checkpoint(ckpt, device=self.device)
        self.collator = build_eval_collator(self.model, self.tok)

    def prepare(self, records, chain_index, spans):
        batch = self.collator(records)
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        partial = batch["attention_mask"].bool().clone()
        cols_per_row = []
        for row, span in enumerate(spans):
            positions = residue_positions_by_chain(batch)[row].get(chain_index, [])
            if span is None:
                cols_per_row.append([])
                continue
            lo, hi = span
            chosen = positions[lo:hi]
            for col in chosen:
                partial[row, col] = False
            cols_per_row.append(chosen)
        return batch, resolve_partial_mask(batch, partial), cols_per_row

    def decode(self, tokens, batch, row, chain_index):
        pic, rm, am = (batch["position_ids_chain"][row], batch["residue_mask"][row],
                       batch["attention_mask"][row])
        cols = [c for c in range(tokens.size(0))
                if am[c] and rm[c] and int(pic[c].item()) == chain_index]
        if not cols:
            return ""
        idx = torch.tensor(cols, device=tokens.device, dtype=torch.long)
        return decode_residue_span(tokens.index_select(0, idx), self.tok, 0, len(cols))

    def generate(self, records, chain_index, spans, max_iter):
        batch, partial, _ = self.prepare(records, chain_index, spans)
        tokens, _ = run_grammar_generate(self.model, batch, partial_mask=partial,
                                         max_iter=max_iter, sampling_strategy="gumbel_argmax")
        return [self.decode(tokens[r], batch, r, chain_index) for r in range(len(records))]

    def roundtrip(self, records, chain_index):
        """No forward pass at all; the fusion id remap is inverted so this reads
        back exactly what the production decode path would see."""
        batch, _, _ = self.prepare(records, chain_index, [None] * len(records))
        tokens = _inverse_remap_llada_tokens(self.model, batch["input_ids"])
        return [self.decode(tokens[r], batch, r, chain_index) for r in range(len(records))]

    @torch.no_grad()
    def marginals(self, records, chain_index, spans):
        """One forward pass over the masked state. No sampling, no iteration.

        Returns per row a list of (true_residue, top1_residue, p_true, p_top1)
        for each masked position, in sequence order.
        """
        batch, partial, cols_per_row = self.prepare(records, chain_index, spans)
        gen_mask = build_generation_mask(batch, partial)
        mask_id = int(self.model.config.mask_token_id)
        out_tokens, _ = initialize_output_tokens(batch["input_ids"], gen_mask, mask_id)
        ts = _inference_timesteps(1, 1, self.model.config.time_epsilon, self.device)
        logits = _model_logits(model=self.model, batch=batch, output_tokens=out_tokens,
                              generation_mask=gen_mask, mask_token_id=mask_id,
                              timesteps=ts, cfg_scale=0.0, partial_mask=partial)
        probs = torch.softmax(logits.float(), dim=-1)
        results = []
        for row, cols in enumerate(cols_per_row):
            row_out = []
            for col in cols:
                true_id = int(batch["input_ids"][row, col].item())
                p = probs[row, col]
                p_true = float(p[true_id].item())
                p_top1, top1_id = p.max(dim=-1)
                true_res = self.decode_one(true_id)
                top1_res = self.decode_one(int(top1_id.item()))
                row_out.append((true_res, top1_res, p_true, float(p_top1.item())))
            results.append(row_out)
        return results

    def decode_one(self, token_id: int) -> str:
        t = _inverse_remap_llada_tokens(
            self.model, torch.tensor([[token_id]], device=self.device))
        return decode_residue_span(t[0], self.tok, 0, 1)


def core_records(rows, layout):
    recs = []
    for cdr3b, ep, mhc in rows:
        if layout == "tcr_single":
            recs.append(BioSeqRecord(chains=[BioSeqChain(cdr3b, "tcr_beta")],
                                     task_type="tcr", source="v2"))
        elif layout == "tcr_pmhc":
            recs.append(BioSeqRecord(
                chains=[BioSeqChain(mhc, "mhc"), BioSeqChain(ep, "peptide"),
                        BioSeqChain(cdr3b, "tcr_beta")],
                task_type="tcr_pmhc", source="v2"))
        else:
            raise ValueError(layout)
    return recs, {"tcr_single": 0, "tcr_pmhc": 2}[layout]


def batched(fn, items, bs):
    out = []
    for i in range(0, len(items), bs):
        out.extend(fn(items[i:i + bs]))
    return out


def aar(res):
    return float(np.mean([sum(a == b for a, b in zip(t, p)) / len(t) for t, p in res]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--n-full", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    H = Harness(args.checkpoint)
    rep: dict = {"checkpoint": args.checkpoint}

    hold = pd.read_csv(ROOT / "data/tcr_papers_v2/dataset/holdout.csv", low_memory=False)
    hold = hold[hold["cdr3b"].notna() & hold["epitope_seq"].notna() & hold["mhc_seq"].notna()]
    hold = hold[hold["cdr3b"].astype(str).str.len().between(12, 18)]
    for c in ("cdr3b", "epitope_seq", "mhc_seq"):
        hold = hold[hold[c].astype(str).str.fullmatch(f"[{AA}]+")]
    rows = list(zip(hold["cdr3b"].astype(str), hold["epitope_seq"].astype(str),
                    hold["mhc_seq"].astype(str)))[: args.n]
    print(f"[corpus] {len(rows)} holdout CDR3b, len mean {np.mean([len(r[0]) for r in rows]):.2f}")

    train = pd.read_csv(ROOT / "data/tcr_papers_v2/dataset/train.csv", low_memory=False)
    tr = [c for c in train["cdr3b"].dropna().astype(str).tolist() if len(c) > 5]
    top_j = [s for s, _ in collections.Counter(s[-5:] for s in tr).most_common(30)]

    # ================== E1: round-trip with the id remap fixed ============ #
    print("\n[E1] decode round-trip (no forward pass), fusion id remap inverted")
    rep["E1"] = {}
    for layout in ("tcr_single", "tcr_pmhc"):
        recs, tgt = core_records(rows[:40], layout)
        got = H.roundtrip(recs, tgt)
        ex = float(np.mean([a[0] == b for a, b in zip(rows[:40], got)]))
        rep["E1"][layout] = ex
        print(f"     {layout:11s} exact={ex:.4f}   e.g. {rows[0][0]} -> {got[0]}")

    # ============ E2: suffix-5, decoding-budget sweep ===================== #
    print("\n[E2] suffix-5 mask on bare CDR3b, max_iter sweep (H1: schedule)")
    rep["E2"] = {}
    for layout in ("tcr_single", "tcr_pmhc"):
        rep["E2"][layout] = {}
        for mi in (1, 5, 16, 32):
            def go(chunk, layout=layout, mi=mi):
                recs, tgt = core_records(chunk, layout)
                spans = [(len(c[0]) - 5, len(c[0])) for c in chunk]
                got = H.generate(recs, tgt, spans, max_iter=mi)
                return [(c[0][-5:], g[-5:]) for c, g in zip(chunk, got) if len(g) == len(c[0])]
            res = batched(go, rows, args.batch_size)
            a = aar(res)
            isj = float(np.mean([p in top_j for _, p in res]))
            rep["E2"][layout][mi] = {"aar": a, "pred_is_top30_J": isj, "n": len(res)}
            print(f"     {layout:11s} max_iter={mi:3d}: AAR={a:.3f} pred_is_J={isj:.3f} n={len(res)}")

    # ============ E3: raw marginals, zero decoding ======================== #
    print("\n[E3] one forward pass on the masked state; no sampling (H1 vs H3)")
    rep["E3"] = {}
    for layout in ("tcr_single", "tcr_pmhc"):
        def marg(chunk, layout=layout):
            recs, tgt = core_records(chunk, layout)
            spans = [(len(c[0]) - 5, len(c[0])) for c in chunk]
            return H.marginals(recs, tgt, spans)
        res = batched(marg, rows, args.batch_size)
        by_off = collections.defaultdict(list)
        for row_out in res:
            for i, (true_res, top1, p_true, p_top1) in enumerate(row_out):
                by_off[5 - i].append((true_res == top1, p_true, p_top1))
        entry = {}
        for off in sorted(by_off, reverse=True):
            v = by_off[off]
            entry[off] = {"top1_correct": float(np.mean([x[0] for x in v])),
                          "mean_p_true": float(np.mean([x[1] for x in v])),
                          "mean_p_top1": float(np.mean([x[2] for x in v])),
                          "n": len(v)}
        rep["E3"][layout] = entry
        print(f"     {layout} (offset from C-term: top1_acc / p_true / p_top1)")
        for off in sorted(entry, reverse=True):
            e = entry[off]
            print(f"       offset {off}: top1={e['top1_correct']:.3f}  "
                  f"p_true={e['mean_p_true']:.3f}  p_top1={e['mean_p_top1']:.3f}")
        # what does the joint argmax 5-mer look like?
        preds = ["".join(x[1] for x in row_out) for row_out in res]
        isj = float(np.mean([p in top_j for p in preds]))
        rep["E3"][layout]["argmax_5mer_is_top30_J"] = isj
        print(f"       argmax 5-mer is a top-30 J motif: {isj:.3f}   e.g. {preds[:5]}")

    # ============ E4: right-context controls (H2) ========================= #
    print("\n[E4] right-context controls (H2)")
    rep["E4"] = {}
    # (c) same 5-residue mask, but mid-sequence in the bare layout
    for layout in ("tcr_single", "tcr_pmhc"):
        def mid(chunk, layout=layout):
            recs, tgt = core_records(chunk, layout)
            spans = []
            for c in chunk:
                L = len(c[0])
                lo = (L - 5) // 2
                spans.append((lo, lo + 5))
            got = H.generate(recs, tgt, spans, max_iter=5)
            out = []
            for c, g, sp in zip(chunk, got, spans):
                if len(g) == len(c[0]):
                    out.append((c[0][sp[0]:sp[1]], g[sp[0]:sp[1]]))
            return out
        res = batched(mid, rows, args.batch_size)
        rep["E4"][f"{layout}_middle5"] = {"aar": aar(res), "n": len(res)}
        print(f"     {layout:11s} MIDDLE-5 (both sides visible): AAR={aar(res):.3f} n={len(res)}")

    # (d) full-length beta: J motif with vs without downstream context
    ots = pd.read_csv(ROOT / "data/ots_paired_clean/splits/holdout.csv", low_memory=False)
    pairs = []
    for r in ots.itertuples():
        beta, alpha, core = str(r.cleaned_chain1_seq), str(r.cleaned_chain2_seq), str(r.chain1_CDR3)
        if not core or core == "nan" or len(core) < 8:
            continue
        if set(beta) - set(AA) or set(alpha) - set(AA):
            continue
        off = beta.find(core)
        if off < 5 or off + len(core) > len(beta) - 8:
            continue
        pairs.append((alpha, beta, core, off))
        if len(pairs) >= args.n_full:
            break
    print(f"     {len(pairs)} OTS pairs for the full-length controls")

    def full_run(chunk, mode):
        recs = [BioSeqRecord(chains=[BioSeqChain(a, "tcr_alpha"), BioSeqChain(b, "tcr_beta")],
                             task_type="tcr", source="v2") for a, b, _, _ in chunk]
        spans, truths = [], []
        for _, b, core, off in chunk:
            end = off + len(core)
            if mode == "with_right_ctx":
                spans.append((end - 5, end))
            else:  # mask the J motif AND everything downstream of it
                spans.append((end - 5, len(b)))
            truths.append(b[end - 5:end])
        got = H.generate(recs, 1, spans, max_iter=16)
        out = []
        for (_, b, core, off), g, t in zip(chunk, got, truths):
            if len(g) != len(b):
                continue
            end = off + len(core)
            out.append((t, g[end - 5:end]))
        return out

    for mode, label in (("with_right_ctx", "J motif, FR4 visible"),
                        ("no_right_ctx", "J motif, everything downstream masked")):
        res = batched(lambda ch, m=mode: full_run(ch, m), pairs, 12)
        a = aar(res)
        isj = float(np.mean([p in top_j for _, p in res]))
        rep["E4"][mode] = {"aar": a, "pred_is_top30_J": isj, "n": len(res)}
        print(f"     full-length {label:38s} AAR={a:.3f} pred_is_J={isj:.3f} n={len(res)}")
        for t, p in res[:4]:
            print(f"        {t} -> {p}")

    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2, default=str))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
