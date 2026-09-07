"""Region-resolved infill probe: does the model know the CDR3b J-gene motif?

Setting-B generation confounds two things -- whether the model *knows* the
germline C-terminal motif and whether iterative masked-diffusion decoding
manages to place it.  This probe pins down the first: everything except a small
window stays visible, so the answer is nearly determined by the context.

Three windows per sequence, run separately:
  jmotif  the last ``--motif-len`` residues (germline J, the thing missing
          from Setting-B output)
  vmotif  the first 3 residues (germline V, which Setting-B output gets right)
  middle  a same-width window at the centre (non-germline junction, the
          genuinely hard part -- the natural upper bound on difficulty)

If jmotif recovery is high, the knowledge is there and the generation gap is a
decoding/pipeline problem.  If it collapses toward the marginal-frequency
baseline while vmotif stays high, the model never learned the J side.

Both conventions are measured, because the training corpus stores anchor-free
IMGT cores while every eval reference is a full ``C..F`` junction:
  core      ASSLGTDTQY      (what training rows look like)
  anchored  CASSLGTDTQYF    (what eval references look like)

Reuses the same generate path as ``downstream.grammar.tcr_generation --mode
infill`` so nothing here is a re-implementation of the model interface.
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

from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, BioSeqRecord  # noqa: E402
from downstream.grammar.common import (  # noqa: E402
    build_eval_collator,
    load_grammar_checkpoint,
    run_grammar_generate,
)
from downstream.grammar.masks import cdr3b_span_partial_mask  # noqa: E402
from downstream.grammar.metrics import decode_residue_span  # noqa: E402

TRAIN_CSV = PROJECT_ROOT / "data" / "tcr_papers_v2" / "dataset" / "train.csv"
HOLDOUT_CSV = PROJECT_ROOT / "data" / "tcr_papers_v2" / "dataset" / "holdout.csv"


def load_cores(path: Path, n: int, seed: int) -> list[str]:
    df = pd.read_csv(path, low_memory=False)
    cores = [c for c in df["cdr3b"].dropna().astype(str).tolist() if 8 <= len(c) <= 20]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cores), size=min(n, len(cores)), replace=False)
    return [cores[i] for i in idx]


def marginal_baseline(train_cores: list[str], motif_len: int) -> dict[int, str]:
    """Frequency-only predictor: most common residue per offset-from-C-terminus.

    The comparison point for "did the model learn anything beyond base rates".
    Indexed from the end so it lines up with the J motif at any length.
    """
    by_offset: dict[int, collections.Counter] = {
        off: collections.Counter() for off in range(1, motif_len + 1)
    }
    for c in train_cores:
        for off in range(1, motif_len + 1):
            if len(c) >= off:
                by_offset[off][c[-off]] += 1
    return {off: cnt.most_common(1)[0][0] for off, cnt in by_offset.items() if cnt}


def spans_for(seqs: list[str], region: str, motif_len: int) -> list[tuple[int, int]]:
    out = []
    for s in seqs:
        L = len(s)
        if region == "jmotif":
            out.append((L - motif_len, L))
        elif region == "vmotif":
            out.append((0, 3))
        else:
            lo = (L - motif_len) // 2
            out.append((lo, lo + motif_len))
    return out


def _decode_chain(token_ids, batch, row, chain_index, tokenizer) -> str:
    pic = batch["position_ids_chain"][row]
    rm = batch["residue_mask"][row]
    am = batch["attention_mask"][row]
    cols = [c for c in range(token_ids.size(0))
            if am[c] and rm[c] and int(pic[c].item()) == chain_index]
    if not cols:
        return ""
    idx = torch.tensor(cols, device=token_ids.device, dtype=torch.long)
    return decode_residue_span(token_ids.index_select(0, idx), tokenizer, 0, len(cols))


@torch.no_grad()
def run_region(model, collator, tokenizer, seqs, spans, device, batch_size, max_iter):
    preds: list[str] = []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start:start + batch_size]
        chunk_spans = spans[start:start + batch_size]
        records = [
            BioSeqRecord(chains=[BioSeqChain(s, "tcr_beta")],
                         task_type="tcr", source="probe")
            for s in chunk
        ]
        batch = collator(records)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        partial = cdr3b_span_partial_mask(batch, chain_index=0, span=chunk_spans)
        tokens, _ = run_grammar_generate(
            model, batch, partial_mask=partial, max_iter=max_iter,
            sampling_strategy="gumbel_argmax", temperature=1.0)
        for row, s in enumerate(chunk):
            full = _decode_chain(tokens[row], batch, row, 0, tokenizer)
            lo, hi = chunk_spans[row]
            preds.append(full[lo:hi] if len(full) == len(s) else "")
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--motif-len", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conventions", default="core,anchored")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    label = args.label or Path(args.checkpoint).name
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[{label}] device={device}")
    model, tokenizer = load_grammar_checkpoint(args.checkpoint, device=device)
    model.eval()
    collator = build_eval_collator(model, tokenizer)

    train_cores = load_cores(TRAIN_CSV, 200_000, args.seed)
    eval_src = HOLDOUT_CSV if HOLDOUT_CSV.exists() else TRAIN_CSV
    cores = load_cores(eval_src, args.n, args.seed + 1)
    base = marginal_baseline(train_cores, args.motif_len)
    print(f"[{label}] marginal J predictor (offset->aa): "
          + "".join(base[o] for o in sorted(base, reverse=True)))

    results = {"label": label, "checkpoint": args.checkpoint,
               "eval_source": eval_src.name, "n": len(cores),
               "motif_len": args.motif_len, "max_iter": args.max_iter, "runs": {}}

    for convention in args.conventions.split(","):
        seqs = cores if convention == "core" else ["C" + c + "F" for c in cores]
        for region in ("jmotif", "vmotif", "middle"):
            spans = spans_for(seqs, region, args.motif_len)
            preds = run_region(model, collator, tokenizer, seqs, spans, device,
                               args.batch_size, args.max_iter)
            truth = [s[lo:hi] for s, (lo, hi) in zip(seqs, spans)]
            ok = [(t, p) for t, p in zip(truth, preds) if p]
            hit = sum(sum(a == b for a, b in zip(t, p)) for t, p in ok)
            tot = sum(len(t) for t, _ in ok)
            row = {"aar": hit / max(1, tot),
                   "exact_motif": float(np.mean([t == p for t, p in ok])) if ok else 0.0,
                   "n_positions": tot, "n_usable": len(ok), "n_total": len(truth),
                   "examples": [{"truth": t, "pred": p} for t, p in ok[:6]]}
            if region == "jmotif":
                bhit = sum(1 for t, _ in ok for i, ch in enumerate(t)
                           if base.get(len(t) - i) == ch)
                row["marginal_baseline_aar"] = bhit / max(1, tot)
            results["runs"][f"{convention}/{region}"] = row
            extra = (f"  (marginal baseline {row['marginal_baseline_aar']*100:.2f}%)"
                     if region == "jmotif" else "")
            print(f"[{label}] {convention:9s} {region:7s} AAR={row['aar']*100:6.2f}%  "
                  f"exact={row['exact_motif']*100:6.2f}%  n={len(ok)}/{len(truth)}{extra}")
            print("    e.g. " + "  ".join(f"{d['truth']}->{d['pred']}"
                                          for d in row["examples"][:5]))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
