"""Is v3's 126k-token vocabulary a handicap on a ~20-way problem?

The two model generations differ in output vocabulary, not just in data:

  esmc189000  decoder.wte = (49, 960)   -- a compact residue-level vocabulary
  v3          reuses the full LLaDA text vocabulary (mask id 126336) with a
              grammar<->LLaDA id remap

At inference ``_model_logits`` calls ``mask_forbidden_target_logits``, which
zeroes everything outside the legal target set, so a miscalibrated model is
rescued after the fact. During training there is no such rescue. If v3 leaves a
large share of its raw probability mass outside the 20 amino acids, then the
softmax is poorly calibrated over its own vocabulary and the forbidden-token
mask is doing real work -- which would make the oversized vocabulary a genuine
optimisation handicap rather than a cosmetic difference.

Reports, at masked CDR3b residue positions from ONE forward pass:
  * raw probability mass on the 20 amino-acid tokens vs elsewhere
  * how often the raw argmax is not even a legal residue
  * how often masking changes the argmax
"""

from __future__ import annotations

import argparse
import os
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, BioSeqRecord  # noqa: E402
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (  # noqa: E402
    BioSeqEncoderDiffusionModel,
    apply_decoder_corruption_to_encoder,
    forbidden_diffusion_target_token_ids,
    mask_forbidden_target_logits,
)
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import (  # noqa: E402
    build_generation_mask,
    initialize_output_tokens,
    resolve_partial_mask,
)
from downstream.grammar.common import (  # noqa: E402
    _inverse_remap_llada_tokens,
    build_eval_collator,
    load_grammar_checkpoint,
)
from downstream.grammar.masks import (  # noqa: E402
    residue_positions_by_chain,
    tcr_generation_partial_mask,
)

AA = "ACDEFGHIKLMNPQRSTVWY"
PAPERS = ROOT / "data" / "tcr_papers_v2" / "dataset"


def both_logits(model, batch, out_tokens, gen_mask, mask_id, partial, t):
    """Return (raw, masked) logits from the SAME code path as generation.

    Reimplementing ``_model_logits`` risks passing a different kwarg set, so
    instead call it twice and neutralise the forbidden-token mask on the first
    call. That keeps the two tensors byte-comparable apart from the mask.
    """
    import dllm.pipelines.qwen3_vl_arch.sampling_bioseq as sb

    orig = sb.mask_forbidden_target_logits
    call = dict(model=model, batch=batch, output_tokens=out_tokens,
                generation_mask=gen_mask, mask_token_id=mask_id,
                timesteps=t, cfg_scale=0.0, partial_mask=partial)
    try:
        sb.mask_forbidden_target_logits = lambda logits, forbidden: logits
        raw = sb._model_logits(**call).clone()
    finally:
        sb.mask_forbidden_target_logits = orig
    masked = sb._model_logits(**call).clone()
    return raw, masked


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("-n", "--n", type=int, default=32)
    args = ap.parse_args()

    label = args.label or Path(args.checkpoint).name
    print(f"threads={_THREADS}  label={label}")

    df = pd.read_csv(PAPERS / "holdout.csv", low_memory=False)
    df = df[df["cdr3b"].notna() & df["epitope_seq"].notna() & df["cdr3a"].isna()
            & df["mhc_seq"].isna()]
    df = df[df["cdr3b"].astype(str).str.len() >= 11].sample(n=args.n, random_state=0)
    rows = [(str(r.cdr3b).strip().upper(), str(r.epitope_seq).strip().upper())
            for r in df.itertuples()]

    model, tok = load_grammar_checkpoint(args.checkpoint, device="cpu")
    col = build_eval_collator(model, tok)
    vocab = int(model.config.vocab_size) if hasattr(model.config, "vocab_size") else -1
    print(f"model={type(model).__name__}  vocab_size={vocab}  mask_id={model.config.mask_token_id}")

    recs = [BioSeqRecord(
        chains=[BioSeqChain(ep, "peptide"), BioSeqChain(cb, "tcr_beta")],
        task_type="tcr_epitope", source="probe") for cb, ep in rows]
    batch = {k: v for k, v in col(recs).items()}
    partial = resolve_partial_mask(batch, tcr_generation_partial_mask(batch, target_chain_indices=1))
    gen_mask = build_generation_mask(batch, partial)
    mask_id = int(model.config.mask_token_id)
    out_tokens, _ = initialize_output_tokens(batch["input_ids"], gen_mask, mask_id)

    lg, lg_masked = both_logits(model, batch, out_tokens, gen_mask, mask_id,
                                partial, torch.tensor([1.0]))
    forbidden = forbidden_diffusion_target_token_ids(model.config)

    # Which output ids are the 20 amino acids? The grammar tokenizer gives the
    # grammar ids; for the fusion models those are remapped into LLaDA id space,
    # so invert the model's own remap buffer rather than guessing.
    grammar_ids = tok.encode_residues(AA)
    inv = getattr(model, "llada_to_grammar_ids", None)
    aa_ids = set()
    for gid in grammar_ids:
        if inv is None:
            aa_ids.add(int(gid))
            continue
        hit = (inv == int(gid)).nonzero().flatten().tolist()
        aa_ids.update(int(h) for h in hit)
    aa_idx = torch.tensor(sorted(aa_ids), dtype=torch.long)
    print(f"legal amino-acid token ids found: {len(aa_ids)}  "
          f"forbidden-set size: {len(forbidden) if forbidden is not None else 0}")

    by_chain = residue_positions_by_chain(batch)
    probs = torch.softmax(lg.float(), dim=-1)
    mass, illegal, changed, tot = [], 0, 0, 0
    for i in range(len(recs)):
        for c in by_chain[i][1]:
            if not bool(gen_mask[i, c]):
                continue
            p = probs[i, c]
            mass.append(float(p.index_select(0, aa_idx).sum()))
            am_raw = int(lg[i, c].argmax())
            am_msk = int(lg_masked[i, c].argmax())
            illegal += int(am_raw not in aa_ids)
            changed += int(am_raw != am_msk)
            tot += 1
    print(f"\n masked CDR3b positions: {tot}")
    print(f"  raw 概率质量落在 20 种氨基酸上 : {np.mean(mass)*100:6.2f}%  "
          f"(中位 {np.median(mass)*100:.2f}%)")
    print(f"  raw argmax 不是合法残基的比例    : {illegal/max(1,tot)*100:6.2f}%")
    print(f"  forbidden-mask 改变了 argmax     : {changed/max(1,tot)*100:6.2f}%")


if __name__ == "__main__":
    main()
