"""Is a sequence's prediction affected by its batch-mates?

Both stored T4 artifacts were produced at large batch sizes
(``run_bioseq_all.py``: 256 unconditional / 100 conditional) while fresh runs at
smaller batch sizes score much better, and ``_sample_lengths`` puts CDR3b of
length 9..23 in one batch -- so every batch is padded.  If attention or position
ids leak across the pad boundary, output quality would degrade with batch size
and every recorded T4 number would be a padding artifact rather than a model
result.

This is a determinism test, not a sampling test: it compares raw logits for the
*same* target sequence when run alone versus padded next to much longer
batch-mates.  Identical logits (up to float noise) clear the data path; a
material difference localises a real bug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, BioSeqRecord  # noqa: E402
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import (  # noqa: E402
    _model_logits,
    build_generation_mask,
    initialize_output_tokens,
    resolve_partial_mask,
)
from downstream.grammar.common import build_eval_collator, load_grammar_checkpoint  # noqa: E402
from downstream.grammar.masks import (  # noqa: E402
    residue_positions_by_chain,
    tcr_generation_partial_mask,
)


def placeholder(length: int) -> str:
    return "C" + "A" * (length - 2) + "F"


@torch.no_grad()
def logits_for_target(model, collator, device, lengths: list[int], target_row: int):
    """Return logits at the target row's residue positions, plus that row's cols."""
    records = [
        BioSeqRecord(chains=[BioSeqChain(placeholder(L), "tcr_beta")],
                     task_type="tcr", source="probe")
        for L in lengths
    ]
    batch = collator(records)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    partial = tcr_generation_partial_mask(batch, target_chain_indices=0)
    partial = resolve_partial_mask(batch, partial)
    gen_mask = build_generation_mask(batch, partial)
    tokens, _ = initialize_output_tokens(batch["input_ids"], gen_mask,
                                         int(model.config.mask_token_id))
    timesteps = torch.tensor([1.0], device=device)
    logits = _model_logits(
        model=model, batch=batch, output_tokens=tokens, generation_mask=gen_mask,
        mask_token_id=int(model.config.mask_token_id), timesteps=timesteps,
        cfg_scale=0.0, partial_mask=partial)
    cols = residue_positions_by_chain(batch)[target_row].get(0, [])
    return logits[target_row, cols].float().cpu(), cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target-len", type=int, default=13)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model, tokenizer = load_grammar_checkpoint(args.checkpoint, device=device)
    model.eval()
    collator = build_eval_collator(model, tokenizer)
    T = args.target_len

    alone, cols_alone = logits_for_target(model, collator, device, [T], 0)
    print(f"target length {T}: {len(cols_alone)} residue positions, alone in batch")

    scenarios = {
        "padded with 3 much longer (len 23)": [T, 23, 23, 23],
        "padded with 7 mixed (9..23)":        [T, 9, 11, 14, 17, 20, 22, 23],
        "same length x4 (no padding)":        [T, T, T, T],
    }
    for label, lengths in scenarios.items():
        got, cols = logits_for_target(model, collator, device, lengths, 0)
        if cols != cols_alone:
            print(f"  {label:36s} POSITION SHIFT: {cols} vs {cols_alone}")
            continue
        delta = (got - alone).abs()
        # argmax agreement is the decision-relevant view; max-abs is the raw view
        agree = (got.argmax(-1) == alone.argmax(-1)).float().mean().item()
        print(f"  {label:36s} max|dlogit|={delta.max():.6f}  "
              f"mean|dlogit|={delta.mean():.6f}  argmax_agreement={agree*100:.1f}%")


if __name__ == "__main__":
    main()
