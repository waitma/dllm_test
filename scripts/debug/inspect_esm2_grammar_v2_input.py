"""Inspect ESM2 grammar-v2 data input to diagnose the step-1 NaN.

Run:
    source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/inspect_esm2_grammar_v2_input.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.data import GrammarDataModule  # noqa: E402
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (  # noqa: E402
    apply_decoder_corruption_to_encoder,
    sample_bioseq_diffusion_noise,
)

ESM2_MASK_ID = 32
ESM2_VOCAB = 33


def build_loader():
    args = SimpleNamespace(
        sources="oas,ots,tcr,ppi",
        batch_size=4,
        grammar_data_dir=PROJECT_ROOT / "data/bioseq_grammar_v1",
        oas_weight=3.9,
        ots_weight=3.6,
        ppi_weight=1.4,
        tcr_weight=1.0,
        split="train",
        source_seed=0,
        max_sequence_length=2112,
        num_workers=0,
        tokenizer_path=None,  # ESM2 path: Esm2SequenceTokenizer
    )
    dm = GrammarDataModule.from_args(args)
    tokenizer = dm.build_tokenizer()
    loader = dm.loader(tokenizer, split="train", source_seed=0, epoch_size=64)
    return dm, tokenizer, loader


def describe_batch(step, batch, tokenizer):
    enc = batch["encoder_input_ids"]
    enc_attn = batch["encoder_attention_mask"]
    enc_res = batch["encoder_residue_mask"]
    enc_chain = batch["encoder_chain_mask"]
    dec = batch["input_ids"]
    print(f"\n===== step {step} =====")
    print("tasks:", batch.get("task_type_ids").tolist() if "task_type_ids" in batch else "n/a")
    print(f"decoder input_ids: shape={tuple(dec.shape)} min={int(dec.min())} max={int(dec.max())} "
          f"(decoder vocab={tokenizer.vocab_size})")
    print(f"encoder_input_ids: shape={tuple(enc.shape)} min={int(enc.min())} max={int(enc.max())} "
          f"(ESM2 vocab={ESM2_VOCAB})")
    over = (enc >= ESM2_VOCAB).sum().item()
    print(f"  encoder ids >= {ESM2_VOCAB} (OUT OF RANGE for ESM2): {over}")
    # per-row attention sums
    flat_attn = enc_attn.reshape(-1, enc_attn.shape[-1])
    row_sums = flat_attn.sum(-1)
    print(f"  encoder rows: {flat_attn.shape[0]}, min attended per row={int(row_sums.min())}, "
          f"rows fully masked (sum==0): {int((row_sums == 0).sum())}")
    print(f"  chain_mask sum={int(enc_chain.sum())}/{enc_chain.numel()}")

    # Now mimic training: sample noise and corrupt encoder
    noised_input_ids, labels, corruption_mask, timesteps = sample_bioseq_diffusion_noise(
        batch=batch, mask_token_id=ESM2_MASK_ID, time_epsilon=1e-3
    )
    noised_enc = apply_decoder_corruption_to_encoder(
        batch=batch, corruption_mask=corruption_mask, mask_token_id=ESM2_MASK_ID
    )
    print(f"  timesteps: min={float(timesteps.min()):.3f} max={float(timesteps.max()):.3f}")
    # For each encoder row, compute mask_ratio_observed = (#mask among attended) / (#attended)
    n_b, n_c, n_l = noised_enc.shape
    flat_ids = noised_enc.reshape(-1, n_l)
    is_mask = (flat_ids == ESM2_MASK_ID).int()
    attended = flat_attn.int()
    mask_in_attended = (is_mask * attended).sum(-1)
    attended_count = attended.sum(-1).clamp_min(0)
    # token_dropout denom = attended_count; ratio uses src_lengths=attended_count
    ratio = torch.where(attended_count > 0, mask_in_attended.float() / attended_count.float(),
                        torch.zeros_like(attended_count, dtype=torch.float32))
    full_mask_rows = ((ratio >= 1.0) & (attended_count > 0)).sum().item()
    near_full = ((ratio >= 0.99) & (attended_count > 0)).sum().item()
    print(f"  noised encoder rows with mask_ratio==1.0 (token_dropout -> div0): {full_mask_rows}")
    print(f"  noised encoder rows with mask_ratio>=0.99: {near_full}")
    print(f"  max mask_ratio over rows: {float(ratio.max()):.4f}")
    return full_mask_rows


def main():
    torch.manual_seed(0)
    dm, tokenizer, loader = build_loader()
    print("tokenizer:", type(tokenizer.base_tokenizer).__name__,
          "vocab_size=", tokenizer.vocab_size,
          "mask_id=", tokenizer.mask_token_id,
          "pad_id=", tokenizer.pad_token_id,
          "cls_id=", tokenizer.cls_token_id,
          "eos_id=", tokenizer.eos_token_id)

    it = iter(loader)
    total_full = 0
    for step in range(6):
        batch = next(it)
        total_full += describe_batch(step, batch, tokenizer)
    print(f"\nTOTAL fully-masked encoder rows across 6 steps: {total_full}")


if __name__ == "__main__":
    main()
