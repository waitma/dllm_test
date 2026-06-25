"""Probe step-0 loss finiteness for grammar no_encoder training across rank seeds.

Run:
  conda activate protenix_abtcr
  python scripts/debug/probe_no_encoder_step0_nan.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.bioseq.train_qwen3_vl_bioseq_ddp import (
    build_config,
    build_loader,
    build_model,
    build_tokenizer,
    compute_training_output,
)


def make_args() -> Namespace:
    return Namespace(
        tokenizer_path=PROJECT_ROOT / "model_weights/esmc/ESMC-300M",
        grammar_data_dir=PROJECT_ROOT / "data/bioseq_grammar_v1",
        model_type="no_encoder",
        vocab_size=None,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        max_position_embeddings=2304,
        max_chain_positions=64,
        max_chain_roles=32,
        max_task_types=32,
        time_epsilon=0.001,
        loss_norm="token",
        qk_norm=False,
        gradient_checkpointing=True,
        initializer_range=0.02,
        sources="oas,ots,tcr,ppi",
        oas_weight=3.9,
        ots_weight=3.6,
        ppi_weight=1.4,
        tcr_weight=1.0,
        nanobody_weight=1.0,
        processed_v2_weight=1.0,
        split="train",
        source_seed=0,
        epoch_size=None,
        limit_per_source=None,
        batch_size=16,
        num_workers=0,
        max_sequence_length=2112,
        deduplicate_within_batch=False,
        seed=42,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--bf16", action="store_true", default=True)
    args_cli = parser.parse_args()

    device = torch.device(args_cli.device)
    args = make_args()
    tokenizer = build_tokenizer(args)
    print(
        f"vocab={tokenizer.vocab_size} mask_id={tokenizer.mask_token_id} "
        f"pad_id={tokenizer.pad_token_id}"
    )
    config = build_config(args, tokenizer)
    forbidden = config.forbidden_target_token_ids
    print(f"config mask={config.mask_token_id} forbidden={forbidden}")

    model = build_model(args, config).to(device).train()

    for rank in range(args_cli.ranks):
        loader = build_loader(args, tokenizer, split="train", source_seed=rank)
        for batch_idx in range(args_cli.batches):
            batch = next(iter(loader))
            eligible = batch.get("diffusion_eligible_mask", batch.get("diffusion_loss_mask"))
            eligible_counts = eligible.sum(dim=1).tolist()
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            try:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args_cli.bf16):
                    output = compute_training_output(model, model, batch)
                loss_val = float(output.loss.item())
                finite = bool(torch.isfinite(output.loss).item())
                logits_finite = bool(torch.isfinite(output.logits).all().item())
                print(
                    f"rank_seed={rank} batch={batch_idx} loss={loss_val:.4f} finite={finite} "
                    f"logits_finite={logits_finite} eligible_min={min(eligible_counts)} "
                    f"eligible_max={max(eligible_counts)} seq_len={batch['input_ids'].shape[1]}"
                )
                if not finite:
                    bad = (~torch.isfinite(output.logits)).nonzero(as_tuple=False)[:5]
                    print("  nonfinite logits indices (first 5):", bad.tolist())
                    return
            except Exception as exc:
                print(f"rank_seed={rank} batch={batch_idx} ERROR {type(exc).__name__}: {exc}")
                return


if __name__ == "__main__":
    main()
