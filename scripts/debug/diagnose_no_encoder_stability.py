"""Stress-test BioSeq no-encoder diffusion training for numerical stability.

Reproduces the 300M/600M/1B failure mode (degeneration to predicting the
``<mask>`` token at corrupted positions) and verifies that excluding special
tokens from the denoising objective removes that attractor.

Run (single GPU):
    /vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/python \
        /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/diagnose_no_encoder_stability.py \
        --size mid --steps 1500 --lr 3e-4 --forbidden-mask 0

Compare with ``--forbidden-mask 1`` to confirm the fix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader  # noqa: E402

from dllm.pipelines.qwen3_vl_arch.data import (  # noqa: E402
    BioSeqQwenDataCollator,
    BioSeqViewSampler,
    Esm2SequenceTokenizer,
    SourceWithWeight,
    WeightedMixtureDataset,
    default_source_configs,
    source_from_config,
)
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (  # noqa: E402
    BioSeqDiffusionTransformerConfig,
    BioSeqNoEncoderDiffusionModel,
    compute_masked_cross_entropy,
    forbidden_diffusion_target_token_ids,
    sample_bioseq_diffusion_noise,
)

SIZES = {
    "38m": dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048),
    "mid": dict(hidden_size=768, num_hidden_layers=16, num_attention_heads=12, intermediate_size=3072),
    "300m": dict(hidden_size=1024, num_hidden_layers=18, num_attention_heads=16, intermediate_size=4096),
    "600m": dict(hidden_size=1536, num_hidden_layers=16, num_attention_heads=24, intermediate_size=6144),
    "1b": dict(hidden_size=1792, num_hidden_layers=20, num_attention_heads=28, intermediate_size=7168),
}


def build_loader(args, tokenizer):
    configs = [c for c in default_source_configs(split="train", max_records=args.limit_per_source) if c.name in args.sources.split(",")]
    sources = [SourceWithWeight(source_from_config(c), 1.0) for c in configs]
    records = WeightedMixtureDataset(sources, epoch_size=None, seed=0)
    collator = BioSeqQwenDataCollator(
        tokenizer=tokenizer,
        view_sampler=BioSeqViewSampler(allowed_views=("full_denoise",), seed=0),
        max_chain_length=args.max_chain_length,
        max_sequence_length=args.max_sequence_length,
        single_view_per_batch=False,
        require_homogeneous_task=False,
    )
    return DataLoader(records, batch_size=args.batch_size, collate_fn=collator, num_workers=0, drop_last=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(SIZES), default="mid")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="constant")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--forbidden-mask", type=int, choices=[0, 1], default=1)
    p.add_argument("--qk-norm", type=int, choices=[0, 1], default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--sources", type=str, default="oas,ots,nanobody,processed_v2")
    p.add_argument("--limit-per-source", type=int, default=4096)
    p.add_argument("--max-chain-length", type=int, default=320)
    p.add_argument("--max-sequence-length", type=int, default=1024)
    p.add_argument("--vocab-size", type=int, default=64)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    tokenizer = Esm2SequenceTokenizer()
    cfg = BioSeqDiffusionTransformerConfig(
        vocab_size=args.vocab_size,
        **SIZES[args.size],
        dropout=0.1,
        max_position_embeddings=4096,
        mask_token_id=32,
        pad_token_id=1,
        qk_norm=bool(args.qk_norm),
        gradient_checkpointing=True,
    )
    model = BioSeqNoEncoderDiffusionModel(cfg).to(device).train()
    nparams = sum(p.numel() for p in model.parameters())
    forbidden = forbidden_diffusion_target_token_ids(cfg) if args.forbidden_mask else None
    print(
        f"size={args.size} params={nparams/1e6:.1f}M device={device} lr={args.lr} sched={args.lr_scheduler} "
        f"forbidden_mask={args.forbidden_mask} qk_norm={args.qk_norm} mask_id={cfg.mask_token_id}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loader = build_loader(args, tokenizer)
    import math

    def lr_at(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        if args.lr_scheduler == "cosine":
            prog = min(max((step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1), 0.0), 1.0)
            return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))
        return args.lr

    step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    worst_loss = 0.0
    collapsed_at = None
    while step < args.steps:
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.bf16 else torch.autocast(device_type="cpu", enabled=False):
                noised, labels, corr, t = sample_bioseq_diffusion_noise(batch, mask_token_id=cfg.mask_token_id, time_epsilon=cfg.time_epsilon)
                out = model(
                    input_ids=noised,
                    attention_mask=batch.get("attention_mask"),
                    chain_role_ids=batch.get("chain_role_ids"),
                    position_ids_inner=batch.get("position_ids_inner"),
                    position_ids_chain=batch.get("position_ids_chain"),
                    task_type_ids=batch.get("task_type_ids"),
                    timesteps=t,
                )
                loss = compute_masked_cross_entropy(out.logits, labels, loss_norm=cfg.loss_norm, forbidden_token_ids=forbidden)
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf"))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step % args.log_interval == 0:
                    with torch.no_grad():
                        cm = corr.bool()
                        raw_pred = out.logits.float().argmax(-1)
                        frac_mask = (raw_pred[cm] == cfg.mask_token_id).float().mean().item() if cm.any() else 0.0
                        lmax = out.logits.float().abs().max().item()
                    worst_loss = max(worst_loss, float(loss))
                    if collapsed_at is None and frac_mask > 0.5:
                        collapsed_at = step
                    print(
                        f"step={step:5d} loss={float(loss):8.4f} grad_norm={float(gnorm):8.3f} "
                        f"logits_max={lmax:8.2f} raw_pred_mask_frac={frac_mask:.3f} lr={optimizer.param_groups[0]['lr']:.2e}",
                        flush=True,
                    )
                    if not torch.isfinite(gnorm):
                        print(f"NONFINITE grad at step={step}", flush=True)
                step += 1
                if step >= args.steps:
                    break
    print(f"DONE size={args.size} forbidden_mask={args.forbidden_mask} worst_loss={worst_loss:.3f} collapsed_at={collapsed_at}", flush=True)


if __name__ == "__main__":
    main()
