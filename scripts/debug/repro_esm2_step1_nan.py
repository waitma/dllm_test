"""Reproduce the ESM2 grammar-v2 step-1 NaN on a single GPU.

Mirrors the DDP trainer's per-step math (noise, encoder corruption, forward,
loss, backward, grad-clip, optimizer) to localize where non-finite values arise.

Run:
    source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/repro_esm2_step1_nan.py
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
    BioSeqDiffusionTransformerConfig,
    BioSeqEncoderDiffusionModel,
    apply_decoder_corruption_to_encoder,
    compute_masked_cross_entropy,
    forbidden_diffusion_target_token_ids,
    sample_bioseq_diffusion_noise,
)

ESM2_DIR = PROJECT_ROOT / "model_weights/esm2/esm2_t33_650M_UR50D"
DEVICE = "cuda:0"


def build():
    args = SimpleNamespace(
        sources="oas,ots,tcr,ppi",
        batch_size=4,
        grammar_data_dir=PROJECT_ROOT / "data/bioseq_grammar_v1",
        oas_weight=3.9, ots_weight=3.6, ppi_weight=1.4, tcr_weight=1.0,
        split="train", source_seed=0, max_sequence_length=2112,
        num_workers=0, tokenizer_path=None,
    )
    dm = GrammarDataModule.from_args(args)
    tokenizer = dm.build_tokenizer()
    loader = dm.loader(tokenizer, split="train", source_seed=0, epoch_size=64)
    config = BioSeqDiffusionTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=1280,
        num_hidden_layers=28,
        num_attention_heads=16,
        intermediate_size=5120,
        dropout=0.1,
        max_position_embeddings=2304,
        pad_token_id=int(tokenizer.pad_token_id),
        mask_token_id=int(tokenizer.mask_token_id),
        qk_norm=True,
        gradient_checkpointing=True,
    )
    model = BioSeqEncoderDiffusionModel.from_hf_encoder(
        decoder_config=config, encoder_name_or_path=str(ESM2_DIR),
        local_files_only=True, trust_remote_code=True, freeze_encoder=False,
    )
    return dm, tokenizer, loader, config, model


def to_dev(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


def check(name, t):
    if not torch.is_tensor(t):
        return
    f = torch.isfinite(t).all().item()
    print(f"    [{'OK ' if f else 'NAN'}] {name}: finite={f} "
          f"min={t.float().min().item():.4g} max={t.float().max().item():.4g} "
          f"absmax={t.float().abs().max().item():.4g}")


def run_step(model, config, batch, tag, bf16=True):
    print(f"\n--- {tag} (bf16={bf16}) ---")
    noised_input_ids, labels, corruption_mask, timesteps = sample_bioseq_diffusion_noise(
        batch=batch, mask_token_id=config.mask_token_id, time_epsilon=1e-3
    )
    noised_encoder_input_ids = apply_decoder_corruption_to_encoder(
        batch=batch, corruption_mask=corruption_mask, mask_token_id=config.mask_token_id
    )
    kwargs = dict(
        input_ids=noised_input_ids,
        attention_mask=batch.get("attention_mask"),
        position_ids_inner=batch.get("position_ids_inner"),
        position_ids_chain=batch.get("position_ids_chain"),
        timesteps=timesteps,
        chain_ids=batch.get("chain_ids"),
        residue_mask=batch.get("residue_mask"),
        encoder_input_ids=noised_encoder_input_ids,
        encoder_attention_mask=batch.get("encoder_attention_mask"),
        encoder_residue_mask=batch.get("encoder_residue_mask"),
        encoder_chain_mask=batch.get("encoder_chain_mask"),
    )
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cuda", enabled=False)
    with ctx:
        # Inspect encoder output directly first
        chain_cond = model.encode_chain_tokens(
            encoder_input_ids=noised_encoder_input_ids,
            encoder_attention_mask=batch.get("encoder_attention_mask"),
            encoder_residue_mask=batch.get("encoder_residue_mask"),
            encoder_chain_mask=batch.get("encoder_chain_mask"),
        )
        check("encoder chain_token_condition", chain_cond)
        output = model(**kwargs)
        check("decoder logits", output.logits)
        forbidden = forbidden_diffusion_target_token_ids(config)
        loss = compute_masked_cross_entropy(
            output.logits, labels, loss_norm=config.loss_norm, forbidden_token_ids=forbidden
        )
    print(f"    loss={loss.item():.4f} finite={torch.isfinite(loss).item()}")
    return loss


def main():
    torch.manual_seed(0)
    dm, tokenizer, loader, config, model = build()
    model = model.to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.decoder.parameters() if p.requires_grad], "lr": 1e-4},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": 2e-5},
        ],
        weight_decay=0.0,
    )
    it = iter(loader)

    # Step 0: 4 micro-batches (grad_accum=4), then clip + opt.step
    print("==================== STEP 0 ====================")
    opt.zero_grad(set_to_none=True)
    for micro in range(4):
        batch = to_dev(next(it))
        loss = run_step(model, config, batch, f"step0 micro{micro}")
        (loss / 4).backward()
    # grad norm
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"\nstep0 grad_norm (pre-clip) = {gn.item():.4f}  finite={torch.isfinite(gn).item()}")
    # check for non-finite grads
    bad = []
    for n, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad.append(n)
    print(f"params with non-finite grad: {len(bad)}")
    for n in bad[:20]:
        print("   ", n)
    opt.step()
    # check weights after step
    badw = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    print(f"params with non-finite WEIGHTS after opt.step: {len(badw)}")
    for n in badw[:20]:
        print("   ", n)

    # Step 1
    print("\n==================== STEP 1 ====================")
    opt.zero_grad(set_to_none=True)
    for micro in range(4):
        batch = to_dev(next(it))
        loss = run_step(model, config, batch, f"step1 micro{micro}")
        if not torch.isfinite(loss):
            print("    >>> non-finite loss at step1; stopping detailed trace")
            break


if __name__ == "__main__":
    main()
