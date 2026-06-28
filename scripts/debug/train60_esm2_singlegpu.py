"""Faithful single-GPU training loop (60 steps) to test ESM2 grammar-v2 stability.

Forward+backward+clip+AdamW with the real LR schedule, train mode, gradient
checkpointing. Reports any non-finite loss/grad.

Run:
    source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/train60_esm2_singlegpu.py
"""

from __future__ import annotations

import math
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
WARMUP, MAXSTEPS, BASE_LR, ENC_LR, MINR = 1000, 10000, 1e-4, 2e-5, 0.1


def lr_at(step, base):
    if step < WARMUP:
        return base * float(step + 1) / WARMUP
    progress = min(max((step - WARMUP) / (MAXSTEPS - WARMUP), 0.0), 1.0)
    return base * (MINR + (1 - MINR) * 0.5 * (1 + math.cos(math.pi * progress)))


def build():
    args = SimpleNamespace(
        sources="oas,ots,tcr,ppi", batch_size=4,
        grammar_data_dir=PROJECT_ROOT / "data/bioseq_grammar_v1",
        oas_weight=3.9, ots_weight=3.6, ppi_weight=1.4, tcr_weight=1.0,
        split="train", source_seed=0, max_sequence_length=2112,
        num_workers=0, tokenizer_path=None,
    )
    dm = GrammarDataModule.from_args(args)
    tok = dm.build_tokenizer()
    config = BioSeqDiffusionTransformerConfig(
        vocab_size=tok.vocab_size, hidden_size=1280, num_hidden_layers=28,
        num_attention_heads=16, intermediate_size=5120, dropout=0.1,
        max_position_embeddings=2304, pad_token_id=int(tok.pad_token_id),
        mask_token_id=int(tok.mask_token_id), qk_norm=True, gradient_checkpointing=True,
    )
    model = BioSeqEncoderDiffusionModel.from_hf_encoder(
        decoder_config=config, encoder_name_or_path=str(ESM2_DIR),
        local_files_only=True, trust_remote_code=True, freeze_encoder=False,
    ).to(DEVICE)
    model.train()
    loader = dm.loader(tok, split="train", source_seed=0, epoch_size=None)
    return dm, tok, config, model, loader


def to_dev(b):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}


def micro(model, config, batch):
    nids, labels, cmask, ts = sample_bioseq_diffusion_noise(
        batch=batch, mask_token_id=config.mask_token_id, time_epsilon=1e-3
    )
    nenc = apply_decoder_corruption_to_encoder(batch=batch, corruption_mask=cmask, mask_token_id=config.mask_token_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=nids, attention_mask=batch.get("attention_mask"),
            position_ids_inner=batch.get("position_ids_inner"),
            position_ids_chain=batch.get("position_ids_chain"), timesteps=ts,
            chain_ids=batch.get("chain_ids"), residue_mask=batch.get("residue_mask"),
            encoder_input_ids=nenc, encoder_attention_mask=batch.get("encoder_attention_mask"),
            encoder_residue_mask=batch.get("encoder_residue_mask"),
            encoder_chain_mask=batch.get("encoder_chain_mask"),
        )
        forbidden = forbidden_diffusion_target_token_ids(config)
        loss = compute_masked_cross_entropy(out.logits, labels, loss_norm=config.loss_norm, forbidden_token_ids=forbidden)
    return loss


def main():
    torch.manual_seed(42)
    dm, tok, config, model, loader = build()
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.decoder.parameters() if p.requires_grad], "lr": BASE_LR, "initial_lr": BASE_LR},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": ENC_LR, "initial_lr": ENC_LR},
        ],
        weight_decay=0.0,
    )
    it = iter(loader)
    accum = 4
    for step in range(60):
        opt.param_groups[0]["lr"] = lr_at(step, BASE_LR)
        opt.param_groups[1]["lr"] = lr_at(step, ENC_LR)
        opt.zero_grad(set_to_none=True)
        losses = []
        nonfinite = False
        for m in range(accum):
            batch = to_dev(next(it))
            loss = micro(model, config, batch)
            if not torch.isfinite(loss):
                print(f"!!! step {step} micro {m}: NON-FINITE loss={loss.item()} "
                      f"task={int(batch['task_type_ids'][0])} enc={tuple(batch['encoder_input_ids'].shape)}")
                nonfinite = True
                break
            (loss / accum).backward()
            losses.append(loss.item())
        if nonfinite:
            print("STOP: reproduced non-finite loss")
            return
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            print(f"!!! step {step}: non-finite grad_norm={gn.item()}")
            return
        opt.step()
        if step % 5 == 0 or step < 3:
            print(f"step {step}: loss={sum(losses)/len(losses):.4f} grad_norm={gn.item():.2f} "
                  f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
    print("\n60 steps completed with no non-finite loss/grad (single GPU)")


if __name__ == "__main__":
    main()
