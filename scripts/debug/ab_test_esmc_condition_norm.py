"""Controlled A/B test: does ESMC condition LayerNorm fix the plateau?

Same data, same seed, single GPU. Two arms trained from identical init:
  A: baseline (ESMC post-norm condition, replacement injection) -- current code
  B: + LayerNorm(condition) before injection (proposed fix, gamma init=1)
  C: zero condition (reference: decoder gets nothing at residue positions)

If B's train loss drops well below A while A plateaus high, the root cause
(ESMC final-norm gamma~0.038 attenuating features+grads) and the fix are confirmed.

Run:
    export PATH="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin:$PATH"
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/ab_test_esmc_condition_norm.py
"""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dllm.pipelines.qwen3_vl_arch.data.mixture as mixture  # noqa: E402
from dllm.pipelines.qwen3_vl_arch.data import GrammarDataModule  # noqa: E402
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (  # noqa: E402
    BioSeqDiffusionTransformerConfig,
    BioSeqEncoderDiffusionModel,
    apply_decoder_corruption_to_encoder,
    compute_masked_cross_entropy,
    forbidden_diffusion_target_token_ids,
    sample_bioseq_diffusion_noise,
)

ESMC_DIR = PROJECT_ROOT / "model_weights/esmc/ESMC-300M"
DEVICE = "cuda:0"
STEPS = 1000
LOG_EVERY = 50
SEED = 42
ARMS = ["A", "B", "C"]


def build_args():
    return SimpleNamespace(
        sources="oas,ots,tcr,ppi", batch_size=4,
        grammar_data_dir=PROJECT_ROOT / "data/bioseq_grammar_v1",
        oas_weight=3.9, ots_weight=3.6, ppi_weight=1.4, tcr_weight=1.0,
        split="train", source_seed=0, max_sequence_length=2112, num_workers=0,
        tokenizer_path=ESMC_DIR,
    )


def to_dev(b):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}


def build_model(tok):
    config = BioSeqDiffusionTransformerConfig(
        vocab_size=tok.vocab_size, hidden_size=960, num_hidden_layers=28,
        num_attention_heads=16, intermediate_size=3840, dropout=0.1,
        max_position_embeddings=2304, pad_token_id=int(tok.pad_token_id),
        mask_token_id=int(tok.mask_token_id), qk_norm=True, gradient_checkpointing=True,
    )
    model = BioSeqEncoderDiffusionModel.from_esmc(
        decoder_config=config, encoder_name_or_path=str(ESMC_DIR),
        local_files_only=True, trust_remote_code=True, freeze_encoder=False,
    )
    return config, model


def install_cond_hook(model, mode, hidden):
    """mode in {A, B, C}. Patches decoder.forward to transform encoder_condition."""
    decoder = model.decoder
    orig_forward = BioSeqEncoderDiffusionModel.__dict__  # not used
    base_decoder_forward = decoder.forward

    cond_norm = None
    if mode == "B":
        cond_norm = nn.LayerNorm(hidden).to(DEVICE)
    elif mode == "R":  # RMSNorm: scale-to-unit + learnable per-dim gamma, NO centering, NO beta
        cond_norm = nn.RMSNorm(hidden).to(DEVICE)

    def patched(self, *args, encoder_condition=None, **kwargs):
        if encoder_condition is not None:
            if mode in ("B", "R"):
                encoder_condition = cond_norm(encoder_condition)
            elif mode == "S":  # fixed scalar scale-up to ~unit, no params, no centering
                encoder_condition = encoder_condition * 24.0
            elif mode == "C":
                encoder_condition = torch.zeros_like(encoder_condition)
        return base_decoder_forward(*args, encoder_condition=encoder_condition, **kwargs)

    decoder.forward = types.MethodType(patched, decoder)
    return cond_norm


def run_arm(mode, init_state, tok, config):
    torch.manual_seed(SEED)
    model = BioSeqEncoderDiffusionModel.from_esmc(
        decoder_config=config, encoder_name_or_path=str(ESMC_DIR),
        local_files_only=True, trust_remote_code=True, freeze_encoder=False,
    ).to(DEVICE)
    model.load_state_dict(init_state)
    model.train()
    cond_norm = install_cond_hook(model, mode, config.hidden_size)

    params = [
        {"params": list(model.decoder.parameters()), "lr": 1e-4},
        {"params": list(model.encoder.parameters()), "lr": 2e-5},
    ]
    if cond_norm is not None:
        params.append({"params": list(cond_norm.parameters()), "lr": 1e-4})
    opt = torch.optim.AdamW(params)

    mixture.distributed_worker_shard = (lambda: (0, 1))
    torch.manual_seed(SEED)
    loader = dm.loader(tok, split="train", source_seed=0, epoch_size=None)
    it = iter(loader)
    forbidden = forbidden_diffusion_target_token_ids(config)
    losses = []
    running = []
    step = 0
    while step < STEPS:
        batch = to_dev(next(it))
        nids, labels, cmask, ts = sample_bioseq_diffusion_noise(
            batch=batch, mask_token_id=config.mask_token_id, time_epsilon=1e-3,
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
            loss = compute_masked_cross_entropy(out.logits, labels, loss_norm=config.loss_norm, forbidden_token_ids=forbidden)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running.append(float(loss.item()))
        if (step + 1) % LOG_EVERY == 0:
            avg = sum(running) / len(running)
            losses.append((step + 1, avg))
            print(f"  [{mode}] step {step+1:4d}  loss(avg{LOG_EVERY})={avg:.4f}", flush=True)
            running = []
        step += 1
    del model, opt
    torch.cuda.empty_cache()
    return losses


if __name__ == "__main__":
    args = build_args()
    dm = GrammarDataModule.from_args(args)
    tok = dm.build_tokenizer()
    config, init_model = build_model(tok)
    init_state = copy.deepcopy(init_model.state_dict())
    del init_model
    print(f"ESMC final-norm gamma mean = {0.0378:.4f} (measured). STEPS={STEPS}, seed={SEED}")
    results = {}
    for mode in ARMS:
        print(f"=== Arm {mode} ===", flush=True)
        results[mode] = run_arm(mode, init_state, tok, config)
    print("\n=== SUMMARY (avg train loss) ===")
    labels = {"A": "A raw", "S": "S scale24", "R": "R RMSNorm", "B": "B LayerNorm", "C": "C zero(no-enc)"}
    steps = [s for s, _ in results[ARMS[0]]]
    header = "step  | " + "".join(f"{labels[m]:>11}" for m in ARMS)
    print(header)
    for i, s in enumerate(steps):
        row = f"{s:5d} | " + "".join(f"{results[m][i][1]:11.4f}" for m in ARMS)
        print(row)
