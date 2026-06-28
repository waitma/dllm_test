"""Definitive test on TRAINED checkpoints: does each model actually use its
encoder condition? Compares val loss with the condition intact vs zeroed out.

If a model relies on its condition, zeroing it spikes the loss. If the model
learned to ignore the condition (the ESMC failure hypothesis: the condition is
too small in the residual stream to matter), zeroing changes nothing.

Run:
    export PATH="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin:$PATH"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/condition_reliance_trained.py
"""

from __future__ import annotations

import gc
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

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

DEVICE = "cuda:0"
N_BATCHES = 30
SEED = 123

MODELS = {
    "ESM2-650M": dict(
        ckpt=PROJECT_ROOT / "output/grammar_v2_esm2_650m/best.pt",
        encoder=PROJECT_ROOT / "model_weights/esm2/esm2_t33_650M_UR50D",
        kind="esm2", tokenizer_path=None,
        hidden=1280, layers=28, heads=16, inter=5120,
    ),
    "ESMC-300M": dict(
        ckpt=PROJECT_ROOT / "output/grammar_v2_esmc300m/best.pt",
        encoder=PROJECT_ROOT / "model_weights/esmc/ESMC-300M",
        kind="esmc", tokenizer_path=PROJECT_ROOT / "model_weights/esmc/ESMC-300M",
        hidden=960, layers=28, heads=16, inter=3840,
    ),
    "ESMC-600M": dict(
        ckpt=PROJECT_ROOT / "output/grammar_v2_esmc600m/best.pt",
        encoder=PROJECT_ROOT / "model_weights/esmc/ESMC-600M",
        kind="esmc", tokenizer_path=PROJECT_ROOT / "model_weights/esmc/ESMC-600M",
        hidden=1152, layers=28, heads=18, inter=4608,
    ),
}


def to_dev(b):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}


def install_zero_hook(model, zero: bool, scale: float = 1.0):
    decoder = model.decoder
    base = decoder.forward

    def patched(self, *args, encoder_condition=None, **kwargs):
        if encoder_condition is not None:
            if zero:
                encoder_condition = torch.zeros_like(encoder_condition)
            elif scale != 1.0:
                encoder_condition = encoder_condition * scale
        return base(*args, encoder_condition=encoder_condition, **kwargs)

    decoder.forward = types.MethodType(patched, decoder)
    return base


def restore(model, base):
    model.decoder.forward = base


@torch.no_grad()
def eval_loss(model, config, batches, *, zero=False, scale=1.0):
    base = install_zero_hook(model, zero, scale)
    forbidden = forbidden_diffusion_target_token_ids(config)
    tot, cnt = 0.0, 0
    cond_absmax = 0.0
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for batch in batches:
        torch.manual_seed(SEED + cnt)  # same noise across conditions per batch index
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
            if out.encoder_condition is not None:
                cond_absmax = max(cond_absmax, float(out.encoder_condition.float().abs().max()))
        tot += float(loss.item()); cnt += 1
    restore(model, base)
    return tot / max(cnt, 1), cond_absmax


def run_model(name, cfg):
    print(f"\n=== {name} ===", flush=True)
    args = SimpleNamespace(
        sources="oas,ots,tcr,ppi", batch_size=4,
        grammar_data_dir=PROJECT_ROOT / "data/bioseq_grammar_v1",
        oas_weight=3.9, ots_weight=3.6, ppi_weight=1.4, tcr_weight=1.0,
        split="valid", source_seed=0, max_sequence_length=2112, num_workers=0,
        tokenizer_path=cfg["tokenizer_path"],
    )
    dm = GrammarDataModule.from_args(args)
    tok = dm.build_tokenizer()
    config = BioSeqDiffusionTransformerConfig(
        vocab_size=tok.vocab_size, hidden_size=cfg["hidden"], num_hidden_layers=cfg["layers"],
        num_attention_heads=cfg["heads"], intermediate_size=cfg["inter"], dropout=0.1,
        max_position_embeddings=2304, pad_token_id=int(tok.pad_token_id),
        mask_token_id=int(tok.mask_token_id), qk_norm=True, gradient_checkpointing=False,
    )
    if cfg["kind"] == "esm2":
        model = BioSeqEncoderDiffusionModel.from_hf_encoder(
            decoder_config=config, encoder_name_or_path=str(cfg["encoder"]),
            local_files_only=True, trust_remote_code=True, freeze_encoder=False)
    else:
        model = BioSeqEncoderDiffusionModel.from_esmc(
            decoder_config=config, encoder_name_or_path=str(cfg["encoder"]),
            local_files_only=True, trust_remote_code=True, freeze_encoder=False)
    payload = torch.load(cfg["ckpt"], map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"  load: missing={len(missing)} unexpected={len(unexpected)} (step={payload.get('step')})")
    model.to(DEVICE).eval()

    mixture.distributed_worker_shard = (lambda: (0, 1))
    loader = dm.loader(tok, split="valid", source_seed=0, epoch_size=None)
    it = iter(loader)
    batches = [to_dev(next(it)) for _ in range(N_BATCHES)]

    l_norm, cam = eval_loss(model, config, batches, zero=False)
    l_zero, _ = eval_loss(model, config, batches, zero=True)
    print(f"  ckpt step={payload.get('step')}  cond_absmax={cam:.3f}")
    print(f"  val loss  WITH condition  = {l_norm:.4f}")
    print(f"  val loss  condition ZEROED= {l_zero:.4f}")
    print(f"  delta (zeroed - with)     = {l_zero - l_norm:+.4f}   <- how much the model relies on its condition")
    del model
    gc.collect(); torch.cuda.empty_cache()
    return name, l_norm, l_zero


if __name__ == "__main__":
    print(f"Condition-reliance test on trained checkpoints (N_BATCHES={N_BATCHES})")
    rows = []
    for name, cfg in MODELS.items():
        if not cfg["ckpt"].is_file():
            print(f"skip {name}: no ckpt at {cfg['ckpt']}")
            continue
        rows.append(run_model(name, cfg))
    print("\n=== SUMMARY ===")
    print(f"{'model':12} {'loss_with':>10} {'loss_zeroed':>12} {'delta':>8}")
    for name, lw, lz in rows:
        print(f"{name:12} {lw:10.4f} {lz:12.4f} {lz-lw:+8.4f}")
