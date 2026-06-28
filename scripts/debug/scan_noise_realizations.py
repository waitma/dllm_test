"""Same data across ranks, different noise per rank: scan many noise realizations
on each early batch to catch the (batch, noise) combo that yields a non-finite
ESM2 forward loss. Mirrors the real run (source_seed=0, batch_size=4).

Run:
    source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/scan_noise_realizations.py
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
TASK = {1: "antibody", 2: "tcr", 3: "tcr_epitope", 5: "ppi"}
NOISES = 96


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
        mask_token_id=int(tok.mask_token_id), qk_norm=True, gradient_checkpointing=False,
    )
    model = BioSeqEncoderDiffusionModel.from_hf_encoder(
        decoder_config=config, encoder_name_or_path=str(ESM2_DIR),
        local_files_only=True, trust_remote_code=True, freeze_encoder=False,
    ).to(DEVICE).eval()
    loader = dm.loader(tok, split="train", source_seed=0, epoch_size=None)
    return dm, tok, config, model, loader


def to_dev(b):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def fwd_loss(model, config, batch):
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
    logits_finite = torch.isfinite(out.logits).all().item()
    return loss.item(), logits_finite, ts, cmask, labels


def main():
    dm, tok, config, model, loader = build()
    it = iter(loader)
    for bidx in range(8):
        batch = to_dev(next(it))
        task = int(batch["task_type_ids"][0].item())
        n_nonfinite = 0
        worst = 0.0
        worst_info = None
        for s in range(NOISES):
            torch.manual_seed(1000 + bidx * 1000 + s)
            loss, lfin, ts, cmask, labels = fwd_loss(model, config, batch)
            mag = abs(loss)
            if not (loss == loss) or mag == float("inf") or mag > 1e6 or not lfin:
                n_nonfinite += 1
                if worst_info is None:
                    worst_info = (s, loss, lfin, float(ts.min()), float(ts.max()), int(cmask.sum()))
            worst = max(worst, mag if mag == mag else worst)
        flag = "  <<<< NON-FINITE/HUGE" if n_nonfinite else ""
        print(f"batch {bidx} task={TASK.get(task,task):11s} enc={tuple(batch['encoder_input_ids'].shape)} "
              f"dec={batch['input_ids'].shape[1]} | nonfinite/huge {n_nonfinite}/{NOISES} worst|loss|={worst:.4g}{flag}")
        if worst_info:
            s, loss, lfin, tmin, tmax, nc = worst_info
            print(f"      first hit: noise_seed_idx={s} loss={loss:.4g} logits_finite={lfin} "
                  f"t=[{tmin:.3f},{tmax:.3f}] corrupted={nc}")
    print("\ndone")


if __name__ == "__main__":
    main()
