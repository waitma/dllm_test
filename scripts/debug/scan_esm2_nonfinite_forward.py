"""Forward-only long scan to catch a batch that yields a non-finite ESM2 loss.

Run:
    source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/scan_esm2_nonfinite_forward.py
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
    return dm, tok, config, model


def to_dev(b):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def fwd(model, config, batch):
    nids, labels, cmask, ts = sample_bioseq_diffusion_noise(
        batch=batch, mask_token_id=config.mask_token_id, time_epsilon=1e-3
    )
    nenc = apply_decoder_corruption_to_encoder(batch=batch, corruption_mask=cmask, mask_token_id=config.mask_token_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        chain_cond = model.encode_chain_tokens(
            encoder_input_ids=nenc,
            encoder_attention_mask=batch.get("encoder_attention_mask"),
            encoder_residue_mask=batch.get("encoder_residue_mask"),
            encoder_chain_mask=batch.get("encoder_chain_mask"),
        )
        cond_finite = torch.isfinite(chain_cond).all().item()
        out = model(
            input_ids=nids, attention_mask=batch.get("attention_mask"),
            position_ids_inner=batch.get("position_ids_inner"),
            position_ids_chain=batch.get("position_ids_chain"), timesteps=ts,
            chain_ids=batch.get("chain_ids"), residue_mask=batch.get("residue_mask"),
            encoder_input_ids=nenc, encoder_attention_mask=batch.get("encoder_attention_mask"),
            encoder_residue_mask=batch.get("encoder_residue_mask"),
            encoder_chain_mask=batch.get("encoder_chain_mask"),
        )
        logits_finite = torch.isfinite(out.logits).all().item()
        forbidden = forbidden_diffusion_target_token_ids(config)
        loss = compute_masked_cross_entropy(out.logits, labels, loss_norm=config.loss_norm, forbidden_token_ids=forbidden)
    return loss.item(), cond_finite, logits_finite, labels


def main():
    torch.manual_seed(0)
    dm, tok, config, model = build()
    forbidden = set(forbidden_diffusion_target_token_ids(config))
    n_bad = 0
    for seed in (0, 1, 2, 3, 11, 23, 99):
        loader = dm.loader(tok, split="train", source_seed=seed, epoch_size=4000)
        it = iter(loader)
        for i in range(220):
            try:
                batch = to_dev(next(it))
            except StopIteration:
                break
            loss, cfin, lfin, labels = fwd(model, config, batch)
            if (not cfin) or (not lfin) or (not torch.isfinite(torch.tensor(loss))) or abs(loss) > 1e6:
                task = int(batch["task_type_ids"][0].item())
                # forbidden labels among targets?
                lab = labels
                fcount = sum(int((lab == f).sum().item()) for f in forbidden)
                print(f">>> seed={seed} batch={i} task={TASK.get(task,task)} "
                      f"enc={tuple(batch['encoder_input_ids'].shape)} "
                      f"loss={loss:.4g} cond_finite={cfin} logits_finite={lfin} forbidden_labels={fcount}")
                n_bad += 1
                if n_bad >= 6:
                    print("stopping after 6 hits")
                    return
        print(f"seed {seed}: done 220 batches, hits so far={n_bad}")
    print(f"\nTotal non-finite/huge-loss batches: {n_bad}")


if __name__ == "__main__":
    main()
