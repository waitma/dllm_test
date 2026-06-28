"""Find which batch makes the ESM2 grammar-v2 encoder produce non-finite GRADIENTS.

Forward loss can be finite while backward yields NaN/Inf (e.g. long PPI chains
through ESM2 attention in bf16 + gradient checkpointing). In 8-GPU DDP a single
rank's non-finite grad is all-reduced to every rank, so all ranks see a NaN loss
on the next step. This isolates the offending batch on one GPU.

Run:
    source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/find_esm2_nan_grad_batch.py
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
TASK_NAMES = {1: "antibody", 2: "tcr", 3: "tcr_epitope", 5: "ppi"}


def build():
    args = SimpleNamespace(
        sources="oas,ots,tcr,ppi", batch_size=4,
        grammar_data_dir=PROJECT_ROOT / "data/bioseq_grammar_v1",
        oas_weight=3.9, ots_weight=3.6, ppi_weight=1.4, tcr_weight=1.0,
        split="train", source_seed=0, max_sequence_length=2112,
        num_workers=0, tokenizer_path=None,
    )
    dm = GrammarDataModule.from_args(args)
    tokenizer = dm.build_tokenizer()
    config = BioSeqDiffusionTransformerConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=1280, num_hidden_layers=28,
        num_attention_heads=16, intermediate_size=5120, dropout=0.1,
        max_position_embeddings=2304, pad_token_id=int(tokenizer.pad_token_id),
        mask_token_id=int(tokenizer.mask_token_id), qk_norm=True,
        gradient_checkpointing=True,
    )
    model = BioSeqEncoderDiffusionModel.from_hf_encoder(
        decoder_config=config, encoder_name_or_path=str(ESM2_DIR),
        local_files_only=True, trust_remote_code=True, freeze_encoder=False,
    ).to(DEVICE)
    model.train()
    return dm, tokenizer, config, model


def to_dev(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


def one_batch(model, config, batch, bf16=True):
    noised_input_ids, labels, corruption_mask, timesteps = sample_bioseq_diffusion_noise(
        batch=batch, mask_token_id=config.mask_token_id, time_epsilon=1e-3
    )
    noised_encoder_input_ids = apply_decoder_corruption_to_encoder(
        batch=batch, corruption_mask=corruption_mask, mask_token_id=config.mask_token_id
    )
    kwargs = dict(
        input_ids=noised_input_ids, attention_mask=batch.get("attention_mask"),
        position_ids_inner=batch.get("position_ids_inner"),
        position_ids_chain=batch.get("position_ids_chain"), timesteps=timesteps,
        chain_ids=batch.get("chain_ids"), residue_mask=batch.get("residue_mask"),
        encoder_input_ids=noised_encoder_input_ids,
        encoder_attention_mask=batch.get("encoder_attention_mask"),
        encoder_residue_mask=batch.get("encoder_residue_mask"),
        encoder_chain_mask=batch.get("encoder_chain_mask"),
    )
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cuda", enabled=False)
    with ctx:
        output = model(**kwargs)
        forbidden = forbidden_diffusion_target_token_ids(config)
        loss = compute_masked_cross_entropy(
            output.logits, labels, loss_norm=config.loss_norm, forbidden_token_ids=forbidden
        )
    loss.backward()
    grad_finite = True
    first_bad = None
    total_sq = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            grad_finite = False
            if first_bad is None:
                first_bad = n
        else:
            total_sq += p.grad.float().pow(2).sum().item()
    return loss.item(), grad_finite, first_bad, (total_sq ** 0.5 if grad_finite else float("nan"))


def main():
    torch.manual_seed(0)
    dm, tokenizer, config, model = build()
    loader = dm.loader(tokenizer, split="train", source_seed=0, epoch_size=400)
    it = iter(loader)
    print("scanning batches (forward+backward each, isolated grad)...")
    n_bad = 0
    for i in range(60):
        batch = to_dev(next(it))
        task = int(batch["task_type_ids"][0].item()) if "task_type_ids" in batch else -1
        enc_shape = tuple(batch["encoder_input_ids"].shape)
        dec_len = batch["input_ids"].shape[1]
        model.zero_grad(set_to_none=True)
        loss, gfin, bad, gnorm = one_batch(model, config, batch)
        flag = "" if gfin else "  <<<< NON-FINITE GRAD"
        print(f"batch {i:02d} task={TASK_NAMES.get(task,task):10s} enc={enc_shape} dec_len={dec_len} "
              f"loss={loss:.3f} grad_finite={gfin} gnorm={gnorm:.2f}{flag}")
        if not gfin:
            print(f"          first non-finite grad param: {bad}")
            n_bad += 1
            if n_bad >= 4:
                print("stopping after 4 bad batches")
                break
    print(f"\nTotal non-finite-grad batches: {n_bad}")


if __name__ == "__main__":
    main()
