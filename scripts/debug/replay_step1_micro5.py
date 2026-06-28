"""Exact replay of step=1 micro=5 per DDP rank (6th batch, natural torch RNG).

Real training: torch.manual_seed(42+rank) once, then sample_bioseq_diffusion_noise
uses torch.rand without re-seeding. Each rank reads shard (rank, 8).

Run:
    source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/replay_step1_micro5.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dllm.pipelines.qwen3_vl_arch.data.mixture as mixture  # noqa: E402
from dllm.pipelines.qwen3_vl_arch.data import GrammarDataModule, Esm2SequenceTokenizer  # noqa: E402
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
WORLD = 8
GRAD_ACCUM = 4
TARGET_STEP = 1
TARGET_MICRO = 5  # global micro index; step1 micro5 = 6th forward (0-indexed batch 5)
TASK = {1: "antibody", 2: "tcr", 3: "tcr_epitope", 5: "ppi"}
BASE_SEED = 42


def build_model(tok):
    config = BioSeqDiffusionTransformerConfig(
        vocab_size=tok.vocab_size, hidden_size=1280, num_hidden_layers=28,
        num_attention_heads=16, intermediate_size=5120, dropout=0.1,
        max_position_embeddings=2304, pad_token_id=int(tok.pad_token_id),
        mask_token_id=int(tok.mask_token_id), qk_norm=True, gradient_checkpointing=True,
    )
    model = BioSeqEncoderDiffusionModel.from_hf_encoder(
        decoder_config=config, encoder_name_or_path=str(ESM2_DIR),
        local_files_only=True, trust_remote_code=True, freeze_encoder=False,
    ).to(DEVICE).train()
    return config, model


def to_dev(b):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}


def forward_loss(model, config, batch):
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
    return loss, chain_cond, labels, cmask, ts


def replay_rank(rank: int, model, config, dm, tok):
    mixture.distributed_worker_shard = (lambda r=rank: (lambda: (r, WORLD)))()
    torch.manual_seed(BASE_SEED + rank)
    loader = dm.loader(tok, split="train", source_seed=0, epoch_size=None)
    it = iter(loader)
    target_batch = None
    for micro in range(TARGET_MICRO + 1):
        batch = to_dev(next(it))
        if micro == TARGET_MICRO:
            target_batch = batch
        elif micro < GRAD_ACCUM:
            # step 0 micros: run forward+backward to match weight state... skip for now (lr tiny)
            loss, _, _, _, _ = forward_loss(model, config, batch)
            if not torch.isfinite(loss):
                return micro, batch, loss, "nonfinite at step0 replay"
            (loss / GRAD_ACCUM).backward()
    assert target_batch is not None
    loss, chain_cond, labels, cmask, ts = forward_loss(model, config, target_batch)
    task = int(target_batch["task_type_ids"][0].item())
    info = {
        "task": TASK.get(task, task),
        "enc_shape": tuple(target_batch["encoder_input_ids"].shape),
        "dec_len": target_batch["input_ids"].shape[1],
        "loss": float(loss.item()),
        "finite": bool(torch.isfinite(loss).item()),
        "cond_finite": bool(torch.isfinite(chain_cond).all().item()),
        "cond_absmax": float(chain_cond.float().abs().max().item()),
        "t_min": float(ts.min()), "t_max": float(ts.max()),
        "corrupted": int(cmask.sum()),
    }
    return TARGET_MICRO, target_batch, loss, info


def main():
    args = SimpleNamespace(
        sources="oas,ots,tcr,ppi", batch_size=4,
        grammar_data_dir=PROJECT_ROOT / "data/bioseq_grammar_v1",
        oas_weight=3.9, ots_weight=3.6, ppi_weight=1.4, tcr_weight=1.0,
        split="train", source_seed=0, max_sequence_length=2112,
        num_workers=0, tokenizer_path=None,
    )
    dm = GrammarDataModule.from_args(args)
    tok = dm.build_tokenizer()
    config, model = build_model(tok)
    opt = torch.optim.AdamW(
        [
            {"params": list(model.decoder.parameters()), "lr": 1e-4},
            {"params": list(model.encoder.parameters()), "lr": 2e-5},
        ],
    )

    print(f"Replaying step={TARGET_STEP} micro={TARGET_MICRO} (batch index {TARGET_MICRO}) per rank")
    print(f"model token_dropout={getattr(model.encoder.config, 'token_dropout', 'n/a')}")

    for rank in range(WORLD):
        model.load_state_dict(
            BioSeqEncoderDiffusionModel.from_hf_encoder(
                decoder_config=config, encoder_name_or_path=str(ESM2_DIR),
                local_files_only=True, trust_remote_code=True, freeze_encoder=False,
            ).state_dict()
        )
        model.train()
        opt.zero_grad(set_to_none=True)
        # Full step-0 optimizer replay for this rank
        mixture.distributed_worker_shard = (lambda r=rank: (lambda: (r, WORLD)))()
        torch.manual_seed(BASE_SEED + rank)
        loader = dm.loader(tok, split="train", source_seed=0, epoch_size=None)
        it = iter(loader)
        for micro in range(GRAD_ACCUM):
            batch = to_dev(next(it))
            loss, _, _, _, _ = forward_loss(model, config, batch)
            if not torch.isfinite(loss):
                print(f"rank {rank}: step0 micro{micro} NON-FINITE loss={loss.item()}")
                break
            (loss / GRAD_ACCUM).backward()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            # step 1 micro 4 and 5
            for micro in range(GRAD_ACCUM, TARGET_MICRO + 1):
                batch = to_dev(next(it))
                loss, chain_cond, labels, cmask, ts = forward_loss(model, config, batch)
                task = int(batch["task_type_ids"][0].item())
                fin = torch.isfinite(loss).item()
                print(
                    f"rank {rank} step1 micro{micro}: task={TASK.get(task,task)} "
                    f"enc={tuple(batch['encoder_input_ids'].shape)} loss={loss.item():.4f} finite={fin} "
                    f"cond_absmax={chain_cond.float().abs().max().item():.3f}"
                )
                if not fin:
                    id2 = Esm2SequenceTokenizer().id_to_token
                    enc = batch["encoder_input_ids"]
                    for bi in range(min(2, enc.shape[0])):
                        for ci in range(enc.shape[1]):
                            row = enc[bi, ci].tolist()
                            seq = "".join(id2.get(t, "?") for t in row if t not in (1,))
                            print(f"  chain b{bi}c{ci}: {seq[:100]}")
                    break
                if micro < TARGET_MICRO:
                    (loss / GRAD_ACCUM).backward()
            continue
        print(f"rank {rank}: failed during step0 replay")


if __name__ == "__main__":
    main()
