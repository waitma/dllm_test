"""Replay rank-5 step=1 micro=5 (suspected NaN rank from Volc logs).

Run:
    export PATH="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin:$PATH"
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/replay_rank5_micro5.py
"""

from __future__ import annotations

import gc
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
from dllm.pipelines.qwen3_vl_arch.training.trainer import lr_at  # noqa: E402

ESM2_DIR = PROJECT_ROOT / "model_weights/esm2/esm2_t33_650M_UR50D"
DEVICE = "cuda:0"
RANK = 5
WORLD = 8
GRAD_ACCUM = 4
TARGET_MICRO = 5
BASE_SEED = 42
TASK = {1: "antibody", 2: "tcr", 3: "tcr_epitope", 5: "ppi"}


def build_args():
    return SimpleNamespace(
        sources="oas,ots,tcr,ppi", batch_size=4,
        grammar_data_dir=PROJECT_ROOT / "data/bioseq_grammar_v1",
        oas_weight=3.9, ots_weight=3.6, ppi_weight=1.4, tcr_weight=1.0,
        split="train", source_seed=0, max_sequence_length=2112, num_workers=0,
        tokenizer_path=None,
    )


def build_model(tok):
    config = BioSeqDiffusionTransformerConfig(
        vocab_size=tok.vocab_size, hidden_size=1280, num_hidden_layers=28,
        num_attention_heads=16, intermediate_size=5120, dropout=0.1,
        max_position_embeddings=2304, pad_token_id=int(tok.pad_token_id),
        mask_token_id=int(tok.mask_token_id), qk_norm=True, gradient_checkpointing=True,
    )
    return config, BioSeqEncoderDiffusionModel.from_hf_encoder(
        decoder_config=config, encoder_name_or_path=str(ESM2_DIR),
        local_files_only=True, trust_remote_code=True, freeze_encoder=False,
    ).to(DEVICE)


def to_dev(b):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}


def set_lrs(opt, step):
    for group in opt.param_groups:
        base = group["initial_lr"]
        group["lr"] = lr_at(
            step, base, warmup_steps=1000, max_steps=10000,
            scheduler="cosine", min_lr_ratio=0.1, warmup_init_lr=1e-7,
        )


def forward_detail(model, config, batch):
    nids, labels, cmask, ts = sample_bioseq_diffusion_noise(
        batch=batch, mask_token_id=config.mask_token_id, time_epsilon=1e-3,
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
    return {
        "loss": loss,
        "chain_cond": chain_cond,
        "logits": out.logits,
        "labels": labels,
        "cmask": cmask,
        "ts": ts,
        "nenc": nenc,
    }


def batch_stats(batch):
    enc = batch["encoder_input_ids"]
    attn = batch["encoder_attention_mask"]
    b, c, l = enc.shape
    enc_lens = attn.sum(dim=-1).tolist() if attn is not None else []
    dec_len = batch["input_ids"].shape[1]
    task = int(batch["task_type_ids"][0].item())
    return {
        "task": TASK.get(task, task),
        "enc_shape": (b, c, l),
        "enc_lens": enc_lens,
        "dec_len": dec_len,
        "chain_mask_sum": int(batch["encoder_chain_mask"].sum()) if batch.get("encoder_chain_mask") is not None else None,
    }


def main():
    args = build_args()
    dm = GrammarDataModule.from_args(args)
    tok = dm.build_tokenizer()
    config, model = build_model(tok)
    model.train()
    opt = torch.optim.AdamW(
        [
            {"params": list(model.decoder.parameters()), "lr": 1e-4, "initial_lr": 1e-4},
            {"params": list(model.encoder.parameters()), "lr": 2e-5, "initial_lr": 2e-5},
        ],
    )

    mixture.distributed_worker_shard = (lambda: (RANK, WORLD))
    torch.manual_seed(BASE_SEED + RANK)
    loader = dm.loader(tok, split="train", source_seed=0, epoch_size=None)
    it = iter(loader)

    for micro in range(TARGET_MICRO + 1):
        batch = to_dev(next(it))
        set_lrs(opt, step=0 if micro < GRAD_ACCUM else 1)
        if micro < GRAD_ACCUM:
            detail = forward_detail(model, config, batch)
            loss = detail["loss"]
            fin = torch.isfinite(loss).item()
            print(f"step0 micro{micro}: {batch_stats(batch)} loss={loss.item():.4f} finite={fin}")
            if not fin:
                print("  chain_cond finite:", torch.isfinite(detail["chain_cond"]).all().item())
                print("  logits finite:", torch.isfinite(detail["logits"]).all().item())
                return
            (loss / GRAD_ACCUM).backward()
            if micro == GRAD_ACCUM - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        elif micro == TARGET_MICRO:
            stats = batch_stats(batch)
            print(f"\n=== TARGET rank{RANK} step1 micro{micro} ===")
            print(stats)
            detail = forward_detail(model, config, batch)
            loss = detail["loss"]
            cc = detail["chain_cond"]
            lg = detail["logits"]
            print(f"loss={loss.item()} finite={torch.isfinite(loss).item()}")
            print(f"chain_cond finite={torch.isfinite(cc).all().item()} absmax={cc.float().abs().max().item():.3f}")
            print(f"logits finite={torch.isfinite(lg).all().item()} absmax={lg.float().abs().max().item():.3f}")
            if not torch.isfinite(loss):
                # per-token loss breakdown
                forbidden = forbidden_diffusion_target_token_ids(config)
                from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import mask_forbidden_target_logits
                import torch.nn.functional as F
                logits = mask_forbidden_target_logits(lg, forbidden)
                tl = F.cross_entropy(logits.reshape(-1, logits.size(-1)), detail["labels"].reshape(-1), ignore_index=-100, reduction="none").view_as(detail["labels"])
                bad = ~torch.isfinite(tl)
                print(f"non-finite token losses: {bad.sum().item()}")
                if bad.any():
                    idx = bad.nonzero()[0]
                    print(f"  first bad at {idx.tolist()}, label={detail['labels'][idx[0], idx[1]].item()}")
            # also test fp32 encoder
            gc.collect()
            torch.cuda.empty_cache()
            with torch.autocast("cuda", enabled=False):
                nenc = detail["nenc"]
                with torch.no_grad():
                    cc32 = model.encode_chain_tokens(
                        encoder_input_ids=nenc.float() if nenc.dtype != torch.float32 else nenc,
                        encoder_attention_mask=batch.get("encoder_attention_mask"),
                        encoder_residue_mask=batch.get("encoder_residue_mask"),
                        encoder_chain_mask=batch.get("encoder_chain_mask"),
                    )
                print(f"fp32 encoder cond finite={torch.isfinite(cc32).all().item()}")
        else:
            detail = forward_detail(model, config, batch)
            loss = detail["loss"]
            print(f"step1 micro{micro}: loss={loss.item():.4f} finite={torch.isfinite(loss).item()}")
            (loss / GRAD_ACCUM).backward()

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
