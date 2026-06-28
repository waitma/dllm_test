"""Replicate each DDP rank's data shard (num_shards=8) and hunt the batch that
produces a non-finite ESM2 forward loss. In the real run each rank reads
dataset.shard(index=rank, num_shards=8) with random.Random(seed+rank), so ranks
see disjoint records; a single-GPU loader (shard 0) never sees the bad one.

Run:
    source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/hunt_rank_shard_nan.py
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
WORLD = 8
BATCHES_PER_RANK = 8
NOISES = 40


def build_model(tok):
    config = BioSeqDiffusionTransformerConfig(
        vocab_size=tok.vocab_size, hidden_size=1280, num_hidden_layers=28,
        num_attention_heads=16, intermediate_size=5120, dropout=0.1,
        max_position_embeddings=2304, pad_token_id=int(tok.pad_token_id),
        mask_token_id=int(tok.mask_token_id), qk_norm=True, gradient_checkpointing=False,
    )
    model = BioSeqEncoderDiffusionModel.from_hf_encoder(
        decoder_config=config, encoder_name_or_path=str(ESM2_DIR),
        local_files_only=True, trust_remote_code=True, freeze_encoder=False,
    ).to(DEVICE).train()  # train mode: faithful (decoder dropout active)
    return config, model


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
            encoder_input_ids=nenc, encoder_attention_mask=batch.get("encoder_attention_mask"),
            encoder_residue_mask=batch.get("encoder_residue_mask"), encoder_chain_mask=batch.get("encoder_chain_mask"),
        )
        cond_absmax = chain_cond.float().abs().max().item()
        cond_finite = torch.isfinite(chain_cond).all().item()
        out = model(
            input_ids=nids, attention_mask=batch.get("attention_mask"),
            position_ids_inner=batch.get("position_ids_inner"),
            position_ids_chain=batch.get("position_ids_chain"), timesteps=ts,
            chain_ids=batch.get("chain_ids"), residue_mask=batch.get("residue_mask"),
            encoder_input_ids=nenc, encoder_attention_mask=batch.get("encoder_attention_mask"),
            encoder_residue_mask=batch.get("encoder_residue_mask"), encoder_chain_mask=batch.get("encoder_chain_mask"),
        )
        lfin = torch.isfinite(out.logits).all().item()
        forbidden = forbidden_diffusion_target_token_ids(config)
        loss = compute_masked_cross_entropy(out.logits, labels, loss_norm=config.loss_norm, forbidden_token_ids=forbidden)
    return loss.item(), cond_finite, cond_absmax, lfin


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

    hits = 0
    for rank in range(WORLD):
        mixture.distributed_worker_shard = (lambda r=rank: (lambda: (r, WORLD)))()
        loader = dm.loader(tok, split="train", source_seed=0, epoch_size=None)
        it = iter(loader)
        for bidx in range(BATCHES_PER_RANK):
            try:
                batch = to_dev(next(it))
            except StopIteration:
                break
            task = int(batch["task_type_ids"][0].item())
            worst = 0.0
            bad = None
            for s in range(NOISES):
                torch.manual_seed(7000 + rank * 100 + s)
                loss, cfin, cabs, lfin = fwd(model, config, batch)
                if (loss != loss) or abs(loss) > 1e6 or not cfin or not lfin:
                    bad = (s, loss, cfin, cabs, lfin)
                    break
                worst = max(worst, abs(loss), cabs)
            if bad is not None:
                s, loss, cfin, cabs, lfin = bad
                print(f">>> RANK {rank} batch {bidx} task={TASK.get(task,task)} "
                      f"enc={tuple(batch['encoder_input_ids'].shape)} dec={batch['input_ids'].shape[1]} "
                      f"loss={loss:.4g} cond_finite={cfin} cond_absmax={cabs:.4g} logits_finite={lfin}")
                # dump the chains
                enc = batch["encoder_input_ids"]
                from dllm.pipelines.qwen3_vl_arch.data import Esm2SequenceTokenizer
                base = Esm2SequenceTokenizer()
                id2 = base.id_to_token
                for bi in range(enc.shape[0]):
                    for ci in range(enc.shape[1]):
                        row = enc[bi, ci].tolist()
                        seq = "".join(id2.get(t, "?") for t in row if t not in (1,))
                        print(f"      b{bi}c{ci}: {seq[:120]}")
                hits += 1
                break
        if hits:
            break
        print(f"rank {rank}: scanned {BATCHES_PER_RANK} batches, no hit (worst|val|~{worst:.2f})")
    print(f"\nhits={hits}")


if __name__ == "__main__":
    main()
