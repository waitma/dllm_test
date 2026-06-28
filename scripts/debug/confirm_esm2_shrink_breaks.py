"""ESM2 reference test: is encoder-condition MAGNITUDE the sole cause?

ESM2 trains fine (val 0.72); ESMC plateaus (val 1.9). If the only relevant
difference is the residual-stream magnitude of the injected condition
(ESM2 L2~9.5 vs ESMC L2~1.3), then ARTIFICIALLY shrinking ESM2's condition to
ESMC scale must BREAK ESM2 the same way, and re-normalizing must recover it.

Arms (ESM2-650M, identical init/data/seed):
  N  : ESM2 as-is                         (control -> should descend)
  S  : ESM2 condition x0.14 (-> ESMC L2)  (-> should plateau like broken ESMC)
  SL : LayerNorm(condition x0.14)         (the fix -> should recover)

Run:
    export PATH="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin:$PATH"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
    python scripts/debug/confirm_esm2_shrink_breaks.py
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

ESM2_DIR = PROJECT_ROOT / "model_weights/esm2/esm2_t33_650M_UR50D"
DEVICE = "cuda:0"
STEPS = 180
LOG_EVERY = 20
SEED = 42
SHRINK = 0.14  # ESM2 L2 9.5 * 0.14 ~= 1.33 ~= ESMC L2 1.3
ARMS = ["N", "S", "SL"]


def build_args():
    return SimpleNamespace(
        sources="oas,ots,tcr,ppi", batch_size=2,
        grammar_data_dir=PROJECT_ROOT / "data/bioseq_grammar_v1",
        oas_weight=3.9, ots_weight=3.6, ppi_weight=1.4, tcr_weight=1.0,
        split="train", source_seed=0, max_sequence_length=2112, num_workers=0,
        tokenizer_path=None,
    )


def to_dev(b):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}


def make_config(tok):
    return BioSeqDiffusionTransformerConfig(
        vocab_size=tok.vocab_size, hidden_size=1280, num_hidden_layers=28,
        num_attention_heads=16, intermediate_size=5120, dropout=0.1,
        max_position_embeddings=2304, pad_token_id=int(tok.pad_token_id),
        mask_token_id=int(tok.mask_token_id), qk_norm=True, gradient_checkpointing=True,
    )


def new_model(config):
    return BioSeqEncoderDiffusionModel.from_hf_encoder(
        decoder_config=config, encoder_name_or_path=str(ESM2_DIR),
        local_files_only=True, trust_remote_code=True, freeze_encoder=False,
    ).to(DEVICE)


def install_hook(model, mode, hidden):
    decoder = model.decoder
    base_forward = decoder.forward
    cond_norm = nn.LayerNorm(hidden).to(DEVICE) if mode == "SL" else None

    def patched(self, *args, encoder_condition=None, **kwargs):
        if encoder_condition is not None:
            if mode == "S":
                encoder_condition = encoder_condition * SHRINK
            elif mode == "SL":
                encoder_condition = cond_norm(encoder_condition * SHRINK)
        return base_forward(*args, encoder_condition=encoder_condition, **kwargs)

    decoder.forward = types.MethodType(patched, decoder)
    return cond_norm


def run_arm(mode, init_state, config, dm, tok):
    torch.manual_seed(SEED)
    model = new_model(config)
    model.load_state_dict(init_state)
    model.train()
    cond_norm = install_hook(model, mode, config.hidden_size)
    groups = [
        {"params": list(model.decoder.parameters()), "lr": 1e-4},
        {"params": list(model.encoder.parameters()), "lr": 2e-5},
    ]
    if cond_norm is not None:
        groups.append({"params": list(cond_norm.parameters()), "lr": 1e-4})
    opt = torch.optim.AdamW(groups)

    mixture.distributed_worker_shard = (lambda: (0, 1))
    torch.manual_seed(SEED)
    loader = dm.loader(tok, split="train", source_seed=0, epoch_size=None)
    it = iter(loader)
    forbidden = forbidden_diffusion_target_token_ids(config)
    out_log = []
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
            out_log.append((step + 1, avg))
            print(f"  [{mode}] step {step+1:4d} loss(avg{LOG_EVERY})={avg:.4f}", flush=True)
            running = []
        step += 1
    del model, opt
    torch.cuda.empty_cache()
    return out_log


if __name__ == "__main__":
    args = build_args()
    dm = GrammarDataModule.from_args(args)
    tok = dm.build_tokenizer()
    config = make_config(tok)
    torch.manual_seed(SEED)
    init_state = copy.deepcopy(new_model(config).state_dict())
    print(f"ESM2 reference test: shrink={SHRINK} (ESM2 L2~9.5 -> ~1.3 == ESMC). STEPS={STEPS}")
    results = {}
    for mode in ARMS:
        print(f"=== Arm {mode} ===", flush=True)
        results[mode] = run_arm(mode, init_state, config, dm, tok)
    print("\n=== SUMMARY (avg train loss) ===")
    labels = {"N": "N esm2", "S": "S shrunk", "SL": "SL shrunk+LN"}
    steps = [s for s, _ in results[ARMS[0]]]
    print("step  | " + "".join(f"{labels[m]:>13}" for m in ARMS))
    for i, s in enumerate(steps):
        print(f"{s:5d} | " + "".join(f"{results[m][i][1]:13.4f}" for m in ARMS))
