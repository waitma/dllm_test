"""Probe max per-GPU batch size for grammar_v1 BioSeq training configs.

Run on a single GPU with production sequence length, bf16, and gradient checkpointing.
Target on 8 GPUs: effective batch 128 => batch_size * grad_accum = 16.

Example:
    /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
        /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/probe_grammar_v1_batch_size.py
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.bioseq.train_qwen3_vl_bioseq_ddp import (  # noqa: E402
    build_config,
    build_loader,
    build_model,
    build_tokenizer,
    compute_training_output,
    move_batch,
)


DATA_DIR = PROJECT_ROOT / "data/bioseq_grammar_v1"
ESMC_300M = PROJECT_ROOT / "model_weights/esmc/ESMC-300M"
ESMC_600M = PROJECT_ROOT / "model_weights/esmc/ESMC-600M"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    encoder_path: Path | None = None


SPECS = [
    ModelSpec("no_encoder_38m", "no_encoder", 512, 8, 8, 2048),
    ModelSpec("no_encoder_300m", "no_encoder", 1024, 18, 16, 4096),
    ModelSpec("no_encoder_600m", "no_encoder", 1536, 16, 24, 6144),
    ModelSpec("no_encoder_1b", "no_encoder", 1792, 20, 28, 7168),
    ModelSpec("encoder_esmc300m", "encoder", 512, 8, 8, 2048, ESMC_300M),
    ModelSpec("encoder_esmc600m", "encoder", 512, 8, 8, 2048, ESMC_600M),
]


def make_args(spec: ModelSpec, batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        grammar_data_dir=DATA_DIR,
        model_type=spec.model_type,
        encoder_path=spec.encoder_path,
        tokenizer_path=ESMC_300M,
        freeze_encoder=False,
        encoder_use_flash_attn=False,
        sources="oas,ots,tcr,ppi",
        oas_weight=3.9,
        ots_weight=3.6,
        tcr_weight=1.0,
        ppi_weight=1.4,
        nanobody_weight=1.0,
        processed_v2_weight=1.0,
        limit_per_source=128,
        split="train",
        source_seed=0,
        epoch_size=None,
        deduplicate_within_batch=True,
        batch_size=batch_size,
        hidden_size=spec.hidden_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        intermediate_size=spec.intermediate_size,
        dropout=0.1,
        max_sequence_length=2112,
        max_position_embeddings=2304,
        max_chain_positions=8,
        max_chain_roles=8,
        max_task_types=8,
        vocab_size=None,
        time_epsilon=1e-3,
        loss_norm="token",
        qk_norm=False,
        gradient_checkpointing=True,
        initializer_range=0.02,
        num_workers=0,
    )


def gpu_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def recommend(max_bs: int) -> str:
    pairs = [(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)]
    for bs, ga in pairs:
        if bs <= max_bs:
            return f"batch_size={bs} grad_accum={ga} (eff_8gpu={bs * ga * 8})"
    return "OOM at batch_size=1"


def pick_stress_batch(spec: ModelSpec, batch_size: int, device: torch.device, scan_batches: int = 40):
    args = make_args(spec, batch_size)
    tokenizer = build_tokenizer(args)
    loader = build_loader(args, tokenizer)
    best = None
    best_len = -1
    for idx, batch in enumerate(loader):
        seq_len = int(batch["input_ids"].shape[1])
        if seq_len > best_len:
            best_len = seq_len
            best = batch
        if idx + 1 >= scan_batches:
            break
    if best is None:
        raise RuntimeError("no batches produced")
    return args, tokenizer, move_batch(best, device), best_len


def try_batch(spec: ModelSpec, batch_size: int, device: torch.device) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    args, tokenizer, batch, seq_len = pick_stress_batch(spec, batch_size, device)
    config = build_config(args, tokenizer)
    model = build_model(args, config).to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)

    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = compute_training_output(model, model, batch)
            assert output.loss is not None
            output.loss.backward()
        optimizer.step()
        ok = True
        err = None
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            ok = False
            err = "OOM"
        else:
            raise
    finally:
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        del model, batch, optimizer, tokenizer, config, args
        gc.collect()
        torch.cuda.empty_cache()

    return {"batch_size": batch_size, "ok": ok, "error": err, "peak_gb": round(peak, 2), "seq_len": seq_len}


def probe_spec(spec: ModelSpec, device: torch.device, candidates: list[int]) -> dict:
    print(f"\n=== {spec.name} ({spec.model_type}) ===", flush=True)
    results = []
    max_ok = 0
    for bs in candidates:
        row = try_batch(spec, bs, device)
        results.append(row)
        status = "OK" if row["ok"] else row["error"]
        print(
            f"  batch_size={bs:2d} -> {status:3s} peak={row['peak_gb']:.2f}GB seq_len={row['seq_len']}",
            flush=True,
        )
        if row["ok"]:
            max_ok = bs
        elif row["error"] == "OOM" and max_ok > 0:
            break
    return {"spec": spec.name, "max_batch_size": max_ok, "recommendation": recommend(max_ok), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--candidates", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda:0")
    print(f"GPU={torch.cuda.get_device_name(0)} total={gpu_total_gb():.1f}GB seq_len=2112 bf16+ckpt", flush=True)

    specs = [s for s in SPECS if not args.only or s.name in set(args.only)]
    summary = [probe_spec(spec, device, args.candidates) for spec in specs]

    print("\n=== Summary ===", flush=True)
    for row in summary:
        print(f"  {row['spec']:22s} max_bs={row['max_batch_size']:2d} -> {row['recommendation']}", flush=True)


if __name__ == "__main__":
    main()
