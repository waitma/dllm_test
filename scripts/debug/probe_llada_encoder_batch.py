"""Calibrate the ~0.1B LLaDA backbone and probe max per-GPU batch on one 80G A100.

Builds ``BioSeqLLaDAEncoderDiffusionModel`` (ESMC encoder + LLaDA backbone), prints
the LLaDA (decoder) parameter count for the given layer/width, then runs real
forward+backward+AdamW steps at each candidate batch size and reports peak CUDA
memory. Use it to pick a per-GPU microbatch that fills 80G (not a fixed global 128).

Run:
    PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test \
    /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
    scripts/debug/probe_llada_encoder_batch.py \
        --encoder-path /vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-300M \
        --num-hidden-layers 9 --intermediate-size 2560 --num-attention-heads 16 \
        --batch-sizes 16,24,32,48,64 --probe-batches 12
"""

from __future__ import annotations

import argparse
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
    BioSeqLLaDAEncoderDiffusionModel,
)


def infer_encoder_hidden(encoder_path: Path) -> int:
    import json

    with (encoder_path / "config.json").open() as handle:
        config = json.load(handle)
    for key in ("hidden_size", "d_model", "embed_dim", "encoder_embed_dim"):
        if key in config:
            return int(config[key])
    raise ValueError(f"cannot infer encoder hidden size from {encoder_path}")


def build_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        grammar_data_dir=Path(args.grammar_data_dir),
        sources=args.sources,
        split="train",
        source_seed=0,
        limit_per_source=args.limit_per_source,
        epoch_size=None,
        batch_size=8,  # overridden per probe
        max_sequence_length=args.max_sequence_length,
        max_protein_length=args.max_protein_length,
        num_workers=0,
        tokenizer_path=Path(args.encoder_path),
        deduplicate_within_batch=False,
        oas_weight=3.9,
        ots_weight=3.6,
        ppi_weight=1.4,
        tcr_weight=1.0,
        nanobody_weight=1.0,
        processed_v2_weight=1.0,
        val_interval=0,
        val_batches=0,
        val_split="valid",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--grammar-data-dir", default=str(PROJECT_ROOT / "data/bioseq_grammar_v1"))
    parser.add_argument("--sources", default="oas,ots,tcr,ppi")
    parser.add_argument("--num-hidden-layers", type=int, required=True)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=2112)
    parser.add_argument("--max-protein-length", type=int, default=1024)
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--batch-sizes", default="16,24,32,48,64")
    parser.add_argument("--probe-batches", type=int, default=12)
    args = parser.parse_args()

    encoder_path = Path(args.encoder_path)
    hidden = infer_encoder_hidden(encoder_path)
    device = torch.device("cuda")

    datamodule = GrammarDataModule.from_args(build_namespace(args))
    tokenizer = datamodule.build_tokenizer()
    vocab_size = int(getattr(tokenizer, "vocab_size"))

    config = BioSeqDiffusionTransformerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        dropout=0.1,
        max_position_embeddings=args.max_sequence_length + 192,
        pad_token_id=int(tokenizer.pad_token_id),
        mask_token_id=int(tokenizer.mask_token_id),
        gradient_checkpointing=True,
    )
    model = BioSeqLLaDAEncoderDiffusionModel.from_esmc(
        decoder_config=config,
        encoder_name_or_path=str(encoder_path),
        local_files_only=True,
        trust_remote_code=True,
        freeze_encoder=False,
    ).to(device)
    model.train()

    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(
        f"encoder_hidden={hidden} vocab={vocab_size} "
        f"llada_layers={args.num_hidden_layers} heads={args.num_attention_heads} "
        f"intermediate={args.intermediate_size}"
    )
    print(
        f"LLaDA backbone params = {decoder_params/1e6:.1f}M | "
        f"ESMC encoder params = {encoder_params/1e6:.1f}M | "
        f"total = {(decoder_params+encoder_params)/1e6:.1f}M"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    for batch_size in batch_sizes:
        datamodule.batch_size = batch_size
        loader = datamodule.train_loader(tokenizer)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_seq = 0
        ok = True
        try:
            iterator = iter(loader)
            for _ in range(args.probe_batches):
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in next(iterator).items()}
                max_seq = max(max_seq, int(batch["input_ids"].shape[1]))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model.compute_loss(batch)
                output.loss.backward()
                optimizer.step()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                ok = False
                torch.cuda.empty_cache()
            else:
                raise
        peak = torch.cuda.max_memory_allocated() / 1e9
        status = f"peak={peak:.1f}GB/80GB max_seq={max_seq}" if ok else "OOM"
        print(f"  batch_size={batch_size:>3}: {status}")


if __name__ == "__main__":
    main()
