"""Multi-node / multi-GPU trainer for BioSeq foundation diffusion.

This entry point trains the new BioSeq foundation-model path under
``dllm/pipelines/qwen3_vl_arch``. It supports the no-encoder model and the
ESMC/ESM feature-conditioned path behind the same DDP training loop.

Examples
--------
Single-process smoke run::

    python /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py \
        --model-type no_encoder --limit-per-source 64 --batch-size 2 --max-steps 4 \
        --hidden-size 64 --num-hidden-layers 2 --num-attention-heads 4 --intermediate-size 128

Single node, 8 GPUs::

    torchrun --standalone --nproc_per_node=8 \
        /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py \
        --model-type no_encoder --batch-size 8 --max-steps 100000 --bf16

Multi node, 2 nodes x 8 GPUs::

    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py \
        --model-type no_encoder --batch-size 8 --max-steps 1000000 --bf16
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.data import (  # noqa: E402
    DEFAULT_GRAMMAR_DATA_DIR,
    GrammarDataModule,
    TOKEN_CLASS_NAMES,
)
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (  # noqa: E402
    BioSeqDiffusionOutput,
    BioSeqDiffusionTransformerConfig,
    BioSeqEncoderDiffusionModel,
    BioSeqNoEncoderDiffusionModel,
    apply_decoder_corruption_to_encoder,
    compute_masked_cross_entropy,
    forbidden_diffusion_target_token_ids,
    sample_bioseq_diffusion_noise,
)
from dllm.pipelines.qwen3_vl_arch.training import (  # noqa: E402
    BioSeqTrainer,
    TrainStepFns,
    lr_at,
    move_batch,
    setup_distributed,
    setup_wandb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data = parser.add_argument_group("data")
    data.add_argument("--split", type=str, default="train")
    data.add_argument("--sources", type=str, default="oas,ots,tcr,ppi")
    data.add_argument("--limit-per-source", type=int, default=None)
    data.add_argument("--epoch-size", type=int, default=None, help="Records emitted per rank/worker epoch; None means infinite stream.")
    data.add_argument("--batch-size", type=int, default=8, help="Mixed-task microbatch size per process.")
    data.add_argument("--max-sequence-length", type=int, default=None)
    data.add_argument(
        "--deduplicate-within-batch",
        action="store_true",
        help="Skip duplicate records within a task-homogeneous batch.",
    )
    data.add_argument("--source-seed", type=int, default=0)
    data.add_argument("--oas-weight", type=float, default=1.0)
    data.add_argument("--ots-weight", type=float, default=1.0)
    data.add_argument("--nanobody-weight", type=float, default=1.0)
    data.add_argument("--processed-v2-weight", type=float, default=1.0)
    data.add_argument("--tcr-weight", type=float, default=1.0)
    data.add_argument("--ppi-weight", type=float, default=1.0)
    data.add_argument("--grammar-data-dir", type=Path, default=DEFAULT_GRAMMAR_DATA_DIR)
    data.add_argument("--tokenizer-path", type=Path, default=None, help="Optional HF tokenizer path, e.g. an ESMC snapshot.")

    model = parser.add_argument_group("model")
    model.add_argument("--model-type", choices=["no_encoder", "encoder", "esm2"], default="no_encoder")
    model.add_argument(
        "--encoder-path",
        type=Path,
        default=Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-300M"),
    )
    model.add_argument("--freeze-encoder", action="store_true")
    model.add_argument("--encoder-use-flash-attn", action="store_true")
    model.add_argument("--vocab-size", type=int, default=None)
    model.add_argument("--hidden-size", type=int, default=512)
    model.add_argument(
        "--align-hidden-size-to-encoder",
        action="store_true",
        help="For no_encoder ablations, force decoder hidden_size to match --encoder-path latent dim.",
    )
    model.add_argument("--num-hidden-layers", type=int, default=8)
    model.add_argument("--num-attention-heads", type=int, default=8)
    model.add_argument("--intermediate-size", type=int, default=2048)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--qk-norm", action="store_true", help="Apply RMSNorm to query/key per head for attention stability.")
    model.add_argument("--gradient-checkpointing", action="store_true")
    model.add_argument("--initializer-range", type=float, default=0.02)
    model.add_argument("--max-position-embeddings", type=int, default=4096)
    model.add_argument("--max-chain-positions", type=int, default=64)
    model.add_argument("--max-chain-roles", type=int, default=32)
    model.add_argument("--max-task-types", type=int, default=32)
    model.add_argument("--time-epsilon", type=float, default=1e-3)
    model.add_argument("--loss-norm", choices=["token", "sequence", "batch"], default="token")

    optim = parser.add_argument_group("optim")
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--encoder-lr", type=float, default=None)
    optim.add_argument("--weight-decay", type=float, default=0.01)
    optim.add_argument("--grad-accum", type=int, default=1)
    optim.add_argument("--max-steps", type=int, default=1000)
    optim.add_argument("--warmup-steps", type=int, default=0)
    optim.add_argument("--warmup-init-lr", type=float, default=1e-7)
    optim.add_argument("--lr-scheduler", choices=["constant", "cosine", "polynomial"], default="constant")
    optim.add_argument("--min-lr-ratio", type=float, default=0.1)
    optim.add_argument("--grad-clip", type=float, default=1.0)
    optim.add_argument("--bf16", action="store_true")
    optim.add_argument("--seed", type=int, default=42)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    runtime.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers. Default 0 is the safe path for the memory-mapped Arrow "
            "sources: each extra worker is a separate process that shards the infinite "
            "weighted stream independently, so >0 risks per-rank first-batch desync and "
            "DDP collective hangs under torchrun."
        ),
    )
    runtime.add_argument("--log-interval", type=int, default=10)
    runtime.add_argument("--save-interval", type=int, default=200)
    runtime.add_argument(
        "--save-top-k",
        type=int,
        default=10,
        help="Keep up to K full checkpoints with the lowest val/loss under output_dir/checkpoints/. 0 disables.",
    )
    runtime.add_argument("--val-interval", type=int, default=1000, help="Run validation every N optimizer steps; 0 disables.")
    runtime.add_argument("--val-batches", type=int, default=20, help="Validation micro-batches per validation pass.")
    runtime.add_argument("--val-split", type=str, default="valid")
    runtime.add_argument("--resume", type=str, default="auto", help="'auto', 'none', or an explicit checkpoint path.")
    runtime.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output/qwen3_vl_bioseq_ddp")
    runtime.add_argument(
        "--find-unused-parameters",
        action="store_true",
        help="Enable DDP unused-parameter detection. Encoder mode enables this automatically.",
    )
    runtime.add_argument(
        "--debug-ddp-timing",
        action="store_true",
        help="Log per-rank batch/forward/backward/optimizer timing for DDP hang diagnosis.",
    )

    wandb_group = parser.add_argument_group("wandb")
    wandb_group.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    wandb_group.add_argument("--wandb-project", type=str, default="bioseq-qwen3-vl")
    wandb_group.add_argument("--wandb-entity", type=str, default=None)
    wandb_group.add_argument("--wandb-run-name", type=str, default=None)
    wandb_group.add_argument("--wandb-dir", type=Path, default=None)
    return parser.parse_args()


def build_tokenizer(args: argparse.Namespace):
    return GrammarDataModule.from_args(args).build_tokenizer()


def infer_vocab_size(tokenizer: Any, requested_vocab_size: int | None) -> int:
    if requested_vocab_size is not None:
        return int(requested_vocab_size)
    value = getattr(tokenizer, "vocab_size", None)
    if value is not None:
        return int(value)
    hf_tokenizer = getattr(tokenizer, "tokenizer", None)
    if hf_tokenizer is not None:
        return int(len(hf_tokenizer))
    return 33


def infer_encoder_hidden_size_from_path(encoder_path: Path) -> int:
    config_path = encoder_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Encoder config not found: {config_path}")
    with config_path.open() as handle:
        config = json.load(handle)
    for key in ("hidden_size", "d_model", "embed_dim", "encoder_embed_dim"):
        if key in config:
            return int(config[key])
    raise ValueError(f"Could not infer encoder hidden size from {config_path}")


def resolve_hidden_size(args: argparse.Namespace) -> int:
    if args.model_type in {"encoder", "esm2"}:
        return infer_encoder_hidden_size_from_path(args.encoder_path)
    if args.align_hidden_size_to_encoder:
        return infer_encoder_hidden_size_from_path(args.encoder_path)
    return int(args.hidden_size)


def build_config(args: argparse.Namespace, tokenizer: Any) -> BioSeqDiffusionTransformerConfig:
    hidden_size = resolve_hidden_size(args)
    return BioSeqDiffusionTransformerConfig(
        vocab_size=infer_vocab_size(tokenizer, args.vocab_size),
        hidden_size=hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        max_position_embeddings=args.max_position_embeddings,
        max_chain_positions=args.max_chain_positions,
        max_chain_roles=args.max_chain_roles,
        max_task_types=args.max_task_types,
        pad_token_id=int(tokenizer.pad_token_id),
        mask_token_id=int(tokenizer.mask_token_id),
        time_epsilon=args.time_epsilon,
        loss_norm=args.loss_norm,
        qk_norm=args.qk_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        initializer_range=args.initializer_range,
    )


def build_model(args: argparse.Namespace, config: BioSeqDiffusionTransformerConfig) -> torch.nn.Module:
    if args.model_type == "no_encoder":
        return BioSeqNoEncoderDiffusionModel(config)
    if args.model_type == "esm2":
        return BioSeqEncoderDiffusionModel.from_hf_encoder(
            decoder_config=config,
            encoder_name_or_path=str(args.encoder_path),
            local_files_only=True,
            trust_remote_code=True,
            freeze_encoder=args.freeze_encoder,
        )
    return BioSeqEncoderDiffusionModel.from_esmc(
        decoder_config=config,
        encoder_name_or_path=str(args.encoder_path),
        local_files_only=True,
        trust_remote_code=True,
        freeze_encoder=args.freeze_encoder,
        use_flash_attn=args.encoder_use_flash_attn,
    )


def build_loader(
    args: argparse.Namespace,
    tokenizer: Any,
    *,
    split: str | None = None,
    source_seed: int | None = None,
    epoch_size: int | None = None,
) -> DataLoader:
    return GrammarDataModule.from_args(args).loader(
        tokenizer,
        split=split,
        source_seed=source_seed,
        epoch_size=epoch_size,
    )


def build_validation_loader(args: argparse.Namespace, tokenizer: Any) -> DataLoader | None:
    return GrammarDataModule.from_args(args).val_loader(tokenizer)


def optimizer_for_model(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if isinstance(model, BioSeqEncoderDiffusionModel):
        encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
        params = [
            {"params": [p for p in model.decoder.parameters() if p.requires_grad], "lr": args.lr},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": encoder_lr},
        ]
        params = [group for group in params if group["params"]]
        return torch.optim.AdamW(params, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def should_find_unused_parameters(args: argparse.Namespace) -> bool:
    return bool(args.find_unused_parameters or args.model_type in {"encoder", "esm2"})


def compute_training_output(
    train_model: torch.nn.Module,
    module: torch.nn.Module,
    batch: dict[str, Any],
) -> BioSeqDiffusionOutput:
    config = module.config
    noised_input_ids, labels, corruption_mask, timesteps = sample_bioseq_diffusion_noise(
        batch=batch,
        mask_token_id=config.mask_token_id,
        time_epsilon=config.time_epsilon,
    )
    kwargs: dict[str, Any] = {
        "input_ids": noised_input_ids,
        "attention_mask": batch.get("attention_mask"),
        "position_ids_inner": batch.get("position_ids_inner"),
        "position_ids_chain": batch.get("position_ids_chain"),
        "timesteps": timesteps,
    }
    if batch.get("chain_ids") is not None:
        kwargs["chain_ids"] = batch.get("chain_ids")
        kwargs["residue_mask"] = batch.get("residue_mask")
    noised_encoder_input_ids = None
    if isinstance(module, BioSeqEncoderDiffusionModel):
        noised_encoder_input_ids = apply_decoder_corruption_to_encoder(
            batch=batch,
            corruption_mask=corruption_mask,
            mask_token_id=config.mask_token_id,
        )
        encoder_kwargs: dict[str, Any] = {
            "encoder_input_ids": noised_encoder_input_ids,
            "encoder_attention_mask": batch.get("encoder_attention_mask"),
            "encoder_residue_mask": batch.get("encoder_residue_mask"),
            "encoder_chain_mask": batch.get("encoder_chain_mask"),
        }
        if batch.get("encoder_position_ids") is not None:
            encoder_kwargs["encoder_position_ids"] = batch.get("encoder_position_ids")
        kwargs.update(encoder_kwargs)
    output = train_model(**kwargs)
    forbidden = forbidden_diffusion_target_token_ids(config)
    loss = compute_masked_cross_entropy(
        output.logits,
        labels,
        loss_norm=config.loss_norm,
        forbidden_token_ids=forbidden,
    )
    return replace(
        output,
        loss=loss,
        noised_input_ids=noised_input_ids,
        labels=labels,
        corruption_mask=corruption_mask,
        timesteps=timesteps,
        noised_encoder_input_ids=noised_encoder_input_ids,
    )


def diffusion_eligible_token_count(batch: dict[str, Any]) -> torch.Tensor:
    loss_mask = batch.get("diffusion_loss_mask", batch.get("diffusion_target_mask"))
    if loss_mask is None:
        input_ids = batch.get("input_ids")
        device = input_ids.device if torch.is_tensor(input_ids) else torch.device("cpu")
        return torch.zeros((), device=device, dtype=torch.float32)

    explicit_eligible_mask = batch.get("diffusion_eligible_mask")
    eligible_mask = (
        explicit_eligible_mask.bool()
        if explicit_eligible_mask is not None
        else loss_mask.bool()
    )
    attention_mask = batch.get("attention_mask")
    residue_mask = batch.get("residue_mask")
    if attention_mask is not None:
        eligible_mask = eligible_mask & attention_mask.bool()
    if explicit_eligible_mask is None and residue_mask is not None:
        eligible_mask = eligible_mask & residue_mask.bool()
    return eligible_mask.sum().to(dtype=torch.float32)


def token_class_loss_metrics(
    output: BioSeqDiffusionOutput,
    batch: dict[str, Any],
) -> dict[str, float]:
    if output.labels is None or "token_class_ids" not in batch:
        return {}
    token_loss = torch.nn.functional.cross_entropy(
        output.logits.detach().float().reshape(-1, output.logits.shape[-1]),
        output.labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(output.labels)
    metrics: dict[str, float] = {}
    for class_id, name in TOKEN_CLASS_NAMES.items():
        mask = output.labels.ne(-100) & batch["token_class_ids"].eq(class_id)
        if mask.any():
            metrics[f"train/loss_{name}"] = float(token_loss[mask].mean().item())
            metrics[f"train/corrupted_{name}_tokens"] = float(mask.sum().item())
    return metrics


def loss_logging_denominator(output: BioSeqDiffusionOutput, batch: dict[str, Any], loss_norm: str) -> torch.Tensor:
    if loss_norm == "token" and output.corruption_mask is not None:
        return output.corruption_mask.detach().sum().to(dtype=torch.float32).clamp_min(1.0)
    input_ids = batch.get("input_ids")
    if torch.is_tensor(input_ids):
        return torch.tensor(float(input_ids.shape[0]), device=input_ids.device, dtype=torch.float32).clamp_min(1.0)
    device = output.loss.device if output.loss is not None else torch.device("cpu")
    return torch.ones((), device=device, dtype=torch.float32)


@torch.no_grad()
def evaluate_validation(
    train_model: torch.nn.Module,
    module: torch.nn.Module,
    val_iter: Any,
    args: argparse.Namespace,
    device: torch.device,
    distributed: bool,
) -> dict[str, float]:
    train_model.eval()
    loss_numerator = torch.zeros((), device=device, dtype=torch.float32)
    loss_denominator = torch.zeros((), device=device, dtype=torch.float32)
    corrupted_sum = torch.zeros((), device=device, dtype=torch.float32)
    eligible_sum = torch.zeros((), device=device, dtype=torch.float32)
    batch_count = torch.zeros((), device=device, dtype=torch.float32)

    for _ in range(args.val_batches):
        batch = move_batch(next(val_iter), device)
        autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.bf16 else nullcontext()
        with autocast:
            output = compute_training_output(train_model, module, batch)
        assert output.loss is not None
        denominator = loss_logging_denominator(output, batch, module.config.loss_norm)
        loss_numerator += output.loss.detach().to(dtype=torch.float32) * denominator
        loss_denominator += denominator
        if output.corruption_mask is not None:
            corrupted_sum += output.corruption_mask.detach().sum().to(dtype=torch.float32)
        eligible_sum += diffusion_eligible_token_count(batch).detach()
        batch_count += 1.0

    values = torch.stack([loss_numerator, loss_denominator, corrupted_sum, eligible_sum, batch_count])
    if distributed:
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)

    total_loss_denominator = max(float(values[1].item()), 1.0)
    total_batches = max(float(values[4].item()), 1.0)
    total_eligible = max(float(values[3].item()), 1.0)
    val_loss = float(values[0].item()) / total_loss_denominator
    val_corrupted_tokens = float(values[2].item()) / total_batches
    val_eligible_tokens = float(values[3].item()) / total_batches
    val_corruption_rate = float(values[2].item()) / total_eligible
    train_model.train()
    return {
        "val/loss": val_loss,
        "val/loss_denominator": total_loss_denominator,
        "val/corrupted_tokens": val_corrupted_tokens,
        "val/eligible_tokens": val_eligible_tokens,
        "val/corruption_rate": val_corruption_rate,
        "val/batches": total_batches,
        "val_loss": val_loss,
        "val_loss_denominator": total_loss_denominator,
        "val_corrupted_tokens": val_corrupted_tokens,
        "val_eligible_tokens": val_eligible_tokens,
        "val_corruption_rate": val_corruption_rate,
        "val_batches": total_batches,
    }


def main() -> None:
    args = parse_args()
    ctx = setup_distributed(args.device)
    torch.manual_seed(args.seed + ctx.rank)
    torch.set_float32_matmul_precision("high")

    datamodule = GrammarDataModule.from_args(args)
    tokenizer = datamodule.build_tokenizer()
    config = build_config(args, tokenizer)
    model = build_model(args, config).to(ctx.device)
    optimizer = optimizer_for_model(model, args)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    train_model: torch.nn.Module = model
    if ctx.distributed:
        ddp_kwargs = {"find_unused_parameters": should_find_unused_parameters(args)}
        if ctx.device.type == "cuda":
            train_model = DistributedDataParallel(
                model, device_ids=[ctx.local_rank], output_device=ctx.local_rank, **ddp_kwargs
            )
        else:
            train_model = DistributedDataParallel(model, **ddp_kwargs)

    step_fns = TrainStepFns(
        compute_output=compute_training_output,
        loss_denominator=loss_logging_denominator,
        eligible_token_count=diffusion_eligible_token_count,
        token_class_metrics=token_class_loss_metrics,
        evaluate_validation=evaluate_validation,
    )
    wandb_run = setup_wandb(args, ctx.rank, ctx.world_size)
    trainer = BioSeqTrainer(
        args=args,
        ctx=ctx,
        train_model=train_model,
        optimizer=optimizer,
        step_fns=step_fns,
        wandb_run=wandb_run,
    )
    trainer.resume()
    loader = datamodule.train_loader(tokenizer)
    val_loader = datamodule.val_loader(tokenizer)
    val_iter = iter(val_loader) if val_loader is not None else None
    trainer.fit(loader, val_iter)


if __name__ == "__main__":
    main()
