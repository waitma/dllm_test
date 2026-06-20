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
import math
import os
import sys
import time
from collections import Counter
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
    BioSeqQwenDataCollator,
    BioSeqViewSampler,
    DEFAULT_GRAMMAR_DATA_DIR,
    Esm2SequenceTokenizer,
    GrammarArrowSource,
    GrammarArrowSourceConfig,
    GrammarBioSeqCollator,
    GrammarTokenizer,
    HuggingFaceEsmTokenizerAdapter,
    SourceWithWeight,
    TaskHomogeneousBatchDataset,
    TOKEN_CLASS_NAMES,
    WeightedMixtureDataset,
    default_source_configs,
    source_from_config,
)
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (  # noqa: E402
    BioSeqDiffusionOutput,
    BioSeqDiffusionTransformerConfig,
    BioSeqEncoderDiffusionModel,
    BioSeqNoEncoderDiffusionModel,
    apply_decoder_corruption_to_encoder,
    compute_masked_cross_entropy,
    sample_bioseq_diffusion_noise,
)


FOUNDATION_TRAINING_VIEWS = ("full_denoise",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data = parser.add_argument_group("data")
    data.add_argument("--input-format", choices=["legacy", "grammar_v1"], default="legacy")
    data.add_argument("--split", type=str, default="train")
    data.add_argument("--sources", type=str, default="oas,ots,nanobody,processed_v2")
    data.add_argument("--limit-per-source", type=int, default=None)
    data.add_argument("--epoch-size", type=int, default=None, help="Records emitted per rank/worker epoch; None means infinite stream.")
    data.add_argument("--batch-size", type=int, default=8, help="Mixed-task microbatch size per process.")
    data.add_argument("--max-chain-length", type=int, default=512)
    data.add_argument("--max-sequence-length", type=int, default=None)
    data.add_argument(
        "--full-denoise-probability",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    data.add_argument(
        "--deduplicate-within-batch",
        action="store_true",
        help="Deprecated compatibility flag; mixed-task training does not deduplicate inside batches.",
    )
    data.add_argument("--source-seed", type=int, default=0)
    data.add_argument("--view-seed", type=int, default=0)
    data.add_argument("--oas-weight", type=float, default=1.0)
    data.add_argument("--ots-weight", type=float, default=1.0)
    data.add_argument("--nanobody-weight", type=float, default=1.0)
    data.add_argument("--processed-v2-weight", type=float, default=1.0)
    data.add_argument("--tcr-weight", type=float, default=1.0)
    data.add_argument("--ppi-weight", type=float, default=1.0)
    data.add_argument("--grammar-data-dir", type=Path, default=DEFAULT_GRAMMAR_DATA_DIR)
    data.add_argument("--tokenizer-path", type=Path, default=None, help="Optional HF tokenizer path, e.g. an ESMC snapshot.")

    model = parser.add_argument_group("model")
    model.add_argument("--model-type", choices=["no_encoder", "encoder"], default="no_encoder")
    model.add_argument("--encoder-path", type=Path, default=Path("/c20250601/mj/model_weights/esmc/ESMC-300M"))
    model.add_argument("--freeze-encoder", action="store_true")
    model.add_argument("--encoder-use-flash-attn", action="store_true")
    model.add_argument("--vocab-size", type=int, default=None)
    model.add_argument("--hidden-size", type=int, default=512)
    model.add_argument("--num-hidden-layers", type=int, default=8)
    model.add_argument("--num-attention-heads", type=int, default=8)
    model.add_argument("--intermediate-size", type=int, default=2048)
    model.add_argument("--dropout", type=float, default=0.1)
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
    optim.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="constant")
    optim.add_argument("--min-lr-ratio", type=float, default=0.1)
    optim.add_argument("--grad-clip", type=float, default=1.0)
    optim.add_argument("--bf16", action="store_true")
    optim.add_argument("--seed", type=int, default=42)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    runtime.add_argument("--num-workers", type=int, default=2)
    runtime.add_argument("--log-interval", type=int, default=10)
    runtime.add_argument("--save-interval", type=int, default=200)
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

    wandb_group = parser.add_argument_group("wandb")
    wandb_group.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    wandb_group.add_argument("--wandb-project", type=str, default="bioseq-qwen3-vl")
    wandb_group.add_argument("--wandb-entity", type=str, default=None)
    wandb_group.add_argument("--wandb-run-name", type=str, default=None)
    wandb_group.add_argument("--wandb-dir", type=Path, default=None)
    return parser.parse_args()


def setup_distributed(device_mode: str) -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if device_mode == "cpu":
        use_cuda = False
    elif device_mode == "cuda":
        use_cuda = True
    else:
        use_cuda = torch.cuda.is_available()
    if use_cuda and local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA device(s) are visible. "
            "Use --device cpu for CPU DDP smoke tests, or launch with nproc_per_node <= visible GPUs."
        )
    if distributed:
        backend = "nccl" if use_cuda else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        if use_cuda:
            torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    return distributed, rank, world_size, local_rank, device


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, message: str) -> None:
    if is_main(rank):
        print(message, flush=True)


def move_batch(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: move_batch(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def setup_wandb(args: argparse.Namespace, rank: int, world_size: int):
    if not is_main(rank) or args.wandb_mode == "disabled":
        return None
    wandb_dir = args.wandb_dir or (args.output_dir / "wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_INIT_TIMEOUT", "60")
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        log(rank, f"[wandb] import failed ({exc}); continuing without wandb")
        return None

    config = vars(args).copy()
    config["world_size"] = world_size
    config["effective_batch"] = args.batch_size * args.grad_accum * world_size
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)

    for mode in (args.wandb_mode, "offline"):
        try:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                dir=str(wandb_dir),
                mode=mode,
                config=config,
            )
            log(rank, f"[wandb] initialized mode={mode} dir={wandb_dir}")
            return run
        except Exception as exc:  # noqa: BLE001
            log(rank, f"[wandb] init failed mode={mode}: {exc}")
    return None


def build_tokenizer(args: argparse.Namespace):
    base_tokenizer = (
        Esm2SequenceTokenizer()
        if args.tokenizer_path is None
        else HuggingFaceEsmTokenizerAdapter.from_pretrained(args.tokenizer_path, local_files_only=True)
    )
    if args.input_format == "grammar_v1":
        return GrammarTokenizer(base_tokenizer)
    return base_tokenizer


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


def build_config(args: argparse.Namespace, tokenizer: Any) -> BioSeqDiffusionTransformerConfig:
    return BioSeqDiffusionTransformerConfig(
        vocab_size=infer_vocab_size(tokenizer, args.vocab_size),
        hidden_size=args.hidden_size,
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
        gradient_checkpointing=args.gradient_checkpointing,
        initializer_range=args.initializer_range,
    )


def build_model(args: argparse.Namespace, config: BioSeqDiffusionTransformerConfig) -> torch.nn.Module:
    if args.model_type == "no_encoder":
        return BioSeqNoEncoderDiffusionModel(config)
    return BioSeqEncoderDiffusionModel.from_esmc(
        decoder_config=config,
        encoder_name_or_path=str(args.encoder_path),
        local_files_only=True,
        trust_remote_code=True,
        freeze_encoder=args.freeze_encoder,
        use_flash_attn=args.encoder_use_flash_attn,
    )


def source_weight(args: argparse.Namespace, name: str) -> float:
    return {
        "oas": args.oas_weight,
        "ots": args.ots_weight,
        "nanobody": args.nanobody_weight,
        "processed_v2": args.processed_v2_weight,
        "tcr": args.tcr_weight,
        "ppi": args.ppi_weight,
    }.get(name, 1.0)


def build_loader(
    args: argparse.Namespace,
    tokenizer: Any,
    *,
    split: str | None = None,
    source_seed: int | None = None,
    view_seed: int | None = None,
    epoch_size: int | None = None,
) -> DataLoader:
    split = args.split if split is None else split
    source_seed = args.source_seed if source_seed is None else source_seed
    view_seed = args.view_seed if view_seed is None else view_seed
    epoch_size = args.epoch_size if epoch_size is None else epoch_size
    requested_sources = {item.strip() for item in args.sources.split(",") if item.strip()}
    if args.input_format == "grammar_v1":
        configs = [
            GrammarArrowSourceConfig(
                name=name,
                path=args.grammar_data_dir,
                split=split,
                weight=source_weight(args, name),
                max_records=args.limit_per_source,
            )
            for name in sorted(requested_sources)
        ]
        sources = [
            SourceWithWeight(GrammarArrowSource(config), weight=config.weight)
            for config in configs
        ]
        records = WeightedMixtureDataset(sources, epoch_size=epoch_size, seed=source_seed)
        batches = TaskHomogeneousBatchDataset(
            records,
            batch_size=args.batch_size,
            drop_last=True,
            deduplicate_within_batch=args.deduplicate_within_batch,
        )
        collator = GrammarBioSeqCollator(
            tokenizer=tokenizer,
            max_sequence_length=args.max_sequence_length or 2112,
        )
        return DataLoader(
            batches,
            batch_size=None,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    configs = [
        config
        for config in default_source_configs(split=split, max_records=args.limit_per_source)
        if getattr(config, "name", "") in requested_sources
    ]
    if not configs:
        raise ValueError(f"No sources selected from --sources={args.sources!r}")
    sources = [
        SourceWithWeight(source_from_config(config), weight=source_weight(args, getattr(config, "name", "")))
        for config in configs
    ]
    records = WeightedMixtureDataset(sources, epoch_size=epoch_size, seed=source_seed)
    collator = BioSeqQwenDataCollator(
        tokenizer=tokenizer,
        view_sampler=BioSeqViewSampler(
            allowed_views=FOUNDATION_TRAINING_VIEWS,
            seed=view_seed,
        ),
        max_chain_length=args.max_chain_length,
        max_sequence_length=args.max_sequence_length,
        single_view_per_batch=False,
        require_homogeneous_task=False,
    )
    return DataLoader(
        records,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def build_validation_loader(args: argparse.Namespace, tokenizer: Any) -> DataLoader | None:
    if args.val_interval <= 0 or args.val_batches <= 0:
        return None
    return build_loader(
        args,
        tokenizer,
        split=args.val_split,
        source_seed=args.source_seed + 10_000,
        view_seed=args.view_seed + 10_000,
        epoch_size=None,
    )


def lr_at(
    step: int,
    base_lr: float,
    warmup_steps: int,
    *,
    max_steps: int | None = None,
    scheduler: str = "constant",
    min_lr_ratio: float = 0.1,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if scheduler == "cosine":
        if max_steps is None or max_steps <= warmup_steps:
            raise ValueError("cosine LR scheduling requires max_steps > warmup_steps")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0 and 1")
        progress = min(max((step - warmup_steps) / float(max_steps - warmup_steps), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
    return base_lr


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


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def should_find_unused_parameters(args: argparse.Namespace) -> bool:
    return bool(args.find_unused_parameters or args.model_type == "encoder")


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
        "chain_ids": batch.get("chain_ids"),
        "chain_role_ids": batch.get("chain_role_ids"),
        "position_ids_inner": batch.get("position_ids_inner"),
        "position_ids_chain": batch.get("position_ids_chain"),
        "task_type_ids": batch.get("task_type_ids"),
        "timesteps": timesteps,
    }
    noised_encoder_input_ids = None
    if isinstance(module, BioSeqEncoderDiffusionModel):
        noised_encoder_input_ids = apply_decoder_corruption_to_encoder(
            batch=batch,
            corruption_mask=corruption_mask,
            mask_token_id=config.mask_token_id,
        )
        kwargs.update(
            {
                "encoder_input_ids": noised_encoder_input_ids,
                "encoder_attention_mask": batch.get("encoder_attention_mask"),
                "encoder_residue_mask": batch.get("encoder_residue_mask"),
                "encoder_chain_mask": batch.get("encoder_chain_mask"),
                "encoder_position_ids": batch.get("encoder_position_ids"),
            }
        )
    output = train_model(**kwargs)
    loss = compute_masked_cross_entropy(output.logits, labels, loss_norm=config.loss_norm)
    return replace(
        output,
        loss=loss,
        noised_input_ids=noised_input_ids,
        labels=labels,
        corruption_mask=corruption_mask,
        timesteps=timesteps,
        noised_encoder_input_ids=noised_encoder_input_ids,
    )


def count_names(names: list[str]) -> dict[str, int]:
    if not names:
        return {}
    return dict(Counter(str(name) for name in names))


def format_name_counts(names: list[str]) -> str:
    counts = count_names(names)
    if not counts:
        return "unknown"
    return ",".join(f"{name}:{count}" for name, count in sorted(counts.items()))


def wandb_count_metrics(prefix: str, names: list[str]) -> dict[str, int]:
    return {f"{prefix}/{name}": count for name, count in count_names(names).items()}


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


def cuda_memory_metrics(device: torch.device, distributed: bool) -> dict[str, float]:
    if device.type != "cuda":
        return {}

    scale = 1024.0**3
    values = torch.tensor(
        [
            torch.cuda.memory_allocated(device) / scale,
            torch.cuda.memory_reserved(device) / scale,
            torch.cuda.max_memory_allocated(device) / scale,
            torch.cuda.max_memory_reserved(device) / scale,
            torch.cuda.get_device_properties(device).total_memory / scale,
        ],
        device=device,
        dtype=torch.float32,
    )
    if distributed:
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    return {
        "perf/gpu_mem_allocated_gb": float(values[0].item()),
        "perf/gpu_mem_reserved_gb": float(values[1].item()),
        "perf/gpu_mem_peak_allocated_gb": float(values[2].item()),
        "perf/gpu_mem_peak_reserved_gb": float(values[3].item()),
        "perf/gpu_mem_total_gb": float(values[4].item()),
    }


def any_rank_nonfinite(value: torch.Tensor, distributed: bool) -> bool:
    flag = (~torch.isfinite(value.detach())).to(device=value.device, dtype=torch.int32)
    if distributed:
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


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


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    output_dir: Path,
    tag: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    for name in {f"{tag}.pt", "latest.pt"}:
        target = output_dir / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, target)


def maybe_resume(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank: int,
) -> int:
    if args.resume in (None, "", "none"):
        return 0
    path = args.output_dir / "latest.pt" if args.resume == "auto" else Path(args.resume)
    if not path.is_file():
        log(rank, f"[resume] no checkpoint at {path}; starting from step 0")
        return 0
    payload = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    step = int(payload.get("step", 0))
    log(rank, f"[resume] loaded {path}; resuming at step {step}")
    return step


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank, device = setup_distributed(args.device)
    torch.manual_seed(args.seed + rank)
    torch.set_float32_matmul_precision("high")

    tokenizer = build_tokenizer(args)
    config = build_config(args, tokenizer)
    model = build_model(args, config).to(device)
    optimizer = optimizer_for_model(model, args)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    train_model: torch.nn.Module = model
    if distributed:
        ddp_kwargs = {"find_unused_parameters": should_find_unused_parameters(args)}
        if device.type == "cuda":
            train_model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, **ddp_kwargs)
        else:
            train_model = DistributedDataParallel(model, **ddp_kwargs)

    start_step = maybe_resume(args, train_model, optimizer, device, rank)
    loader = build_loader(args, tokenizer)
    val_loader = build_validation_loader(args, tokenizer)
    val_iter = iter(val_loader) if val_loader is not None else None
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.bf16 else nullcontext()
    wandb_run = setup_wandb(args, rank, world_size)

    log(
        rank,
        f"world_size={world_size} device={device} model_type={args.model_type} "
        f"input_format={args.input_format} parameters={sum(p.numel() for p in model.parameters())} "
        f"effective_batch={args.batch_size * args.grad_accum * world_size} "
        f"training_views={','.join(FOUNDATION_TRAINING_VIEWS)}",
    )
    step = start_step
    micro_step = 0
    window_optimizer_steps = 0
    window_loss_numerator = torch.zeros((), device=device, dtype=torch.float32)
    window_loss_denominator = torch.zeros((), device=device, dtype=torch.float32)
    window_corrupted_tokens = torch.zeros((), device=device, dtype=torch.float32)
    window_eligible_tokens = torch.zeros((), device=device, dtype=torch.float32)
    window_start = time.time()
    train_model.train()
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for batch in loader:
            batch = move_batch(batch, device)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(
                    step,
                    group["initial_lr"],
                    args.warmup_steps,
                    max_steps=args.max_steps,
                    scheduler=args.lr_scheduler,
                    min_lr_ratio=args.min_lr_ratio,
                )

            with autocast:
                output = compute_training_output(train_model, unwrap_model(train_model), batch)
                assert output.loss is not None
                loss = output.loss / args.grad_accum
            if any_rank_nonfinite(output.loss, distributed):
                raise FloatingPointError(f"non-finite training loss detected at step={step}")
            loss.backward()
            denominator = loss_logging_denominator(output, batch, unwrap_model(train_model).config.loss_norm)
            window_loss_numerator += output.loss.detach().to(dtype=torch.float32) * denominator
            window_loss_denominator += denominator
            if output.corruption_mask is not None:
                window_corrupted_tokens += output.corruption_mask.detach().sum().to(dtype=torch.float32)
            window_eligible_tokens += diffusion_eligible_token_count(batch).detach()
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                max_grad_norm = args.grad_clip if args.grad_clip > 0 else float("inf")
                grad_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_grad_norm)
                if any_rank_nonfinite(grad_norm, distributed):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f"non-finite gradient norm detected at step={step}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                window_optimizer_steps += 1

                if step % args.log_interval == 0:
                    elapsed = max(time.time() - window_start, 1e-6)
                    samples_per_sec = (
                        args.batch_size * args.grad_accum * world_size * max(window_optimizer_steps, 1) / elapsed
                    )
                    token_counts = torch.stack(
                        [
                            window_loss_numerator,
                            window_loss_denominator,
                            window_corrupted_tokens,
                            window_eligible_tokens,
                        ]
                    )
                    if distributed:
                        torch.distributed.all_reduce(token_counts, op=torch.distributed.ReduceOp.SUM)
                    optimizer_steps = max(window_optimizer_steps, 1)
                    train_loss = float(token_counts[0].item()) / max(float(token_counts[1].item()), 1.0)
                    corrupted = int(round(float(token_counts[2].item()) / optimizer_steps))
                    eligible = int(round(float(token_counts[3].item()) / optimizer_steps))
                    corruption_rate = float(token_counts[2].item()) / max(float(token_counts[3].item()), 1.0)
                    view_summary = format_name_counts(batch.get("view_names", []))
                    task_summary = format_name_counts(batch.get("task_groups", []))
                    memory_metrics = cuda_memory_metrics(device, distributed)
                    memory_summary = ""
                    if memory_metrics:
                        memory_summary = (
                            f" mem_peak={memory_metrics['perf/gpu_mem_peak_allocated_gb']:.1f}GB"
                            f"/{memory_metrics['perf/gpu_mem_total_gb']:.1f}GB"
                        )
                    log(
                        rank,
                        f"step={step} loss={train_loss:.4f} tasks={task_summary} views={view_summary} "
                        f"corrupted={corrupted} eligible={eligible} corrupt_rate={corruption_rate:.3f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} grad_norm={float(grad_norm.item()):.4f} "
                        f"samples/s={samples_per_sec:.1f}{memory_summary}",
                    )
                    if wandb_run is not None:
                        metrics = {
                            "train/loss": train_loss,
                            "train/loss_denominator": float(token_counts[1].item()),
                            "train/corrupted_tokens": corrupted,
                            "train/eligible_tokens": eligible,
                            "train/corruption_rate": corruption_rate,
                            "train/corrupted_tokens_window": float(token_counts[2].item()),
                            "train/eligible_tokens_window": float(token_counts[3].item()),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/grad_norm": float(grad_norm.item()),
                            "perf/samples_per_sec": samples_per_sec,
                        }
                        metrics.update(memory_metrics)
                        metrics.update(token_class_loss_metrics(output, batch))
                        metrics.update(wandb_count_metrics("batch_views", batch.get("view_names", [])))
                        metrics.update(wandb_count_metrics("batch_tasks", batch.get("task_groups", [])))
                        metrics.update(wandb_count_metrics("batch_sources", batch.get("sources", [])))
                        wandb_run.log(metrics, step=step)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    window_start = time.time()
                    window_optimizer_steps = 0
                    window_loss_numerator.zero_()
                    window_loss_denominator.zero_()
                    window_corrupted_tokens.zero_()
                    window_eligible_tokens.zero_()

                if val_iter is not None and args.val_interval > 0 and step > 0 and step % args.val_interval == 0:
                    val_metrics = evaluate_validation(
                        train_model=train_model,
                        module=unwrap_model(train_model),
                        val_iter=val_iter,
                        args=args,
                        device=device,
                        distributed=distributed,
                    )
                    log(
                        rank,
                        f"step={step} val_loss={val_metrics['val/loss']:.4f} "
                        f"val_corrupted={val_metrics['val/corrupted_tokens']:.1f} "
                        f"val_eligible={val_metrics['val/eligible_tokens']:.1f} "
                        f"val_corrupt_rate={val_metrics['val/corruption_rate']:.3f}",
                    )
                    if wandb_run is not None:
                        wandb_run.log(val_metrics, step=step)

                if args.save_interval and step > 0 and step % args.save_interval == 0:
                    if is_main(rank):
                        save_checkpoint(train_model, optimizer, step, args, args.output_dir, tag="latest")
                        log(rank, f"saved checkpoint -> {args.output_dir / 'latest.pt'}")
                    if distributed:
                        torch.distributed.barrier()

                step += 1
                if step >= args.max_steps:
                    break
        if args.epoch_size is None:
            break

    if is_main(rank):
        save_checkpoint(train_model, optimizer, step, args, args.output_dir, tag="final")
        log(rank, f"saved final checkpoint -> {args.output_dir / 'final.pt'}")
    if wandb_run is not None:
        wandb_run.finish()
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
