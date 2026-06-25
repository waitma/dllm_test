"""Model-agnostic DDP trainer for BioSeq foundation diffusion.

``BioSeqTrainer`` owns everything that is *not* model/data specific: distributed
context, the LR schedule, the gradient-accumulation loop, non-finite guards,
window logging + wandb, validation orchestration, and checkpoint/resume. The
model-specific pieces (how to compute the diffusion loss, denominators, eligible
token counts, per-class metrics, and validation) are injected via
:class:`TrainStepFns`, so the loop never imports an entry-point script.

This module is imported by the entry point; it is not run directly. See
``/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py``.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn.parallel import DistributedDataParallel

from .checkpointing import ValLossTopKCheckpointManager, load_resume_payload


@dataclass
class DistributedContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


@dataclass
class TrainStepFns:
    """Model/data-specific callables injected into the generic loop."""

    compute_output: Callable[[torch.nn.Module, torch.nn.Module, dict[str, Any]], Any]
    loss_denominator: Callable[[Any, dict[str, Any], str], torch.Tensor]
    eligible_token_count: Callable[[dict[str, Any]], torch.Tensor]
    token_class_metrics: Callable[[Any, dict[str, Any]], dict[str, float]]
    evaluate_validation: Callable[..., dict[str, float]]


def setup_distributed(device_mode: str) -> DistributedContext:
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
    return DistributedContext(
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


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


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def any_rank_nonfinite(value: torch.Tensor, distributed: bool) -> bool:
    flag = (~torch.isfinite(value.detach())).to(device=value.device, dtype=torch.int32)
    if distributed:
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


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


def lr_at(
    step: int,
    base_lr: float,
    warmup_steps: int,
    *,
    max_steps: int | None = None,
    scheduler: str = "constant",
    min_lr_ratio: float = 0.1,
    warmup_init_lr: float = 1e-7,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        if scheduler == "polynomial":
            return warmup_init_lr + (base_lr - warmup_init_lr) * float(step) / float(warmup_steps)
        return base_lr * float(step + 1) / float(warmup_steps)
    if scheduler in {"cosine", "polynomial"}:
        if max_steps is None or max_steps <= warmup_steps:
            raise ValueError(f"{scheduler} LR scheduling requires max_steps > warmup_steps")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0 and 1")
        progress = min(max((step - warmup_steps) / float(max_steps - warmup_steps), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
    return base_lr


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


class BioSeqTrainer:
    """Generic grad-accumulation DDP loop driven by injected step functions."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        ctx: DistributedContext,
        train_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step_fns: TrainStepFns,
        wandb_run: Any = None,
    ) -> None:
        self.args = args
        self.ctx = ctx
        self.train_model = train_model
        self.optimizer = optimizer
        self.step_fns = step_fns
        self.wandb_run = wandb_run
        self.start_step = 0
        save_top_k = int(getattr(args, "save_top_k", 10))
        self.topk_checkpoints = (
            ValLossTopKCheckpointManager(args.output_dir, save_top_k=save_top_k)
            if save_top_k > 0
            else None
        )

    @staticmethod
    def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _build_payload(self, step: int, *, val_loss: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_state_dict": self.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(self.args).items()
            },
        }
        if val_loss is not None:
            payload["val_loss"] = float(val_loss)
        return payload

    @property
    def module(self) -> torch.nn.Module:
        return unwrap_model(self.train_model)

    def _dbg(self, message: str) -> None:
        if self.args.debug_ddp_timing:
            print(f"[rank={self.ctx.rank}] {message}", flush=True)

    def save_checkpoint(self, step: int, tag: str, *, val_loss: float | None = None) -> None:
        output_dir: Path = self.args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self._build_payload(step, val_loss=val_loss)
        for name in {f"{tag}.pt", "latest.pt"}:
            self._atomic_save(output_dir / name, payload)

    def resume(self) -> int:
        args = self.args
        rank = self.ctx.rank
        if args.resume in (None, "", "none"):
            self.start_step = 0
            return 0
        path = args.output_dir / "latest.pt" if args.resume == "auto" else Path(args.resume)
        if not path.is_file():
            log(rank, f"[resume] no checkpoint at {path}; starting from step 0")
            self.start_step = 0
            return 0
        payload = torch.load(path, map_location="cpu")
        step, vocab_expanded = load_resume_payload(
            self.module,
            self.optimizer,
            payload,
            log=lambda message: log(rank, message),
        )
        if not vocab_expanded:
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.ctx.device)
        log(rank, f"[resume] loaded {path}; resuming at step {step}")
        self.start_step = step
        return step

    def fit(self, loader: Any, val_iter: Any) -> None:
        args = self.args
        ctx = self.ctx
        device = ctx.device
        distributed = ctx.distributed
        rank = ctx.rank
        world_size = ctx.world_size
        train_model = self.train_model
        optimizer = self.optimizer
        module = self.module
        wandb_run = self.wandb_run

        if distributed and args.num_workers > 0:
            log(
                rank,
                f"[warn] num_workers={args.num_workers} with DDP: the weighted Arrow stream is "
                "sharded per worker process and can desync the first batch across ranks "
                "(risking NCCL collective timeouts). num_workers=0 is the validated setting.",
            )
        if self.topk_checkpoints is not None and args.val_interval <= 0:
            log(
                rank,
                "[warn] save_top_k is enabled but val_interval<=0; top-k val-loss checkpoints will never be written.",
            )
        autocast = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if args.bf16
            else nullcontext()
        )
        if distributed:
            torch.distributed.barrier()

        log(
            rank,
            f"world_size={world_size} device={device} model_type={args.model_type} "
            f"input_format=grammar_v1 parameters={sum(p.numel() for p in module.parameters())} "
            f"effective_batch={args.batch_size * args.grad_accum * world_size}",
        )

        step = self.start_step
        micro_step = 0
        window_optimizer_steps = 0
        window_loss_numerator = torch.zeros((), device=device, dtype=torch.float32)
        window_loss_denominator = torch.zeros((), device=device, dtype=torch.float32)
        window_corrupted_tokens = torch.zeros((), device=device, dtype=torch.float32)
        window_eligible_tokens = torch.zeros((), device=device, dtype=torch.float32)
        window_start = time.time()
        train_model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_wait_start = time.time()
        while step < args.max_steps:
            for batch in loader:
                self._dbg(
                    f"batch_ready step={step} micro={micro_step} wait_s={time.time() - batch_wait_start:.2f}"
                )
                batch_wait_start = time.time()
                batch = move_batch(batch, device)
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(
                        step,
                        group["initial_lr"],
                        args.warmup_steps,
                        max_steps=args.max_steps,
                        scheduler=args.lr_scheduler,
                        min_lr_ratio=args.min_lr_ratio,
                        warmup_init_lr=args.warmup_init_lr,
                    )

                forward_start = time.time()
                with autocast:
                    output = self.step_fns.compute_output(train_model, module, batch)
                    assert output.loss is not None
                    loss = output.loss / args.grad_accum
                self._dbg(
                    f"forward_done step={step} micro={micro_step} loss={float(output.loss.item()):.4f} "
                    f"forward_s={time.time() - forward_start:.2f}"
                )
                backward_start = time.time()
                if any_rank_nonfinite(output.loss, distributed):
                    raise FloatingPointError(f"non-finite training loss detected at step={step}")
                loss.backward()
                self._dbg(
                    f"backward_done step={step} micro={micro_step} backward_s={time.time() - backward_start:.2f}"
                )
                denominator = self.step_fns.loss_denominator(output, batch, module.config.loss_norm)
                window_loss_numerator += output.loss.detach().to(dtype=torch.float32) * denominator
                window_loss_denominator += denominator
                if output.corruption_mask is not None:
                    window_corrupted_tokens += output.corruption_mask.detach().sum().to(dtype=torch.float32)
                window_eligible_tokens += self.step_fns.eligible_token_count(batch).detach()
                micro_step += 1

                if micro_step % args.grad_accum == 0:
                    clip_start = time.time()
                    max_grad_norm = args.grad_clip if args.grad_clip > 0 else float("inf")
                    grad_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_grad_norm)
                    self._dbg(
                        f"clip_grad_done step={step} grad_norm={float(grad_norm.item()):.4f} "
                        f"clip_s={time.time() - clip_start:.2f}"
                    )
                    if any_rank_nonfinite(grad_norm, distributed):
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(f"non-finite gradient norm detected at step={step}")
                    opt_start = time.time()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    self._dbg(f"optimizer_step_done step={step} opt_s={time.time() - opt_start:.2f}")
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
                            metrics.update(self.step_fns.token_class_metrics(output, batch))
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
                        val_metrics = self.step_fns.evaluate_validation(
                            train_model=train_model,
                            module=module,
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
                        if self.topk_checkpoints is not None and is_main(rank):
                            val_loss = float(val_metrics["val/loss"])
                            saved = self.topk_checkpoints.maybe_save(
                                step,
                                val_loss,
                                self._build_payload(step, val_loss=val_loss),
                                save_payload=self._atomic_save,
                            )
                            if saved is not None:
                                topk_rank = next(
                                    index + 1
                                    for index, entry in enumerate(self.topk_checkpoints.entries)
                                    if entry.path == saved.path
                                )
                                best = self.topk_checkpoints.best_entry
                                best_note = ""
                                if best is not None and best.path == saved.path:
                                    best_note = f" (new best -> {args.output_dir / 'best.pt'})"
                                log(
                                    rank,
                                    f"saved top-k checkpoint rank={topk_rank}/"
                                    f"{self.topk_checkpoints.save_top_k} "
                                    f"step={step} val_loss={val_loss:.4f} -> {saved.path}{best_note}",
                                )
                        if distributed and self.topk_checkpoints is not None:
                            torch.distributed.barrier()

                    if args.save_interval and step > 0 and step % args.save_interval == 0:
                        if is_main(rank):
                            self.save_checkpoint(step, tag="latest")
                            log(rank, f"saved checkpoint -> {args.output_dir / 'latest.pt'}")
                        if distributed:
                            torch.distributed.barrier()

                    step += 1
                    if step >= args.max_steps:
                        break
            if args.epoch_size is None:
                break

        if is_main(rank):
            self.save_checkpoint(step, tag="final")
            log(rank, f"saved final checkpoint -> {args.output_dir / 'final.pt'}")
        if wandb_run is not None:
            wandb_run.finish()
        if distributed:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
