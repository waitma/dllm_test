"""Multi-node / multi-GPU trainer for the exact Ophiuchus-Ab diffusion model.

This entry point trains :class:`MultiChainOphiuchusAbModel` on a mixture of the
local immune corpora (OAS paired antibody, OTS paired TCR, nanobody VHH) with
**variable sequence length** (no fixed 150/128 padding) via
:class:`MultiChainDynamicCollator`.

It is launchable with ``torchrun`` for single-node multi-GPU and multi-node
multi-GPU, and also runs as a plain ``python`` process (single GPU or CPU) for
quick smoke checks.

Examples
--------
Single process smoke check (CPU/GPU)::

    python /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_bioseq_ddp.py \
        --limit-per-source 200 --batch-size 2 --max-steps 4 --max-length 160

Single node, 8 GPUs::

    torchrun --standalone --nproc_per_node=8 \
        /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_bioseq_ddp.py \
        --batch-size 16 --max-steps 100000 --bf16

Multi node (2 nodes x 8 GPUs)::

    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_bioseq_ddp.py \
        --batch-size 16 --max-steps 1000000 --bf16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.bioseq import (
    Esm2ProteinTokenizer,
    MultiChainOphiuchusAbModel,
    OphiuchusAbTrainStepConfig,
    compute_ophiuchus_ab_training_loss,
    load_ophiuchus_checkpoint,
)
from dllm.pipelines.bioseq.datasets import build_mixed_immune_dataset, default_immune_specs
from dllm.pipelines.bioseq.ophiuchus.collator import MultiChainDynamicCollator
from dllm.pipelines.bioseq.ophiuchus.model import OphiuchusAbBackbone


def select_source_specs(specs, sources: str):
    requested = [item.strip() for item in sources.split(",") if item.strip()]
    if not requested:
        raise ValueError("--sources must select at least one source")

    by_name = {spec.name: spec for spec in specs}
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown --sources entries: {unknown}; available={sorted(by_name)}")
    return [by_name[name] for name in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_argument_group("data")
    data.add_argument("--oas-dir", type=Path, default=default_immune_specs()[0].path)
    data.add_argument("--ots-dir", type=Path, default=default_immune_specs()[1].path)
    data.add_argument("--nanobody-dir", type=Path, default=default_immune_specs()[2].path)
    data.add_argument("--split", type=str, default="train")
    data.add_argument("--sources", type=str, default="oas,ots,nanobody",
                      help="Comma-separated subset of: oas,ots,nanobody.")
    data.add_argument("--limit-per-source", type=int, default=None,
                      help="Cap rows loaded per dataset (use small values for smoke runs).")
    data.add_argument("--oas-weight", type=float, default=1.0)
    data.add_argument("--ots-weight", type=float, default=1.0)
    data.add_argument("--nanobody-weight", type=float, default=1.0)
    data.add_argument("--max-length", type=int, default=512, help="Per-chain token cap (incl <cls>/<eos>).")

    model = parser.add_argument_group("model")
    model.add_argument("--checkpoint-path", type=Path, default=None,
                       help="Optional Ophiuchus-Ab .ckpt to initialize the backbone.")
    model.add_argument("--init-multimer", action="store_true",
                       help="Initialize multimer attention from self attention before training.")

    optim = parser.add_argument_group("optim")
    optim.add_argument("--batch-size", type=int, default=8, help="Per-process batch size.")
    optim.add_argument("--grad-accum", type=int, default=1)
    optim.add_argument("--max-steps", type=int, default=1000)
    optim.add_argument("--lr", type=float, default=4e-5)
    optim.add_argument("--weight-decay", type=float, default=0.01)
    optim.add_argument("--warmup-steps", type=int, default=0)
    optim.add_argument("--grad-clip", type=float, default=1.0)
    optim.add_argument("--bf16", action="store_true")
    optim.add_argument("--seed", type=int, default=42)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--num-workers", type=int, default=2)
    runtime.add_argument("--log-interval", type=int, default=1)
    runtime.add_argument("--save-interval", type=int, default=200,
                         help="Save a full checkpoint (weights+optimizer+step) every N steps; 0 disables.")
    runtime.add_argument("--resume", type=str, default="auto",
                         help="'auto' resumes <output-dir>/latest.pt if present, a path resumes that file, 'none' disables.")
    runtime.add_argument("--output-dir", type=Path,
                         default=PROJECT_ROOT / ".models/bioseq/ophiuchus-ab-mixed")
    runtime.add_argument("--find-unused-parameters", action="store_true")

    wandb_group = parser.add_argument_group("wandb")
    wandb_group.add_argument("--wandb-mode", type=str, default="online",
                             choices=["online", "offline", "disabled"])
    wandb_group.add_argument("--wandb-project", type=str, default="bioseq-ophiuchus")
    wandb_group.add_argument("--wandb-entity", type=str, default=None)
    wandb_group.add_argument("--wandb-run-name", type=str, default=None)
    wandb_group.add_argument("--wandb-dir", type=Path, default=None,
                             help="Directory for wandb run files; defaults to <output-dir>/wandb.")
    return parser.parse_args()


def setup_wandb(args: argparse.Namespace, rank: int, world_size: int):
    """Initialize wandb on rank 0 only, with a safe offline/disabled fallback.

    A cluster node may not reach api.wandb.ai; rather than letting wandb block or
    crash a 16-GPU job, we fall back online -> offline -> disabled and keep
    training. Returns the wandb run handle or None.
    """
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

    config = {
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum * world_size,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "bf16": args.bf16,
        "world_size": world_size,
        "limit_per_source": args.limit_per_source,
        "sources": args.sources,
    }
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
            log(rank, f"[wandb] initialized (mode={mode}) dir={wandb_dir}")
            return run
        except Exception as exc:  # noqa: BLE001
            log(rank, f"[wandb] init failed (mode={mode}): {exc}")
    log(rank, "[wandb] disabled after fallback failures")
    return None


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    """Initialize the process group from torchrun env vars when present."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()

    distributed = world_size > 1
    if distributed:
        backend = "nccl" if use_cuda else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        if use_cuda:
            torch.cuda.set_device(local_rank)

    if use_cuda:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return distributed, rank, world_size, local_rank, device


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, message: str) -> None:
    if is_main(rank):
        print(message, flush=True)


def move_batch(batch, device: torch.device):
    if isinstance(batch, dict):
        return {key: move_batch(value, device) for key, value in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


def build_model(args: argparse.Namespace, device: torch.device) -> MultiChainOphiuchusAbModel:
    backbone = OphiuchusAbBackbone()
    if args.init_multimer:
        backbone.init_multimer_attention()
    model = MultiChainOphiuchusAbModel(net=backbone)
    if args.checkpoint_path is not None:
        missing, unexpected = load_ophiuchus_checkpoint(model, args.checkpoint_path, device="cpu")
        log(int(os.environ.get("RANK", "0")),
            f"loaded checkpoint {args.checkpoint_path} (missing={len(missing)} unexpected={len(unexpected)})")
    return model.to(device)


def lr_at(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    return base_lr


def _unwrap_backbone(model: MultiChainOphiuchusAbModel):
    return model.net.module if isinstance(model.net, DistributedDataParallel) else model.net


def save_checkpoint(
    model: MultiChainOphiuchusAbModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    args: argparse.Namespace,
    output_dir: Path,
    tag: str,
) -> None:
    """Atomically save a full training checkpoint (weights + optimizer + step).

    Writes ``<tag>.pt`` and refreshes ``latest.pt`` so a preempted/restarted job
    can resume. The temp-file + os.replace pattern avoids leaving a half-written
    checkpoint if the node is killed mid-save.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone = _unwrap_backbone(model)
    payload = {
        "backbone_state_dict": backbone.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    for name in {f"{tag}.pt", "latest.pt"}:
        target = output_dir / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, target)


def maybe_resume(
    args: argparse.Namespace,
    model: MultiChainOphiuchusAbModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank: int,
) -> int:
    """Resume weights+optimizer+step from a checkpoint, returning the start step.

    ``--resume auto`` loads ``<output-dir>/latest.pt`` when it exists (the safe
    default for preemptible jobs); ``--resume <path>`` loads an explicit file;
    ``--resume none`` disables resuming.
    """
    if args.resume in (None, "none", ""):
        return 0
    if args.resume == "auto":
        path = args.output_dir / "latest.pt"
    else:
        path = Path(args.resume)
    if not path.is_file():
        log(rank, f"[resume] no checkpoint at {path}; starting from step 0")
        return 0

    payload = torch.load(path, map_location="cpu")
    _unwrap_backbone(model).load_state_dict(payload["backbone_state_dict"])
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    start_step = int(payload.get("step", 0))
    log(rank, f"[resume] loaded {path} -> resuming at step {start_step}")
    return start_step


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank, device = setup_distributed()
    torch.manual_seed(args.seed + rank)
    torch.set_float32_matmul_precision("high")

    log(rank, f"world_size={world_size} device={device} distributed={distributed}")

    specs = select_source_specs(default_immune_specs(
        oas_dir=args.oas_dir,
        ots_dir=args.ots_dir,
        nanobody_dir=args.nanobody_dir,
        oas_weight=args.oas_weight,
        ots_weight=args.ots_weight,
        nanobody_weight=args.nanobody_weight,
    ), args.sources)
    dataset, counts = build_mixed_immune_dataset(
        specs=specs, split=args.split, max_rows_per_source=args.limit_per_source
    )
    log(rank, f"dataset sources={counts} total={len(dataset)}")

    collator = MultiChainDynamicCollator(tokenizer=Esm2ProteinTokenizer(), max_length=args.max_length)

    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args, device)
    if distributed:
        ddp_kwargs = {"find_unused_parameters": args.find_unused_parameters}
        if device.type == "cuda":
            model.net = DistributedDataParallel(model.net, device_ids=[local_rank], output_device=local_rank, **ddp_kwargs)
        else:
            model.net = DistributedDataParallel(model.net, **ddp_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step_config = OphiuchusAbTrainStepConfig()

    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if args.bf16
        else nullcontext()
    )

    start_step = maybe_resume(args, model, optimizer, device, rank)

    wandb_run = setup_wandb(args, rank, world_size)
    global_batch = args.batch_size * world_size

    model.train()
    step = start_step
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    log(rank, f"starting training at step {start_step}")
    window_start = time.time()
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            batch = move_batch(batch, device)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, args.lr, args.warmup_steps)

            with autocast:
                result = compute_ophiuchus_ab_training_loss(model, batch, step_config)
                loss = result.loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                elapsed = max(time.time() - window_start, 1e-6)
                samples_per_sec = global_batch * (args.log_interval if step > 0 else 1) / elapsed
                cur_lr = optimizer.param_groups[0]["lr"]
                log(
                    rank,
                    f"step={step} loss={result.loss.item():.4f} "
                    f"heavy={result.heavy_loss.item():.4f} light={result.light_loss.item():.4f} "
                    f"lr={cur_lr:.2e} samples/s={samples_per_sec:.1f}",
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": result.loss.item(),
                            "train/heavy_loss": result.heavy_loss.item(),
                            "train/light_loss": result.light_loss.item(),
                            "train/lr": cur_lr,
                            "perf/samples_per_sec": samples_per_sec,
                            "train/epoch": epoch,
                        },
                        step=step,
                    )
                window_start = time.time()
            if args.save_interval and step > 0 and step % args.save_interval == 0:
                if is_main(rank):
                    save_checkpoint(model, optimizer, step, epoch, args, args.output_dir, tag="latest")
                    log(rank, f"saved checkpoint at step {step} -> {args.output_dir / 'latest.pt'}")
                if distributed:
                    torch.distributed.barrier()

            step += 1
            if step >= args.max_steps:
                break
        epoch += 1

    if is_main(rank):
        save_checkpoint(model, optimizer, step, epoch, args, args.output_dir, tag="final")
        log(rank, f"saved final checkpoint to {args.output_dir / 'final.pt'}")
    if wandb_run is not None:
        wandb_run.finish()

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
