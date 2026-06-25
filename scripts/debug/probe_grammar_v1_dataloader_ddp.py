"""Probe grammar_v1 DataLoader behavior under simulated multi-rank DDP sharding.

Uses CPU + gloo (no GPU required). Detects per-rank batch stalls from
TaskHomogeneousBatchDataset + WeightedMixtureDataset sharding.

Example:
    torchrun --standalone --nproc_per_node=8 \
        /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/probe_grammar_v1_dataloader_ddp.py \
        --num-workers 2 --batches-per-rank 8 --batch-size 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.bioseq.train_qwen3_vl_bioseq_ddp import build_loader, build_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches-per-rank", type=int, default=8)
    parser.add_argument("--limit-per-source", type=int, default=2000)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--num-workers-single", action="store_true", help="Also run a single-process baseline.")
    return parser.parse_args()


def make_train_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        grammar_data_dir=PROJECT_ROOT / "data/bioseq_grammar_v1",
        split="train",
        sources="oas,ots,tcr,ppi",
        limit_per_source=args.limit_per_source,
        epoch_size=None,
        batch_size=args.batch_size,
        max_sequence_length=2112,
        deduplicate_within_batch=False,
        source_seed=0,
        oas_weight=3.9,
        ots_weight=3.6,
        tcr_weight=1.0,
        ppi_weight=1.4,
        nanobody_weight=1.0,
        processed_v2_weight=1.0,
        num_workers=args.num_workers,
        tokenizer_path=PROJECT_ROOT / "model_weights/esmc/ESMC-300M",
    )


def setup_distributed() -> tuple[int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.distributed.init_process_group(backend="gloo", init_method="env://")
    return rank, world_size


def probe_rank(args: argparse.Namespace) -> None:
    rank, world_size = setup_distributed()
    train_args = make_train_args(args)
    tokenizer = build_tokenizer(train_args)
    loader = build_loader(train_args, tokenizer)
    iterator = iter(loader)
    batch_times: list[float] = []
    task_groups_seen: list[str] = []

    for batch_idx in range(args.batches_per_rank):
        wait_start = time.time()
        batch = next(iterator)
        elapsed = time.time() - wait_start
        batch_times.append(elapsed)
        tasks = batch.get("task_groups", [])
        task_summary = tasks[0] if tasks else "?"
        task_groups_seen.append(str(task_summary))
        print(
            f"[rank={rank}/{world_size} workers={args.num_workers}] "
            f"batch={batch_idx} wait_s={elapsed:.2f} task={task_summary} "
            f"seq_len={batch['input_ids'].shape[1]}",
            flush=True,
        )
        if elapsed > args.timeout_sec:
            raise TimeoutError(
                f"rank={rank} batch={batch_idx} exceeded timeout {args.timeout_sec}s "
                f"(wait_s={elapsed:.2f})"
            )

    max_wait = max(batch_times)
    torch.distributed.barrier()
    if rank == 0:
        print(
            f"[summary workers={args.num_workers}] fetched {args.batches_per_rank} batches/rank; "
            f"max_single_batch_wait_s={max_wait:.2f}",
            flush=True,
        )
    torch.distributed.destroy_process_group()


def probe_single(args: argparse.Namespace) -> None:
    train_args = make_train_args(args)
    tokenizer = build_tokenizer(train_args)
    loader = build_loader(train_args, tokenizer)
    iterator = iter(loader)
    print(f"[single workers={args.num_workers}] starting", flush=True)
    for batch_idx in range(args.batches_per_rank):
        wait_start = time.time()
        batch = next(iterator)
        elapsed = time.time() - wait_start
        tasks = batch.get("task_groups", [])
        task_summary = tasks[0] if tasks else "?"
        print(
            f"[single workers={args.num_workers}] batch={batch_idx} wait_s={elapsed:.2f} "
            f"task={task_summary} seq_len={batch['input_ids'].shape[1]}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if "RANK" in os.environ:
        probe_rank(args)
        return
    if args.num_workers_single:
        probe_single(args)
        return
    raise SystemExit("Launch with torchrun --standalone --nproc_per_node=N ...")


if __name__ == "__main__":
    main()
