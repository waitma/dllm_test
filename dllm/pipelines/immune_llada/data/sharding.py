"""Neutral DDP/DataLoader worker sharding for iterable data sources.

Use ``distributed_worker_shard()`` inside an iterable dataset to obtain the
single global ``(shard_index, num_shards)`` pair for the current worker.
"""

from __future__ import annotations

import torch
from torch.utils.data import get_worker_info


def distributed_worker_shard() -> tuple[int, int]:
    """Return shard info across DDP ranks and DataLoader workers."""

    worker = get_worker_info()
    worker_id = worker.id if worker is not None else 0
    num_workers = worker.num_workers if worker is not None else 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank * num_workers + worker_id, world_size * num_workers


__all__ = ["distributed_worker_shard"]
