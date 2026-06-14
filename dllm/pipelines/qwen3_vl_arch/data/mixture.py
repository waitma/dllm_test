from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .records import BioSeqRecord


class RecordSource(IterableDataset):
    def iter_records(self, shard_index: int = 0, num_shards: int = 1) -> Iterator[BioSeqRecord]:
        raise NotImplementedError


@dataclass(frozen=True)
class SourceWithWeight:
    source: RecordSource
    weight: float = 1.0


def bioseq_task_group(record: BioSeqRecord) -> str:
    roles = set(record.chain_roles)
    has_antigen = "antigen" in roles
    if has_antigen and roles & {"antibody_heavy", "antibody_light"}:
        return "antibody_antigen"
    if has_antigen and "nanobody_vhh" in roles:
        return "nanobody_antigen"
    if "nanobody_vhh" in roles:
        return "nanobody"
    return record.task_type


def bioseq_record_fingerprint(record: BioSeqRecord) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    return record.task_type, tuple(record.chain_roles), tuple(record.sequences)


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


class SequentialMultiSourceDataset(IterableDataset):
    def __init__(self, sources: Sequence[RecordSource]) -> None:
        super().__init__()
        self.sources = list(sources)

    def __iter__(self) -> Iterator[BioSeqRecord]:
        shard_index, num_shards = distributed_worker_shard()
        for source in self.sources:
            yield from source.iter_records(shard_index=shard_index, num_shards=num_shards)


class WeightedMixtureDataset(IterableDataset):
    """Infinite or epoch-sized weighted stream over resettable record sources."""

    def __init__(
        self,
        sources: Sequence[SourceWithWeight],
        epoch_size: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not sources:
            raise ValueError("WeightedMixtureDataset requires at least one source")
        self.sources = list(sources)
        self.epoch_size = epoch_size
        self.seed = seed

    def __iter__(self) -> Iterator[BioSeqRecord]:
        shard_index, num_shards = distributed_worker_shard()
        rng = random.Random(self.seed + shard_index)
        weights = [max(item.weight, 0.0) for item in self.sources]
        if not any(weights):
            raise ValueError("At least one source weight must be positive")

        iterators = [
            item.source.iter_records(shard_index=shard_index, num_shards=num_shards)
            for item in self.sources
        ]
        emitted = 0
        while self.epoch_size is None or emitted < self.epoch_size:
            index = rng.choices(range(len(self.sources)), weights=weights, k=1)[0]
            try:
                record = next(iterators[index])
            except StopIteration:
                iterators[index] = self.sources[index].source.iter_records(
                    shard_index=shard_index,
                    num_shards=num_shards,
                )
                record = next(iterators[index])
            yield record
            emitted += 1


class TaskHomogeneousBatchDataset(IterableDataset):
    """Group a record stream into task-homogeneous batches."""

    def __init__(
        self,
        records: Iterable[BioSeqRecord],
        batch_size: int,
        task_key_fn: Callable[[BioSeqRecord], str] = bioseq_task_group,
        drop_last: bool = True,
        deduplicate_within_batch: bool = False,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.records = records
        self.batch_size = batch_size
        self.task_key_fn = task_key_fn
        self.drop_last = drop_last
        self.deduplicate_within_batch = deduplicate_within_batch

    def __iter__(self) -> Iterator[list[BioSeqRecord]]:
        buffers: dict[str, list[BioSeqRecord]] = {}
        fingerprints: dict[str, set[tuple[str, tuple[str, ...], tuple[str, ...]]]] = {}
        for record in self.records:
            task_key = self.task_key_fn(record)
            record_key = bioseq_record_fingerprint(record)
            task_buffer = buffers.setdefault(task_key, [])
            task_fingerprints = fingerprints.setdefault(task_key, set())
            if self.deduplicate_within_batch and record_key in task_fingerprints:
                continue
            task_buffer.append(record)
            task_fingerprints.add(record_key)
            if len(task_buffer) == self.batch_size:
                yield list(task_buffer)
                task_buffer.clear()
                task_fingerprints.clear()

        if not self.drop_last:
            for task_key in sorted(buffers):
                if buffers[task_key]:
                    yield list(buffers[task_key])
