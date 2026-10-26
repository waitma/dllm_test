"""Run with activated pllm: USE_TF=0 python -B -m pytest THIS_FILE -q."""
from collections import Counter

import pytest

from scripts.research.diffusion_source_sampling import CappedSourceBatchSampler, source_quotas


def test_actual_source_cap_and_total():
    lengths = [68071, 2160824, 2092446, 39227, 59168, 1828249, 1971]
    quotas = source_quotas(lengths)
    assert sum(quotas) == 65536
    assert min(quotas) > 0
    assert all(q / 65536 <= 4 * n / sum(lengths) for q, n in zip(quotas, lengths))
    assert quotas[-1] / 65536 < .0013


def test_rank_partition_resume_and_source_coverage():
    lengths, world, batch = [3, 17, 51], 4, 12
    def make(rank, start=0):
        return CappedSourceBatchSampler(lengths, sampling=dict(period=64),
            global_batch=batch, rank=rank, world_size=world, seed=42,
            start_step=start, max_steps=43)
    streams = [list(make(r)) for r in range(world)]
    for start in (1, 5, 6, 21, 42):
        for rank in range(world):
            assert list(make(rank, start)) == streams[rank][start:]
    merged = [x for batches in zip(*streams) for group in zip(*batches) for x in group]
    sampler = make(0)
    assert merged == [x for batch_ in CappedSourceBatchSampler(lengths,
        sampling=dict(period=64), global_batch=batch, rank=0, world_size=1,
        seed=42, max_steps=43) for x in batch_]
    counts = []
    for size, offset in zip(lengths, sampler.base.offsets):
        values = [x for x in merged if offset <= x < offset + size]
        counts.append(len(values))
        for start in range(0, len(values)-size+1, size):
            assert sorted(values[start:start+size]) == list(range(offset, offset+size))
    assert counts == sampler.exposures(43)
    assert Counter(sampler.schedule) == dict(enumerate(sampler.quotas))


def test_unrepresentable_cap_is_rejected():
    with pytest.raises(ValueError, match="Period too short"):
        source_quotas([1, 100000], period=32)
