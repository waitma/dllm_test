"""Check exact legacy sampling, DDP partition and warmup-resume behavior.

Activate pllm; python -B -m pytest
scripts/tests/immune_llada/test_diffusion_continuation_data.py -q.
"""
import pytest
import torch

from dllm.pipelines.immune_llada.data.balanced_sampling import FullCoverageSourceSampler
from scripts.research.diffusion_continuation_data import (
    BalancedSourceBatchSampler, source_lengths, warmup_cosine_scheduler)


def test_global_stream_matches_exploration_across_epochs_and_rank_resume():
    lengths, batch, world = [3, 19, 47], 12, 4
    legacy = FullCoverageSourceSampler(lengths, effective_batch_size=batch, seed=42)
    expected = []
    for epoch in range(3):
        legacy.set_epoch(epoch)
        expected.extend(legacy)
    steps = len(expected) // batch
    ranks = [list(BalancedSourceBatchSampler(lengths, global_batch=batch, rank=rank,
                  world_size=world, seed=42, max_steps=steps)) for rank in range(world)]
    merged = [value for batches in zip(*ranks) for group in zip(*batches) for value in group]
    assert merged == expected
    for rank in range(world):
        resumed = list(BalancedSourceBatchSampler(lengths, global_batch=batch, rank=rank,
                       world_size=world, seed=42, start_step=13, max_steps=steps))
        assert resumed == ranks[rank][13:]
    sampler = BalancedSourceBatchSampler(lengths, global_batch=batch, rank=0,
                                         world_size=world, seed=42, max_steps=steps)
    assert sum(sampler.exposures(steps)) == len(expected)
    for source, (size, offset) in enumerate(zip(lengths, legacy.offsets)):
        values = merged[source::len(lengths)]
        for start in range(0, len(values) - size + 1, size):
            assert sorted(values[start:start+size]) == list(range(offset, offset+size))


def test_counts_use_shards_and_reject_noncontiguous_sources():
    shards = [dict(split="train", source=s, records=n) for s, n in [("a", 2), ("a", 3), ("b", 7)]]
    index = dict(shards=shards, records={"train": 12})
    assert source_lengths(index) == {"a": 5, "b": 7}
    index["shards"] = [shards[0], shards[2], shards[1]]
    with pytest.raises(ValueError, match="contiguous"):
        source_lengths(index)


def test_warmup_and_resume_have_same_next_learning_rate():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    scheduler = warmup_cosine_scheduler(optimizer, 20, 4)
    rates = []
    for step in range(21):
        rates.append(optimizer.param_groups[0]["lr"])
        if step == 9:
            import copy
            saved = copy.deepcopy((optimizer.state_dict(), scheduler.state_dict()))
        optimizer.step()
        scheduler.step()
    assert rates[0] == pytest.approx(1e-7)
    assert rates[4] == pytest.approx(1e-5)
    assert rates[20] == pytest.approx(1e-6)
    optimizer.load_state_dict(saved[0])
    scheduler.load_state_dict(saved[1])
    assert optimizer.param_groups[0]["lr"] == rates[9]
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == rates[10]
