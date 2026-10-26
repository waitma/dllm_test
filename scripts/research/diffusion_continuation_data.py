"""Resume the existing equal-source stream directly at a native-DDP batch.

Activate pllm; run python -B -m pytest
scripts/tests/immune_llada/test_diffusion_continuation_data.py -q.
The sampling policy matches FullCoverageSourceSampler; only sharding/cursor
handling differs, because this trainer uses native DDP rather than Accelerate.
"""
from collections import OrderedDict
import math

import torch
from torch.utils.data import Sampler

from dllm.pipelines.immune_llada.data.balanced_sampling import FullCoverageSourceSampler


def source_lengths(index):
    """Resolve dataset offsets from shard provenance, never record.source labels."""
    lengths = OrderedDict()
    last = None
    for shard in index["shards"]:
        if shard["split"] != "train":
            continue
        name = shard["source"]
        if name in lengths and name != last:
            raise ValueError("Equal-source offsets require contiguous source shards")
        lengths[name] = lengths.get(name, 0) + shard["records"]
        last = name
    if sum(lengths.values()) != index["records"]["train"]:
        raise ValueError("Source counts do not match the bound training index")
    return lengths


class BalancedSourceBatchSampler(Sampler):
    """One deterministic global stream, partitioned exactly once over ranks.

    Source permutations, seed arithmetic and round-robin order match the old
    exploration sampler. Resume reconstructs per-source cycles/cursors directly;
    DataLoader prefetch cannot advance the checkpoint's optimizer-step cursor.
    """

    def __init__(self, lengths, *, global_batch, rank, world_size, seed,
                 start_step=0, max_steps):
        if (world_size <= 0 or global_batch % world_size or not 0 <= rank < world_size
                or not 0 <= start_step <= max_steps):
            raise ValueError("Invalid distributed batch or resume cursor")
        self.base = FullCoverageSourceSampler(list(lengths),
                                             effective_batch_size=global_batch, seed=seed)
        self.global_batch, self.rank, self.world_size = global_batch, rank, world_size
        self.start_step, self.max_steps = start_step, max_steps

    def __len__(self):
        return self.max_steps - self.start_step

    def exposures(self, step):
        total, sources = step * self.global_batch, len(self.base.source_lengths)
        return [(total + sources - 1 - source) // sources for source in range(sources)]

    def source_at(self, ordinal):
        return ordinal % len(self.base.source_lengths)

    def __iter__(self):
        base = self.base
        sources = len(base.source_lengths)
        previous = self.exposures(self.start_step)
        cycles = [draws // size for draws, size in zip(previous, base.source_lengths)]
        cursors = [draws % size for draws, size in zip(previous, base.source_lengths)]

        def permutation(source):
            generator = torch.Generator().manual_seed(
                (base.seed + cycles[source] * 1_000_003 + source * 97_409) % (2**63 - 1))
            return torch.randperm(base.source_lengths[source], generator=generator)

        orders = [permutation(source) for source in range(sources)]
        for step in range(self.start_step, self.max_steps):
            indices = []
            for offset in range(self.global_batch):
                source = self.source_at(step * self.global_batch + offset)
                if cursors[source] == base.source_lengths[source]:
                    cycles[source] += 1
                    orders[source] = permutation(source)
                    cursors[source] = 0
                index = base.offsets[source] + int(orders[source][cursors[source]])
                cursors[source] += 1
                if offset % self.world_size == self.rank:
                    indices.append(index)
            yield indices


def warmup_cosine_scheduler(optimizer, total_steps, warmup_steps, initial_fraction=.01,
                            final_fraction=.1):
    """Start above zero, warm up to the configured LR, then decay smoothly."""
    if not (0 < warmup_steps < total_steps and 0 <= initial_fraction <= 1
            and 0 <= final_fraction <= 1):
        raise ValueError("Invalid warmup/cosine schedule")

    def multiplier(step):
        if step < warmup_steps:
            return initial_fraction + (1 - initial_fraction) * step / warmup_steps
        progress = min(1., (step - warmup_steps) / (total_steps - warmup_steps))
        return final_fraction + (1 - final_fraction) * .5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
