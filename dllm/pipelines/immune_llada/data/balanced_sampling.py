"""Full-coverage, equal-source sampling for the prepared ConcatDataset.

Run the CPU regressions with the training environment's Python:
python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_source_balanced_training.py -q
"""

from __future__ import annotations

import math
from itertools import accumulate
from typing import Iterator

import torch
from torch.utils.data import Sampler


class FullCoverageSourceSampler(Sampler[int]):
    """Round-robin sources, independently shuffled without replacement.

    Accelerate shards the resulting ordinary microbatches across ranks; this
    sampler must NOT shard a second time. Every rank constructs the identical
    stream. One coverage epoch is long enough for every source to contribute
    at least ``max(source_lengths)`` rows, padded to complete optimizer batches.
    Small sources reshuffle only after exhausting their own permutation, even
    across coverage-epoch boundaries (an epoch does not reset their cursors).
    Restarting at (epoch, skipped_batches) reconstructs the exact same stream.
    """

    def __init__(self, source_lengths: list[int], *, effective_batch_size: int, seed: int = 42) -> None:
        self.source_lengths = tuple(int(n) for n in source_lengths)
        if not self.source_lengths or min(self.source_lengths) <= 0:
            raise ValueError("Source-balanced training requires every selected source to be nonempty")
        self.effective_batch_size = int(effective_batch_size)
        if self.effective_batch_size < len(self.source_lengths):
            raise ValueError("Effective batch size must be at least the number of sources")
        self.seed = int(seed)
        self.epoch = 0
        self.offsets = (0, *tuple(accumulate(self.source_lengths))[:-1])
        self.steps_per_coverage_epoch = math.ceil(
            max(self.source_lengths) * len(self.source_lengths) / self.effective_batch_size
        )
        self.num_samples = self.steps_per_coverage_epoch * self.effective_batch_size

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        source_count = len(self.source_lengths)
        start = self.epoch * self.num_samples
        # Random-access reconstruction of the continuous per-source stream.
        # No replay of earlier epochs or worker-local state is needed on resume.
        previous_draws = [(start + source_count - 1 - source) // source_count for source in range(source_count)]
        cycles = [draws // size for draws, size in zip(previous_draws, self.source_lengths)]
        cursors = [draws % size for draws, size in zip(previous_draws, self.source_lengths)]

        def permutation(source: int) -> torch.Tensor:
            generator = torch.Generator().manual_seed(
                (self.seed + cycles[source] * 1_000_003 + source * 97_409) % (2**63 - 1)
            )
            return torch.randperm(self.source_lengths[source], generator=generator)

        # Keep permutations as compact tensors, not millions of Python ints.
        permutations = [permutation(source) for source in range(source_count)]
        for position in range(self.num_samples):
            source = (start + position) % source_count
            if cursors[source] == self.source_lengths[source]:
                cycles[source] += 1
                permutations[source] = permutation(source)
                cursors[source] = 0
            yield self.offsets[source] + int(permutations[source][cursors[source]])
            cursors[source] += 1
