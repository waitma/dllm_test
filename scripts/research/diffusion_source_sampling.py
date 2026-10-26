"""Bound small-source amplification while retaining deterministic exact resume.

Activate pllm; USE_TF=0 python -B -m pytest
scripts/tests/immune_llada/test_diffusion_source_sampling.py -q.
The optional policy changes the record sampling objective, not loss normalization.
"""
import math

import numpy as np
import torch

from scripts.research.diffusion_continuation_data import BalancedSourceBatchSampler

POLICY = "capped_temperature_continuous_without_replacement_then_reshuffle"
LEGACY = "equal_source_continuous_without_replacement_then_reshuffle"


def source_quotas(lengths, *, alpha=.75, amplification_cap=4., period=65536):
    if (not lengths or any(type(n) is not int or n <= 0 for n in lengths)
            or not math.isfinite(alpha) or not 0 <= alpha <= 1
            or not math.isfinite(amplification_cap) or amplification_cap <= 1
            or type(period) is not int or period < len(lengths)):
        raise ValueError("Invalid capped source sampling configuration")
    natural = np.array(lengths, dtype=np.float64) / sum(lengths)
    caps = np.floor(amplification_cap * natural * period).astype(np.int64)
    if np.any(caps < 1) or caps.sum() < period:
        raise ValueError("Period too short to represent all sources under the cap")
    weights = np.array(lengths, dtype=np.float64) ** alpha
    lower, upper = 0., period / weights.min()
    for _ in range(100):
        scale = (lower + upper) / 2
        if np.minimum(scale * weights, caps).sum() < period:
            lower = scale
        else:
            upper = scale
    ideal = np.minimum(upper * weights, caps)
    quotas = np.floor(ideal).astype(np.int64)
    for source in np.argsort(-(ideal - quotas), kind="stable"):
        if quotas.sum() == period:
            break
        if quotas[source] < caps[source]:
            quotas[source] += 1
    if quotas.sum() != period or np.any(quotas < 1) or np.any(quotas > caps):
        raise ValueError("Invalid source apportionment")
    return quotas.tolist()


class CappedSourceBatchSampler(BalancedSourceBatchSampler):
    """A shuffled periodic source schedule; each source cycles without replacement.

    Prefix counts recover every cursor from the optimizer step, independent of
    DataLoader prefetch and rank. The cap applies to full-period frequencies;
    individual minibatches fluctuate. Record permutations change every source epoch.
    """

    def __init__(self, lengths, *, sampling, **kwargs):
        super().__init__(lengths, **kwargs)
        self.quotas = source_quotas(lengths, **sampling)
        self.period = sum(self.quotas)
        order = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(self.quotas))
        generator = torch.Generator().manual_seed(self.base.seed + 7_310_017)
        self.schedule = order[torch.randperm(self.period, generator=generator)].tolist()
        self.prefix = np.stack([np.r_[0, np.cumsum(np.array(self.schedule) == s)]
                                for s in range(len(lengths))])

    def source_at(self, ordinal):
        return self.schedule[ordinal % self.period]

    def exposures(self, step):
        cycles, cursor = divmod(step * self.global_batch, self.period)
        return [cycles * q + int(self.prefix[s, cursor]) for s, q in enumerate(self.quotas)]


def make_sampler(lengths, settings, *, rank, world, start_step):
    kwargs = dict(global_batch=settings["global_batch"], rank=rank, world_size=world,
                  seed=settings["seed"], start_step=start_step, max_steps=settings["max_steps"])
    policy = settings["sampler"]
    if policy == POLICY:
        return CappedSourceBatchSampler(lengths, sampling=settings["source_sampling"], **kwargs)
    if policy == LEGACY:
        return BalancedSourceBatchSampler(lengths, **kwargs)
    raise ValueError("Unknown source sampling policy: " + policy)
