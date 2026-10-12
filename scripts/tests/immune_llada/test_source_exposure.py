"""Activate pllm, then USE_TF=0 python -m pytest -q scripts/tests/immune_llada/test_source_exposure.py.

Uses small integer populations only; never opens sequences or launches training.
"""
from bisect import bisect_right
from collections import Counter
from itertools import accumulate, islice
import pytest
import torch

from scripts.architecture_prototypes.source_exposure import allocation, audit, counts_from_manifest
from dllm.pipelines.immune_llada.data.balanced_sampling import FullCoverageSourceSampler


def test_counts_aggregate_manifest_shards_and_sort_sources():
    manifest = {'splits': {'train': {'records': 9, 'shards': [
        {'source': 'b', 'records': 2}, {'source': 'a', 'records': 4}, {'source': 'b', 'records': 3}]}}}
    assert counts_from_manifest(manifest) == {'a': 4, 'b': 5}
    manifest['splits']['train']['records'] = 10
    with pytest.raises(ValueError):
        counts_from_manifest(manifest)


@pytest.mark.parametrize('counts', [{}, {'a': 0}, {'a': -1}, {'a': True}, {'a': 1.5}])
def test_invalid_counts_rejected(counts):
    with pytest.raises(ValueError):
        allocation(counts, 8, .5)


@pytest.mark.parametrize('alpha', [0, .5, .75, 1])
def test_exact_budget_deterministic_order_and_weights(alpha):
    plan = allocation({'big': 100, 'small': 4}, 257, alpha)
    assert sum(plan['apportioned_presentations'].values()) == 257
    assert sum(plan['probability'].values()) == pytest.approx(1)
    assert plan['probability']['big']/plan['probability']['small'] == pytest.approx(25**alpha)
    assert plan == allocation({'small': 4, 'big': 100}, 257, alpha)


def test_actual_current_sampler_prefixes_and_rng_preservation():
    counts = {'a': 3, 'b': 13, 'c': 51}
    state = torch.random.get_rng_state().clone()
    report = audit(counts, (1, 5), global_batch=8, seed=12)
    assert torch.equal(state, torch.random.get_rng_state())
    names, sizes = list(counts), list(counts.values())
    ends = list(accumulate(sizes))
    natural = torch.randperm(sum(sizes), generator=torch.Generator().manual_seed(12)).tolist()
    equal = list(islice(FullCoverageSourceSampler(sizes, effective_batch_size=8, seed=12), 40))
    for step in (1, 5):
        for method, stream in (('natural', natural), ('equal_source', equal)):
            indices = stream[:step*8]
            expected = Counter(names[bisect_right(ends, index)] for index in indices)
            measured = report['stages'][str(step)][method]['sources']
            assert {s: item['presentations'] for s, item in measured.items()} == {s: expected[s] for s in names}
            for i, source in enumerate(names):
                actual = [index for index in indices if bisect_right(ends, index) == i]
                assert measured[source]['unique_records'] == len(set(actual))
                assert measured[source]['repeated_presentations'] == len(actual)-len(set(actual))
    assert report['training_certification'] is False
    assert report['stages']['5']['tempered_proposals'][0]['status'] == 'proposal_only_sampler_not_implemented'


def test_beyond_first_epoch_and_unsorted_sources_rejected():
    with pytest.raises(ValueError, match='first epoch'):
        audit({'a': 3}, (2,), global_batch=8)
    with pytest.raises(ValueError, match='sorted'):
        audit({'b': 10, 'a': 10}, (1,), global_batch=8)
