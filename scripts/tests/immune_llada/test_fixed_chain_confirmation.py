"""Activate pllm; python -m pytest scripts/tests/immune_llada/test_fixed_chain_confirmation.py -q.

Analytic fixtures check paired populations and uncertainty without reserved data.
"""
import copy
import json
import pytest
from scripts.research.analyze_fixed_chain_confirmation import (
    VIEWS, primary_change, bootstrap_primary, generation_summary,
)


def metrics(scale=1.):
    result = {}
    for view in VIEWS:
        records = [{'row': i, 'cluster': str(i // 2), 'totals': [0, 0, scale * 20, 10]}
                   for i in range(6)]
        result[view] = {'macro': {'aa_loss': scale * 2}, 'records': {'source': records}}
    return result


def test_exact_paired_improvement_and_cluster_interval():
    control, candidate = metrics(), metrics(.9)
    assert primary_change(candidate, control) == pytest.approx(-.1)
    assert bootstrap_primary([candidate] * 3, [control] * 3, replicates=64) == pytest.approx([-.1, -.1])
    assert bootstrap_primary([control] * 3, [control] * 3, replicates=64) == [0., 0.]


def test_bootstrap_rejects_unpaired_targets():
    control, candidate = metrics(), metrics(.9)
    candidate[VIEWS[0]]['records']['source'][0]['totals'][3] += 1
    with pytest.raises(ValueError, match='counts differ'):
        bootstrap_primary([candidate], [control], replicates=2)


def test_all_invalid_generation_is_not_zero_length_error():
    result = generation_summary({'records': [{'source': 'test', 'chains': [{'length_error': None}]}]})
    assert result['validity'] == 0 and result['valid_only_length_mae'] is None
    json.dumps(result, allow_nan=False)


def test_confirmation_cannot_open_without_frozen_plan(tmp_path):
    from scripts.research.run_fixed_chain_confirmation import validate_plan
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps({'schema': 'fixed_chain.confirmation.v1', 'status': 'draft'}))
    with pytest.raises(ValueError, match='must be frozen'):
        validate_plan(path, 'candidate')
