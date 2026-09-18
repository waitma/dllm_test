"""Run: python -m pytest -q scripts/tests/immune_llada/test_tcr_generation_result_io.py.

CPU scorer/writer and immutable recovery regressions; never load model weights.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from downstream.benchmark.tcr_generation_bench import scoring
from downstream.benchmark.tcr_generation_bench.result_io import prepare_metrics_for_json, NOVELTY
from downstream.grammar.ab_features import write_json
from scripts.downstream import run_tcr_native_generation as generation


def fixture(monkeypatch, conditioning):
    monkeypatch.setattr(scoring, 'load_conditioning_by_epitope', lambda: conditioning)
    entries = {key: dict(epitope=key, category='benchmark14', ref_binders=['CASSF', 'CATF'])
               for key in ('a', 'b')}
    designs = {key: ['CASSF', 'CATF'] * 50 for key in entries}
    result = scoring.score_conditional(designs, entries, k=100,
                                      greedy_map={key: 'CASSF' for key in entries})
    return result, {key for key in entries if not conditioning.get(key)}


@pytest.mark.parametrize('conditioning', [{}, {'a': ['CASSF']}, {'a': ['CASSF'], 'b': ['CATF']}])
def test_real_scorer_strict_writer_roundtrip(tmp_path, monkeypatch, conditioning):
    result, missing = fixture(monkeypatch, conditioning)
    fixed = prepare_metrics_for_json(result, missing)
    write_json(tmp_path / 'metrics.json', fixed)
    text = (tmp_path / 'metrics.json').read_text()
    assert 'NaN' not in text and 'Infinity' not in text
    assert json.loads(text) == fixed
    assert fixed['summary']['overall']['n_scored_by_metric'][NOVELTY] == 2-len(missing)
    assert fixed['summary']['overall']['n_pmhc_scored'] == 2
    for key, item in fixed['per_pmhc'].items():
        assert item['precision'] == result['per_pmhc'][key]['precision']
        assert (item[NOVELTY] is None) == (key in missing)
    if missing:
        assert result['per_pmhc'][next(iter(missing))][NOVELTY] != result['per_pmhc'][next(iter(missing))][NOVELTY]
        assert fixed['metric_availability']['per_pmhc_unavailable']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), None])
@pytest.mark.parametrize('section,key', [('per_pmhc', 'a'), ('summary', 'overall')])
def test_primary_nonfinite_still_fails(monkeypatch, value, section, key):
    result, missing = fixture(monkeypatch, {})
    result[section][key]['precision'] = value
    with pytest.raises(ValueError):
        prepare_metrics_for_json(result, missing)


@pytest.mark.parametrize('case', ['unexpected_nan', 'optional_inf', 'invented_zero', 'wrong_count'])
def test_optional_missing_is_not_blanket_sanitization(monkeypatch, case):
    result, missing = fixture(monkeypatch, {} if case != 'unexpected_nan' else {'a': ['CASSF']})
    if case == 'unexpected_nan':
        result['per_pmhc']['a'][NOVELTY] = float('nan')
    elif case == 'optional_inf':
        result['per_pmhc']['a'][NOVELTY] = float('inf')
    elif case == 'invented_zero':
        result['per_pmhc']['a'][NOVELTY] = 0.
    else:
        result['summary']['overall']['n_scored_by_metric'][NOVELTY] = 1
    with pytest.raises(ValueError):
        prepare_metrics_for_json(result, missing)


def test_shared_strict_writer_not_relaxed(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / 'invalid.json', {'metric': float('nan')})
    assert not (tmp_path / 'invalid.json').exists()


def test_recovery_identity_rejects_model_sampler_and_reference_drift():
    runner = str(generation.ROOT / 'scripts/downstream/run_tcr_native_generation.py')
    helper = str(generation.BENCH / 'tcr_generation_bench/result_io.py')
    old = {'checkpoint': 'ckpt49000', 'protocol': {'max_iter': 32},
           'code_and_scoring_sha256': {runner: generation.LEGACY_RUNNER_SHA, 'sampler': 'same', 'refs': 'same'}}
    new = deepcopy(old)
    new['code_and_scoring_sha256'].update({runner: 'repaired', helper: 'new'})
    generation.verify_recovery_identity(old, new)
    for field in ('sampler', 'refs'):
        bad = deepcopy(new)
        bad['code_and_scoring_sha256'][field] = 'changed'
        with pytest.raises(ValueError):
            generation.verify_recovery_identity(old, bad)
    new['protocol']['max_iter'] = 8
    with pytest.raises(ValueError):
        generation.verify_recovery_identity(old, new)


def test_recovery_partial_raw_is_rejected(tmp_path):
    from scripts.downstream.audit_tcr_49000_generation_results import validate_raw
    (tmp_path / 'designs.jsonl').write_text('')
    with pytest.raises(AssertionError):
        validate_raw('benchmark14', tmp_path)


def test_recovery_never_constructs_sampler_and_refuses_overwrite(tmp_path, monkeypatch):
    from scripts.downstream import audit_tcr_49000_generation_results as audit
    source, output = tmp_path / 'old', tmp_path / 'new'
    source.mkdir()
    (tmp_path / 'generation_preflight').mkdir()
    identity = {'fake': 'identity'}
    original = dict(status='failed', dataset='benchmark14', paired_generation=False,
                    identity=identity, error='ValueError: Out of range float values')
    write_json(source / 'run_manifest.json', original)
    (source / 'designs.jsonl').write_text('retained bytes\n')
    write_json(tmp_path / 'generation_preflight/passed.json', dict(status='passed', identity=identity))
    gate = tmp_path / 'new_gate.json'
    write_json(gate, dict(status='passed', identity=identity))
    monkeypatch.setattr(generation, 'OUT', tmp_path)
    monkeypatch.setattr(generation, 'identity', lambda: identity)
    monkeypatch.setattr(generation, 'verify_recovery_identity', lambda *a: None)
    monkeypatch.setattr(generation, 'output_for', lambda task: output)
    monkeypatch.setattr(audit, 'validate_raw', lambda *a: ({}, [], {}, {}))
    monkeypatch.setattr(generation, 'score_result', lambda *a: {'finite': 1.})
    def forbidden(*a, **kw):
        raise AssertionError('recovery must not load weights or regenerate')
    monkeypatch.setattr(generation, 'V5BetaSampler', forbidden)
    monkeypatch.setattr(generation, 'assert_checkpoint', forbidden)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    generation.recover('benchmark14', source, gate)
    assert json.loads((output / 'run_manifest.json').read_text())['status'] == 'success'
    assert (output / 'designs.jsonl').read_bytes() == before['designs.jsonl']
    with pytest.raises(FileExistsError):
        generation.recover('benchmark14', source, gate)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
