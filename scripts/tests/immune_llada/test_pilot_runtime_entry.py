"""Artificial CPU rejection tests; no GPU/data acceptance marker is generated.

Activate pllm:
USE_TF=0 python -m pytest scripts/tests/immune_llada/test_pilot_runtime_entry.py -q
Real subprocess tests intentionally fail admission and retain no checkpoints.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

from scripts.architecture_prototypes import pilot_runtime_entry_v1 as entry


def make_pool(tmp_path):
    pool = tmp_path/'pool'
    (pool/'runs').mkdir(parents=True)
    (pool/'profiles').mkdir()
    (pool/'pool.json').write_text(json.dumps({'source_root': str(tmp_path)}))
    return pool


def argv(pool, *, output=None):
    return ['--variant', entry.VARIANT, '--mode', 'profile-synthetic', '--steps', '1',
            '--profile-dir', str(pool/'profiles'/entry.VARIANT),
            '--output-dir', str(output or pool/'runs'/'test-job')]


def test_fixed_transport_resolves_model_profile(tmp_path):
    pool = make_pool(tmp_path)
    result = entry.validate_transport(argv(pool))
    assert result == (pool, pool/'runs'/'test-job', pool/'profiles'/entry.MODEL_VARIANT)


@pytest.mark.parametrize('extra', [
    ['--data-dir', '/tmp/real-data'], ['--resume-from-checkpoint', '/tmp/checkpoint'],
    ['--variant', entry.VARIANT], ['--steps', '2'], ['--shell', 'anything'],
])
def test_extra_transport_cannot_select_data_or_execution(tmp_path, extra):
    with pytest.raises(ValueError, match='exactly five'):
        entry.validate_transport(argv(make_pool(tmp_path))+extra)


@pytest.mark.parametrize('option,value', [('--variant', 'r20_refined_control'),
    ('--mode', 'train'), ('--steps', '2')])
def test_model_training_is_not_pool_diagnostic(tmp_path, option, value):
    values = argv(make_pool(tmp_path))
    values[values.index(option)+1] = value
    with pytest.raises(ValueError, match='fixed D01'):
        entry.validate_transport(values)


def test_output_and_profile_cannot_cross_pool(tmp_path):
    pool = make_pool(tmp_path)
    values = argv(pool)
    values[values.index('--profile-dir')+1] = str(tmp_path/'other'/'profiles'/entry.VARIANT)
    with pytest.raises(ValueError, match='another pool'):
        entry.validate_transport(values)
    output = pool/'runs'/'test-job'
    output.mkdir()
    with pytest.raises(ValueError, match='fresh'):
        entry.validate_transport(argv(pool))


@pytest.mark.parametrize('content', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', '[]'])
def test_json_rejects_ambiguous_or_nonfinite_evidence(tmp_path, content):
    path = tmp_path/'evidence.json'
    path.write_text(content)
    with pytest.raises(ValueError):
        entry.strict_json(path)


def test_json_rejects_symlink(tmp_path):
    target = tmp_path/'target.json'
    target.write_text('{}')
    path = tmp_path/'alias.json'
    path.symlink_to(target)
    with pytest.raises(ValueError, match='symlink'):
        entry.strict_json(path)


def test_missing_immutable_runtime_blocks_before_project_import(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, 'ROOT', tmp_path/'absent-runtime')
    with pytest.raises(ValueError, match='Missing'):
        entry.load_runtime()


def receipt(command):
    return dict(schema=entry.PROCESS_SCHEMA, argv=command, returncode=0,
                cleanup_completed=True, completion_observed_by_parent=True)


@pytest.mark.parametrize('key,value', [('returncode', 1), ('returncode', False),
    ('cleanup_completed', False), ('completion_observed_by_parent', False),
    ('argv', ['unexpected-target'])])
def test_child_report_cannot_substitute_observed_normal_exit(key, value):
    command = ['fixed-python', 'fixed-producer']
    value_receipt = receipt(command)
    value_receipt[key] = value
    with pytest.raises(ValueError, match='exit normally'):
        entry.validate_observed_exit(value_receipt, command, .1)


@pytest.mark.parametrize('elapsed', [-1, float('nan'), float('inf'), entry.TIMEOUT_SECONDS + .001, True])
def test_fixed_deadline_is_not_extended(elapsed):
    command = ['fixed-python', 'fixed-producer']
    with pytest.raises(ValueError, match='deadline'):
        entry.validate_observed_exit(receipt(command), command, elapsed)


def test_data_gate_precedes_runtime_or_producer(monkeypatch, tmp_path):
    from scripts.architecture_prototypes import certified_pilot_execution as execution
    from scripts.architecture_prototypes import certified_pilot as pilot
    from scripts.architecture_prototypes import pilot_runtime_package as package
    calls = []
    def gate(*args, **kwargs):
        calls.append('original_data_gate')
        raise ValueError('uncertified data')
    def forbidden(*args, **kwargs):
        calls.append('runtime_or_model')
        raise AssertionError('must not run after refused data')
    monkeypatch.setattr(pilot, 'assert_ready', gate)
    monkeypatch.setattr(package, 'verify_runtime', forbidden)
    monkeypatch.setattr(pilot, 'make_model', forbidden)
    request = pilot.PilotRequest(entry.MODEL_VARIANT, tmp_path/'data', tmp_path/'selection',
                                tmp_path/'profile', tmp_path/'training')
    with pytest.raises(ValueError, match='uncertified data'):
        entry.execute_admitted(request, tmp_path/'gpu-evidence')
    assert calls == ['original_data_gate']
    assert not request.output_dir.exists()


@pytest.mark.parametrize('code', [0, 2])
def test_real_cpu_child_without_gpu_proof_never_signs_complete(monkeypatch, tmp_path, code):
    from scripts.architecture_prototypes import cuda_attention_probe as processes
    pool = make_pool(tmp_path)
    runtime = tmp_path/'CPU_REJECTION_TEST_DOUBLE'
    source = runtime/entry.PRODUCER
    source.parent.mkdir(parents=True)
    source.write_text('import sys\nprint("CPU TEST DOUBLE: no CUDA executed")\nsys.exit('+str(code)+')\n')
    (runtime/entry.MANIFEST).write_text('{"kind":"CPU_REJECTION_TEST_DOUBLE"}')
    manifest = {'source_root':str(tmp_path)}
    package = SimpleNamespace(read_json=entry.strict_json)
    producer = SimpleNamespace(verify_candidate=lambda *args: entry.strict_json(pool/'runs'/'test-job'/'gpu_release_summary.json'))
    execution = SimpleNamespace(GPU_SCHEMA='unused_for_refused_test')
    monkeypatch.setattr(entry, 'ROOT', runtime)
    monkeypatch.setattr(entry, 'load_runtime', lambda: (manifest,package,execution,producer,processes))
    with pytest.raises(ValueError):
        entry.run_artificial(argv(pool))
    output = pool/'runs'/'test-job'
    assert entry.strict_json(output/'FAILED.json')['status'] == 'failed'
    observed = entry.strict_json(output/'process_evidence.json')
    assert observed['returncode'] == code and observed['cleanup_completed'] is True
    assert 'CPU TEST DOUBLE' in (output/'producer.log').read_text()
    assert not (output/'GPU_RELEASE_COMPLETE.json').exists()
    assert not (output/'COMPLETE.json').exists()
