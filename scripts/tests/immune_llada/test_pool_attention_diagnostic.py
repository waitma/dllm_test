"""Activate pllm; python -m pytest -q scripts/tests/immune_llada/test_pool_attention_diagnostic.py.

CPU contract tests only: synthetic worker JSON is explicitly a test double and
exists only in pytest temporary directories. No GPU or actual child is launched.
Real child lifecycle tests live in test_cuda_attention_probe_lifecycle_cpu.py.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import torch

from scripts.architecture_prototypes import pool_attention_diagnostic as adapter

probe = adapter.probe


@pytest.fixture(autouse=True)
def forbid_gpu_and_real_processes(monkeypatch):
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        pytest.fail('CPU adapter test attempted a GPU operation or real child')
    for name in ('is_available', 'set_device', 'synchronize', 'empty_cache', 'get_device_properties'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(probe, 'run_cuda_case', forbidden)
    monkeypatch.setattr(probe, 'run_case_process', forbidden)


@pytest.fixture
def location(tmp_path):
    pool = tmp_path/'pool'
    (pool/'runs').mkdir(parents=True)
    (pool/'profiles').mkdir()
    return pool/'runs'/'one', pool/'profiles'/adapter.VARIANT


def envelope():
    return dict(variant=adapter.VARIANT, **adapter.TRANSPORT)


def worker_fixture(case, sources, status):
    """Entirely fabricated CUDA metadata for dispatch validation, never GPU proof."""
    row = probe.empty_result(case, status, 'complete', sources)
    snapshots = {name: torch.ones(2) for name in ('output', 'q_grad', 'k_grad', 'v_grad', 'bias_parameter_grad')}
    operators = (['aten::_scaled_dot_product_attention_math'] if case.backend == 'math' else
        ['aten::_scaled_dot_product_efficient_attention', 'aten::_scaled_dot_product_efficient_attention_backward'])
    row.update(gpu_execution='completed', elapsed_seconds=.25,
        environment=dict(cuda_available=True, torch_version='CPU TEST DOUBLE', python='CPU TEST DOUBLE',
            gpu='CPU TEST DOUBLE, NOT A GPU', cuda_build='CPU TEST DOUBLE', total_memory_bytes=1024,
            device_index=0, activation_dtype=case.precision, leaf_dtype='float32', capability=[0, 0],
            gpu_uuid=None, cudnn=0), input_sha256=probe.input_digest(probe.make_inputs(case)),
        correctness=probe.compare_snapshots(snapshots, snapshots, case.precision),
        backend_evidence=dict(requested=case.backend, observed=case.backend, operators=operators,
            cuda_kernel_names=['CPU TEST DOUBLE, NOT A CUDA KERNEL']),
        timing=dict(warmup=5, iterations=20, raw_total_ms=[1.]*20, wall_seconds=.1,
            **{key: probe.timing_summary([1.]*20) for key in ('forward_ms', 'backward_ms', 'total_ms')}),
        memory=dict(baseline_allocated_bytes=100, peak_allocated_bytes=160,
            incremental_peak_allocated_bytes=60, peak_reserved_bytes=180))
    if status != 'passed':
        row.update(phase='candidate_profile_and_gradients', gpu_execution='attempted', timing=None, memory=None,
            error=dict(type='RuntimeError', message='CPU TEST DOUBLE: requested backend unsupported'))
        if status == 'numerical_mismatch':
            changed = {key: value+1 for key, value in snapshots.items()}
            row['correctness'] = probe.compare_snapshots(snapshots, changed, case.precision)
            row['error']['type'] = 'NumericalMismatch'
        else:
            row['correctness'] = row['backend_evidence'] = None
    return row


def mock_workers(monkeypatch, statuses=None, mutate=None, returncode=None):
    statuses = statuses or ['passed']*4
    calls = []
    def child(command, *, stdout, timeout):
        index = len(calls)
        calls.append(command)
        assert 0 < timeout <= adapter.CASE_TIMEOUT
        spec = json.loads(Path(command[command.index('--worker-spec')+1]).read_text())
        case = probe.ProbeCase(**spec['case'])
        row = worker_fixture(case, spec['source_fingerprints'], statuses[index])
        if mutate:
            mutate(index, row)
        probe.write_json(Path(command[command.index('--worker-result')+1]), row)
        return SimpleNamespace(returncode=(returncode if returncode is not None else 0 if row['status']=='passed' else 2))
    monkeypatch.setattr(probe, 'run_case_process', child)
    return calls


def run(location):
    output, profile = location
    return adapter.run_diagnostic(envelope(), output_dir=output, profile_dir=profile)


def argv(location):
    output, profile = location
    return ['--variant', adapter.VARIANT, '--mode', 'profile-synthetic', '--steps', '1',
            '--profile-dir', str(profile), '--output-dir', str(output)]


def test_fixed_protocol_and_full_artifact_binding(monkeypatch, location):
    calls = mock_workers(monkeypatch)
    before = torch.get_rng_state().clone()
    report = run(location)
    output, profile = location
    assert torch.equal(before, torch.get_rng_state())
    assert len(calls) == 4 and report['execution_completed'] and report['all_cases_passed']
    assert report['completed_case_count'] == report['case_count'] == 4
    assert report['source_sha256'] == report['source_sha256_before'] == report['source_sha256_after']
    assert len(report['source_sha256']) == 3
    assert all(not Path(key).is_absolute() for key in report['source_sha256'])
    assert report['kind'] == adapter.KIND and report['training_started'] is False
    assert report['biological_quality_claim'] is False and 'global_step' not in report
    assert not profile.exists() and not (output/'FAILED.json').exists()
    marker = json.loads((output/'COMPLETE.json').read_text())
    assert marker['summary_sha256'] == hashlib.sha256((output/'summary.json').read_bytes()).hexdigest()
    assert marker['probe_summary_sha256'] == hashlib.sha256((output/'probe'/'summary.json').read_bytes()).hexdigest()
    for name, digest in report['evidence_sha256'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest() == digest
    plan = json.loads((output/'probe'/'plan.json').read_text())
    assert plan['case_timeout_seconds'] == 120 and plan['total_timeout_seconds'] == 600 and plan['dry_run'] is False
    assert [(case['backend'], case['precision']) for case in plan['cases']] == [
        ('math','float32'), ('math','bfloat16'), ('efficient','float32'), ('efficient','bfloat16')]
    for case in plan['cases']:
        assert (case['batch_size'],case['heads'],case['length'],case['head_dim'],case['warmup'],case['iterations']) == (1,12,256,64,5,20)


@pytest.mark.parametrize('status', ['unsupported', 'oom', 'numerical_mismatch'])
def test_valid_negative_is_completed_but_not_gpu_pass(monkeypatch, location, status):
    mock_workers(monkeypatch, ['passed']*3+[status])
    assert adapter.main(argv(location)) == 0
    report = json.loads((location[0]/'summary.json').read_text())
    assert report['execution_completed'] is True and report['all_cases_passed'] is False
    assert report['completed_case_count'] == 4 and status in report['case_statuses'].values()
    assert (location[0]/'COMPLETE.json').is_file()


@pytest.mark.parametrize('status', ['timeout', 'unavailable', 'failed', 'backend_mismatch', 'not_run_budget', 'dry_run'])
def test_execution_failures_never_complete(monkeypatch, location, status):
    mock_workers(monkeypatch, ['passed']*3+[status])
    assert adapter.main(argv(location)) == 2
    report = json.loads((location[0]/'summary.json').read_text())
    assert not report['execution_completed'] and not report['all_cases_passed']
    assert report['completed_case_count'] == 3
    assert not (location[0]/'COMPLETE.json').exists() and (location[0]/'FAILED.json').is_file()


@pytest.mark.parametrize('change', [dict(variant='r20'), dict(mode='train'), dict(steps=2), dict(steps=True),
    dict(steps=1.), dict(data_dir='x'), dict(resume_from_checkpoint='x'), dict(shell='echo x'), dict(timeout=999)])
def test_envelope_is_exact(change):
    job = envelope()
    job.update(change)
    with pytest.raises(adapter.EvidenceError):
        adapter.validate_envelope(job)


@pytest.mark.parametrize('args', [['--data-dir','x'], ['--resume-from-checkpoint','x'], ['--worker-spec','x'],
    ['--case-timeout','999'], ['--dry-run'], ['--output-d','x']])
def test_parser_rejects_unapproved_controls(location, args):
    with pytest.raises(SystemExit) as caught:
        adapter.parser().parse_args(argv(location)+args)
    assert caught.value.code == 2


@pytest.mark.parametrize('kind', ['relative', 'existing', 'wrong_pool', 'wrong_variant', 'symlink'])
def test_paths_prevent_overwrite_or_redirect(monkeypatch, location, kind):
    output, profile = location
    if kind == 'relative': output = Path('runs/job')
    if kind == 'existing': output.mkdir()
    if kind == 'wrong_pool': profile = profile.parent.parent.parent/'other'/'profiles'/adapter.VARIANT
    if kind == 'wrong_variant': profile = profile.with_name('another')
    if kind == 'symlink':
        target = output.parent.parent/'target'
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    with pytest.raises(adapter.EvidenceError):
        adapter.run_diagnostic(envelope(), output_dir=output, profile_dir=profile)


def tamper_plan(monkeypatch, operation):
    original = probe.execute_plan
    def changed(*args, **kwargs):
        summary = original(*args, **kwargs)
        operation(Path(args[1]), summary)
        return summary
    monkeypatch.setattr(probe, 'execute_plan', changed)


@pytest.mark.parametrize('kind', ['returned', 'duplicate', 'missing', 'counts', 'comparison', 'spec', 'result',
    'budget', 'plan_dry_run', 'source', 'float_case', 'nonfinite', 'duplicate_key', 'missing_log'])
def test_corrupt_or_inconsistent_evidence_rejected(monkeypatch, location, kind):
    mock_workers(monkeypatch)
    def change(directory, summary):
        if kind == 'returned':
            summary['status'] = 'different'
            return
        case_dir = next(directory.glob('00_*'))
        if kind in {'spec', 'result', 'budget', 'plan_dry_run'}:
            path = (case_dir/('spec.json' if kind=='spec' else 'result.json')
                    if kind in {'spec','result'} else directory/'plan.json')
            obj = json.loads(path.read_text())
            if kind in {'spec', 'result'}: obj['case']['seed'] += 1
            elif kind == 'budget': obj['total_timeout_seconds'] += 1
            else: obj['dry_run'] = True
            probe.write_json(path, obj)
            return
        if kind == 'missing_log':
            (case_dir/'worker.log').unlink()
            return
        if kind == 'nonfinite':
            (directory/'summary.json').write_text('{"bad":1e999}')
            return
        if kind == 'duplicate_key':
            (directory/'summary.json').write_text('{"bad":1,"bad":2}')
            return
        if kind == 'duplicate': summary['results'][1] = deepcopy(summary['results'][0])
        elif kind == 'missing': summary['results'].pop()
        elif kind == 'counts': summary['status_counts']['passed'] = 3
        elif kind == 'comparison': summary['matched_comparisons'] = []
        elif kind == 'source': summary['source_fingerprints'] = {}
        elif kind == 'float_case':
            summary['results'][0]['case']['batch_size'] = 1.
            probe.write_json(case_dir/'result.json', summary['results'][0])
        probe.write_json(directory/'summary.json', summary)
    tamper_plan(monkeypatch, change)
    report = run(location)
    assert not report['execution_completed'] and not report['all_cases_passed']
    assert report['error'] is not None and not (location[0]/'COMPLETE.json').exists()


@pytest.mark.parametrize('kind', ['parent_phase', 'no_cuda', 'wrong_input', 'no_error', 'false_mismatch', 'missing_child'])
def test_negative_string_is_insufficient_evidence(monkeypatch, location, kind):
    status = 'numerical_mismatch' if kind == 'false_mismatch' else 'unsupported'
    def change(index, row):
        if index != 0: return
        if kind == 'parent_phase': row['phase'] = 'parent_execution'
        elif kind == 'no_cuda': row['gpu_execution'] = 'not_started'
        elif kind == 'wrong_input': row['input_sha256'] = 'wrong'
        elif kind == 'no_error': row['error'] = None
        elif kind == 'false_mismatch':
            for item in row['correctness']['comparisons'].values(): item['passed'] = True
    mock_workers(monkeypatch, [status]+['passed']*3, mutate=change)
    if kind == 'missing_child':
        def remove(directory, summary):
            summary['results'][0].pop('child_process')
            probe.write_json(next(directory.glob('00_*'))/'result.json', summary['results'][0])
            probe.write_json(directory/'summary.json', summary)
        tamper_plan(monkeypatch, remove)
    report = run(location)
    assert not report['execution_completed'] and report['error'] is not None


@pytest.mark.parametrize('returncode', [0, 1, -15, -9, 143])
def test_abnormal_negative_worker_exit_cannot_complete(monkeypatch, location, returncode):
    mock_workers(monkeypatch, ['unsupported']*4, returncode=returncode)
    report = run(location)
    assert not report['execution_completed'] and report['completed_case_count'] == 0
    assert set(report['case_statuses'].values()) == {'failed'}


def test_source_changed_after_four_passes_is_overall_failure(monkeypatch, location):
    mock_workers(monkeypatch)
    before = adapter.source_sha256()
    calls = []
    def sources():
        calls.append(1)
        return before if len(calls)==1 else dict(before, **{next(iter(before)): '0'*64})
    monkeypatch.setattr(adapter, 'source_sha256', sources)
    report = run(location)
    assert report['completed_case_count'] == 4 and set(report['case_statuses'].values()) == {'passed'}
    assert not report['execution_completed'] and not report['all_cases_passed']
    assert report['error']['type'] == 'SourceChanged' and not (location[0]/'COMPLETE.json').exists()


def test_source_unreadable_after_execution_writes_failed(monkeypatch, location):
    mock_workers(monkeypatch)
    before = adapter.source_sha256()
    calls = []
    def sources():
        calls.append(1)
        if len(calls) > 1: raise FileNotFoundError('CPU fixture source removed')
        return before
    monkeypatch.setattr(adapter, 'source_sha256', sources)
    report = run(location)
    assert report['completed_case_count'] == 4 and not report['execution_completed']
    assert report['error']['type'] == 'SourceReadError'
    assert (location[0]/'FAILED.json').exists() and not (location[0]/'COMPLETE.json').exists()


def test_runtime_changes_between_completed_cases_fail(monkeypatch, location):
    def change(index, row):
        if index == 1: row['environment']['gpu'] = 'DIFFERENT CPU TEST DOUBLE'
    mock_workers(monkeypatch, mutate=change)
    report = run(location)
    assert not report['execution_completed'] and 'Runtime/GPU changed' in report['error']['message']


@pytest.mark.parametrize('kind', ['missing', 'negative', 'bool', 'false_finite', 'aggregate'])
def test_incomplete_mismatch_measurements_rejected(monkeypatch, location, kind):
    def change(index, row):
        if index: return
        item = row['correctness']['comparisons']['output']
        if kind == 'missing': item.pop('max_abs_error')
        elif kind == 'negative': item['difference_l2'] = -1
        elif kind == 'bool': item['difference_l2'] = True
        elif kind == 'false_finite': item['finite'] = False
        elif kind == 'aggregate': row['correctness']['passed'] = True
    mock_workers(monkeypatch, ['numerical_mismatch']+['passed']*3, mutate=change)
    report = run(location)
    assert not report['execution_completed'] and report['error'] is not None


def test_nonfinite_mismatch_with_null_metrics_is_completed_negative(monkeypatch, location):
    def change(index, row):
        if index: return
        values = {name: torch.tensor([float('nan')]) for name in
                  ('output', 'q_grad', 'k_grad', 'v_grad', 'bias_parameter_grad')}
        row['correctness'] = probe.compare_snapshots(values, values, row['case']['precision'])
    mock_workers(monkeypatch, ['numerical_mismatch']+['passed']*3, mutate=change)
    report = run(location)
    assert report['execution_completed'] and not report['all_cases_passed']


def test_changed_request_evidence_rejected_at_final_check(monkeypatch, location):
    mock_workers(monkeypatch)
    def change(directory, summary):
        (directory.parent/'execution_request.json').write_text('{}')
    tamper_plan(monkeypatch, change)
    report = run(location)
    assert report['completed_case_count'] == 4 and not report['execution_completed']
    assert report['error']['type'] == 'EvidenceChanged'


@pytest.mark.parametrize('error', [probe.CaseCleanupError('CPU fixture cleanup failed'),
    probe.ProbeInterrupted(signal.SIGTERM), KeyboardInterrupt()])
def test_cleanup_failure_or_interrupt_never_complete(monkeypatch, location, error):
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr(probe, 'execute_plan', fail)
    code = adapter.main(argv(location))
    assert code == (143 if isinstance(error, probe.ProbeInterrupted) else 130 if isinstance(error, KeyboardInterrupt) else 2)
    assert not (location[0]/'COMPLETE.json').exists()
    report = json.loads((location[0]/'FAILED.json').read_text())
    assert not report['execution_completed'] and not report['all_cases_passed']
