"""Fabricated D01 JSON reporting tests; no GPU, model, runtime or corpus runs.

Activate pllm; python -m pytest -q scripts/tests/immune_llada/test_typed_trainer_runtime_reporting.py
These temporary metadata fixtures test presentation and byte bindings only and
must never be used as a GPU acceptance or biological data certificate.
"""
import hashlib
import json

import pytest

from scripts import immune_diagnostic_reporting as reporting
from scripts import immune_research_supervision as supervision
from scripts import monitor_immune_research_pool as monitor


def put(path, value):
    """Mark fabricated metadata; preserve the exact fixed job envelope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if set(value) != {'variant', 'mode', 'steps'}:
        value = {**value, 'fabricated_reporting_only':True}
    path.write_text(json.dumps(value, allow_nan=False))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def present(run, summary):
    put(run/'summary.json', summary)
    put(run/'COMPLETE.json', dict(schema=reporting.TRAINER_SCHEMA, kind=reporting.TRAINER_KIND,
        variant=reporting.TRAINER_VARIANT, status='completed', summary_sha256=sha(run/'summary.json'),
        **{key:summary[key] for key in ('execution_completed', 'functional_gpu_passed',
             'completed_optimizer_window_count', 'runtime_manifest_sha256', 'artifact_sha256')}))


@pytest.fixture
def artificial_report(tmp_path, monkeypatch):
    root = tmp_path/'artificial_repo'
    root.mkdir()
    put(root/'FABRICATED_REPORTING_ONLY.json', dict(scope='all descendants are reporting fixtures, not GPU/data acceptance'))
    monkeypatch.setattr(supervision, 'ROOT', root)
    pool = root/'artificial_pool'
    for name in ('pending', 'running', 'finished', 'results', 'workers', 'profiles', 'runs', 'logs'):
        (pool/name).mkdir(parents=True)
    job = dict(variant=reporting.TRAINER_VARIANT, mode='profile-synthetic', steps=1)
    put(pool/'finished/job.json', job)
    put(pool/'results/job.json', dict(state='succeeded', job=job))
    runtime = root/'output/immune_pilot_runtimes'/reporting.TRAINER_RUNTIME
    sources = {}
    for name in (reporting.TRAINER_ENTRY, reporting.TRAINER_PRODUCER):
        source = runtime/name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('# fabricated JSON reporting source, never executed\n')
        sources[name] = sha(source)
    manifest = dict(schema='immune.pilot_runtime_source_package.v1', runtime_root=str(runtime),
        source_root=str(root), runtime_files_sha256=sources)
    put(runtime/'pilot_runtime_manifest.json', manifest)
    manifest_sha = sha(runtime/'pilot_runtime_manifest.json')
    windows = [dict(step=step, microsteps=8, records=256, finite_gradients=True,
                    observed_gradient_norm=.3, mean_micro_loss=1./step) for step in range(1, 4)]
    run = pool/'runs/job'
    gpu = dict(schema='immune.certified_pilot_gpu_release.v1', kind='full_pilot_runtime_gpu_acceptance',
        variant='r20_refined_control', cuda_execution=True,
        proof_extension=dict(schema='immune.pilot_gpu_release_producer.v1',
            runtime_manifest_sha256=manifest_sha, wrapper_sha256=sources[reporting.TRAINER_ENTRY],
            producer_sha256=sources[reporting.TRAINER_PRODUCER],
            training=dict(optimizer_steps=3, windows=windows)))
    put(run/'gpu_release_summary.json', gpu)
    put(run/'gpu_release_raw.json', dict(formal_training=dict(optimizer_steps=3, windows=windows)))
    put(run/'PRODUCER_SUCCESS.json', dict(status='producer_completed_not_parent_gpu_acceptance'))
    put(run/'process_evidence.json', dict(schema='immune.pilot_gpu_process_receipt.v1', returncode=0,
        cleanup_completed=True, completion_observed_by_parent=True))
    put(run/'GPU_RELEASE_COMPLETE.json', dict(schema=gpu['schema'], status='passed',
        summary_sha256=sha(run/'gpu_release_summary.json'), runtime_manifest_sha256=manifest_sha,
        raw_sha256=sha(run/'gpu_release_raw.json'), producer_success_sha256=sha(run/'PRODUCER_SUCCESS.json'),
        process_evidence_sha256=sha(run/'process_evidence.json')))
    summary = dict(schema=reporting.TRAINER_SCHEMA, kind=reporting.TRAINER_KIND,
        variant=reporting.TRAINER_VARIANT, transport=dict(mode='profile-synthetic', steps=1),
        dataset_kind='artificial_trainer_fixture', biological_quality_claim=False,
        corpus_training_started=False, training_started=True, optimizer_window_count=3,
        completed_optimizer_window_count=3, optimizer_windows=windows,
        execution_completed=True, functional_gpu_passed=True, runtime_root=str(runtime),
        runtime_manifest_sha256=manifest_sha, source_sha256=sources,
        artifact_sha256={name:sha(run/name) for name in reporting.TRAINER_ARTIFACTS})
    present(run, summary)
    return root, pool, run, runtime, summary


def test_report_shows_three_actual_windows_and_artificial_training(artificial_report, tmp_path):
    _, pool, run, runtime, summary = artificial_report
    before = {path:path.read_bytes() for path in pool.rglob('*.json')}
    document = tmp_path/'artificial_supervision.md'
    record = supervision.update(pool, document)['records']['job']
    assert record['kind'] == reporting.TRAINER_KIND and record['alerts'] == []
    assert record['source_status'] == 'matched' and record['source_root'] == str(runtime)
    assert record['step'] is None and record['requested_steps'] is None
    assert record['training_started'] is True and record['corpus_training_started'] is False
    assert record['biological_quality_claim'] is False
    assert record['optimizer_windows'] == summary['optimizer_windows']
    assert '3/3 optimizer windows' in document.read_text() and 'corpus_training_started=False' in document.read_text()
    assert all(path.read_bytes() == value for path, value in before.items())


def test_monitor_discards_profile_and_uses_windows_instead_of_transport_step(artificial_report):
    _, pool, _, _, _ = artificial_report
    previous = dict(records={'job':dict(profile=dict(recommended_micro_batch=256),
                                       trainer_state=dict(global_step=9999))})
    state = monitor.snapshot(pool, previous)
    record = state['records']['job']
    rendered = monitor.render(state, 30)
    assert 'profile' not in record and monitor.progress(record) is None
    assert '3/3 optimizer windows' in rendered and '| 1/1 |' not in rendered
    assert 'corpus\\_training\\_started' in rendered and '| 3 | 8 | 256 | True |' in rendered


@pytest.mark.parametrize('mutation,expected', [
    ('source', 'source_changed'), ('summary', 'run_summary_hash_mismatch'),
    ('artifact', 'trainer_diagnostic_artifact_hash_mismatch'),
    ('raw_windows', 'trainer_diagnostic_gpu_evidence_mismatch'),
    ('manifest', 'trainer_diagnostic_runtime_identity_mismatch'),
    ('false_count', 'trainer_diagnostic_inconsistent_windows'),
    ('corpus_claim', 'trainer_diagnostic_identity_or_claim_mismatch'),
    ('process_bool', 'trainer_diagnostic_process_mismatch'),
    ('transport_bool', 'trainer_diagnostic_identity_or_claim_mismatch'),
])
def test_mutated_artificial_metadata_cannot_be_presented_as_valid(artificial_report, mutation, expected):
    _, pool, run, runtime, summary = artificial_report
    if mutation == 'source':
        (runtime/reporting.TRAINER_PRODUCER).write_text('# changed\n')
    elif mutation == 'summary':
        with (run/'summary.json').open('a') as handle:
            handle.write('\n')
    elif mutation == 'artifact':
        with (run/'gpu_release_raw.json').open('a') as handle:
            handle.write('\n')
    elif mutation == 'raw_windows':
        put(run/'gpu_release_raw.json', dict(formal_training=dict(optimizer_steps=1, windows=summary['optimizer_windows'][:1])))
        summary['artifact_sha256']['gpu_release_raw.json'] = sha(run/'gpu_release_raw.json')
        present(run, summary)
    elif mutation == 'manifest':
        with (runtime/'pilot_runtime_manifest.json').open('a') as handle:
            handle.write('\n')
    elif mutation == 'false_count':
        summary['completed_optimizer_window_count'] = 1
        present(run, summary)
    elif mutation == 'corpus_claim':
        summary['corpus_training_started'] = True
        present(run, summary)
    elif mutation == 'transport_bool':
        summary['transport']['steps'] = True
        present(run, summary)
    else:
        process = json.loads((run/'process_evidence.json').read_text())
        process['returncode'] = False
        put(run/'process_evidence.json', process)
        summary['artifact_sha256']['process_evidence.json'] = sha(run/'process_evidence.json')
        present(run, summary)
    assert expected in supervision.inspect(pool)['records']['job']['alerts']


def test_pending_d01_does_not_infer_training_or_gpu_execution(artificial_report):
    _, pool, run, _, _ = artificial_report
    (pool/'results/job.json').unlink()
    (pool/'finished/job.json').rename(pool/'pending/job.json')
    (run/'summary.json').unlink()
    (run/'COMPLETE.json').unlink()
    record = supervision.inspect(pool)['records']['job']
    assert record['alerts'] == [] and record['training_started'] is None
    assert record['execution_completed'] is None and record['source_status'] == 'pending'
    assert record['corpus_training_started'] is False
    assert '?/3 optimizer windows' in monitor.render(monitor.snapshot(pool), 30)


def test_worker_success_without_completion_is_visible_failure(artificial_report):
    _, pool, run, _, _ = artificial_report
    (run/'COMPLETE.json').unlink()
    assert 'trainer_diagnostic_inconsistent_worker_success' in supervision.inspect(pool)['records']['job']['alerts']


def test_numeric_completion_flag_cannot_replace_boolean_evidence(artificial_report):
    _, pool, run, _, _ = artificial_report
    completion = json.loads((run/'COMPLETE.json').read_text())
    completion['functional_gpu_passed'] = 1
    put(run/'COMPLETE.json', completion)
    assert 'trainer_diagnostic_completion_mismatch' in supervision.inspect(pool)['records']['job']['alerts']


def version_sources(root, name):
    """Fabricated reporting-only source bytes; never a runtime package/execution."""
    runtime = root/'output/immune_pilot_runtimes'/name
    sources = {}
    for relative in (reporting.TRAINER_ENTRY, reporting.TRAINER_PRODUCER):
        path = runtime/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fabricated reporting-only '+name+' source, never executed\n')
        sources[relative] = sha(path)
    put(runtime/'pilot_runtime_manifest.json', dict(schema='immune.pilot_runtime_source_package.v1',
        runtime_root=str(runtime), source_root=str(root), runtime_files_sha256=sources))
    return runtime, sources


def retarget_report(root, run, summary, name):
    runtime, sources = version_sources(root, name)
    manifest_sha = sha(runtime/'pilot_runtime_manifest.json')
    gpu = json.loads((run/'gpu_release_summary.json').read_text())
    gpu['proof_extension'].update(runtime_manifest_sha256=manifest_sha,
        wrapper_sha256=sources[reporting.TRAINER_ENTRY], producer_sha256=sources[reporting.TRAINER_PRODUCER])
    put(run/'gpu_release_summary.json', gpu)
    proof = json.loads((run/'GPU_RELEASE_COMPLETE.json').read_text())
    proof.update(runtime_manifest_sha256=manifest_sha, summary_sha256=sha(run/'gpu_release_summary.json'))
    put(run/'GPU_RELEASE_COMPLETE.json', proof)
    summary.update(runtime_root=str(runtime), runtime_manifest_sha256=manifest_sha, source_sha256=sources,
        artifact_sha256={name:sha(run/name) for name in reporting.TRAINER_ARTIFACTS})
    present(run, summary)
    return runtime


def test_legacy_v1_summary_keeps_its_sources_when_v2_also_exists(artificial_report):
    root, pool, run, runtime, summary = artificial_report
    other, _ = version_sources(root, 'r20_r27_pilot_v2_20261001')
    record = supervision.inspect(pool)['records']['job']
    assert record['source_root'] == str(runtime) and record['source_root'] != str(other)
    assert record['runtime_version'] == 'r20_r27_pilot_v1_20261001'
    assert record['runtime_identity_source'] == 'summary'
    assert record['source_status'] == 'matched' and record['alerts'] == []
    assert record['optimizer_windows'] == summary['optimizer_windows']


@pytest.mark.parametrize('runtime_name', ['r20_r27_pilot_v2_20261001', 'r20_r27_pilot_v3_20261001'])
def test_current_summary_checks_its_bound_sources_artifacts_and_windows(artificial_report, runtime_name):
    root, pool, run, _, summary = artificial_report
    runtime = retarget_report(root, run, summary, runtime_name)
    record = supervision.inspect(pool)['records']['job']
    assert record['source_root'] == str(runtime) and record['runtime_version'] == runtime_name
    assert record['source_status'] == 'matched' and record['alerts'] == []
    assert record['optimizer_windows'] == summary['optimizer_windows']
    assert reporting.optimizer_progress(record) == '3/3 optimizer windows'
    # Missing summary source maps must still reject; request fallback is only
    # available when the summary itself is absent.
    summary.pop('source_sha256')
    present(run, summary)
    assert 'trainer_diagnostic_missing_source_evidence' in supervision.inspect(pool)['records']['job']['alerts']


def test_unknown_noncanonical_and_symlink_roots_reject_without_request_fallback(artificial_report):
    root, pool, run, runtime, summary = artificial_report
    put(run/'request.json', dict(runtime_root=str(runtime), fabricated_reporting_only=True))
    before = {path:path.read_bytes() for path in pool.rglob('*.json')}
    unknowns = [str(root/'output/immune_pilot_runtimes/r20_r27_pilot_v4_20261001'),
                str(runtime)+'/../'+runtime.name, 'relative/runtime']
    link = root/'output/immune_pilot_runtimes/r20_r27_pilot_v2_20261001'
    link.symlink_to(runtime, target_is_directory=True)
    unknowns.append(str(link))
    for value in unknowns:
        summary['runtime_root'] = value
        present(run, summary)
        record = supervision.inspect(pool)['records']['job']
        assert record['source_root'] is None and record['runtime_version'] is None
        assert record['source_status'] == 'rejected' and record['runtime_identity_source'] == 'summary'
        assert 'trainer_diagnostic_runtime_identity_mismatch' in record['alerts']
        assert record['execution_completed'] is None and record['optimizer_windows'] == []
    assert all(path.read_bytes() == value for path,value in before.items()
               if path.name not in {'summary.json','COMPLETE.json'})


def test_failed_without_summary_uses_each_pinned_request_runtime_without_gpu_inference(artificial_report):
    root, pool, run, legacy, summary = artificial_report
    current, _ = version_sources(root, 'r20_r27_pilot_v2_20261001')
    for name in ('summary.json','COMPLETE.json',*reporting.TRAINER_ARTIFACTS):
        (run/name).unlink()
    put(pool/'results/job.json', dict(state='failed', job=dict(variant=reporting.TRAINER_VARIANT,
        mode='profile-synthetic', steps=1)))
    latest, _ = version_sources(root, 'r20_r27_pilot_v3_20261001')
    for runtime in (legacy,current,latest):
        request = dict(schema=reporting.TRAINER_SCHEMA, kind=reporting.TRAINER_KIND,
            variant=reporting.TRAINER_VARIANT, transport=dict(mode='profile-synthetic',steps=1),
            runtime_root=str(runtime), runtime_manifest_sha256=sha(runtime/'pilot_runtime_manifest.json'),
            biological_quality_claim=False)
        put(run/'request.json', request)
        before = {path:path.read_bytes() for path in pool.rglob('*.json')}
        record = supervision.inspect(pool)['records']['job']
        assert record['source_root'] == str(runtime) and record['runtime_version'] == runtime.name
        assert record['runtime_identity_source'] == 'request' and record['source_status'] == 'matched'
        assert record['source_files'] == 2 and record['alerts'] == ['trainer_diagnostic_execution_failed']
        assert record['execution_completed'] is None and record['functional_gpu_passed'] is None
        assert record['optimizer_windows'] == [] and reporting.optimizer_progress(record) == '?/3 optimizer windows'
        assert record['training_started'] is None and record['corpus_training_started'] is False
        assert all(path.read_bytes() == value for path,value in before.items())
