"""Artificial JSON contracts, not CUDA evidence. Activate pllm, then run:
python -m pytest scripts/tests/immune_llada/test_typed_diagnostic_reporting.py -q
"""
import hashlib
import json

import pytest

from scripts import immune_diagnostic_reporting as protocol
from scripts import immune_research_supervision as supervision
from scripts import monitor_immune_research_pool as monitor


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch, statuses=None, *, succeeded=True):
    root = tmp_path/'repo'
    root.mkdir()
    monkeypatch.setattr(supervision, 'ROOT', root)
    source = root/'adapter.py'
    source.write_text('# artificial monitor fixture; never GPU evidence\n')
    pool = root/'pool'
    for name in ('pending', 'running', 'finished', 'results', 'workers', 'profiles', 'runs', 'logs'):
        (pool/name).mkdir(parents=True, exist_ok=True)
    job = dict(variant=protocol.VARIANT, mode='profile-synthetic', steps=1)
    put(pool/'finished/job.json', job)
    put(pool/'results/job.json', dict(state='succeeded' if succeeded else 'failed', job=job))
    run = pool/'runs/job'
    statuses = statuses or dict(a='passed', b='passed', c='passed', d='passed')
    put(run/'probe/summary.json', dict(results=[dict(case_id=key, status=value) for key, value in statuses.items()]))
    count = sum(value in protocol.COMPLETED_STATUSES for value in statuses.values())
    complete = count == 4
    summary = dict(schema_version=protocol.SCHEMA, kind=protocol.KIND, variant=protocol.VARIANT,
        transport=dict(mode='profile-synthetic', steps=1), training_started=False, biological_quality_claim=False,
        case_count=4, completed_case_count=count, case_statuses=statuses,
        execution_completed=complete, all_cases_passed=complete and all(s == 'passed' for s in statuses.values()),
        source_sha256={'adapter.py': sha(source)}, probe_summary_sha256=sha(run/'probe/summary.json'), error=None)
    summary.update(source_sha256_before=dict(summary['source_sha256']), source_sha256_after=dict(summary['source_sha256']),
                   evidence_sha256={'probe/summary.json': sha(run/'probe/summary.json')})
    write_report(run, summary)
    return pool, source, run, summary


def write_report(run, summary):
    put(run/'summary.json', summary)
    if summary['execution_completed']:
        keys = ('schema_version', 'kind', 'variant', 'training_started', 'biological_quality_claim',
                'case_count', 'completed_case_count', 'case_statuses', 'execution_completed',
                'all_cases_passed', 'source_sha256', 'probe_summary_sha256')
        put(run/'COMPLETE.json', dict(status='completed', summary_sha256=sha(run/'summary.json'),
                                     **{key: summary[key] for key in keys}))


@pytest.mark.parametrize('negative', sorted(protocol.COMPLETED_STATUSES-{'passed'}))
def test_completed_negative_is_reported_without_training_failure(tmp_path, monkeypatch, negative):
    pool, source, run, summary = fixture(tmp_path, monkeypatch,
        dict(a='passed', b=negative, c='passed', d='passed'))
    before = {path: path.read_bytes() for path in pool.rglob('*.json')}
    document = tmp_path/'supervision.md'
    record = supervision.update(pool, document)['records']['job']
    assert record['execution_completed'] is True and record['all_cases_passed'] is False
    assert record['step'] is None and record['requested_steps'] is None
    assert record['profile_status'] is None and record['micro_batch'] is None
    assert record['alerts'] == [] and record['source_status'] == 'matched'
    assert '4/4 cases' in document.read_text() and negative in record['case_statuses'].values()
    assert all(path.read_bytes() == content for path, content in before.items())


def test_all_passed_has_case_progress_and_discards_stale_profile(tmp_path, monkeypatch):
    pool, source, run, summary = fixture(tmp_path, monkeypatch)
    previous = dict(records={'job': dict(profile=dict(recommended_micro_batch=256))})
    state = monitor.snapshot(pool, previous)
    record = state['records']['job']
    assert monitor.progress(record) is None and 'profile' not in record
    document = monitor.render(state, 30)
    assert '4/4 cases' in document and '| 1/1 |' not in document
    assert 'all\\_cases\\_passed' in document
    assert supervision.inspect(pool)['records']['job']['all_cases_passed'] is True


@pytest.mark.parametrize('status', ['backend_mismatch', 'timeout', 'unavailable', 'not_run_budget'])
def test_unfinished_cases_cannot_be_promoted_by_worker_success(tmp_path, monkeypatch, status):
    pool, _, _, _ = fixture(tmp_path, monkeypatch, dict(a='passed', b=status, c='passed', d='passed'))
    record = supervision.inspect(pool)['records']['job']
    assert record['completed_case_count'] == 3
    assert 'diagnostic_inconsistent_worker_success' in record['alerts']
    assert 'diagnostic_execution_failed' in record['alerts']
    assert 'inconsistent_completion' not in record['alerts']


@pytest.mark.parametrize('mutation,expected', [
    ('source', 'source_changed'), ('summary', 'run_summary_hash_mismatch'),
    ('probe', 'diagnostic_probe_hash_mismatch'), ('duplicate_case', 'diagnostic_probe_case_mismatch'),
    ('false_step', 'diagnostic_identity_or_claim_mismatch'),
    ('wrong_flag', 'diagnostic_inconsistent_completion'),
])
def test_tampered_evidence_is_detected(tmp_path, monkeypatch, mutation, expected):
    pool, source, run, summary = fixture(tmp_path, monkeypatch)
    if mutation == 'source':
        source.write_text('# changed\n')
    elif mutation == 'summary':
        with (run/'summary.json').open('a') as handle:
            handle.write('\n')
    elif mutation == 'probe':
        (run/'probe/summary.json').write_text('{}')
    elif mutation == 'duplicate_case':
        put(run/'probe/summary.json', dict(results=[dict(case_id='a', status='passed')]*4))
        summary['probe_summary_sha256'] = sha(run/'probe/summary.json')
        write_report(run, summary)
    elif mutation == 'false_step':
        summary['global_step'] = 1
        write_report(run, summary)
    else:
        summary['all_cases_passed'] = False
        write_report(run, summary)
    assert expected in supervision.inspect(pool)['records']['job']['alerts']


def test_pending_diagnostic_has_no_training_or_profile_assumptions(tmp_path, monkeypatch):
    pool, _, run, _ = fixture(tmp_path, monkeypatch)
    (pool/'results/job.json').unlink()
    (pool/'finished/job.json').rename(pool/'pending/job.json')
    (run/'summary.json').unlink()
    (run/'COMPLETE.json').unlink()
    record = supervision.inspect(pool)['records']['job']
    assert record['alerts'] == [] and record['step'] is None
    assert record['execution_completed'] is None and record['source_status'] == 'pending'


def test_whole_run_failure_after_completed_cases_is_not_a_contradiction(tmp_path, monkeypatch):
    pool, _, run, summary = fixture(tmp_path, monkeypatch, succeeded=False)
    (run/'COMPLETE.json').unlink()
    summary.update(execution_completed=False, all_cases_passed=False, error='source changed after cases')
    write_report(run, summary)
    record = supervision.inspect(pool)['records']['job']
    assert record['completed_case_count'] == 4
    assert record['alerts'] == ['diagnostic_execution_failed']


def test_corrupted_cases_not_counted_by_status_string_alone(tmp_path, monkeypatch):
    pool, _, run, summary = fixture(tmp_path, monkeypatch, succeeded=False)
    (run/'COMPLETE.json').unlink()
    summary.update(completed_case_count=2, execution_completed=False, all_cases_passed=False,
                   error='two case artifacts failed verification')
    write_report(run, summary)
    assert supervision.inspect(pool)['records']['job']['alerts'] == ['diagnostic_execution_failed']
