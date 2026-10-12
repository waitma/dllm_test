"""Read-only typed operator and artificial Trainer reports; no Torch/GPU imports.

Activate pllm, then run:
  python -m pytest scripts/tests/immune_llada/test_typed_diagnostic_reporting.py -q
The pool monitor and supervisor use this module. It never executes a diagnostic,
modifies queue entries, certifies model memory, or interprets cases as train steps.
"""
import hashlib
import json
from pathlib import Path
import re


KIND = 'cuda_attention_backend_probe'
VARIANT = 'd00_cuda_attention_probe_v1'
SCHEMA = 'immune.pool_attention_diagnostic.v1'
COMPLETED_STATUSES = {'passed', 'unsupported', 'oom', 'numerical_mismatch'}
FAILURE_STATUSES = {'backend_mismatch', 'unavailable', 'timeout', 'not_run_budget', 'failed', 'dry_run'}
TRAINER_KIND = 'trainer_gpu_runtime_functional_diagnostic'
TRAINER_VARIANT = 'd01_pilot_runtime_gpu_r20_v1'
TRAINER_SCHEMA = 'immune.pool_trainer_gpu_runtime.v1'
TRAINER_RUNTIME = 'r20_r27_pilot_v1_20261001'
TRAINER_RUNTIMES = frozenset({TRAINER_RUNTIME, 'r20_r27_pilot_v2_20261001',
    'r20_r27_pilot_v3_20261001'})
TRAINER_ARTIFACTS = frozenset({'GPU_RELEASE_COMPLETE.json', 'gpu_release_summary.json',
    'gpu_release_raw.json', 'process_evidence.json', 'PRODUCER_SUCCESS.json'})
TRAINER_ENTRY = 'scripts/architecture_prototypes/pilot_runtime_entry_v1.py'
TRAINER_PRODUCER = 'scripts/architecture_prototypes/pilot_gpu_release_producer_v1.py'
CANDIDATE_PACKAGE = 'scripts/architecture_prototypes/pilot_candidate_runtime_package_v1.py'
CANDIDATE_INSPECTOR = 'scripts/architecture_prototypes/pilot_candidate_runtime_inspect_v1.py'
CANDIDATE_TRAINERS = {
    'r21_refined_head_gate': dict(package_name='r21_pilot_v1_20261001',
        entry='scripts/architecture_prototypes/pilot_runtime_entry_r21_v1.py',
        producer='scripts/architecture_prototypes/pilot_gpu_release_producer_r21_v1.py'),
    'r24_refined_fixed_scale': dict(package_name='r24_pilot_v1_20261001',
        entry='scripts/architecture_prototypes/pilot_runtime_entry_r24_v1.py',
        producer='scripts/architecture_prototypes/pilot_gpu_release_producer_r24_v1.py'),
    'r26_refined_local_global': dict(package_name='r26_pilot_v1_20261001',
        entry='scripts/architecture_prototypes/pilot_runtime_entry_r26_v1.py',
        producer='scripts/architecture_prototypes/pilot_gpu_release_producer_r26_v1.py'),
}


def is_trainer_diagnostic(job, summary, completion=None):
    candidate = job.get('variant') in CANDIDATE_TRAINERS
    exact = (candidate and job == dict(variant=job['variant'], mode='profile-synthetic', steps=1)
             and type(job.get('steps')) is int)
    return (job.get('variant') == TRAINER_VARIANT or exact or summary.get('kind') == TRAINER_KIND
            or (completion or {}).get('kind') == TRAINER_KIND)


def is_diagnostic(job, summary, completion=None):
    return (job.get('variant') == VARIANT or summary.get('kind') == KIND
            or (completion or {}).get('kind') == KIND or is_trainer_diagnostic(job, summary, completion))


def optimizer_progress(summary):
    count = summary.get('completed_optimizer_window_count')
    return f'{count if type(count) is int and 0 <= count <= 3 else "?"}/3 optimizer windows'


def case_progress(summary):
    if is_trainer_diagnostic({}, summary):
        return optimizer_progress(summary)
    count = summary.get('completed_case_count')
    return f'{count if type(count) is int and 0 <= count <= 4 else "?"}/4 cases'


def _read(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _sha(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _trainer_runtime(source_root, declared_root, variant=TRAINER_VARIANT):
    """Choose only a declared canonical root in the fixed read-only whitelist."""
    names = (CANDIDATE_TRAINERS[variant]['package_name'],) if variant in CANDIDATE_TRAINERS else (
        TRAINER_RUNTIMES if variant == TRAINER_VARIANT else ())
    for name in names:
        runtime = source_root/'output/immune_pilot_runtimes'/name
        if declared_root == str(runtime):
            if source_root == source_root.resolve() and runtime == runtime.resolve():
                return runtime
            return None
    return None


def _trainer_sources(record, runtime, source_root, declared_sha, sources=None, *, request_manifest_sources=False,
                     variant=TRAINER_VARIANT):
    manifest_path = runtime/'pilot_runtime_manifest.json'
    manifest = _read(manifest_path)
    if (manifest.get('schema') != 'immune.pilot_runtime_source_package.v1'
            or manifest.get('runtime_root') != str(runtime) or manifest.get('source_root') != str(source_root)
            or _sha(manifest_path) != declared_sha):
        record['alerts'].append('trainer_diagnostic_runtime_identity_mismatch')
    if variant in CANDIDATE_TRAINERS:
        identity = CANDIDATE_TRAINERS[variant]
        additions = manifest.get('additions_sha256')
        files = manifest.get('runtime_files_sha256')
        required = {identity['entry'], identity['producer'], CANDIDATE_PACKAGE, CANDIDATE_INSPECTOR}
        if (manifest.get('candidate_variant') != variant or manifest.get('package_name') != identity['package_name']
                or type(files) is not dict or len(files) != 528
                or type(additions) is not dict or set(additions) != required
                or any(files.get(name) != value for name, value in (additions or {}).items())):
            record['alerts'].append('trainer_diagnostic_runtime_identity_mismatch')
    # A failed run has no summary source map. Its pinned request manifest can
    # still bind source bytes without inventing GPU execution or formal windows.
    if sources is None and request_manifest_sources:
        sources = manifest.get('runtime_files_sha256')
    if type(sources) is not dict or not sources or sources != manifest.get('runtime_files_sha256'):
        record['alerts'].append('trainer_diagnostic_missing_source_evidence')
        return
    record['source_files'] = len(sources)
    for relative, expected in sources.items():
        if type(relative) is not str:
            record['source_mismatches'].append(str(relative))
            continue
        path = runtime/relative
        if (Path(relative).is_absolute() or path != path.resolve() or not path.is_relative_to(runtime)
                or not path.is_file() or type(expected) is not str
                or not re.fullmatch('[0-9a-f]{64}', expected) or _sha(path) != expected):
            record['source_mismatches'].append(relative)
    record['source_status'] = 'mismatch' if record['source_mismatches'] else 'matched'
    if record['source_mismatches']:
        record['alerts'].append('source_changed')


def inspect_trainer_diagnostic(job, state, run, summary, completion, source_root):
    """Present byte-bound observed windows; never issue a GPU/data certificate."""
    alerts = []
    request = {} if summary else _read(run/'request.json')
    declaration = summary if summary else request
    variant = job.get('variant')
    runtime = _trainer_runtime(source_root, declaration.get('runtime_root'), variant)
    identity = CANDIDATE_TRAINERS.get(variant, {})
    entry = identity.get('entry', TRAINER_ENTRY)
    producer = identity.get('producer', TRAINER_PRODUCER)
    model_variant = variant if variant in CANDIDATE_TRAINERS else 'r20_refined_control'
    record = dict(kind=TRAINER_KIND, variant=job.get('variant'), mode=job.get('mode'),
        requested_steps=None, step=None, metrics={}, initial_metrics={}, final_metrics={},
        trace=[], train_probe={}, source_root=str(runtime) if runtime else None, source_files=0,
        source_status='pending', source_mismatches=[], profile_status=None, micro_batch=None,
        profile_peak_gib=None, profile_oom_batches=[], completion=completion,
        optimizer_window_count=3, completed_optimizer_window_count=None, optimizer_windows=[],
        execution_completed=None, functional_gpu_passed=None, training_started=None,
        dataset_kind='artificial_trainer_fixture', corpus_training_started=False,
        biological_quality_claim=False, alerts=alerts,
        runtime_version=runtime.name if runtime else None,
        runtime_identity_source='summary' if summary else ('request' if request else None))
    expected_job = dict(variant=variant, mode='profile-synthetic', steps=1)
    if (variant != TRAINER_VARIANT and variant not in CANDIDATE_TRAINERS
            or job != expected_job or type(job.get('steps')) is not int):
        alerts.append('trainer_diagnostic_invalid_envelope')
    if declaration and runtime is None:
        record['source_status'] = 'rejected'
        alerts.append('trainer_diagnostic_runtime_identity_mismatch')
        if state == 'failed':
            alerts.append('trainer_diagnostic_execution_failed')
        return record
    if not summary:
        if request and runtime:
            transport = request.get('transport')
            if (request.get('schema') != TRAINER_SCHEMA or request.get('kind') != TRAINER_KIND
                    or request.get('variant') != variant
                    or type(transport) is not dict or transport != dict(mode='profile-synthetic', steps=1)
                    or type(transport.get('steps')) is not int
                    or request.get('biological_quality_claim') is not False):
                alerts.append('trainer_diagnostic_request_identity_mismatch')
            record['runtime_manifest_sha256'] = request.get('runtime_manifest_sha256')
            _trainer_sources(record, runtime, source_root, request.get('runtime_manifest_sha256'),
                             request_manifest_sources=True, variant=variant)
        if completion or state == 'succeeded':
            alerts.append('trainer_diagnostic_missing_summary')
        if state == 'failed':
            alerts.append('trainer_diagnostic_execution_failed')
        return record
    record.update({key: summary.get(key) for key in ('completed_optimizer_window_count', 'optimizer_windows',
        'execution_completed', 'functional_gpu_passed', 'training_started', 'runtime_manifest_sha256')})
    transport = summary.get('transport')
    if (summary.get('schema') != TRAINER_SCHEMA or summary.get('kind') != TRAINER_KIND
            or summary.get('variant') != variant
            or type(transport) is not dict or transport != dict(mode='profile-synthetic', steps=1)
            or type(transport.get('steps')) is not int
            or summary.get('dataset_kind') != 'artificial_trainer_fixture'
            or summary.get('biological_quality_claim') is not False
            or summary.get('corpus_training_started') is not False
            or type(summary.get('training_started')) is not bool
            or any(key in summary for key in ('global_step', 'trace', 'recommended_micro_batch'))):
        alerts.append('trainer_diagnostic_identity_or_claim_mismatch')
    windows = summary.get('optimizer_windows')
    count = summary.get('completed_optimizer_window_count')
    valid_windows = (type(windows) is list and type(count) is int and count == len(windows) and 0 <= count <= 3
        and type(summary.get('optimizer_window_count')) is int and summary['optimizer_window_count'] == 3
        and all(type(window) is dict and type(window.get('step')) is int and window['step'] == index
                and type(window.get('microsteps')) is int and window['microsteps'] == 8
                and type(window.get('records')) is int and window['records'] == 256
                and window.get('finite_gradients') is True for index, window in enumerate(windows, 1)))
    if not valid_windows or (summary.get('execution_completed') is True
                            and (count != 3 or summary.get('training_started') is not True
                                 or summary.get('functional_gpu_passed') is not True)):
        alerts.append('trainer_diagnostic_inconsistent_windows')
    sources = summary.get('source_sha256')
    source_map = sources if type(sources) is dict else {}
    _trainer_sources(record, runtime, source_root, summary.get('runtime_manifest_sha256'), sources, variant=variant)
    artifacts = summary.get('artifact_sha256')
    if type(artifacts) is not dict or set(artifacts) != TRAINER_ARTIFACTS:
        alerts.append('trainer_diagnostic_missing_artifact_evidence')
    else:
        if any(type(expected) is not str or not re.fullmatch('[0-9a-f]{64}', expected)
               or (run/name) != (run/name).resolve() or _sha(run/name) != expected
               for name, expected in artifacts.items()):
            alerts.append('trainer_diagnostic_artifact_hash_mismatch')
        gpu = _read(run/'gpu_release_summary.json')
        proof = _read(run/'GPU_RELEASE_COMPLETE.json')
        process = _read(run/'process_evidence.json')
        raw = _read(run/'gpu_release_raw.json')
        extension = gpu.get('proof_extension', {})
        extension = extension if type(extension) is dict else {}
        training = extension.get('training', {})
        training = training if type(training) is dict else {}
        formal = raw.get('formal_training', {})
        formal = formal if type(formal) is dict else {}
        if (gpu.get('schema') != 'immune.certified_pilot_gpu_release.v1'
                or gpu.get('kind') != 'full_pilot_runtime_gpu_acceptance' or gpu.get('variant') != model_variant
                or gpu.get('cuda_execution') is not True or extension.get('schema') != 'immune.pilot_gpu_release_producer.v1'
                or extension.get('runtime_manifest_sha256') != summary.get('runtime_manifest_sha256')
                or extension.get('wrapper_sha256') != source_map.get(entry)
                or extension.get('producer_sha256') != source_map.get(producer)
                or training.get('windows') != windows or training.get('optimizer_steps') != count
                or formal.get('windows') != windows or formal.get('optimizer_steps') != count):
            alerts.append('trainer_diagnostic_gpu_evidence_mismatch')
        if (proof.get('schema') != gpu.get('schema') or proof.get('status') != 'passed'
                or proof.get('summary_sha256') != _sha(run/'gpu_release_summary.json')
                or proof.get('runtime_manifest_sha256') != summary.get('runtime_manifest_sha256')
                or proof.get('raw_sha256') != _sha(run/'gpu_release_raw.json')
                or proof.get('producer_success_sha256') != _sha(run/'PRODUCER_SUCCESS.json')
                or proof.get('process_evidence_sha256') != _sha(run/'process_evidence.json')):
            alerts.append('trainer_diagnostic_gpu_completion_mismatch')
        if (process.get('schema') != 'immune.pilot_gpu_process_receipt.v1'
                or type(process.get('returncode')) is not int or process['returncode'] != 0
                or process.get('cleanup_completed') is not True
                or process.get('completion_observed_by_parent') is not True):
            alerts.append('trainer_diagnostic_process_mismatch')
    if completion:
        if (completion.get('schema') != TRAINER_SCHEMA or completion.get('kind') != TRAINER_KIND
                or completion.get('variant') != variant or completion.get('status') != 'completed'
                or type(completion.get('completed_optimizer_window_count')) is not int
                or any(completion.get(key) is not summary.get(key) for key in
                       ('execution_completed', 'functional_gpu_passed'))
                or any(completion.get(key) != summary.get(key) for key in
                       ('completed_optimizer_window_count', 'runtime_manifest_sha256', 'artifact_sha256'))
                or 'global_step' in completion):
            alerts.append('trainer_diagnostic_completion_mismatch')
        if _sha(run/'summary.json') != completion.get('summary_sha256'):
            alerts.append('run_summary_hash_mismatch')
    if state == 'succeeded' and (not completion or summary.get('execution_completed') is not True
                               or summary.get('functional_gpu_passed') is not True or not valid_windows or count != 3):
        alerts.append('trainer_diagnostic_inconsistent_worker_success')
    if state == 'failed' or summary.get('execution_completed') is False:
        alerts.append('trainer_diagnostic_execution_failed')
    return record


def inspect_diagnostic(job, state, run, summary, completion, source_root):
    """Check report identity and byte bindings; kernel validation is the adapter's job."""
    if is_trainer_diagnostic(job, summary, completion):
        return inspect_trainer_diagnostic(job, state, run, summary, completion, source_root)
    alerts = []
    record = dict(kind=KIND, variant=job.get('variant'), mode=job.get('mode'),
        requested_steps=None, step=None, metrics={}, initial_metrics={}, final_metrics={},
        trace=[], train_probe={}, source_root=str(source_root), source_files=0,
        source_status='pending', source_mismatches=[], profile_status=None, micro_batch=None,
        profile_peak_gib=None, profile_oom_batches=[], completion=completion,
        case_count=4, completed_case_count=None, case_statuses={}, execution_completed=None,
        all_cases_passed=None, training_started=False, biological_quality_claim=False,
        alerts=alerts)
    expected_job = dict(variant=VARIANT, mode='profile-synthetic', steps=1)
    if job != expected_job or type(job.get('steps')) is not int:
        alerts.append('diagnostic_invalid_envelope')
    if not summary:
        if completion or state == 'succeeded':
            alerts.append('diagnostic_missing_summary')
        if state == 'failed':
            alerts.append('diagnostic_execution_failed')
        return record
    record.update({key: summary.get(key) for key in
        ('completed_case_count', 'case_statuses', 'execution_completed', 'all_cases_passed')})
    if (summary.get('schema_version') != SCHEMA or summary.get('kind') != KIND or summary.get('variant') != VARIANT
            or summary.get('training_started') is not False
            or summary.get('biological_quality_claim') is not False
            or any(key in summary for key in ('global_step', 'trace', 'recommended_micro_batch'))):
        alerts.append('diagnostic_identity_or_claim_mismatch')
    sources = summary.get('source_sha256')
    if not isinstance(sources, dict) or not sources:
        alerts.append('diagnostic_missing_source_evidence')
    else:
        record['source_files'] = len(sources)
        for relative, expected in sources.items():
            if not isinstance(relative, str):
                record['source_mismatches'].append(str(relative))
                continue
            path = (source_root / relative).resolve()
            if (Path(relative).is_absolute() or not path.is_relative_to(source_root)
                    or not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)
                    or _sha(path) != expected):
                record['source_mismatches'].append(relative)
        record['source_status'] = 'mismatch' if record['source_mismatches'] else 'matched'
        if record['source_mismatches']:
            alerts.append('source_changed')
    statuses = summary.get('case_statuses')
    valid_statuses = (isinstance(statuses, dict) and len(statuses) == 4
        and all(isinstance(key, str) and isinstance(value, str)
                and value in COMPLETED_STATUSES | FAILURE_STATUSES for key, value in statuses.items()))
    if not valid_statuses:
        alerts.append('diagnostic_invalid_case_statuses')
    else:
        terminal_statuses = sum(value in COMPLETED_STATUSES for value in statuses.values())
        completed = summary.get('completed_case_count')
        execution = summary.get('execution_completed')
        if (type(summary.get('case_count')) is not int or summary['case_count'] != 4
                or type(completed) is not int or not 0 <= completed <= terminal_statuses
                or type(execution) is not bool or (execution and completed != 4)
                or (not execution and terminal_statuses == 4 and not summary.get('error'))
                or summary.get('all_cases_passed') is not (execution is True and all(value == 'passed' for value in statuses.values()))):
            alerts.append('diagnostic_inconsistent_completion')
    artifacts = summary.get('evidence_sha256')
    if not isinstance(artifacts, dict) or not artifacts:
        alerts.append('diagnostic_missing_artifact_evidence')
    else:
        for relative, expected in artifacts.items():
            if not isinstance(relative, str):
                alerts.append('diagnostic_artifact_hash_mismatch')
                break
            path = (run/relative).resolve()
            if (Path(relative).is_absolute() or not path.is_relative_to(run.resolve())
                    or not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)
                    or _sha(path) != expected):
                alerts.append('diagnostic_artifact_hash_mismatch')
                break
    probe_path = run / 'probe' / 'summary.json'
    if _sha(probe_path) != summary.get('probe_summary_sha256') or not probe_path.is_file():
        alerts.append('diagnostic_probe_hash_mismatch')
    else:
        probe = _read(probe_path)
        rows = probe.get('results')
        if (not isinstance(rows, list) or len(rows) != 4
                or not all(isinstance(row, dict) and isinstance(row.get('case_id'), str) for row in rows)
                or len({row['case_id'] for row in rows}) != 4
                or {row['case_id']: row.get('status') for row in rows} != statuses):
            alerts.append('diagnostic_probe_case_mismatch')
    if summary.get('execution_completed') is True and (summary.get('source_sha256_before') != sources
                                                       or summary.get('source_sha256_after') != sources):
        alerts.append('diagnostic_source_stability_mismatch')
    if completion:
        if (completion.get('schema_version') != SCHEMA or completion.get('kind') != KIND or completion.get('variant') != VARIANT
                or completion.get('training_started') is not False
                or completion.get('biological_quality_claim') is not False
                or completion.get('execution_completed') is not summary.get('execution_completed')
                or completion.get('all_cases_passed') is not summary.get('all_cases_passed')
                or any(completion.get(key) != summary.get(key) for key in
                       ('case_count', 'completed_case_count', 'case_statuses', 'source_sha256', 'probe_summary_sha256'))
                or 'global_step' in completion):
            alerts.append('diagnostic_completion_mismatch')
        if _sha(run/'summary.json') != completion.get('summary_sha256'):
            alerts.append('run_summary_hash_mismatch')
    if state == 'succeeded' and (not completion or summary.get('execution_completed') is not True
                               or completion.get('status') != 'completed'):
        alerts.append('diagnostic_inconsistent_worker_success')
    if state == 'failed' or summary.get('execution_completed') is False:
        alerts.append('diagnostic_execution_failed')
    # Negative backend findings are visible statuses, not invented training failures.
    return record
