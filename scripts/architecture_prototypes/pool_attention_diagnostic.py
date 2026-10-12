"""Fixed, typed random-tensor diagnostic; no live pool routing is installed.

Activate pllm; CPU tests:
python -m pytest -q scripts/tests/immune_llada/test_pool_attention_diagnostic.py
Future approved idle-pool execution only (four CUDA cases, no data/model):
python scripts/architecture_prototypes/pool_attention_diagnostic.py \
  --variant d00_cuda_attention_probe_v1 --mode profile-synthetic --steps 1 \
  --profile-dir /absolute/pool/profiles/d00_cuda_attention_probe_v1 \
  --output-dir /absolute/pool/runs/unique_diagnostic_id
The legacy transport names do not describe training or a model memory profile.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.architecture_prototypes import cuda_attention_probe as probe

SCHEMA = 'immune.pool_attention_diagnostic.v1'
KIND = 'cuda_attention_backend_probe'
VARIANT = 'd00_cuda_attention_probe_v1'
TRANSPORT = {'mode': 'profile-synthetic', 'steps': 1}
CASE_TIMEOUT = 120
TOTAL_TIMEOUT = 600
COMPLETED_STATUSES = frozenset({'passed', 'unsupported', 'oom', 'numerical_mismatch'})
REPO = Path(__file__).resolve().parents[2]


class EvidenceError(ValueError):
    pass


class SourceChanged(EvidenceError):
    pass


def fixed_cases():
    return [probe.ProbeCase(backend=backend, precision=precision, batch_size=1, heads=12,
                length=256, head_dim=64, mask='global', bias_kind='table', padding=True,
                local_radius=32, seed=8421, warmup=5, iterations=20, device_index=0).validate()
            for backend in ('math', 'efficient') for precision in ('float32', 'bfloat16')]


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def validate_envelope(job):
    require(type(job) is dict and set(job) == {'variant', 'mode', 'steps'},
            'Only variant/mode/steps are accepted; data, resume and shell are forbidden')
    require(job['variant'] == VARIANT and job['mode'] == TRANSPORT['mode']
            and type(job['steps']) is int and job['steps'] == 1, 'Unsupported diagnostic envelope')
    return dict(job)


def source_sha256():
    paths = (Path(__file__).resolve(), Path(probe.__file__).resolve(), Path(probe.oracle.__file__).resolve())
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def validate_paths(output_dir, profile_dir):
    output, profile = Path(output_dir), Path(profile_dir)
    require(output.is_absolute() and profile.is_absolute(), 'Paths must be absolute')
    require(output == output.resolve() and profile == profile.resolve(), 'Symlinks or noncanonical paths rejected')
    require(output.parent.name == 'runs' and output.parent.is_dir(), 'Output must be a fresh pool/runs/job directory')
    require(profile == output.parent.parent/'profiles'/VARIANT, 'Profile path must belong to the same pool/variant')
    require(not output.exists(), 'Output already exists; no overwrite or resume')
    return output, profile


def strict_json(path):
    require(path.is_file() and not path.is_symlink(), f'Missing or symlink evidence: {path.name}')
    raw = path.read_bytes()
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def constant(value):
        raise EvidenceError(f'Nonfinite JSON constant: {value}')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    canonical(value)  # Also catches overflow such as 1e999, nested anywhere.
    return value, hashlib.sha256(raw).hexdigest()


def validate_comparison_metrics(correctness, precision):
    comparisons = correctness.get('comparisons') or {}
    names = {'output', 'q_grad', 'k_grad', 'v_grad', 'bias_parameter_grad'}
    metrics = {'max_abs_error', 'rms_error', 'difference_l2', 'reference_l2', 'allowed_difference_l2',
               'reference_abs_max', 'candidate_abs_max', 'candidate_l1'}
    require(set(comparisons) == names and correctness.get('tolerances') == probe.TOLERANCES[precision]
            and correctness.get('bias_gradient_present') is True, 'Incomplete numerical comparison identity')
    for item in comparisons.values():
        require(type(item) is dict and set(item) == metrics | {'finite', 'passed'}
                and type(item.get('finite')) is bool and type(item.get('passed')) is bool,
                'Incomplete numerical comparison metrics')
        if item['finite']:
            require(all(type(item[name]) in (int, float) and math.isfinite(item[name]) and item[name] >= 0
                        for name in metrics), 'Invalid finite numerical metrics')
        else:
            require(item['passed'] is False and all(item[name] is None for name in metrics),
                    'Nonfinite comparison must fail and retain null metrics')
    require(correctness.get('passed') is all(item['passed'] for item in comparisons.values()),
            'Numerical aggregate disagrees with comparisons')


def validate_completed_case(result, case, sources, input_sha):
    """Stricter negative-result contract supplements the probe's positive oracle."""
    require(canonical(result.get('case')) == canonical(asdict(case)), 'Case types/configuration changed')
    probe.validate_result(result, case, sources)
    if result['status'] not in COMPLETED_STATUSES:
        return False
    child = result.get('child_process') or {}
    code = 0 if result['status'] == 'passed' else 2
    require(type(child.get('returncode')) is int and child['returncode'] == code
            and type(child.get('expected_returncode')) is int and child['expected_returncode'] == code
            and child.get('protocol_valid') is True
            and child.get('cleanup_completed') is True and child.get('reported_status') == result['status'],
            'Missing normal child exit / successful process-group cleanup evidence')
    env = result.get('environment') or {}
    require(env.get('cuda_available') is True and result.get('gpu_execution') in {'attempted', 'completed'}
            and result.get('input_sha256') == input_sha, 'No matching actual CUDA attempt/input evidence')
    require(all(isinstance(env.get(name), str) and env[name] for name in ('torch_version', 'python', 'gpu', 'cuda_build'))
            and type(env.get('total_memory_bytes')) is int and env['total_memory_bytes'] > 0
            and env.get('device_index') == 0 and env.get('activation_dtype') == case.precision
            and env.get('leaf_dtype') == 'float32', 'Incomplete runtime or precision evidence')
    require(type(result.get('elapsed_seconds')) in (int, float)
            and math.isfinite(result['elapsed_seconds']) and result['elapsed_seconds'] >= 0,
            'Invalid case elapsed time')
    if result['status'] == 'passed':
        require(result.get('phase') == 'complete', 'Passing case did not finish')
        validate_comparison_metrics(result['correctness'], case.precision)
        return True
    error = result.get('error') or {}
    require(all(isinstance(error.get(name), str) and error[name] for name in ('type', 'message')),
            'Negative outcome lacks the worker error')
    require(result.get('phase') in {'inputs', 'math_reference', 'candidate_profile_and_gradients', 'warmup_and_timing'},
            'Negative outcome was not produced during a CUDA case')
    if result['status'] == 'numerical_mismatch':
        correctness = result.get('correctness') or {}
        validate_comparison_metrics(correctness, case.precision)
        comparisons = correctness.get('comparisons') or {}
        require(error['type'] == 'NumericalMismatch' and result['phase'] == 'candidate_profile_and_gradients'
                and correctness.get('passed') is False and correctness.get('bias_gradient_present') is True
                and correctness.get('tolerances') == probe.TOLERANCES[case.precision]
                and set(comparisons) == {'output', 'q_grad', 'k_grad', 'v_grad', 'bias_parameter_grad'}
                and all(type(item.get('passed')) is bool and type(item.get('finite')) is bool
                        for item in comparisons.values())
                and any(item['passed'] is False for item in comparisons.values()),
                'Numerical mismatch lacks a complete failing comparison')
        evidence = result.get('backend_evidence') or {}
        probe.observed_backend(evidence.get('operators', []), evidence.get('cuda_kernel_names', []), case.backend)
        require(evidence.get('requested') == evidence.get('observed') == case.backend, 'Mismatch backend identity')
    return True


def verify_probe_evidence(output, returned, source_map, progress):
    """Read-only verification; progress only counts individually verified cases."""
    directory = output/'probe'
    require(directory.is_dir() and not directory.is_symlink(), 'Missing probe evidence directory')
    hashes = progress['evidence_sha256']
    def read(path):
        value, digest = strict_json(path)
        hashes[str(path.relative_to(output))] = digest
        return value
    sources = {str(REPO/name): digest for name, digest in source_map.items()
               if name != str(Path(__file__).resolve().relative_to(REPO))}
    plan = read(directory/'plan.json')
    cases = fixed_cases()
    for name, expected in {'schema_version': probe.SCHEMA, 'source_fingerprints': sources,
            'cases': [asdict(case) for case in cases], 'case_timeout_seconds': CASE_TIMEOUT,
            'total_timeout_seconds': TOTAL_TIMEOUT, 'dry_run': False}.items():
        require(canonical(plan.get(name)) == canonical(expected), f'Invalid probe plan field: {name}')
    summary = read(directory/'summary.json')
    progress['probe_summary_sha256'] = hashes['probe/summary.json']
    require(canonical(summary) == canonical(returned), 'Returned summary differs from disk summary')
    require(summary.get('schema_version') == probe.SCHEMA and summary.get('source_fingerprints') == sources,
            'Invalid summary source/schema identity')
    results = summary.get('results')
    require(type(results) is list and len(results) == len(cases), 'Exactly four results required')
    require([row.get('case_id') for row in results] == [case.case_id for case in cases],
            'Missing, duplicate, reordered or unknown case identity')
    progress['case_statuses'] = {row['case_id']: row.get('status') for row in results}
    input_sha = probe.input_digest(probe.make_inputs(cases[0]))  # Private CPU RNG; no CUDA.
    common_environment = None
    for index, (case, result) in enumerate(zip(cases, results)):
        case_dir = directory/f'{index:02d}_{case.case_id}'
        require(case_dir.is_dir() and not case_dir.is_symlink(), 'Invalid case directory')
        require(canonical(read(case_dir/'spec.json')) == canonical({'case': asdict(case), 'source_fingerprints': sources}),
                'Case spec differs from fixed plan')
        require(canonical(read(case_dir/'result.json')) == canonical(result), 'Case JSON differs from summary')
        if validate_completed_case(result, case, sources, input_sha):
            environment = {key: result['environment'].get(key) for key in
                           ('torch_version', 'cuda_build', 'gpu', 'gpu_uuid', 'capability', 'cudnn', 'total_memory_bytes')}
            require(common_environment is None or canonical(environment) == canonical(common_environment),
                    'Runtime/GPU changed between completed cases')
            common_environment = environment
            log = case_dir/'worker.log'
            require(log.is_file() and not log.is_symlink(), 'Completed case lacks worker log')
            hashes[str(log.relative_to(output))] = hashlib.sha256(log.read_bytes()).hexdigest()
            progress['completed_case_count'] += 1
    counts = dict(Counter(row['status'] for row in results))
    require(canonical(summary.get('status_counts')) == canonical(counts), 'Summary status counts differ')
    expected_status = 'all_passed' if counts == {'passed': 4} else 'completed_with_nonpassing_cases'
    require(summary.get('status') == expected_status, 'Summary status differs from case results')
    require(canonical(summary.get('matched_comparisons')) == canonical(probe.matched_comparisons(results)),
            'Pairwise comparisons differ from raw results')
    require(type(summary.get('elapsed_seconds')) in (int, float)
            and math.isfinite(summary['elapsed_seconds']) and summary['elapsed_seconds'] >= 0, 'Invalid elapsed time')
    return progress['completed_case_count'] == 4


def run_diagnostic(job, *, output_dir, profile_dir):
    validate_envelope(job)
    output, _ = validate_paths(output_dir, profile_dir)
    before = source_sha256()
    output.mkdir(exist_ok=False)
    progress = dict(case_count=4, completed_case_count=0, case_statuses={}, probe_summary_sha256=None,
                    evidence_sha256={})
    execution_completed, error, interrupted = False, None, None
    probe.write_json(output/'execution_request.json', dict(schema_version=SCHEMA, kind=KIND, envelope=job,
        source_sha256=before, case_timeout_seconds=CASE_TIMEOUT, total_timeout_seconds=TOTAL_TIMEOUT,
        cases=[asdict(case) for case in fixed_cases()]))
    progress['evidence_sha256']['execution_request.json'] = hashlib.sha256((output/'execution_request.json').read_bytes()).hexdigest()
    try:
        returned = probe.execute_plan(fixed_cases(), output/'probe', dry_run=False,
                                      case_timeout=CASE_TIMEOUT, total_timeout=TOTAL_TIMEOUT)
        execution_completed = verify_probe_evidence(output, returned, before, progress)
        if not execution_completed:
            error = dict(type='IncompleteDiagnostic', message='One or more cases did not complete a valid diagnostic attempt')
    except (probe.ProbeInterrupted, KeyboardInterrupt) as caught:
        interrupted = caught if isinstance(caught, probe.ProbeInterrupted) else probe.ProbeInterrupted(signal.SIGINT)
        error = dict(type=type(caught).__name__, message=str(caught))
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    try:
        after = source_sha256()
    except Exception as caught:
        after = {}
        error = dict(type='SourceReadError', message=f'{type(caught).__name__}: {caught}')
    if after and after != before:
        error = dict(type='SourceChanged', message='Adapter/probe/oracle source changed during diagnostic')
        execution_completed = False
    for relative, expected in progress['evidence_sha256'].items():
        path = output/relative
        try:
            unchanged = path.is_file() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
        except OSError:
            unchanged = False
        if not unchanged:
            error = dict(type='EvidenceChanged', message=f'Evidence changed after verification: {relative}')
            execution_completed = False
    if error is not None:
        execution_completed = False
    summary = dict(schema_version=SCHEMA, kind=KIND, variant=VARIANT, transport=TRANSPORT,
        training_started=False, biological_quality_claim=False, **progress,
        execution_completed=execution_completed,
        all_cases_passed=execution_completed and set(progress['case_statuses'].values()) == {'passed'},
        source_sha256=before, source_sha256_before=before, source_sha256_after=after, error=error,
        created_at=datetime.now(timezone.utc).isoformat())
    probe.write_json(output/'summary.json', summary)
    marker = dict(summary, status='completed' if execution_completed else 'failed',
                  summary_sha256=hashlib.sha256((output/'summary.json').read_bytes()).hexdigest())
    probe.write_json(output/('COMPLETE.json' if execution_completed else 'FAILED.json'), marker)
    if interrupted is not None:
        raise interrupted
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--variant', choices=[VARIANT], required=True)
    result.add_argument('--mode', choices=[TRANSPORT['mode']], required=True)
    result.add_argument('--steps', type=int, choices=[1], required=True)
    result.add_argument('--profile-dir', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        summary = run_diagnostic(dict(variant=args.variant, mode=args.mode, steps=args.steps),
                                 output_dir=args.output_dir, profile_dir=args.profile_dir)
        return 0 if summary['execution_completed'] else 2
    except probe.ProbeInterrupted as error:
        print(str(error), file=sys.stderr)
        return 128+error.signum
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
