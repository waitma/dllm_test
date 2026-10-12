"""Enqueue reviewed architecture plans inside the existing eight-GPU allocation.

After activating pllm: python scripts/enqueue_immune_architecture.py --stage profile_synthetic100 --apply
Without --apply, only validate and print. This script never requests GPU resources.
Real pilot stages require both clean-data certificates and a completed diagnostic.
"""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from scripts.immune_research_pool import enqueue, validate


MASKS = ('mask25', 'mask50', 'mask75', 'mask100')


def read_json(path):
    """Reject nested NaN/Infinity, including overflowing JSON exponents."""
    def finite(value):
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'Nonfinite diagnostic evidence: {path}')
    try:
        value = json.loads(path.read_text())
        finite(value)
        if not isinstance(value, dict):
            raise ValueError('Expected a JSON object')
        return value
    except (OSError, ValueError) as exc:
        raise ValueError(f'Invalid evidence {path}: {exc}') from exc


def current_fingerprints():
    # Pure file hashes; do not call verify_profile(), which checks a local GPU.
    from scripts.run_immune_architecture import fingerprints
    return fingerprints()


def production_config():
    from dataclasses import asdict
    from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
    return asdict(ImmuneConfig(ablation='A4'))


def assert_data_ready(directory):
    from examples.llada.immune_pretrain_v3 import assert_ready
    assert_ready(directory)


def mask_metrics(point):
    levels = point['mask_levels']
    if (point.get('mask_protocol') != 'fixed_per_record_selected_tokens_v1'
            or set(levels) != set(MASKS)):
        raise ValueError('All four fixed-mask evaluations are required')
    losses, accuracy = [], []
    for index, name in enumerate(MASKS, 1):
        level = levels[name]
        selected, residues, terminal = (level[key] for key in ('selected_count', 'aa_count', 'terminal_count'))
        if (level['mask_ratio'] != index/4 or any(type(n) is not int for n in (selected, residues, terminal))
                or residues <= 0 or terminal < 0 or selected != residues+terminal):
            raise ValueError('Invalid fixed-mask population or noise ratio')
        loss, aa = level['loss'], level['aa_accuracy']
        if (type(loss) not in (int, float) or loss < 0 or
                type(aa) not in (int, float) or not 0 <= aa <= 1):
            raise ValueError('Invalid fixed-mask loss or AA accuracy')
        losses.append(loss)
        accuracy.append(aa)
    mean = sum(losses)/len(losses)
    reported = point['mask_mean']['loss']
    if type(reported) not in (int, float) or not math.isclose(mean, reported, rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError('Four-mask mean loss is inconsistent with its components')
    return mean, accuracy


def verify_diagnostic(pool, variant):
    """Require an improved 100-step engineering check; not biological validation."""
    profile = pool/'profiles'/variant
    proof = read_json(profile/'COMPLETE.json')
    measured = read_json(profile/'summary.json')
    if proof.get('status') != 'passed' or proof.get('variant') != variant:
        raise ValueError('No passing profile for this variant')
    if proof.get('summary_sha256') != hashlib.sha256((profile/'summary.json').read_bytes()).hexdigest():
        raise ValueError('Profile summary SHA does not match its certificate')
    sources = proof.get('source_sha256')
    if not sources or sources != current_fingerprints():
        raise ValueError('Profile source hashes differ from the current complete fingerprint map')
    batch = measured.get('recommended_micro_batch')
    passed = [item for item in measured.get('measurements', [])
              if item.get('status') == 'passed' and item.get('batch_size') == batch]
    if (measured.get('variant') != variant or measured.get('config') != production_config()
            or measured.get('global_batch') != 256 or type(batch) is not int
            or batch < 1 or 256 % batch or len(passed) != 1):
        raise ValueError('Invalid production profile configuration or recommended microbatch')
    candidate = passed[0]
    used, total = candidate.get('max_allocated_gib'), candidate.get('total_gib')
    if (candidate.get('config') != measured['config']
            or candidate.get('precision') != 'fp32_parameters_adam_bf16_autocast'
            or type(used) not in (int, float) or type(total) not in (int, float)
            or used < 0 or total <= 0 or used/total > .85):
        raise ValueError('Profile recommendation lacks valid precision/memory evidence')
    errors = []
    for path in sorted((pool/'results').glob('*.json')):
        result = read_json(path)
        spec = result.get('job', {})
        if not (spec.get('variant') == variant and spec.get('mode') == 'profile-synthetic'
                and spec.get('steps') == 100 and result.get('state') == 'succeeded'):
            continue
        try:
            if result.get('job_id') != path.stem or result.get('returncode') != 0:
                raise ValueError('Result identity or exit status is inconsistent')
            run = pool/'runs'/path.stem
            complete, report = read_json(run/'COMPLETE.json'), read_json(run/'summary.json')
            if (complete.get('status') != 'completed' or report.get('status') != 'completed'
                    or report.get('variant') != variant or report.get('steps') != 100
                    or report.get('kind') != 'synthetic_functional_diagnostic'
                    or report.get('biological_quality_claim') is not False):
                raise ValueError('No completed same-variant 100-step diagnostic')
            if report.get('source_sha256') != sources:
                raise ValueError('Diagnostic source hashes differ from the profile')
            if (report.get('config') != measured['config']
                    or report.get('micro_batch') != measured['recommended_micro_batch']
                    or report.get('global_batch') != 256
                    or report['micro_batch']*report['grad_accum'] != 256):
                raise ValueError('Diagnostic does not use the profiled production configuration/batch')
            trace = report['trace']
            if [point.get('step') for point in trace] != list(range(5, 101, 5)):
                raise ValueError('Diagnostic trace must reach step 100 at each five-step checkpoint')
            initial_loss, initial_aa = mask_metrics(report['initial'])
            for point in [*trace, report['final']]:
                mask_metrics(point)
                if any(point['mask_levels'][name][key] != report['initial']['mask_levels'][name][key]
                       for name in MASKS for key in ('selected_count', 'aa_count', 'terminal_count')):
                    raise ValueError('Fixed-mask population changed during diagnostic')
            final_loss, final_aa = mask_metrics(report['final'])
            if any(report['final'][key] != trace[-1][key] for key in ('mask_levels', 'mask_mean')):
                raise ValueError('Final metrics disagree with the step-100 trace')
            if final_loss >= initial_loss or not any(b > a for a, b in zip(initial_aa, final_aa)):
                raise ValueError('Diagnostic requires lower four-mask mean loss and AA improvement in at least one mask')
            return path.stem
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f'{path.stem}: {exc}')
    detail = '; '.join(errors) or 'no succeeded same-variant 100-step result'
    raise ValueError(f'No valid completed 100-step engineering diagnostic for {variant}: {detail}')


def enqueue_jobs(pool, jobs, apply=False):
    """Validate the whole stage, then publish once per exact job specification."""
    for job in jobs:
        validate(job)
        if job['mode'] == 'train':
            assert_data_ready(Path(job['data_dir']))
            verify_diagnostic(pool, job['variant'])
    # Serialize helper invocations without changing the pool worker protocol.
    lock = (pool/'architecture_enqueue.lock').open('a') if apply else nullcontext()
    with lock as handle:
        if apply:
            fcntl.flock(handle, fcntl.LOCK_EX)
        existing = []
        for folder in ('pending', 'running', 'finished'):
            for path in (pool/folder).glob('*.json'):
                try:
                    existing.append(json.loads(path.read_text()))
                except FileNotFoundError:
                    pass  # A worker moved it into the next lifecycle folder.
        decisions, seen = [], []
        for job in jobs:
            if job in existing:
                decisions.append(dict(job=job, decision='already_registered'))
            elif job in seen:
                decisions.append(dict(job=job, decision='duplicate_in_plan'))
            elif apply:
                decisions.append(dict(job=job, decision='enqueued', job_id=enqueue(pool, job)))
                existing.append(job)
            else:
                decisions.append(dict(job=job, decision='validated_not_enqueued'))
            seen.append(job)
        return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'train_jobs/immune_architecture_v2/experiments.yml')
    parser.add_argument('--stage', choices=['profile_synthetic100', 'clean_data1000', 'selected5000'], required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    plan = yaml.safe_load(args.plan.read_text())
    if plan['schema'] != 'immune_architecture_plan.v1':
        raise ValueError('Unknown plan format')
    pool = Path(plan['pool_dir'])
    jobs = plan['stages'][args.stage]['jobs']
    if not jobs:
        raise ValueError('No selected methods; record the validation decision before filling this stage')
    print(json.dumps(enqueue_jobs(pool, jobs, args.apply), indent=2))


if __name__ == '__main__':
    main()
