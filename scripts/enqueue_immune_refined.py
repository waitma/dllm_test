"""Validate and enqueue refined experiments in the existing eight-GPU pool.

After activating pllm: python scripts/enqueue_immune_refined.py --stage profile_diagnostic25 --apply
Omit --apply for a read-only preview. No platform resources are submitted.
Artificial 500-step diagnostics and certified real-data pilots are distinct stages.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from scripts.enqueue_immune_architecture import read_json, mask_metrics, assert_data_ready
from scripts.immune_research_pool import enqueue, validate
from scripts.run_immune_refined import fingerprints, build_fixtures
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.immune_llada.research_refined.model import VARIANTS


def verify_profile_evidence(pool, variant):
    if variant not in VARIANTS:
        raise ValueError('Not a refined experiment')
    directory = pool/'profiles'/variant
    proof = read_json(directory/'COMPLETE.json')
    summary = read_json(directory/'summary.json')
    if (proof.get('status') != 'passed' or proof.get('variant') != variant
            or proof.get('source_sha256') != fingerprints()
            or proof.get('summary_sha256') != hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest()
            or summary.get('variant') != variant or summary.get('config') != asdict(ImmuneConfig())
            or summary.get('global_batch') != 256):
        raise ValueError('Missing, stale or incompatible refined profile')
    batch = summary.get('recommended_micro_batch')
    if type(batch) is not int or batch < 1 or 256 % batch:
        raise ValueError('Invalid profiled batch')
    matches = [item for item in summary.get('measurements', []) if item.get('status') == 'passed'
               and item.get('batch_size') == batch]
    if len(matches) != 1:
        raise ValueError('No unique measured batch')
    measured = matches[0]
    used, total = measured.get('max_allocated_gib'), measured.get('total_gib')
    if (measured.get('config') != summary['config'] or measured.get('gpu') != summary.get('gpu')
            or measured.get('precision') != 'fp32_parameters_adam_bf16_autocast'
            or type(used) not in (int, float) or type(total) not in (int, float)
            or not 0 < used <= .85*total):
        raise ValueError('Measured precision, configuration or memory is incompatible')
    return summary


def verify_evaluation(point, fixture, initial=None):
    """Bind learning evidence to the complete heldout strata and stable mask counts."""
    expected = fixture['splits']['valid']
    strata = point['strata']
    keys = {key for key, count in expected['group_length_counts'].items() if count}
    if type(point.get('records')) is not int or point['records'] != expected['records'] or set(strata) != keys:
        raise ValueError('Heldout record count or evaluation strata are incomplete')
    metrics = {'loss': 'selected_count', 'aa_loss': 'aa_count', 'aa_accuracy': 'aa_count',
               'terminal_loss': 'terminal_count', 'terminal_accuracy': 'terminal_count'}
    for name, subset in [('all', point), *sorted(strata.items())]:
        mask_metrics(subset)
        for mask, level in subset['mask_levels'].items():
            for metric, count in metrics.items():
                value = level[metric]
                if (level[count] == 0 and value is not None or level[count] > 0 and
                    (type(value) not in (int, float) or value < 0 or 'accuracy' in metric and value > 1)):
                    raise ValueError('Invalid heldout metric or empty-subset convention')
            if initial is not None:
                reference = initial if name == 'all' else initial['strata'][name]
                if any(level[count] != reference['mask_levels'][mask][count]
                       for count in ('selected_count', 'aa_count', 'terminal_count')):
                    raise ValueError('Heldout fixed-mask population changed')
        for metric in metrics:
            values = [level[metric] for level in subset['mask_levels'].values() if level[metric] is not None]
            mean = subset['mask_mean'][metric]
            if ((not values and mean is not None) or
                (values and (type(mean) not in (int, float) or not math.isclose(mean, sum(values)/len(values), rel_tol=1e-6, abs_tol=1e-7)))):
                raise ValueError('Heldout mask mean is inconsistent')
    for mask, level in point['mask_levels'].items():
        parts = [subset['mask_levels'][mask] for subset in strata.values()]
        for metric, count in metrics.items():
            if sum(part[count] for part in parts) != level[count]:
                raise ValueError('Heldout aggregate population differs from strata')
            mean = sum(part[metric]*part[count] for part in parts if part[count])/level[count] if level[count] else None
            if mean is not None and not math.isclose(mean, level[metric], rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError('Heldout aggregate metrics differ from strata')
    return mask_metrics(point)


def verify_diagnostic(pool, variant, steps, require_learning=False):
    if steps not in (25, 500):
        raise ValueError('Only 25-step profiling or 500-step artificial proofs are supported')
    profile = verify_profile_evidence(pool, variant)
    expected_fixture = build_fixtures(ImmuneConfig()).manifest
    errors = []
    for path in sorted((pool/'results').glob('*.json'), reverse=True):
        result = read_json(path)
        job = result.get('job', {})
        if job.get('variant') != variant or job.get('steps') != steps:
            continue
        if result.get('state') != 'succeeded' or result.get('returncode') != 0:
            continue
        try:
            if result.get('job_id') != path.stem or job.get('mode') != ('profile-synthetic' if steps == 25 else 'train'):
                raise ValueError('Diagnostic job identity or mode mismatch')
            if 'resume_from_checkpoint' in job:
                raise ValueError('Artificial diagnostic must start from scratch')
            if steps == 500:
                from scripts.architecture_prototypes.expanded_fixtures import load_verified_fixture_manifest
                directory = Path(job['data_dir'])
                if not directory.is_absolute() or load_verified_fixture_manifest(directory).manifest != expected_fixture:
                    raise ValueError('500-step job lacks the verified artificial data directory')
            elif 'data_dir' in job:
                raise ValueError('25-step profiling diagnostic must use built-in artificial fixtures')
            run = pool/'runs'/path.stem
            proof, summary = read_json(run/'COMPLETE.json'), read_json(run/'summary.json')
            if (proof.get('status') != 'completed' or proof.get('global_step') != steps
                or proof.get('kind') != 'synthetic_generalization_diagnostic'
                or proof.get('summary_sha256') != hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()
                or proof.get('biological_quality_claim') is not False
                or summary.get('status') != 'completed' or summary.get('variant') != variant
                or summary.get('kind') != 'synthetic_generalization_diagnostic'
                or summary.get('biological_quality_claim') is not False
                or summary.get('dataset_kind') != 'deterministic_artificial_sequences'
                or summary.get('validation_uses_test') is not False
                or summary.get('source_sha256') != fingerprints()
                or summary.get('config') != profile['config'] or summary.get('steps') != steps
                or json.dumps(summary.get('fixture_manifest'), sort_keys=True) != json.dumps(expected_fixture, sort_keys=True)
                or summary.get('global_batch') != 256
                or summary.get('micro_batch') != profile['recommended_micro_batch']
                or summary.get('grad_accum')*summary.get('micro_batch') != 256):
                raise ValueError('Diagnostic proof or protocol mismatch')
            trace = summary.get('trace', [])
            if [point.get('step') for point in trace] != list(range(25, steps+1, 25)):
                raise ValueError('Diagnostic trace is incomplete')
            initial_loss, initial_aa = verify_evaluation(summary['initial'], expected_fixture)
            final_loss, final_aa = verify_evaluation(summary['final'], expected_fixture, summary['initial'])
            for point in trace:
                verify_evaluation(point, expected_fixture, summary['initial'])
                if type(point.get('gradient_norm')) not in (int, float) or point['gradient_norm'] < 0:
                    raise ValueError('Missing finite gradient evidence')
            if any(summary['final'][key] != trace[-1][key] for key in ('mask_levels', 'mask_mean', 'strata', 'records')):
                raise ValueError('Final metrics differ from the last evaluation')
            if require_learning and (final_loss >= initial_loss or not any(a > b for a, b in zip(final_aa, initial_aa))):
                raise ValueError('Heldout artificial diagnostic has not shown learning')
            return summary
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f'{path.stem}: {exc}')
    raise ValueError(f'No completed {steps}-step refined diagnostic for {variant}: '+('; '.join(errors) or 'no matching result'))


def validate_job(pool, job, stage):
    validate(job)
    if job['variant'] not in VARIANTS:
        raise ValueError('Not a refined experiment')
    if stage == 'profile_diagnostic25':
        if job['mode'] != 'profile-synthetic' or job['steps'] != 25:
            raise ValueError('The first stage is a profiled 25-step artificial diagnostic')
    elif stage == 'synthetic500':
        from scripts.architecture_prototypes.expanded_fixtures import load_verified_fixture_manifest
        if job['mode'] != 'train' or job['steps'] != 500 or 'resume_from_checkpoint' in job:
            raise ValueError('Expected a fresh 500-step artificial diagnostic')
        fixtures = load_verified_fixture_manifest(Path(job['data_dir']))
        if fixtures.manifest != build_fixtures(ImmuneConfig()).manifest:
            raise ValueError('Artificial fixture protocol differs')
        verify_diagnostic(pool, job['variant'], 25)
    elif stage == 'real1000':
        if job['mode'] != 'train' or job['steps'] != 1000 or 'resume_from_checkpoint' in job:
            raise ValueError('Expected an initial 1000-step real pilot')
        directory = Path(job['data_dir'])
        if (directory/'synthetic_fixture_manifest.json').exists():
            raise ValueError('Artificial data cannot authorize a real-data stage')
        assert_data_ready(directory)
        verify_diagnostic(pool, job['variant'], 500, require_learning=True)
    else:
        raise ValueError('Unknown stage')


def enqueue_jobs(pool, jobs, stage, apply=False):
    if not jobs:
        raise ValueError('Empty experiment stage')
    lock = (pool/'refined_enqueue.lock').open('a') if apply else nullcontext()
    with lock as handle:
        if apply:
            fcntl.flock(handle, fcntl.LOCK_EX)
        for job in jobs:
            validate_job(pool, job, stage)
        existing = []
        for folder in ('pending', 'running', 'finished'):
            for path in (pool/folder).glob('*.json'):
                try:
                    existing.append(read_json(path))
                except ValueError as exc:
                    if not isinstance(exc.__cause__, FileNotFoundError):
                        raise
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
    parser.add_argument('--plan', type=Path, default=ROOT/'train_jobs/immune_architecture_refined/experiments.yml')
    parser.add_argument('--stage', choices=['profile_diagnostic25', 'synthetic500', 'real1000'], required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    plan = yaml.safe_load(args.plan.read_text())
    if plan.get('schema') != 'immune_refined_plan.v1' or plan.get('global_batch') != 256 or plan.get('request_new_resources') is not False:
        raise ValueError('Incompatible plan')
    print(json.dumps(enqueue_jobs(Path(plan['pool_dir']), plan['stages'][args.stage]['jobs'], args.stage, args.apply), indent=2))


if __name__ == '__main__':
    main()
