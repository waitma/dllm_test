"""Activate pllm; python -B FILE --out ABS [--check-code].

Deliver all fixed CDR ablation results after both checkpoints and platform jobs
finish. No policy is selected. The protocol hash is pinned before inference.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path


# Captured from the frozen protocol before evaluation begins.
FROZEN_PLAN_SHA256 = '60b4ea16fe684c03098b5bd8c283035dfd3835a4cc915dfbfb0a7197698cb22e'
CHECKPOINTS = ('112000', '50000')
MODES = ('cdrh1', 'cdrh2', 'cdrh3')
CONTROLS = ('single_pass', 'original4', 'original8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, expected=None):
    value = sha(path)
    if expected is not None:
        assert value == expected, ('changed file', str(path), value, expected)
    return value


def direction(interval):
    assert len(interval) == 2 and all(math.isfinite(x) for x in interval) and interval[0] <= interval[1]
    return 'benefit' if interval[0] > 0 else 'harm' if interval[1] < 0 else 'inconclusive'


def validate_job_records(out):
    paths = sorted(out.glob('round*_platform.json'))
    assert {path.name for path in paths} == {'round1_platform.json', 'round2_platform.json'}, 'Both platform records are required before final delivery'
    jobs, hashes = [], {}
    for path in paths:
        record = read(path)
        submission_path = path.with_name(path.name.replace('_platform', '_submission'))
        submission = read(submission_path)
        assert record['JobId'] == submission['task_id']
        expected_checkpoint = '112000' if path.name == 'round1_platform.json' else '50000'
        assert submission['checkpoint'] == expected_checkpoint
        assert submission['plan_sha256'] == FROZEN_PLAN_SHA256
        assert record['Status'] == 'Success' and record['End']
        config = Path(submission['config'])
        jobs.append(dict(job_id=record['JobId'], checkpoint=expected_checkpoint,
            status=record['Status'], start=record['Start'], end=record['End'],
            yaml=str(config), yaml_sha256=checked(config, submission['config_sha256'])))
        hashes[str(path)] = checked(path)
        hashes[str(submission_path)] = checked(submission_path)
    assert len({job['job_id'] for job in jobs}) == len(jobs)
    return jobs, hashes


def validate_audit(out, checkpoint, plan, expected_policies):
    folder = out / ('evaluation_' + checkpoint)
    path = folder / 'independent_audit.json'
    report = read(path)
    assert report['status'] == 'passed' and report['checkpoint'] == checkpoint
    assert report['plan_sha256'] == FROZEN_PLAN_SHA256 and report['winner_selected'] is False
    assert report['n'] == 5254 and report['policies'] == 13 and report['predictions'] == 5254 * 13
    assert report['physical_sample_forwards'] == 5254 * 31 and report['query_masks_audited'] == 5254 * 18
    for flag in ('source_reference_recovered', 'probabilities_reconstructed',
                 'actual_feedback_weights_reconstructed', 'sequential_bank_reconstructed'):
        assert report[flag] is True
    assert report['bootstrap']['repeats'] == 50000 and report['bootstrap']['seed'] == 20261016
    assert report['bootstrap']['primary_family_tests'] == 6
    assert report['bootstrap']['primary_adjusted_confidence'] == 1 - .05 / 6
    auditor = out / 'source/audit_cdr_block_ablation.py'
    assert report['auditor_sha256'] == plan['source_sha256'][str(auditor)]
    hashes = {str(path): checked(path)}
    for filename, field in (('predictions.jsonl', 'predictions_sha256'), ('query_audit.jsonl', 'query_audit_sha256'),
                            ('summary.json', 'summary_sha256'), ('identity.json', 'identity_sha256')):
        artifact = folder / filename
        hashes[str(artifact)] = checked(artifact, report[field])
    assert set(report['posterior_sha256']) == {str(p) for p in folder.glob('batch_*.npz')}
    for artifact, digest in report['posterior_sha256'].items():
        hashes[artifact] = checked(artifact, digest)
    for artifact, digest in report['source_data_sha256'].items():
        checked(artifact, digest)
    checkpoint_path = Path(plan['checkpoints'][checkpoint])
    checked(checkpoint_path / 'vocabulary.json', report['vocabulary_sha256'])
    identity = read(folder / 'identity.json')
    assert identity['model']['checkpoint'] == str(checkpoint_path)
    assert identity['plan_sha256'] == FROZEN_PLAN_SHA256 and identity['phase'] == 'ablation'
    summary = read(folder / 'summary.json')
    assert summary['status'] == 'passed' and summary['plan_sha256'] == FROZEN_PLAN_SHA256
    assert summary['n'] == report['n'] and summary['gate_cost_separate']
    assert summary['physical_sample_forwards'] == report['physical_sample_forwards']
    assert set(summary['mean_sequence_aar']) == expected_policies
    assert set(report['results']) == {'oas_new:' + mode for mode in MODES}
    for mode in MODES:
        key = 'oas_new:' + mode
        result = report['results'][key]
        assert result['n'] == plan['counts_by_mode'][mode]
        assert result['groups'] == (4096 if mode == 'cdrh3' else 512)
        assert set(result['mean_sequence_aar']) == set(result['logical_mean_nfe']) == expected_policies
        for name, score in result['mean_sequence_aar'].items():
            assert math.isfinite(score) and 0 <= score <= 100
            assert abs(score - summary['mean_sequence_aar'][name][key]) < 1e-8
            assert result['logical_mean_nfe'][name] == (1 if name == 'single_pass' else 8 if name == 'original8' else 4)
        for name, contrast in result['contrasts'].items():
            estimate = sum(weight * result['mean_sequence_aar'][policy] for policy, weight in contrast['coefficients'].items())
            assert abs(estimate - contrast['delta_pp']) < 1e-8
            direction(contrast['ci95_pp'])
            if contrast['role'] == 'primary':
                assert mode == 'cdrh3'
                assert contrast['bonferroni_family_tests'] == 6
                assert contrast['simultaneous_interval_confidence'] == 1 - .05 / 6
                assert contrast['adjusted_interval_supports_positive_contribution'] == (direction(contrast['ci_bonferroni_pp']) == 'benefit')
            else:
                assert 'ci_bonferroni_pp' not in contrast
    gate_path = out / f'gate_cuda_{checkpoint}.json'
    gate = read(gate_path)
    assert gate['status'] == 'passed' and gate['plan_sha256'] == FROZEN_PLAN_SHA256
    assert gate['identity'] == identity['model'] and gate['checkpoint'] == checkpoint
    for flag in ('no_reference_in_features', 'original_single_pass_exact', 'full_method_matches_frozen_parent_exact',
                 'zero_strength_all_factor_cells_exact', 'sequential_feedback_bank_reconstructed', 'hook_cleanup_exact'):
        assert gate[flag] is True
    hashes[str(gate_path)] = checked(gate_path)
    return report, hashes


def summarize(out):
    assert isinstance(FROZEN_PLAN_SHA256, str) and len(FROZEN_PLAN_SHA256) == 64, 'Pin the protocol before running delivery'
    checked(out / 'plan.json', FROZEN_PLAN_SHA256)
    plan = read(out / 'plan.json')
    assert set(plan['checkpoints']) == set(CHECKPOINTS)
    for field in ('source_sha256', 'external_source_sha256'):
        for path, digest in plan[field].items():
            checked(path, digest)
    evidence = {str(out / 'plan.json'): FROZEN_PLAN_SHA256}
    for filename, field in (('rows.json', 'rows_sha256'), ('data_plan.json', 'data_plan_sha256'),
                            ('data_independent_audit.json', 'data_audit_sha256')):
        evidence[str(out / filename)] = checked(out / filename, plan[field])
    data = read(out / 'data_independent_audit.json')
    assert data['status'] == 'passed' and data['rows'] == 5254
    assert data['counts_by_mode'] == plan['counts_by_mode']
    expected = set(CONTROLS) | {family['name'] + '_prior' + str(weight)
        for family in plan['families'] for weight in family['prior_weights']}
    assert len(expected) == 13
    assert not (out / 'selection.json').exists(), 'This ablation must not select a winner'
    jobs, hashes = validate_job_records(out)
    evidence.update(hashes)
    audits, table, primary, secondary, factorial, baselines = {}, [], [], [], [], []
    for checkpoint in CHECKPOINTS:
        audit, hashes = validate_audit(out, checkpoint, plan, expected)
        audits[checkpoint] = audit
        evidence.update(hashes)
        for mode in MODES:
            result = audit['results']['oas_new:' + mode]
            table.append(dict(checkpoint=checkpoint, dataset='oas_new', mode=mode, n=result['n'], groups=result['groups'],
                aar_percent=result['mean_sequence_aar'], logical_nfe=result['logical_mean_nfe']))
            for name, contrast in result['contrasts'].items():
                row = dict(checkpoint=checkpoint, mode=mode, contrast=name, **contrast)
                if contrast['role'] == 'primary':
                    row['adjusted_direction'] = direction(contrast['ci_bonferroni_pp'])
                    primary.append(row)
                elif contrast['role'] == 'secondary_descriptive':
                    secondary.append(row)
                elif contrast['role'] == 'descriptive_factorial':
                    factorial.append(row)
                elif contrast['role'] == 'descriptive_baseline' and name.startswith(plan['full_policy'] + '_vs_'):
                    baselines.append(row)
    assert len(table) == 6 and len(primary) == 6
    primary_by_name = {}
    for contrast in plan['primary_contrasts']:
        records = [row for row in primary if row['contrast'] == contrast['name']]
        assert {row['checkpoint'] for row in records} == set(CHECKPOINTS)
        directions = {row['checkpoint']: row['adjusted_direction'] for row in records}
        claim = 'benefit' if all(value == 'benefit' for value in directions.values()) else 'harm' if all(value == 'harm' for value in directions.values()) else 'no_consistent_adjusted_direction'
        primary_by_name[contrast['name']] = dict(direction_by_checkpoint=directions, cross_checkpoint_claim=claim,
            full_policy=contrast['positive'], ablated_policy=contrast['negative'])
    output = dict(status='complete', outcome='all_prespecified_ablations_reported', winner_selected=False,
        full_policy=plan['full_policy'], jobs=jobs, table=table, primary_h3=primary,
        component_conclusions=primary_by_name, secondary_descriptive=secondary,
        factorial_descriptive=factorial, full_vs_controls_descriptive=baselines,
        counts=dict(rows_per_checkpoint=5254, policies=13, counts_by_mode=plan['counts_by_mode'],
            total_predictions=sum(audits[c]['predictions'] for c in CHECKPOINTS),
            total_sample_forwards=sum(audits[c]['physical_sample_forwards'] for c in CHECKPOINTS),
            gate_cost_separate=True), bootstrap=plan['bootstrap'], inferential_rule=plan['inferential_rule'],
        multiplicity_interpretation=dict(primary_family_size=6, nominal_family_coverage_at_least=.95,
            per_comparison_adjusted_interval_confidence=1-.05/6,
            note='Each primary interval uses 99.1666667% nominal coverage. Bonferroni targets at least 95% simultaneous family coverage across six comparisons, not 99.1666667% family coverage; percentile-bootstrap calibration is approximate.'),
        secondary_rule=plan['secondary_rule'], sequential_definition=plan['sequential_definition'],
        hard_definition=plan['hard_definition'], evidence_sha256=evidence, script_sha256=sha(Path(__file__)),
        limitations=[
            'A component conclusion requires the same Bonferroni-adjusted direction on both checkpoints. Intervals crossing zero do not establish equivalence.',
            'The six primary H3 component comparisons form the correction family. Secondary settings, factorial effects, control comparisons, and H1/H2 are descriptive.',
            'Every candidate has four logical forwards; original4 matches NFE while single pass uses one and original8 uses eight.',
            'All policies were fixed before scoring; this report does not nominate a best policy.',
            'Sequential feedback is one sweep in fixed group order, with no final output anchor in its feedback bank; conclusions do not apply to every iterative sampler.',
            'Actual saved feedback weights and sequential probability banks were reconstructed; embedding mixture integrity also depends on implementation gates.',
            'OAS supplementary data are separate from the official benchmark, with inherited exact/cluster exclusions rather than universal pretraining decontamination.',
        ])
    path = out / 'ablation_delivery.json'
    if path.exists():
        assert read(path) == output, 'Existing delivery differs; refusing to overwrite'
    else:
        with path.open('x') as stream:
            stream.write(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: output[key] for key in ('status', 'counts', 'primary_h3', 'component_conclusions')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-code', action='store_true')
    args = parser.parse_args()
    if args.check_code:
        ast.parse(Path(__file__).read_text())
        print('PASS: syntax only; no result files read or written')
        return
    assert args.out is not None, '--out required'
    summarize(args.out)


if __name__ == '__main__':
    main()
