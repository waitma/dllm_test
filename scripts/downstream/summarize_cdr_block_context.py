"""Activate pllm; python -B FILE --out ABS [--check-code].

Summarize the complete, independently audited two-checkpoint OAS confirmation.
The selection and protocol hashes below were captured before confirmation results.
This delivery script is separate from the already frozen inference source bundle.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


FROZEN_PLAN_SHA256 = 'fd42371bbd9ffa9746f33a74eb534b3117bd20f940272f8f5fed3c7974974200'
FROZEN_SELECTION_SHA256 = 'd8fc8002d082f51a59aa3f36e35b2faea91e37a91da668ca53537460dfbe05d8'
CHECKPOINTS = ('112000', '50000')
MODES = ('cdrh1', 'cdrh2', 'cdrh3')
BASELINES = ('single_pass', 'original4', 'original8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, expected=None):
    path = Path(path)
    digest = sha(path)
    if expected is not None:
        assert digest == expected, ('changed artifact', str(path), digest, expected)
    return digest


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def validate_platform(out):
    jobs, hashes = [], {}
    for number in (1, 2):
        platform_path = out / f'round{number}_platform.json'
        submission_path = out / f'round{number}_submission.json'
        platform, submission = read(platform_path), read(submission_path)
        assert platform['JobId'] == submission['task_id']
        assert platform['Status'] == 'Success', ('incomplete platform job', platform)
        assert platform['End'] and platform['Start']
        assert timestamp(platform['End']) >= timestamp(platform['Start'])
        config = Path(submission['config'])
        jobs.append(dict(phase='explore' if number == 1 else 'confirm',
            job_id=platform['JobId'], status=platform['Status'], start=platform['Start'], end=platform['End'],
            yaml=str(config), yaml_sha256=checked(config)))
        hashes[str(platform_path)] = checked(platform_path)
        hashes[str(submission_path)] = checked(submission_path)
    assert jobs[0]['job_id'] != jobs[1]['job_id']
    return jobs, hashes


def validate_confirmation(out, checkpoint, plan, selection, counts):
    folder = out / f'confirmation_{checkpoint}'
    audit_path = folder / 'independent_audit.json'
    audit = read(audit_path)
    assert audit['status'] == 'passed' and audit['phase'] == 'confirm'
    assert audit['checkpoint'] == checkpoint and audit['plan_sha256'] == FROZEN_PLAN_SHA256
    assert audit['n'] == sum(counts.values()) and audit['policies'] == 4
    assert audit['predictions'] == audit['n'] * 4
    assert audit['source_reference_recovered'] and audit['probabilities_reconstructed']
    assert audit['bootstrap']['repeats'] == plan['bootstrap']['repeats']
    assert audit['bootstrap']['seed'] == plan['bootstrap']['seed']
    auditor = out / 'source/audit_cdr_block_context.py'
    assert audit['auditor_sha256'] == plan['source_sha256'][str(auditor)]
    hashes = {str(audit_path): checked(audit_path)}
    for filename, field in (('predictions.jsonl', 'predictions_sha256'),
                            ('query_audit.jsonl', 'query_audit_sha256'),
                            ('summary.json', 'summary_sha256'),
                            ('identity.json', 'identity_sha256')):
        path = folder / filename
        hashes[str(path)] = checked(path, audit[field])
    assert set(audit['posterior_sha256']) == {str(p) for p in folder.glob('batch_*.npz')}
    for path, digest in audit['posterior_sha256'].items():
        hashes[path] = checked(path, digest)
    for path, digest in audit['source_data_sha256'].items():
        checked(path, digest)
    identity = read(folder / 'identity.json')
    assert identity['model']['checkpoint'] == plan['checkpoints'][checkpoint]
    assert identity['plan_sha256'] == FROZEN_PLAN_SHA256 and identity['phase'] == 'confirm'
    summary = read(folder / 'summary.json')
    assert summary['status'] == 'passed' and summary['plan_sha256'] == FROZEN_PLAN_SHA256
    assert summary['n'] == audit['n'] and summary['gate_cost_separate']
    assert summary['physical_sample_forwards'] == audit['physical_sample_forwards']
    candidate = selection['selected_name']
    nfe = selection['config']['groups'] + 1
    assert audit['physical_sample_forwards'] == audit['n'] * (nfe + 12)
    assert audit['query_masks_audited'] == audit['n'] * (nfe - 1)
    assert set(audit['results']) == {'oas_new:' + mode for mode in MODES}
    assert set(summary['mean_sequence_aar']) == set(BASELINES) | {candidate}
    for mode in MODES:
        key = 'oas_new:' + mode
        result = audit['results'][key]
        assert result['n'] == counts[mode] and result['groups'] == 2048
        assert set(result['mean_sequence_aar']) == set(BASELINES) | {candidate}
        assert set(result['logical_mean_nfe']) == set(BASELINES) | {candidate}
        for name in BASELINES + (candidate,):
            value = result['mean_sequence_aar'][name]
            assert math.isfinite(value) and 0 <= value <= 100
            assert abs(value - summary['mean_sequence_aar'][name][key]) < 1e-8
        assert result['logical_mean_nfe'] == {'single_pass': 1., 'original4': 4., 'original8': 8., candidate: float(nfe)}
        for base in BASELINES:
            comparison = result['comparisons'][base][candidate]
            delta = result['mean_sequence_aar'][candidate] - result['mean_sequence_aar'][base]
            assert abs(comparison['delta_pp'] - delta) < 1e-8
            interval = comparison['ci95_pp']
            assert len(interval) == 2 and all(math.isfinite(x) for x in interval) and interval[0] <= interval[1]
    gate_path = out / f'gate_cuda_{checkpoint}.json'
    gate = read(gate_path)
    assert gate['status'] == 'passed' and gate['plan_sha256'] == FROZEN_PLAN_SHA256
    assert gate['identity'] == identity['model'] and gate['checkpoint'] == checkpoint
    for field in ('no_reference_in_features', 'original_single_pass_argmax_exact', 'zero_strength_exact',
                  'single_group_exact', 'singleton_partition_equals_legacy_loo_exact', 'hook_cleanup_exact'):
        assert gate[field] is True
    hashes[str(gate_path)] = checked(gate_path)
    return audit, hashes


def summarize(out):
    checked(out / 'plan.json', FROZEN_PLAN_SHA256)
    checked(out / 'selection.json', FROZEN_SELECTION_SHA256)
    plan, selection = read(out / 'plan.json'), read(out / 'selection.json')
    assert selection['eligible'] is True and selection['plan_sha256'] == FROZEN_PLAN_SHA256
    assert set(plan['checkpoints']) == set(CHECKPOINTS)
    for field in ('source_sha256', 'external_source_sha256'):
        for path, digest in plan[field].items():
            checked(path, digest)
    checked(out / 'rows.json', plan['rows_sha256'])
    checked(out / 'data_plan.json', plan['data_plan_sha256'])
    checked(out / 'data_independent_audit.json', plan['data_audit_sha256'])
    data = read(out / 'data_independent_audit.json')
    assert data['status'] == 'passed_scoped_checks'
    assert data['rows_sha256'] == plan['rows_sha256'] and data['data_plan_sha256'] == plan['data_plan_sha256']
    counts = {mode: data['counts']['confirm:' + mode] for mode in MODES}
    assert set(counts.values()) == {2094} and data['confirm_groups'] == 2048
    development_path = out / 'explore_112000/independent_audit.json'
    checked(development_path, selection['development_audit_sha256'])
    development = read(development_path)
    assert development['status'] == 'passed' and development['plan_sha256'] == FROZEN_PLAN_SHA256
    candidate = selection['selected_name']
    config = selection['config']
    assert candidate == config['name'] + '_prior' + str(config['prior_weight'])
    family = next(f for f in plan['families'] if f['name'] == config['name'])
    assert all(config[key] == value for key, value in family.items()) and config['prior_weight'] in plan['prior_weights']
    dev = development['results']['modern:cdrh3']
    candidates = [name for name in dev['mean_sequence_aar'] if name != 'single_pass']
    winner = min(candidates, key=lambda name: (-dev['mean_sequence_aar'][name], dev['logical_mean_nfe'][name], name))
    assert winner == candidate and dev['comparisons']['single_pass'][winner]['delta_pp'] > 0
    assert abs(dev['comparisons']['single_pass'][winner]['delta_pp'] - selection['best_explore_delta_pp']) < 1e-8
    jobs, evidence = validate_platform(out)
    assert timestamp(selection['frozen_utc']) < timestamp(jobs[1]['start'])
    audits, table = {}, []
    for checkpoint in CHECKPOINTS:
        audit, hashes = validate_confirmation(out, checkpoint, plan, selection, counts)
        audits[checkpoint] = audit
        evidence.update(hashes)
        for mode in MODES:
            result = audit['results']['oas_new:' + mode]
            table.append(dict(checkpoint=checkpoint, dataset='oas_new', mode=mode,
                n=result['n'], groups=result['groups'],
                aar_percent={**{name: result['mean_sequence_aar'][name] for name in BASELINES},
                             'candidate': result['mean_sequence_aar'][candidate]},
                comparisons={base: result['comparisons'][base][candidate] for base in BASELINES},
                candidate_logical_nfe=result['logical_mean_nfe'][candidate]))
    assert len(table) == 6
    primary = {row['checkpoint']: row['comparisons']['single_pass']['ci95_pp'][0] > 0 for row in table if row['mode'] == 'cdrh3'}
    secondary = {row['checkpoint'] + ':' + row['mode']: row['comparisons']['single_pass']['delta_pp'] >= -.25
                 for row in table if row['mode'] in ('cdrh1', 'cdrh2')}
    nfe = config['groups'] + 1
    cost_pass = all(row['candidate_logical_nfe'] <= 4 for row in table)
    passed = all(primary.values()) and all(secondary.values()) and cost_pass
    old_path = Path(plan['parent']) / 'confirmation_112000/independent_audit.json'
    previous = read(old_path)
    assert previous['status'] == 'passed'
    old_nfe = previous['results']['modern:cdrh3']['candidate_mean_nfe']
    assert abs(old_nfe - 24.380622837370243) < 1e-10
    evidence.update({str(out / 'plan.json'): FROZEN_PLAN_SHA256,
        str(out / 'selection.json'): FROZEN_SELECTION_SHA256,
        str(development_path): selection['development_audit_sha256'],
        str(out / 'data_plan.json'): plan['data_plan_sha256'],
        str(out / 'data_independent_audit.json'): plan['data_audit_sha256'],
        str(old_path): checked(old_path)})
    report = dict(status='complete', outcome='meets_preregistered_criteria' if passed else 'does_not_meet_preregistered_criteria',
        preregistered_success_passed=passed, selected_name=candidate, config=config,
        jobs=jobs, table=table,
        criteria=dict(protocol_text=plan['success'], h3_ci_lower_strictly_positive_by_checkpoint=primary,
            both_checkpoints_h3_pass=all(primary.values()),
            h1_h2_delta_at_least_minus_0p25pp=secondary, all_secondary_pass=all(secondary.values()),
            maximum_logical_nfe=4, cost_pass=cost_pass, all_conditions_pass=passed),
        cost=dict(candidate_logical_nfe=nfe, single_pass_logical_nfe=1,
            historical_loo_modern_h3_mean_nfe=old_nfe,
            logical_nfe_reduction_vs_historical_loo_percent=100 * (1 - nfe / old_nfe),
            historical_loo_to_candidate_nfe_ratio=old_nfe / nfe,
            note='Historical LOO cost was measured on modern-SAbDab H3; this candidate is confirmed on OAS. This is a logical-forward-count reference across different cohorts, not a measured wall-clock speedup or a matched accuracy comparison.'),
        counts=dict(confirm_pairs=2094, confirm_components=2048, rows_per_checkpoint=6282,
            predictions_per_checkpoint={checkpoint: audits[checkpoint]['predictions'] for checkpoint in CHECKPOINTS},
            total_confirmation_predictions=sum(audits[c]['predictions'] for c in CHECKPOINTS),
            total_confirmation_sample_forwards=sum(audits[c]['physical_sample_forwards'] for c in CHECKPOINTS),
            gate_cost_separate=True, reserve_rows_per_mode=9026),
        bootstrap=plan['bootstrap'], evidence_sha256=evidence,
        script_sha256=sha(Path(__file__)),
        limitations=[
            'All six checkpoint-by-CDR results are included. Development winner confidence intervals do not establish confirmation.',
            'The OAS confirmation supplement is distinct from official SAbDab/Kong/SAb23 benchmarks.',
            'H1/H2 guardrails use point-estimate losses relative to single pass; they are not formal noninferiority tests.',
            'Paired component percentile intervals are unadjusted. Success requires both checkpoint H3 lower bounds above zero.',
            'Original4/8 token outputs were audited, without saved per-step trajectory reconstruction.',
            'Exact heavy/H3 and supplied cluster exclusions do not establish universal homology or wider pretraining decontamination.',
            'This experiment does not establish monotonic AAR as sampling steps increase.',
        ])
    destination = out / 'confirmation_delivery.json'
    if destination.exists():
        assert read(destination) == report, 'Existing delivery differs; refusing to replace it silently'
    else:
        with destination.open('x') as stream:
            stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'outcome', 'table', 'criteria', 'cost', 'counts')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-code', action='store_true')
    args = parser.parse_args()
    if args.check_code:
        ast.parse(Path(__file__).read_text())
        assert len(FROZEN_PLAN_SHA256) == len(FROZEN_SELECTION_SHA256) == 64
        assert set(FROZEN_PLAN_SHA256 + FROZEN_SELECTION_SHA256) <= set('0123456789abcdef')
        print('PASS: syntax and pre-confirmation protocol/selection hash constants; no experiment outputs read or written')
        return
    assert args.out is not None, '--out is required to summarize completed confirmation'
    summarize(args.out)


if __name__ == '__main__':
    main()
