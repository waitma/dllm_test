"""Activate pllm; python -B FILE --out ABS [--check-code].

Deliver the completed two-checkpoint grouped four/eight-forward comparison.
This script is separate from frozen inference sources and never selects a policy.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


FROZEN_PLAN_SHA256 = 'a671a2122df4efd4d9f0483fd08cd2446262edbfdc80ac81af8c1f7e9a143340'
CHECKPOINTS = ('112000', '50000')
MODES = ('cdrh1', 'cdrh2', 'cdrh3')
NEW_POLICIES = {'single_pass', 'grouped4', 'grouped8'}
ALL_POLICIES = NEW_POLICIES | {'original4', 'original8'}
HASH_CACHE = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, expected=None):
    key = str(path)
    if key not in HASH_CACHE:
        HASH_CACHE[key] = sha(path)
    value = HASH_CACHE[key]
    if expected is not None:
        assert value == expected, ('changed artifact', key, value, expected)
    return value


def direction(interval):
    assert len(interval) == 2 and all(math.isfinite(value) for value in interval) and interval[0] <= interval[1]
    return 'benefit' if interval[0] > 0 else 'harm' if interval[1] < 0 else 'inconclusive'


def validate_cpu_recovery(out, plan):
    path = out / 'audit_recovery_record.json'
    record = read(path)
    assert record['status'] == 'passed' and record['plan_sha256'] == FROZEN_PLAN_SHA256
    assert record['original_job_status_preserved'] is True and record['statistical_protocol_unchanged'] is True
    platforms = {str(out / f'round{number}_platform.json') for number in (1, 2)}
    assert set(record['original_platform_sha256']) == platforms
    for artifact, digest in record['original_platform_sha256'].items():
        checked(artifact, digest)
    parent = Path(plan['parent'])
    restored = {str(parent / f'evaluation_{checkpoint}/predictions.jsonl') for checkpoint in CHECKPOINTS}
    assert set(record['restored_parent_predictions']) == restored
    for artifact, metadata in record['restored_parent_predictions'].items():
        assert metadata['expected_sha256'] == metadata['actual_sha256'] == plan['parent_evidence_sha256'][artifact]
        checked(artifact, metadata['expected_sha256'])
    restore = record['restore_job']
    assert restore['status'] == 'Success'
    checked(restore['source_path'], restore['source_sha256'])
    checked(restore['platform_path'], restore['platform_sha256'])
    platform = read(restore['platform_path'])
    assert platform['JobId'] == restore['job_id'] and platform['Status'] == 'Success'
    assert set(record['cpu_audits']) == set(CHECKPOINTS)
    frozen_auditor = out / 'source/audit_cdr_block_budget.py'
    for checkpoint in CHECKPOINTS:
        cpu = record['cpu_audits'][checkpoint]
        assert cpu['exit_code'] == 0 and Path(cpu['auditor_path']) == frozen_auditor
        assert cpu['auditor_sha256'] == plan['source_sha256'][str(frozen_auditor)]
        checked(cpu['auditor_path'], cpu['auditor_sha256'])
        expected = out / f'evaluation_{checkpoint}/independent_audit.json'
        assert Path(cpu['independent_audit_path']) == expected
        checked(expected, cpu['independent_audit_sha256'])
        audit = read(expected)
        assert audit['status'] == 'passed' and audit['checkpoint'] == checkpoint
        assert audit['plan_sha256'] == FROZEN_PLAN_SHA256 and audit['auditor_sha256'] == cpu['auditor_sha256']
        required = {str(out / f'gate_cuda_{checkpoint}.json')}
        required.update(str(expected.parent / filename) for filename in ('predictions.jsonl', 'query_audit.jsonl', 'summary.json', 'identity.json', 'status.json'))
        required.update(str(artifact) for artifact in expected.parent.glob('batch_*.npz'))
        assert required <= set(record['new_artifact_sha256'])
    for artifact, digest in record['new_artifact_sha256'].items():
        checked(artifact, digest)
    hashes = {str(path): checked(path), restore['source_path']: restore['source_sha256'],
              restore['platform_path']: restore['platform_sha256']}
    return dict(mode='cpu_audit_recovered_after_exact_dependency_restoration',
        original_gpu_job_status_preserved=True, original_statistical_protocol_unchanged=True,
        restored_parent_prediction_files_bitwise_identical=True, cpu_original_frozen_audits_passed=True,
        recovery_record=str(path), recovery_record_sha256=hashes[str(path)],
        restore_job_id=restore['job_id'], restore_job_status=restore['status']), hashes


def validate_jobs(out, plan):
    paths = sorted(out.glob('round*_platform.json'))
    assert {path.name for path in paths} == {'round1_platform.json', 'round2_platform.json'}
    jobs, hashes = [], {}
    for number, checkpoint in ((1, '112000'), (2, '50000')):
        path = out / f'round{number}_platform.json'
        submission_path = out / f'round{number}_submission.json'
        platform, submission = read(path), read(submission_path)
        assert platform['JobId'] == submission['task_id']
        assert platform['Status'] in ('Success', 'Failed') and platform['End']
        assert submission['checkpoint'] == checkpoint and submission['plan_sha256'] == FROZEN_PLAN_SHA256
        config = Path(submission['config'])
        jobs.append(dict(checkpoint=checkpoint, job_id=platform['JobId'], status=platform['Status'],
            start=platform['Start'], end=platform['End'], yaml=str(config), yaml_sha256=checked(config, submission['config_sha256'])))
        hashes[str(path)] = checked(path)
        hashes[str(submission_path)] = checked(submission_path)
    assert jobs[0]['job_id'] != jobs[1]['job_id']
    if any(job['status'] == 'Failed' for job in jobs):
        lifecycle, recovery_hashes = validate_cpu_recovery(out, plan)
        hashes.update(recovery_hashes)
    else:
        lifecycle = dict(mode='original_gpu_jobs_and_audits_succeeded', original_gpu_job_status_preserved=True,
                         original_statistical_protocol_unchanged=True)
    return jobs, hashes, lifecycle


def validate_report(out, checkpoint, plan):
    folder = out / ('evaluation_' + checkpoint)
    parent_folder = Path(plan['parent']) / ('evaluation_' + checkpoint)
    audit_path = folder / 'independent_audit.json'
    audit = read(audit_path)
    assert audit['status'] == 'passed' and audit['checkpoint'] == checkpoint and audit['n'] == 5254
    assert audit['plan_sha256'] == FROZEN_PLAN_SHA256 and audit['winner_selected'] is False
    assert audit['policies'] == audit['policies_new'] == 3 and audit['policies_compared'] == 5
    assert audit['predictions'] == audit['new_predictions'] == 5254 * 3
    assert audit['inherited_control_predictions'] == 5254 * 2 and audit['compared_predictions'] == 5254 * 5
    assert audit['physical_sample_forwards'] == 5254 * 11 and audit['query_masks_audited'] == 5254 * 10
    assert audit['gate_cost_separate'] is True
    for flag in ('source_reference_recovered', 'probabilities_reconstructed', 'actual_feedback_weights_reconstructed',
                 'prior_and_grouped4_probabilities_bitwise_equal_to_parent',
                 'grouped4_feedback_weights_and_queries_bitwise_equal_to_parent',
                 'single_pass_and_grouped4_predictions_exact_to_parent'):
        assert audit[flag] is True
    replay = audit['replay']
    assert replay['n'] == 5254 and replay['changed_replayed_prediction_tokens'] == 0
    assert replay['maximum_prior_absolute_difference'] == replay['maximum_grouped4_posterior_absolute_difference'] == 0
    assert audit['bootstrap']['repeats'] == 50000 and audit['bootstrap']['seed'] == 20261016
    assert audit['bootstrap']['family_size'] == 2 and audit['bootstrap']['per_comparison_adjusted_confidence'] == .975
    assert audit['bootstrap']['nominal_family_coverage_at_least'] == .95
    auditor = out / 'source/audit_cdr_block_budget.py'
    assert audit['auditor_sha256'] == plan['source_sha256'][str(auditor)]
    assert audit['parent_audit_sha256'] == plan['parent_audit_sha256'][checkpoint]
    parent_audit_path = parent_folder / 'independent_audit.json'
    checked(parent_audit_path, audit['parent_audit_sha256'])
    previous = read(parent_audit_path)
    hashes = {str(audit_path): checked(audit_path)}
    for filename, field in (('predictions.jsonl', 'predictions_sha256'), ('query_audit.jsonl', 'query_audit_sha256'),
                            ('summary.json', 'summary_sha256'), ('identity.json', 'identity_sha256')):
        path = folder / filename
        hashes[str(path)] = checked(path, audit[field])
    assert set(audit['posterior_sha256']) == {str(path) for path in folder.glob('batch_*.npz')}
    for path, digest in audit['posterior_sha256'].items():
        hashes[path] = checked(path, digest)
    assert audit['parent_replayed_posterior_sha256'] == previous['posterior_sha256']
    for path, digest in audit['parent_replayed_posterior_sha256'].items():
        checked(path, digest)
    assert audit['parent_predictions_sha256'] == previous['predictions_sha256']
    checked(parent_folder / 'predictions.jsonl', audit['parent_predictions_sha256'])
    assert audit['source_data_sha256'] == previous['source_data_sha256']
    for path, digest in audit['source_data_sha256'].items():
        checked(path, digest)
    checkpoint_path = Path(plan['checkpoints'][checkpoint])
    checked(checkpoint_path / 'vocabulary.json', audit['vocabulary_sha256'])
    identity = read(folder / 'identity.json')
    assert identity['model']['checkpoint'] == str(checkpoint_path)
    assert identity['phase'] == 'budget' and identity['plan_sha256'] == FROZEN_PLAN_SHA256
    summary = read(folder / 'summary.json')
    assert summary['status'] == 'passed' and summary['n'] == audit['n']
    assert summary['plan_sha256'] == FROZEN_PLAN_SHA256 and summary['gate_cost_separate']
    assert summary['prior_and_grouped4_exact_to_parent'] is True
    assert summary['physical_sample_forwards'] == audit['physical_sample_forwards']
    assert set(summary['mean_sequence_aar']) == NEW_POLICIES
    assert set(audit['results']) == {'oas_new:' + mode for mode in MODES}
    for mode in MODES:
        key = 'oas_new:' + mode
        result = audit['results'][key]
        assert result['n'] == plan['counts_by_mode'][mode]
        assert result['groups'] == (4096 if mode == 'cdrh3' else 512)
        assert set(result['mean_sequence_aar']) == set(result['logical_mean_nfe']) == ALL_POLICIES
        assert result['logical_mean_nfe'] == {'single_pass': 1., 'grouped4': 4., 'grouped8': 8., 'original4': 4., 'original8': 8.}
        for name, value in result['mean_sequence_aar'].items():
            assert math.isfinite(value) and 0 <= value <= 100
            if name in NEW_POLICIES:
                assert abs(value - summary['mean_sequence_aar'][name][key]) < 1e-8
            if name != 'grouped8':
                old_name = 'soft_frozen_prior0.5' if name == 'grouped4' else name
                assert abs(value - previous['results'][key]['mean_sequence_aar'][old_name]) < 1e-8
        assert len(result['contrasts']) == 7
        for name, contrast in result['contrasts'].items():
            assert name == contrast['positive'] + '_vs_' + contrast['negative']
            delta = result['mean_sequence_aar'][contrast['positive']] - result['mean_sequence_aar'][contrast['negative']]
            assert abs(contrast['delta_pp'] - delta) < 1e-8
            direction(contrast['ci95_pp'])
            if mode == 'cdrh3' and name == 'grouped8_vs_grouped4':
                assert contrast['role'] == 'primary_followup' and contrast['bonferroni_family_tests'] == 2
                assert contrast['per_comparison_adjusted_confidence'] == .975 and contrast['nominal_family_coverage_at_least'] == .95
                assert contrast['adjusted_direction'] == direction(contrast['ci_bonferroni_pp'])
            else:
                assert contrast['role'] == 'descriptive' and 'ci_bonferroni_pp' not in contrast
        changes = result['grouped8_vs_grouped4_changes']
        assert all(type(value) is int and value >= 0 for value in changes.values())
        assert changes['sequences_improved'] + changes['sequences_worsened'] + changes['sequences_equal_aar'] == result['n']
        assert changes['wrong_to_right'] + changes['right_to_wrong'] + changes['changed_wrong_to_wrong'] == changes['changed_residues']
        assert changes['changed_sequences'] <= changes['changed_residues'] and changes['changed_sequences'] <= result['n']
    gate_path = out / ('gate_cuda_' + checkpoint + '.json')
    hashes[str(gate_path)] = checked(gate_path, audit['gate_sha256'])
    gate = read(gate_path)
    assert gate['status'] == 'passed' and gate['checkpoint'] == checkpoint and gate['plan_sha256'] == FROZEN_PLAN_SHA256
    assert gate['identity'] == identity['model']
    for flag in ('no_reference_in_features', 'original_single_pass_exact', 'grouped4_matches_parent_exact',
                 'grouped8_partition_and_weights_audited', 'grouped8_zero_strength_exact', 'hook_cleanup_exact', 'fp32_frozen_model'):
        assert gate[flag] is True
    return audit, hashes


def summarize(out):
    HASH_CACHE.clear()
    checked(out / 'plan.json', FROZEN_PLAN_SHA256)
    plan = read(out / 'plan.json')
    assert set(plan['checkpoints']) == set(CHECKPOINTS)
    assert set(plan['policies']) == NEW_POLICIES and plan['reused_controls'] == ['original4', 'original8']
    for field in ('source_sha256', 'external_source_sha256', 'parent_evidence_sha256'):
        for path, digest in plan[field].items():
            checked(path, digest)
    evidence = {str(out / 'plan.json'): FROZEN_PLAN_SHA256}
    for filename, field in (('rows.json', 'rows_sha256'), ('data_plan.json', 'data_plan_sha256')):
        evidence[str(out / filename)] = checked(out / filename, plan[field])
    parent = Path(plan['parent'])
    checked(parent / 'rows.json', plan['parent_rows_sha256'])
    checked(parent / 'plan.json', plan['parent_plan_sha256'])
    assert plan['rows_sha256'] == plan['parent_rows_sha256']
    data = read(out / 'data_plan.json')
    assert data['rows_identical_to_parent'] is True and data['counts_by_mode'] == plan['counts_by_mode']
    rows = read(out / 'rows.json')
    assert len(rows) == 5254 and rows == read(parent / 'rows.json')
    target_residues = sum(len(row['mask_h']) for row in rows)
    for mode in MODES:
        subset = [row for row in rows if row['mode'] == mode]
        lengths = [len(row['mask_h']) for row in subset]
        expected = dict(n=len(subset), components=len({row['group'] for row in subset}),
            length_below_seven=sum(length < 7 for length in lengths),
            length_at_most_seven=sum(length <= 7 for length in lengths),
            effective_group_count={str(key): value for key, value in sorted(Counter(min(length, 7) for length in lengths).items())},
            empty_group_sample_forwards=sum(max(7-length, 0) for length in lengths))
        assert data['group_stats'][mode] == expected
    assert not (out / 'selection.json').exists()
    jobs, hashes, lifecycle = validate_jobs(out, plan)
    evidence.update(hashes)
    audits, table, primary, descriptive, changes = {}, [], [], [], []
    for checkpoint in CHECKPOINTS:
        audit, hashes = validate_report(out, checkpoint, plan)
        assert audit['replay']['target_residues'] == target_residues
        audits[checkpoint] = audit
        evidence.update(hashes)
        for mode in MODES:
            result = audit['results']['oas_new:' + mode]
            table.append(dict(checkpoint=checkpoint, dataset='oas_new', mode=mode, n=result['n'], groups=result['groups'],
                aar_percent=result['mean_sequence_aar'], logical_nfe=result['logical_mean_nfe']))
            changes.append(dict(checkpoint=checkpoint, mode=mode, **result['grouped8_vs_grouped4_changes']))
            for name, comparison in result['contrasts'].items():
                row = dict(checkpoint=checkpoint, mode=mode, contrast=name, **comparison)
                (primary if comparison['role'] == 'primary_followup' else descriptive).append(row)
    assert len(table) == 6 and len(primary) == 2 and len(changes) == 6
    directions = {row['checkpoint']: row['adjusted_direction'] for row in primary}
    both_improve = all(value == 'benefit' for value in directions.values())
    both_harm = all(value == 'harm' for value in directions.values())
    report = dict(status='complete', outcome='both_checkpoints_improve_vs_grouped4' if both_improve else
        'both_checkpoints_worsen_vs_grouped4' if both_harm else 'no_consistent_adjusted_budget_effect',
        no_selection=True, winner_selected=False, analysis_scope=plan['sample_use'], jobs=jobs,
        execution_lifecycle=lifecycle,
        table=table, primary_h3=primary, descriptive=descriptive, change_counts=changes,
        primary_conclusion=dict(adjusted_direction_by_checkpoint=directions,
            both_checkpoints_improve_vs_grouped4=both_improve, both_checkpoints_worsen_vs_grouped4=both_harm),
        replay=dict(all_prior_and_grouped4_posteriors_weights_queries_exact=True,
            all_single_pass_and_grouped4_predictions_exact=True, target_residues_per_checkpoint=target_residues),
        counts=dict(rows_per_checkpoint=5254, counts_by_mode=plan['counts_by_mode'], policies_new=3, policies_compared=5,
            total_new_predictions=sum(audits[cp]['new_predictions'] for cp in CHECKPOINTS),
            total_inherited_control_predictions=sum(audits[cp]['inherited_control_predictions'] for cp in CHECKPOINTS),
            total_compared_predictions=sum(audits[cp]['compared_predictions'] for cp in CHECKPOINTS),
            total_new_sample_forwards=sum(audits[cp]['physical_sample_forwards'] for cp in CHECKPOINTS),
            gate_cost_separate=True, additional_unused_reserve_rows_consumed=0),
        group_geometry=data['group_stats'], bootstrap=plan['bootstrap'],
        multiplicity_interpretation=dict(per_comparison_adjusted_confidence=.975,
            local_family_size=2, nominal_local_family_coverage_at_least=.95,
            note='Two H3 follow-up contrasts use 97.5% individual percentile intervals for at least 95% nominal local family coverage. Bootstrap calibration is approximate; this does not correct all historical adaptive analyses.'),
        inferential_rule=plan['inferential_rule'], secondary_rule=plan['secondary_rule'],
        budget_definition=plan['budget_definition'], evidence_sha256=evidence, script_sha256=sha(Path(__file__)),
        limitations=plan['limitations'] + [
            'The same previously evaluated rows are reused; this is not fresh independent confirmation.',
            'A positive eight-minus-four result does not imply eight-forward superiority to single pass.',
            'The experiment reports logical model-forward counts, not a measured wall-clock speedup.',
            'All new policies and inherited controls are disclosed; this delivery does not choose a winner.',
        ] + (['Original GPU jobs remain Failed because their post-inference audit encountered deleted parent dependencies. Both parent prediction files were restored to their original hashes, and the unchanged original frozen audits subsequently passed on CPU; no statistical protocol was relaxed.']
             if lifecycle['mode'] == 'cpu_audit_recovered_after_exact_dependency_restoration' else []))
    path = out / 'budget_delivery.json'
    if path.exists():
        assert read(path) == report, 'Existing delivery differs; refusing to overwrite'
    else:
        with path.open('x') as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'outcome', 'counts', 'primary_h3', 'change_counts')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-code', action='store_true')
    args = parser.parse_args()
    if args.check_code:
        ast.parse(Path(__file__).read_text())
        assert len(FROZEN_PLAN_SHA256) == 64 and set(FROZEN_PLAN_SHA256) <= set('0123456789abcdef')
        print('PASS: syntax and pinned protocol digest; no experiment outputs read or written')
        return
    assert args.out is not None, '--out required'
    summarize(args.out)


if __name__ == '__main__':
    main()
