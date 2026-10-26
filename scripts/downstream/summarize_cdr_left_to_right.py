"""Activate pllm; python -B FILE --out ABS [--check-code].

Deliver all strict-left-to-right results after both fixed-checkpoint jobs finish.
This reporting script is outside the frozen inference bundle and selects nothing.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path


FROZEN_PLAN_SHA256 = '056bd5c0d3f9a353141ff1fc710e7ce9c764f2f012f4c69abb012b3f8fe1e999'
CHECKPOINTS = ('112000', '50000')
MODES = ('cdrh1', 'cdrh2', 'cdrh3')
BASELINES = ('single_pass', 'original4', 'original8', 'grouped4', 'grouped8')
AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')
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


def validate_jobs(out):
    assert {p.name for p in out.glob('round*_platform.json')} == {'round1_platform.json', 'round2_platform.json'}
    jobs, hashes = [], {}
    for number, checkpoint in ((1, '112000'), (2, '50000')):
        platform_path = out / f'round{number}_platform.json'
        submission_path = out / f'round{number}_submission.json'
        platform, submission = read(platform_path), read(submission_path)
        assert platform['JobId'] == submission['task_id'] and platform['Status'] == 'Success' and platform['End']
        assert submission['checkpoint'] == checkpoint and submission['plan_sha256'] == FROZEN_PLAN_SHA256
        config = Path(submission['config'])
        jobs.append(dict(checkpoint=checkpoint, job_id=platform['JobId'], status=platform['Status'],
            start=platform['Start'], end=platform['End'], yaml=str(config), yaml_sha256=checked(config, submission['config_sha256'])))
        hashes[str(platform_path)] = checked(platform_path)
        hashes[str(submission_path)] = checked(submission_path)
    assert jobs[0]['job_id'] != jobs[1]['job_id']
    return jobs, hashes


def validate_report(out, checkpoint, plan, data):
    folder = out / ('evaluation_' + checkpoint)
    parent_folder = Path(plan['parent']) / ('evaluation_' + checkpoint)
    ablation_folder = Path(plan['ablation_parent']) / ('evaluation_' + checkpoint)
    path = folder / 'independent_audit.json'
    report = read(path)
    assert report['status'] == 'passed' and report['checkpoint'] == checkpoint and report['n'] == 5254
    assert report['plan_sha256'] == FROZEN_PLAN_SHA256 and report['winner_selected'] is False
    assert report['policies'] == 1 and report['predictions'] == 5254
    assert report['policies_compared'] == 6 and report['compared_predictions'] == 5254*6
    for flag in ('source_reference_recovered', 'observed_context_unchanged', 'strict_left_to_right_history_reconstructed',
                 'current_and_future_targets_remain_masked', 'selected_forward_probabilities_reconstructed',
                 'prior_bitwise_equal_to_parent', 'first_query_reuses_prior', 'decoding_exactly_target_length_without_early_stop',
                 'no_additional_amino_acid_restriction', 'forbidden_predictor_probabilities_are_zero', 'gate_cost_separate'):
        assert report[flag] is True
    for field in ('logical_sample_forwards', 'physical_sample_forwards', 'physical_batch_forwards'):
        assert report[field] == data['cost_per_checkpoint'][field]
    assert report['inactive_row_sample_forwards'] == report['physical_sample_forwards'] - report['logical_sample_forwards']
    assert report['actual_query_input_rows_audited'] == report['physical_sample_forwards']
    assert report['bootstrap']['repeats'] == 50000 and report['bootstrap']['seed'] == 20261016
    assert report['bootstrap']['family_size'] == 2 and report['bootstrap']['per_comparison_adjusted_confidence'] == .975
    assert report['bootstrap']['nominal_family_coverage_at_least'] == .95
    auditor = out / 'source/audit_cdr_left_to_right.py'
    assert report['auditor_sha256'] == plan['source_sha256'][str(auditor)]
    hashes = {str(path): checked(path)}
    for filename, field in (('predictions.jsonl', 'predictions_sha256'), ('summary.json', 'summary_sha256'), ('identity.json', 'identity_sha256')):
        artifact = folder / filename
        hashes[str(artifact)] = checked(artifact, report[field])
    assert set(report['posterior_sha256']) == {str(p) for p in folder.glob('batch_*.npz')}
    for artifact, digest in report['posterior_sha256'].items():
        hashes[artifact] = checked(artifact, digest)
    assert report['parent_audit_sha256'] == plan['parent_audit_sha256'][checkpoint]
    assert report['ablation_audit_sha256'] == plan['ablation_audit_sha256'][checkpoint]
    checked(parent_folder / 'independent_audit.json', report['parent_audit_sha256'])
    checked(ablation_folder / 'independent_audit.json', report['ablation_audit_sha256'])
    parent_audit = read(parent_folder / 'independent_audit.json')
    ablation_audit = read(ablation_folder / 'independent_audit.json')
    assert report['parent_replayed_posterior_sha256'] == parent_audit['posterior_sha256']
    for artifact, digest in report['parent_replayed_posterior_sha256'].items():
        checked(artifact, digest)
    assert report['parent_predictions_sha256'] == parent_audit['predictions_sha256']
    assert report['ablation_predictions_sha256'] == ablation_audit['predictions_sha256']
    checked(parent_folder / 'predictions.jsonl', report['parent_predictions_sha256'])
    checked(ablation_folder / 'predictions.jsonl', report['ablation_predictions_sha256'])
    assert report['source_data_sha256'] == parent_audit['source_data_sha256'] == ablation_audit['source_data_sha256']
    for artifact, digest in report['source_data_sha256'].items():
        checked(artifact, digest)
    checkpoint_path = Path(plan['checkpoints'][checkpoint])
    checked(checkpoint_path / 'vocabulary.json', report['vocabulary_sha256'])
    identity = read(folder / 'identity.json')
    assert identity['model']['checkpoint'] == str(checkpoint_path)
    assert identity['plan_sha256'] == FROZEN_PLAN_SHA256 and identity['phase'] == 'left_to_right'
    summary = read(folder / 'summary.json')
    assert summary['status'] == 'passed' and summary['n'] == report['n']
    assert summary['plan_sha256'] == FROZEN_PLAN_SHA256 and summary['gate_cost_separate'] and summary['prior_exact_to_parent']
    for field in ('logical_sample_forwards', 'physical_sample_forwards', 'physical_batch_forwards'):
        assert summary[field] == report[field]
    assert set(summary['mean_sequence_aar']) == {'left_to_right'}
    assert set(report['results']) == {'oas_new:' + mode for mode in MODES}
    expected_names = {'left_to_right'} | set(BASELINES)
    for mode in MODES:
        key = 'oas_new:' + mode
        result = report['results'][key]
        assert result['n'] == plan['counts_by_mode'][mode] and result['groups'] == (4096 if mode == 'cdrh3' else 512)
        assert set(result['mean_sequence_aar']) == set(result['logical_mean_nfe']) == expected_names
        for name, value in result['mean_sequence_aar'].items():
            assert math.isfinite(value) and 0 <= value <= 100
            old = summary['mean_sequence_aar']['left_to_right'][key] if name == 'left_to_right' else parent_audit['results'][key]['mean_sequence_aar'][name]
            assert abs(value - old) < 1e-8
        assert abs(result['logical_mean_nfe']['left_to_right'] - data['mode_stats'][mode]['mean_logical_nfe']) < 1e-12
        for name in BASELINES:
            assert result['logical_mean_nfe'][name] == (1 if name == 'single_pass' else 8 if name in ('original8', 'grouped8') else 4)
        assert result['left_to_right_physical_mean_forwards'] >= result['logical_mean_nfe']['left_to_right']
        assert set(result['contrasts']) == {'left_to_right_vs_' + baseline for baseline in BASELINES}
        for name, contrast in result['contrasts'].items():
            assert contrast['positive'] == 'left_to_right' and name == 'left_to_right_vs_' + contrast['negative']
            delta = result['mean_sequence_aar']['left_to_right'] - result['mean_sequence_aar'][contrast['negative']]
            assert abs(delta - contrast['delta_pp']) < 1e-8
            direction(contrast['ci95_pp'])
            if mode == 'cdrh3' and contrast['negative'] == 'single_pass':
                assert contrast['role'] == 'primary_followup' and contrast['bonferroni_family_tests'] == 2
                assert contrast['per_comparison_adjusted_confidence'] == .975 and contrast['nominal_family_coverage_at_least'] == .95
                assert contrast['adjusted_direction'] == direction(contrast['ci_bonferroni_pp'])
            else:
                assert contrast['role'] == 'descriptive' and 'ci_bonferroni_pp' not in contrast
        assert set(result['left_to_right_changes_vs_baseline']) == set(BASELINES)
        for counts in result['left_to_right_changes_vs_baseline'].values():
            assert all(type(value) is int and value >= 0 for value in counts.values())
            assert counts['sequences_improved'] + counts['sequences_worsened'] + counts['sequences_equal_aar'] == result['n']
            assert counts['wrong_to_right'] + counts['right_to_wrong'] + counts['changed_wrong_to_wrong'] == counts['changed_residues']
            assert counts['changed_sequences'] <= min(result['n'], counts['changed_residues'])
        assert set(result['non_amino_acid_outputs']) == expected_names
        for counts in result['non_amino_acid_outputs'].values():
            assert counts['residues'] == sum(counts['token_counts'].values())
            assert 0 <= counts['sequences'] <= min(result['n'], counts['residues'])
            assert all(token not in AMINO_ACIDS and type(number) is int and number > 0 for token, number in counts['token_counts'].items())
    gate_path = out / f'gate_cuda_{checkpoint}.json'
    hashes[str(gate_path)] = checked(gate_path, report['gate_sha256'])
    gate = read(gate_path)
    assert gate['status'] == 'passed' and gate['checkpoint'] == checkpoint and gate['identity'] == identity['model']
    assert gate['plan_sha256'] == FROZEN_PLAN_SHA256
    for flag in ('no_reference_in_features', 'original_single_pass_exact', 'strict_ltr_inputs_and_argmax',
                 'first_query_reuses_prior', 'observed_context_unchanged', 'native_bridge_inputs_verified', 'fp32_frozen_model'):
        assert gate[flag] is True
    return report, hashes


def summarize(out):
    HASH_CACHE.clear()
    checked(out / 'plan.json', FROZEN_PLAN_SHA256)
    plan = read(out / 'plan.json')
    assert set(plan['checkpoints']) == set(CHECKPOINTS) and plan['policies'] == ['left_to_right']
    assert plan['reused_controls'] == list(BASELINES)
    for field in ('source_sha256', 'external_source_sha256', 'parent_evidence_sha256'):
        for path, digest in plan[field].items():
            checked(path, digest)
    evidence = {str(out / 'plan.json'): FROZEN_PLAN_SHA256}
    for filename, field in (('rows.json', 'rows_sha256'), ('data_plan.json', 'data_plan_sha256')):
        evidence[str(out / filename)] = checked(out / filename, plan[field])
    checked(Path(plan['parent']) / 'rows.json', plan['parent_rows_sha256'])
    checked(Path(plan['parent']) / 'plan.json', plan['parent_plan_sha256'])
    assert plan['rows_sha256'] == plan['parent_rows_sha256']
    rows, data = read(out / 'rows.json'), read(out / 'data_plan.json')
    assert len(rows) == 5254 and data['rows_identical_to_parent'] and data['counts_by_mode'] == plan['counts_by_mode']
    lengths = [len(row['mask_h']) for row in rows]
    batches = [lengths[start:start+24] for start in range(0, len(rows), 24)]
    cost = dict(logical_sample_forwards=sum(lengths), physical_sample_forwards=sum(len(batch)*max(batch) for batch in batches),
        physical_batch_forwards=sum(max(batch) for batch in batches), batches=len(batches))
    assert data['cost_per_checkpoint'] == cost
    for mode in MODES:
        subset = [row for row in rows if row['mode'] == mode]
        sizes = [len(row['mask_h']) for row in subset]
        assert data['mode_stats'][mode] == dict(n=len(subset), components=len({row['group'] for row in subset}),
            logical_sample_forwards=sum(sizes), mean_logical_nfe=sum(sizes)/len(sizes), min_length=min(sizes), max_length=max(sizes))
    assert not (out / 'selection.json').exists()
    for filename in ('cpu_tests.json', 'cpu_tests.log', 'gate_cpu_112000.json', 'gate_cpu_112000.log',
                     'independent_preflight.json', 'synthetic_audit_validation.json', 'firstbatch_audit.json'):
        artifact = out / filename
        if artifact.exists():
            evidence[str(artifact)] = checked(artifact)
    if (out / 'synthetic_audit_validation.json').exists():
        validation = read(out / 'synthetic_audit_validation.json')
        assert validation['status'] == 'passed' and validation['model_quality_result'] is False
        assert validation['auditor_sha256'] == plan['source_sha256'][str(out / 'source/audit_cdr_left_to_right.py')]
    if (out / 'firstbatch_audit.json').exists():
        validation = read(out / 'firstbatch_audit.json')
        assert validation['status'] == 'passed' and validation['plan_sha256'] == FROZEN_PLAN_SHA256
        assert validation['frozen_auditor_sha256'] == plan['source_sha256'][str(out / 'source/audit_cdr_left_to_right.py')]
        assert {item['checkpoint'] for item in validation['checkpoints']} == set(CHECKPOINTS)
        for item in validation['checkpoints']:
            assert item['status'] == 'passed' and item['n'] == 24
            for flag in ('actual_model_trace', 'source_reference_recovered', 'strict_history_and_context_reconstructed',
                         'selected_argmax_exact', 'prior_bitwise_equal_to_parent', 'corrupted_prefix_rejected'):
                assert item[flag] is True
            for prefix in ('source', 'parent'):
                checked(item[prefix + '_npz'], item[prefix + '_npz_sha256'])
    jobs, hashes = validate_jobs(out)
    evidence.update(hashes)
    audits, table, primary, descriptive, changes, non_aa = {}, [], [], [], [], []
    for checkpoint in CHECKPOINTS:
        audit, hashes = validate_report(out, checkpoint, plan, data)
        audits[checkpoint] = audit
        evidence.update(hashes)
        for mode in MODES:
            result = audit['results']['oas_new:' + mode]
            table.append(dict(checkpoint=checkpoint, mode=mode, n=result['n'], groups=result['groups'],
                aar_percent=result['mean_sequence_aar'], logical_nfe=result['logical_mean_nfe'],
                left_to_right_physical_mean_forwards=result['left_to_right_physical_mean_forwards']))
            changes.append(dict(checkpoint=checkpoint, mode=mode, versus_baselines=result['left_to_right_changes_vs_baseline']))
            non_aa.append(dict(checkpoint=checkpoint, mode=mode, policies=result['non_amino_acid_outputs']))
            for name, contrast in result['contrasts'].items():
                entry = dict(checkpoint=checkpoint, mode=mode, contrast=name, **contrast)
                (primary if contrast['role'] == 'primary_followup' else descriptive).append(entry)
    assert len(table) == len(changes) == len(non_aa) == 6 and len(primary) == 2
    directions = {row['checkpoint']: row['adjusted_direction'] for row in primary}
    both_better = all(value == 'benefit' for value in directions.values())
    both_worse = all(value == 'harm' for value in directions.values())
    report = dict(status='complete', outcome='both_checkpoints_improve_vs_single_pass' if both_better else
        'both_checkpoints_worsen_vs_single_pass' if both_worse else 'no_consistent_adjusted_effect_vs_single_pass',
        no_selection=True, winner_selected=False, analysis_scope=plan['sample_use'], jobs=jobs,
        table=table, primary_h3=primary, descriptive=descriptive, change_counts=changes, non_amino_acid_outputs=non_aa,
        primary_conclusion=dict(adjusted_direction_by_checkpoint=directions,
            both_checkpoints_improve_vs_single_pass=both_better, both_checkpoints_worsen_vs_single_pass=both_worse),
        counts=dict(rows_per_checkpoint=5254, counts_by_mode=plan['counts_by_mode'], policies_new=1, policies_compared=6,
            total_new_predictions=sum(audits[cp]['predictions'] for cp in CHECKPOINTS),
            total_compared_predictions=sum(audits[cp]['compared_predictions'] for cp in CHECKPOINTS),
            total_logical_sample_forwards=sum(audits[cp]['logical_sample_forwards'] for cp in CHECKPOINTS),
            total_physical_sample_forwards=sum(audits[cp]['physical_sample_forwards'] for cp in CHECKPOINTS),
            total_physical_batch_forwards=sum(audits[cp]['physical_batch_forwards'] for cp in CHECKPOINTS),
            total_inactive_row_sample_forwards=sum(audits[cp]['inactive_row_sample_forwards'] for cp in CHECKPOINTS),
            gate_cost_separate=True, additional_unused_reserve_rows_consumed=0),
        per_checkpoint_cost=cost, decoding=plan['decoding'], token_rule=plan['token_rule'],
        budget_definition=plan['budget_definition'], bootstrap=plan['bootstrap'],
        multiplicity_interpretation=dict(per_comparison_adjusted_confidence=.975,
            local_family_size=2, nominal_local_family_coverage_at_least=.95,
            note='97.5% per-comparison intervals target at least 95% nominal simultaneous coverage across the two local H3 tests. Bootstrap calibration is approximate; historical adaptive exploration is not covered.'),
        inferential_rule=plan['inferential_rule'], secondary_rule=plan['secondary_rule'],
        evidence_sha256=evidence, script_sha256=sha(Path(__file__)), limitations=plan['limitations'] + [
            'Actual saved virtual inputs and selected probabilities reconstruct every greedy step. Native model input mapping additionally passed the implementation gate.',
            'Frozen legal EOS/separator tokens remain eligible; fixed-length decoding does not stop when they occur, and non-AA predictions are counted explicitly.',
            'First-forward target probabilities exactly match the parent run, and existing baselines retain their original per-example predictions.',
            'The experiment reports logical and padded physical model-forward counts, not a wall-clock speed comparison.',
        ])
    path = out / 'left_to_right_delivery.json'
    if path.exists():
        assert read(path) == report, 'Existing delivery differs; refusing to overwrite'
    else:
        with path.open('x') as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'outcome', 'counts', 'primary_h3')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-code', action='store_true')
    args = parser.parse_args()
    if args.check_code:
        ast.parse(Path(__file__).read_text())
        assert len(FROZEN_PLAN_SHA256) == 64 and set(FROZEN_PLAN_SHA256) <= set('0123456789abcdef')
        print('PASS: syntax and fixed protocol hash; no experiment outputs read or written')
        return
    assert args.out is not None, '--out required'
    summarize(args.out)


if __name__ == '__main__':
    main()
