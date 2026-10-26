"""Activate pllm; python -B FILE --out ABS --checkpoint 112000|50000.

Independently reconstruct strict left-to-right greedy decoding from actual saved
query inputs and selected forward probabilities. All prior baselines are fixed;
this reused-cohort follow-up never selects a new policy or consumes reserve data.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


BASELINES = ('single_pass', 'original4', 'original8', 'grouped4', 'grouped8')
AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def probability_check(value):
    assert value.dtype == np.float32
    assert np.isfinite(value).all() and (value >= 0).all()
    assert np.allclose(value.sum(-1), 1., atol=1e-5)


def frozen_vocabulary_check(value, token):
    allowed = {token[aa] for aa in AMINO_ACIDS} | {token['<protd>'], token['.']}
    forbidden = sorted(set(range(len(token))) - allowed)
    assert (value[..., forbidden] == 0).all(), 'A forbidden predictor token has nonzero probability'


def initial_virtual(row, token):
    """Independently reconstruct the documented 150/128 virtual chain layout."""
    values = np.full(278, token['<protd>'], dtype=np.int64)
    for sequence, positions, base, limit in ((row['heavy'], row['mask_h'], 0, 148),
                                            (row['light'], row['mask_l'], 150, 126)):
        assert len(sequence) <= limit
        values[base] = token['<prots>']
        values[base+1:base+1+len(sequence)] = [token[aa] for aa in sequence]
        for position in positions:
            values[base+1+position] = token['<mask>']
    return values


def read_ltr_predictions(path, rows, references, vocabulary_size):
    by_id = {row['id']: row for row in rows}
    records = {}
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            key = record['id']
            assert key in by_id and key not in records and record['policy'] == 'left_to_right'
            for field in ('group', 'dataset', 'mode', 'fold'):
                assert record[field] == by_id[key][field]
            reference = references[key]
            assert record['reference_ids'] == reference
            assert record['length'] == record['logical_nfe'] == len(reference) == len(record['prediction_ids'])
            assert all(type(value) is int and 0 <= value < vocabulary_size for value in record['prediction_ids'])
            assert record['correct'] == sum(a == b for a, b in zip(record['prediction_ids'], reference))
            records[key] = record
    assert set(records) == set(by_id)
    return records


def reconstruct_trajectories(folder, parent_folder, rows, cases, token, vocabulary_size, batch_size, parent_hashes):
    by_id = {row['id']: row for row in rows}
    expected_keys = {'ids', 'lengths', 'positions', 'prior', 'query_inputs', 'selected_probs', 'selected_positions'}
    seen, hashes, parent_replay, physical_by_id = set(), {}, {}, {}
    physical_batches = physical_samples = logical_samples = 0
    for path in sorted(folder.glob('batch_*.npz')):
        hashes[str(path)] = sha(path)
        old_path = parent_folder / path.name
        assert sha(old_path) == parent_hashes[str(old_path)]
        parent_replay[str(old_path)] = parent_hashes[str(old_path)]
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == expected_keys
            arrays = {name: archive[name] for name in archive.files}
        with np.load(old_path, allow_pickle=False) as archive:
            previous = {name: archive[name] for name in ('ids', 'lengths', 'positions', 'prior')}
        ids, lengths = arrays['ids'].tolist(), arrays['lengths']
        assert ids == previous['ids'].tolist()
        for field in ('lengths', 'positions', 'prior'):
            assert np.array_equal(arrays[field], previous[field]), (path.name, field, 'parent replay')
        assert 0 < len(ids) <= batch_size and lengths.shape == (len(ids),)
        assert np.issubdtype(lengths.dtype, np.integer) and (lengths > 0).all()
        steps = int(lengths.max())
        assert arrays['positions'].shape == (len(ids), steps)
        assert arrays['prior'].shape == (len(ids), steps, vocabulary_size)
        assert arrays['query_inputs'].shape == (steps, len(ids), 278)
        assert arrays['selected_probs'].shape == (steps, len(ids), vocabulary_size)
        assert arrays['selected_positions'].shape == (steps, len(ids))
        assert arrays['prior'].dtype == arrays['selected_probs'].dtype == np.float32
        for field in ('positions', 'query_inputs', 'selected_positions'):
            assert np.issubdtype(arrays[field].dtype, np.integer)
        state = np.stack([initial_virtual(by_id[key], token) for key in ids])
        initial = state.copy()
        target = np.zeros_like(state, dtype=bool)
        for i, (key, length) in enumerate(zip(ids, lengths.tolist())):
            assert key not in seen and key in by_id
            seen.add(key)
            expected_positions = [position+1 for position in by_id[key]['mask_h']]
            assert expected_positions == sorted(expected_positions) and len(expected_positions) == length
            assert arrays['positions'][i, :length].tolist() == expected_positions
            assert (arrays['positions'][i, length:] == -1).all()
            target[i, expected_positions] = True
            assert (initial[i, expected_positions] == token['<mask>']).all()
            probability_check(arrays['prior'][i, :length])
            frozen_vocabulary_check(arrays['prior'][i, :length], token)
            assert (arrays['prior'][i, length:] == 0).all()
            assert arrays['prior'][i, :length].argmax(-1).tolist() == cases['single_pass'][key]['prediction_ids']
            physical_by_id[key] = steps
        for query in range(steps):
            actual_inputs = arrays['query_inputs'][query]
            assert np.array_equal(actual_inputs, state), (path.name, query, 'actual input history')
            assert np.array_equal(actual_inputs[~target], initial[~target]), (path.name, query, 'observed context')
            for i, (key, length) in enumerate(zip(ids, lengths.tolist())):
                position = arrays['selected_positions'][query, i]
                probabilities = arrays['selected_probs'][query, i]
                if query >= length:
                    assert position == -1 and (probabilities == 0).all()
                    continue
                expected_position = arrays['positions'][i, query]
                assert position == expected_position
                current_and_future = arrays['positions'][i, query:length]
                assert (actual_inputs[i, current_and_future] == token['<mask>']).all()
                probability_check(probabilities)
                frozen_vocabulary_check(probabilities, token)
                if query == 0:
                    assert np.array_equal(probabilities, arrays['prior'][i, 0]), (path.name, key, 'first query uses prior')
                chosen = int(probabilities.argmax(-1))
                assert chosen == cases['left_to_right'][key]['prediction_ids'][query]
                state[i, position] = chosen
        for i, (key, length) in enumerate(zip(ids, lengths.tolist())):
            assert state[i, arrays['positions'][i, :length]].tolist() == cases['left_to_right'][key]['prediction_ids']
        physical_batches += steps
        physical_samples += steps * len(ids)
        logical_samples += int(lengths.sum())
    assert seen == set(by_id) and set(parent_replay) == set(parent_hashes)
    return dict(posterior_sha256=hashes, parent_replayed_posterior_sha256=parent_replay,
        physical_batch_forwards=physical_batches, physical_sample_forwards=physical_samples,
        logical_sample_forwards=logical_samples, inactive_row_sample_forwards=physical_samples-logical_samples,
        physical_by_id=physical_by_id)


def statistics(rows, cases, mode, repeats, seed, vocabulary, physical_by_id):
    names = sorted(cases)
    index = {name: i for i, name in enumerate(names)}
    groups = {group: i for i, group in enumerate(sorted({row['group'] for row in rows}))}
    counts = np.zeros(len(groups))
    sums = np.zeros((len(groups), len(names)))
    for row in rows:
        group = groups[row['group']]
        counts[group] += 1
        for name, column in index.items():
            record = cases[name][row['id']]
            sums[group, column] += 100 * record['correct'] / record['length']
    coefficients = np.zeros((len(names), len(BASELINES)))
    for column, baseline in enumerate(BASELINES):
        coefficients[index['left_to_right'], column] = 1
        coefficients[index[baseline], column] = -1
    means = sums.sum(0) / len(rows)
    draws = np.empty((repeats, len(BASELINES)))
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(groups), 1 / len(groups))
    for start in range(0, repeats, 256):
        weights = rng.multinomial(len(groups), probabilities, size=min(256, repeats-start))
        sampled = (weights @ sums) / (weights @ counts)[:, None]
        draws[start:start + len(weights)] = sampled @ coefficients
    point = means @ coefficients
    contrasts = {}
    for column, baseline in enumerate(BASELINES):
        primary = mode == 'cdrh3' and baseline == 'single_pass'
        result = dict(delta_pp=float(point[column]), ci95_pp=np.percentile(draws[:, column], [2.5, 97.5]).tolist(),
            positive='left_to_right', negative=baseline, role='primary_followup' if primary else 'descriptive')
        if primary:
            interval = np.percentile(draws[:, column], [1.25, 98.75]).tolist()
            result.update(bonferroni_family_tests=2, per_comparison_adjusted_confidence=.975,
                nominal_family_coverage_at_least=.95, ci_bonferroni_pp=interval,
                adjusted_direction='benefit' if interval[0] > 0 else 'harm' if interval[1] < 0 else 'inconclusive')
        contrasts['left_to_right_vs_' + baseline] = result
    changes = {}
    for baseline in BASELINES:
        record = dict(changed_sequences=0, changed_residues=0, wrong_to_right=0, right_to_wrong=0,
            changed_wrong_to_wrong=0, sequences_improved=0, sequences_worsened=0, sequences_equal_aar=0)
        for row in rows:
            before, after = cases[baseline][row['id']], cases['left_to_right'][row['id']]
            record['changed_sequences'] += before['prediction_ids'] != after['prediction_ids']
            delta = after['correct'] - before['correct']
            record['sequences_improved' if delta > 0 else 'sequences_worsened' if delta < 0 else 'sequences_equal_aar'] += 1
            for old, new, ref in zip(before['prediction_ids'], after['prediction_ids'], before['reference_ids']):
                if old != new:
                    record['changed_residues'] += 1
                    record['wrong_to_right' if new == ref else 'right_to_wrong' if old == ref else 'changed_wrong_to_wrong'] += 1
        changes[baseline] = record
    non_amino_acid = {}
    for name in names:
        tokens = Counter()
        affected = 0
        for row in rows:
            invalid = [vocabulary[value] for value in cases[name][row['id']]['prediction_ids'] if vocabulary[value] not in AMINO_ACIDS]
            tokens.update(invalid)
            affected += bool(invalid)
        non_amino_acid[name] = dict(residues=sum(tokens.values()), sequences=affected, token_counts=dict(tokens))
    return dict(n=len(rows), groups=len(groups), mean_sequence_aar={name: float(means[i]) for i, name in enumerate(names)},
        logical_mean_nfe={name: float(np.mean([cases[name][row['id']]['logical_nfe'] for row in rows])) for name in names},
        left_to_right_physical_mean_forwards=float(np.mean([physical_by_id[row['id']] for row in rows])),
        contrasts=contrasts, left_to_right_changes_vs_baseline=changes, non_amino_acid_outputs=non_amino_acid)


def verify_context(out, checkpoint):
    plan = read(out / 'plan.json')
    for field in ('source_sha256', 'external_source_sha256', 'parent_evidence_sha256'):
        for path, digest in plan[field].items():
            assert sha(path) == digest, ('changed frozen dependency', path)
    parent = Path(plan['parent'])
    ablation = Path(plan['ablation_parent'])
    assert sha(parent / 'plan.json') == plan['parent_plan_sha256']
    assert sha(parent / 'rows.json') == plan['parent_rows_sha256']
    assert sha(out / 'rows.json') == plan['rows_sha256'] == plan['parent_rows_sha256']
    assert sha(out / 'data_plan.json') == plan['data_plan_sha256']
    parent_plan = read(parent / 'plan.json')
    assert parent_plan['checkpoints'] == plan['checkpoints']
    assert parent_plan['batch_size'] == plan['batch_size'] == 24
    assert parent_plan['counts_by_mode'] == plan['counts_by_mode'] == {'cdrh1': 526, 'cdrh2': 526, 'cdrh3': 4202}
    rows = read(out / 'rows.json')
    assert len(rows) == 5254 and len({row['id'] for row in rows}) == len(rows)
    assert all(row['split'] == 'ablation' and row['parent_split'] == 'reserve' for row in rows)
    assert plan['policy'] == 'left_to_right' and plan['policies'] == ['left_to_right']
    assert plan['reused_controls'] == list(BASELINES)
    assert plan['bootstrap']['repeats'] == 50000 and plan['bootstrap']['seed'] == 20261016 and plan['bootstrap']['family_size'] == 2
    parent_path = parent / f'evaluation_{checkpoint}/independent_audit.json'
    ablation_path = ablation / f'evaluation_{checkpoint}/independent_audit.json'
    assert sha(parent_path) == plan['parent_audit_sha256'][checkpoint]
    assert sha(ablation_path) == plan['ablation_audit_sha256'][checkpoint]
    parent_audit, ablation_audit = read(parent_path), read(ablation_path)
    assert parent_audit['status'] == ablation_audit['status'] == 'passed'
    assert parent_audit['checkpoint'] == ablation_audit['checkpoint'] == checkpoint
    assert parent_audit['n'] == ablation_audit['n'] == len(rows)
    assert parent_audit['plan_sha256'] == plan['parent_plan_sha256']
    baseline_helper = parent / 'source/audit_cdr_block_budget.py'
    reference_helper = Path(plan['block_parent']) / 'source/audit_cdr_block_context.py'
    for path in (baseline_helper, reference_helper):
        assert plan['external_source_sha256'][str(path)] == sha(path)
    return plan, rows, parent_audit, ablation_audit, module(baseline_helper, 'ltr_baseline_auditor'), module(reference_helper, 'ltr_reference_auditor')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('112000', '50000'), required=True)
    args = parser.parse_args()
    out = args.out
    plan, rows, parent_audit, ablation_audit, baseline_helper, reference_helper = verify_context(out, args.checkpoint)
    folder = out / ('evaluation_' + args.checkpoint)
    parent_folder = Path(plan['parent']) / ('evaluation_' + args.checkpoint)
    ablation_folder = Path(plan['ablation_parent']) / ('evaluation_' + args.checkpoint)
    summary = read(folder / 'summary.json')
    assert summary['status'] == 'passed' and summary['n'] == len(rows)
    assert summary['plan_sha256'] == sha(out / 'plan.json') and summary['gate_cost_separate']
    assert summary['prior_exact_to_parent'] is True
    identity = read(folder / 'identity.json')
    assert identity['model']['checkpoint'] == plan['checkpoints'][args.checkpoint]
    assert identity['plan_sha256'] == sha(out / 'plan.json') and identity['phase'] == 'left_to_right'
    vocabulary = read(Path(plan['checkpoints'][args.checkpoint]) / 'vocabulary.json')
    token = {aa: i for i, aa in enumerate(vocabulary)}
    assert len(token) == len(vocabulary)
    references, source_hashes = reference_helper.recover_references(rows, token)
    assert source_hashes == parent_audit['source_data_sha256'] == ablation_audit['source_data_sha256']
    parent_predictions = parent_folder / 'predictions.jsonl'
    ablation_predictions = ablation_folder / 'predictions.jsonl'
    assert sha(parent_predictions) == parent_audit['predictions_sha256']
    assert sha(ablation_predictions) == ablation_audit['predictions_sha256']
    cases = baseline_helper.read_records(parent_predictions, rows, references, {'single_pass', 'grouped4', 'grouped8'})
    cases.update(baseline_helper.read_records(ablation_predictions, rows, references, {'original4', 'original8'}, retained={'original4', 'original8'}))
    cases['left_to_right'] = read_ltr_predictions(folder / 'predictions.jsonl', rows, references, len(vocabulary))
    trajectory = reconstruct_trajectories(folder, parent_folder, rows, cases, token,
        len(vocabulary), plan['batch_size'], parent_audit['posterior_sha256'])
    for field in ('physical_batch_forwards', 'physical_sample_forwards', 'logical_sample_forwards'):
        assert summary[field] == trajectory[field]
    assert trajectory['logical_sample_forwards'] == sum(len(row['mask_h']) for row in rows)
    assert set(summary['mean_sequence_aar']) == {'left_to_right'}
    results = {}
    for mode in ('cdrh1', 'cdrh2', 'cdrh3'):
        subset = [row for row in rows if row['mode'] == mode]
        key = 'oas_new:' + mode
        assert all(row['dataset'] == 'oas_new' for row in subset)
        result = statistics(subset, cases, mode, plan['bootstrap']['repeats'], plan['bootstrap']['seed'],
                            vocabulary, trajectory['physical_by_id'])
        assert result['n'] == plan['counts_by_mode'][mode] and result['groups'] == (4096 if mode == 'cdrh3' else 512)
        assert abs(result['mean_sequence_aar']['left_to_right'] - summary['mean_sequence_aar']['left_to_right'][key]) < 1e-8
        for name in BASELINES:
            assert abs(result['mean_sequence_aar'][name] - parent_audit['results'][key]['mean_sequence_aar'][name]) < 1e-8
        results[key] = result
    assert set(summary['mean_sequence_aar']['left_to_right']) == set(results)
    gate_path = out / ('gate_cuda_' + args.checkpoint + '.json')
    gate = read(gate_path)
    assert gate['status'] == 'passed' and gate['checkpoint'] == args.checkpoint
    assert gate['plan_sha256'] == sha(out / 'plan.json') and gate['identity'] == identity['model']
    for flag in ('no_reference_in_features', 'original_single_pass_exact', 'strict_ltr_inputs_and_argmax',
                 'first_query_reuses_prior', 'observed_context_unchanged', 'native_bridge_inputs_verified', 'fp32_frozen_model'):
        assert gate[flag] is True
    report = dict(status='passed', checkpoint=args.checkpoint, n=len(rows), policies=1, predictions=len(rows),
        policies_compared=6, compared_predictions=sum(map(len, cases.values())), results=results, winner_selected=False,
        source_reference_recovered=True, observed_context_unchanged=True, strict_left_to_right_history_reconstructed=True,
        current_and_future_targets_remain_masked=True, selected_forward_probabilities_reconstructed=True,
        prior_bitwise_equal_to_parent=True, first_query_reuses_prior=True,
        forbidden_predictor_probabilities_are_zero=True,
        vocabulary_policy_evidence='Frozen predictor/sampler sources and implementation gates retain the existing legal vocabulary; zero allowed probabilities are not rejected because FP32 underflow is possible.',
        decoding_exactly_target_length_without_early_stop=True, no_additional_amino_acid_restriction=True,
        actual_query_input_rows_audited=trajectory['physical_sample_forwards'],
        logical_sample_forwards=trajectory['logical_sample_forwards'], physical_sample_forwards=trajectory['physical_sample_forwards'],
        physical_batch_forwards=trajectory['physical_batch_forwards'], inactive_row_sample_forwards=trajectory['inactive_row_sample_forwards'],
        gate_cost_separate=True, posterior_sha256=trajectory['posterior_sha256'],
        parent_replayed_posterior_sha256=trajectory['parent_replayed_posterior_sha256'], source_data_sha256=source_hashes,
        predictions_sha256=sha(folder / 'predictions.jsonl'), parent_predictions_sha256=sha(parent_predictions),
        ablation_predictions_sha256=sha(ablation_predictions), summary_sha256=sha(folder / 'summary.json'),
        identity_sha256=sha(folder / 'identity.json'), gate_sha256=sha(gate_path),
        vocabulary_sha256=sha(Path(plan['checkpoints'][args.checkpoint]) / 'vocabulary.json'),
        parent_audit_sha256=plan['parent_audit_sha256'][args.checkpoint],
        ablation_audit_sha256=plan['ablation_audit_sha256'][args.checkpoint], plan_sha256=sha(out / 'plan.json'),
        auditor_sha256=sha(Path(__file__)), bootstrap={**plan['bootstrap'], 'paired': True, 'interval': 'percentile',
            'per_comparison_adjusted_confidence': .975, 'nominal_family_coverage_at_least': .95},
        primary_interpretation='Left-to-right minus single-pass H3 at two checkpoints forms the fixed two-test family; each adjusted interval is 97.5%, targeting at least 95% nominal local family coverage.',
        limitations=[
            'This fixed follow-up reuses previously evaluated ablation rows and consumes no unused reserve; it is not independent unseen confirmation.',
            'Observed framework/light-chain context remains available bidirectionally; only masked target residues are filled left to right. This does not change model training or make the backbone autoregressive.',
            'Logical cost equals target length; dense batches execute max target length for every row, including inactive rows. The first query reuses its saved initial forward, with no extra prior call.',
            'The frozen predictor vocabulary is retained, including allowed special tokens. Decoding fills all target positions without early stopping, and non-amino-acid outputs are reported.',
            'All non-primary comparisons are descriptive 95% intervals. Local correction does not control all historical adaptive exploration; percentile-bootstrap coverage is approximate.',
            'Differences from other methods combine decoding order, context policy and compute budget; they do not isolate forward count alone.',
        ])
    destination = folder / 'independent_audit.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='passed', checkpoint=args.checkpoint, predictions=report['predictions'],
        primary_h3=results['oas_new:cdrh3']['contrasts']['left_to_right_vs_single_pass'],
        mean_sequence_aar={key: result['mean_sequence_aar'] for key, result in results.items()},
        logical_sample_forwards=report['logical_sample_forwards'], physical_sample_forwards=report['physical_sample_forwards']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
