"""Activate pllm; python -B FILE --out ABS --checkpoint 112000|50000.

Audit the fixed four-versus-eight-forward grouped-context comparison on already
used ablation rows. Replay the prior/four-forward outputs exactly and reuse only
hash-bound parent controls. This is follow-up analysis, not fresh confirmation.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def module(path):
    spec = importlib.util.spec_from_file_location('frozen_budget_reference_helper', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def probability_check(value):
    assert value.dtype == np.float32
    assert np.isfinite(value).all() and (value >= 0).all()
    assert np.allclose(value.sum(-1), 1., atol=1e-5)


def read_records(path, rows, references, expected_names, retained=None):
    """Check every retained prediction against independently recovered references."""
    by_id = {row['id']: row for row in rows}
    cases = defaultdict(dict)
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            policy, key = record['policy'], record['id']
            if retained is not None and policy not in retained:
                continue
            assert policy in expected_names and key in by_id
            assert key not in cases[policy]
            row, reference = by_id[key], references[key]
            for field in ('group', 'dataset', 'mode', 'fold'):
                assert record[field] == row[field]
            assert record['reference_ids'] == reference
            assert record['length'] == len(reference) == len(record['prediction_ids'])
            assert all(type(token) is int for token in record['prediction_ids'])
            assert record['correct'] == sum(a == b for a, b in zip(record['prediction_ids'], reference))
            nfe = 1 if policy == 'single_pass' else 8 if policy in ('grouped8', 'original8') else 4
            assert record['logical_nfe'] == nfe
            cases[policy][key] = record
    assert set(cases) == set(expected_names)
    assert all(set(records) == set(by_id) for records in cases.values())
    return cases


def reconstruct_arrays(folder, parent_folder, rows, cases, families, vocabulary_size, batch_size, parent_hashes):
    by_id = {row['id']: row for row in rows}
    expected = {'ids', 'lengths', 'positions', 'prior'}
    for family in families:
        name = family['name']
        expected.update((name, name + '__weights', name + '__queries'))
    seen, hashes, inherited = set(), {}, {}
    weights_checked = 0
    for path in sorted(folder.glob('batch_*.npz')):
        hashes[str(path)] = sha(path)
        previous_path = parent_folder / path.name
        assert sha(previous_path) == parent_hashes[str(previous_path)]
        inherited[str(previous_path)] = parent_hashes[str(previous_path)]
        # Materialize each member once: NpzFile indexing repeatedly decompresses.
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == expected
            arrays = {name: archive[name] for name in archive.files}
        with np.load(previous_path, allow_pickle=False) as archive:
            previous = {name: archive[name] for name in ('ids', 'lengths', 'positions', 'prior',
                'soft_frozen', 'soft_frozen__weights', 'soft_frozen__queries')}
        ids, lengths = arrays['ids'].tolist(), arrays['lengths']
        assert ids == previous['ids'].tolist()
        for field in ('lengths', 'positions', 'prior'):
            assert np.array_equal(arrays[field], previous[field]), (path.name, field, 'parent exact replay')
        assert np.array_equal(arrays['grouped4'], previous['soft_frozen']), (path.name, 'posterior exact replay')
        assert np.array_equal(arrays['grouped4__weights'], previous['soft_frozen__weights']), (path.name, 'feedback weight exact replay')
        assert np.array_equal(arrays['grouped4__queries'], previous['soft_frozen__queries']), (path.name, 'query probability exact replay')
        assert 0 < len(ids) <= batch_size and lengths.shape == (len(ids),)
        assert np.issubdtype(lengths.dtype, np.integer) and (lengths > 0).all()
        width = int(lengths.max())
        shape = (len(ids), width, vocabulary_size)
        assert arrays['prior'].shape == shape and arrays['prior'].dtype == np.float32
        assert arrays['positions'].shape == shape[:2] and np.issubdtype(arrays['positions'].dtype, np.integer)
        for family in families:
            name, groups = family['name'], family['groups']
            assert arrays[name].shape == shape and arrays[name].dtype == np.float32
            for suffix in ('__weights', '__queries'):
                assert arrays[name + suffix].shape == (groups,) + shape
                assert arrays[name + suffix].dtype == np.float32
        for i, (key, length) in enumerate(zip(ids, lengths.tolist())):
            assert key in by_id and key not in seen
            seen.add(key)
            assert length == len(by_id[key]['mask_h'])
            assert arrays['positions'][i, :length].tolist() == [p + 1 for p in by_id[key]['mask_h']]
            assert (arrays['positions'][i, length:] == -1).all()
            prior = arrays['prior'][i, :length]
            probability_check(prior)
            assert (arrays['prior'][i, length:] == 0).all()
            assert prior.argmax(-1).tolist() == cases['single_pass'][key]['prediction_ids']
            for family in families:
                name, groups = family['name'], family['groups']
                posterior = arrays[name][i, :length]
                probability_check(posterior)
                assert (arrays[name][i, length:] == 0).all()
                expected_weights = prior ** (1 / family['temperature'])
                expected_weights /= expected_weights.sum(-1, keepdims=True)
                merged = prior.copy()
                counts = np.zeros(length, dtype=int)
                for query in range(groups):
                    scored = np.arange(length) % groups == query
                    feedback = ~scored
                    actual = arrays[name + '__weights'][query, i, :length]
                    predictions = arrays[name + '__queries'][query, i, :length]
                    probability_check(predictions)
                    assert np.isfinite(actual).all() and (actual >= 0).all()
                    assert (actual[scored] == 0).all()
                    assert (arrays[name + '__weights'][query, i, length:] == 0).all()
                    assert (arrays[name + '__queries'][query, i, length:] == 0).all()
                    assert np.allclose(actual[feedback], expected_weights[feedback], atol=1e-6, rtol=1e-5), (key, name, query)
                    assert np.allclose(actual[feedback].sum(-1), 1., atol=1e-5)
                    weights_checked += int(feedback.sum())
                    merged[scored] = predictions[scored]
                    counts += scored
                assert (counts == 1).all()
                assert np.array_equal(merged, posterior), (key, name, 'query assembly')
                weight = family['prior_weight']
                assert (weight * prior + (1-weight) * posterior).argmax(-1).tolist() == cases[name][key]['prediction_ids']
    assert seen == set(by_id)
    assert set(inherited) == set(parent_hashes), 'The follow-up must replay every parent batch'
    return hashes, inherited, weights_checked


def bootstrap_results(rows, cases, mode, repeats, seed):
    names = sorted(cases)
    index = {name: i for i, name in enumerate(names)}
    groups = {group: i for i, group in enumerate(sorted({r['group'] for r in rows}))}
    counts = np.zeros(len(groups))
    sums = np.zeros((len(groups), len(names)))
    for row in rows:
        group = groups[row['group']]
        counts[group] += 1
        for name, column in index.items():
            record = cases[name][row['id']]
            sums[group, column] += 100 * record['correct'] / record['length']
    comparisons = [('grouped8', 'grouped4')]
    comparisons += [(candidate, baseline) for candidate in ('grouped4', 'grouped8')
                    for baseline in ('single_pass', 'original4', 'original8')]
    coefficients = np.zeros((len(names), len(comparisons)))
    for j, (candidate, baseline) in enumerate(comparisons):
        coefficients[index[candidate], j] = 1
        coefficients[index[baseline], j] = -1
    means = sums.sum(0) / len(rows)
    boot = np.empty((repeats, len(comparisons)))
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(groups), 1 / len(groups))
    for start in range(0, repeats, 256):
        weights = rng.multinomial(len(groups), probabilities, size=min(256, repeats-start))
        resampled = (weights @ sums) / (weights @ counts)[:, None]
        boot[start:start + len(weights)] = resampled @ coefficients
    point = means @ coefficients
    contrasts = {}
    for j, (candidate, baseline) in enumerate(comparisons):
        primary = mode == 'cdrh3' and candidate == 'grouped8' and baseline == 'grouped4'
        result = dict(delta_pp=float(point[j]), ci95_pp=np.percentile(boot[:, j], [2.5, 97.5]).tolist(),
            positive=candidate, negative=baseline, role='primary_followup' if primary else 'descriptive')
        if primary:
            interval = np.percentile(boot[:, j], [1.25, 98.75]).tolist()
            result.update(bonferroni_family_tests=2, per_comparison_adjusted_confidence=.975,
                nominal_family_coverage_at_least=.95, ci_bonferroni_pp=interval,
                adjusted_direction='benefit' if interval[0] > 0 else 'harm' if interval[1] < 0 else 'inconclusive')
        contrasts[candidate + '_vs_' + baseline] = result
    changed = dict(changed_sequences=0, changed_residues=0, wrong_to_right=0, right_to_wrong=0,
                   changed_wrong_to_wrong=0, sequences_improved=0, sequences_worsened=0, sequences_equal_aar=0)
    for row in rows:
        old, new = cases['grouped4'][row['id']], cases['grouped8'][row['id']]
        different = old['prediction_ids'] != new['prediction_ids']
        changed['changed_sequences'] += different
        difference = new['correct'] - old['correct']
        changed['sequences_improved' if difference > 0 else 'sequences_worsened' if difference < 0 else 'sequences_equal_aar'] += 1
        for before, after, reference in zip(old['prediction_ids'], new['prediction_ids'], old['reference_ids']):
            if before == after:
                continue
            changed['changed_residues'] += 1
            changed['wrong_to_right' if after == reference else 'right_to_wrong' if before == reference else 'changed_wrong_to_wrong'] += 1
    return dict(n=len(rows), groups=len(groups),
        mean_sequence_aar={name: float(means[i]) for i, name in enumerate(names)},
        logical_mean_nfe={name: float(np.mean([cases[name][row['id']]['logical_nfe'] for row in rows])) for name in names},
        contrasts=contrasts, grouped8_vs_grouped4_changes=changed)


def verify_context(out, checkpoint):
    plan = read(out / 'plan.json')
    for field in ('source_sha256', 'external_source_sha256', 'parent_evidence_sha256'):
        for path, digest in plan[field].items():
            assert sha(path) == digest, ('changed frozen file', path)
    parent = Path(plan['parent'])
    assert sha(parent / 'plan.json') == plan['parent_plan_sha256']
    assert sha(parent / 'rows.json') == plan['parent_rows_sha256']
    assert sha(out / 'rows.json') == plan['rows_sha256'] == plan['parent_rows_sha256']
    assert sha(out / 'data_plan.json') == plan['data_plan_sha256']
    parent_plan = read(parent / 'plan.json')
    assert parent_plan['rows_sha256'] == plan['rows_sha256']
    assert parent_plan['checkpoints'] == plan['checkpoints']
    assert plan['counts_by_mode'] == parent_plan['counts_by_mode'] == {'cdrh1': 526, 'cdrh2': 526, 'cdrh3': 4202}
    rows = read(out / 'rows.json')
    assert len(rows) == 5254 and len({r['id'] for r in rows}) == len(rows)
    assert all(r['split'] == 'ablation' and r['parent_split'] == 'reserve' for r in rows)
    assert plan['bootstrap']['repeats'] == 50000 and plan['bootstrap']['seed'] == 20261016
    assert plan['bootstrap']['family_size'] == 2
    assert plan['reused_controls'] == ['original4', 'original8']
    families = plan['families']
    assert len(families) == 2 and {f['name'] for f in families} == {'grouped4', 'grouped8'}
    for family in families:
        assert family['groups'] == (3 if family['name'] == 'grouped4' else 7)
        assert family['context'] == 'soft' and family['feedback'] == 'frozen'
        assert family['rule'] == 'interleaved' and family['temperature'] == .5
        assert family['strength'] == .5 and family['prior_weight'] == .5
    parent_audit_path = parent / ('evaluation_' + checkpoint) / 'independent_audit.json'
    assert sha(parent_audit_path) == plan['parent_audit_sha256'][checkpoint]
    parent_audit = read(parent_audit_path)
    assert parent_audit['status'] == 'passed' and parent_audit['n'] == len(rows)
    assert parent_audit['checkpoint'] == checkpoint and parent_audit['plan_sha256'] == plan['parent_plan_sha256']
    assert parent_audit['winner_selected'] is False
    helper_path = Path(plan['block_parent']) / 'source/audit_cdr_block_context.py'
    assert plan['external_source_sha256'][str(helper_path)] == sha(helper_path)
    helper = module(helper_path)
    return plan, rows, families, parent_audit, helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('112000', '50000'), required=True)
    args = parser.parse_args()
    out = args.out
    plan, rows, families, parent_audit, helper = verify_context(out, args.checkpoint)
    folder = out / ('evaluation_' + args.checkpoint)
    parent_folder = Path(plan['parent']) / ('evaluation_' + args.checkpoint)
    summary = read(folder / 'summary.json')
    assert summary['status'] == 'passed' and summary['n'] == len(rows)
    assert summary['plan_sha256'] == sha(out / 'plan.json') and summary['gate_cost_separate']
    assert summary['prior_and_grouped4_exact_to_parent'] is True
    identity = read(folder / 'identity.json')
    assert identity['model']['checkpoint'] == plan['checkpoints'][args.checkpoint]
    assert identity['plan_sha256'] == sha(out / 'plan.json') and identity['phase'] == 'budget'
    vocabulary = read(Path(plan['checkpoints'][args.checkpoint]) / 'vocabulary.json')
    token = {aa: i for i, aa in enumerate(vocabulary)}
    assert len(token) == len(vocabulary)
    references, source_hashes = helper.recover_references(rows, token)
    assert source_hashes == parent_audit['source_data_sha256']
    new_names = {'single_pass', 'grouped4', 'grouped8'}
    cases = read_records(folder / 'predictions.jsonl', rows, references, new_names)
    old_path = parent_folder / 'predictions.jsonl'
    assert sha(old_path) == parent_audit['predictions_sha256']
    retained = {'single_pass', 'soft_frozen_prior0.5', 'original4', 'original8'}
    previous = read_records(old_path, rows, references, retained, retained=retained)
    for current, old in (('single_pass', 'single_pass'), ('grouped4', 'soft_frozen_prior0.5')):
        for row in rows:
            now_record, old_record = cases[current][row['id']], previous[old][row['id']]
            assert now_record['prediction_ids'] == old_record['prediction_ids'], ('prediction replay', current, row['id'])
            assert now_record['correct'] == old_record['correct']
    for name in ('original4', 'original8'):
        cases[name] = previous[name]
    arrays, old_arrays, weight_count = reconstruct_arrays(folder, parent_folder, rows, cases,
        families, len(vocabulary), plan['batch_size'], parent_audit['posterior_sha256'])
    query_count = helper.audit_queries(folder, {r['id']: r for r in rows}, families)
    assert query_count == len(rows) * 10
    assert summary['physical_sample_forwards'] == len(rows) * 11
    assert set(summary['mean_sequence_aar']) == new_names
    results = {}
    for mode in ('cdrh1', 'cdrh2', 'cdrh3'):
        subset = [r for r in rows if r['mode'] == mode]
        key = 'oas_new:' + mode
        assert all(r['dataset'] == 'oas_new' for r in subset)
        result = bootstrap_results(subset, cases, mode, plan['bootstrap']['repeats'], plan['bootstrap']['seed'])
        assert result['n'] == plan['counts_by_mode'][mode]
        assert result['groups'] == (4096 if mode == 'cdrh3' else 512)
        for name in new_names:
            assert abs(result['mean_sequence_aar'][name] - summary['mean_sequence_aar'][name][key]) < 1e-8
        for name, old_name in (('single_pass', 'single_pass'), ('grouped4', 'soft_frozen_prior0.5'),
                               ('original4', 'original4'), ('original8', 'original8')):
            assert abs(result['mean_sequence_aar'][name] - parent_audit['results'][key]['mean_sequence_aar'][old_name]) < 1e-8
        assert result['logical_mean_nfe'] == {'single_pass': 1., 'grouped4': 4., 'grouped8': 8., 'original4': 4., 'original8': 8.}
        results[key] = result
    assert all(set(scores) == set(results) for scores in summary['mean_sequence_aar'].values())
    gate_path = out / ('gate_cuda_' + args.checkpoint + '.json')
    gate = read(gate_path)
    assert gate['status'] == 'passed' and gate['checkpoint'] == args.checkpoint
    assert gate['plan_sha256'] == sha(out / 'plan.json') and gate['identity'] == identity['model']
    for flag in ('no_reference_in_features', 'original_single_pass_exact', 'grouped4_matches_parent_exact',
                 'grouped8_partition_and_weights_audited', 'grouped8_zero_strength_exact', 'hook_cleanup_exact'):
        assert gate[flag] is True
    report = dict(status='passed', checkpoint=args.checkpoint, n=len(rows), policies=3, predictions=len(rows) * 3,
        policies_new=3, policies_compared=5,
        new_predictions=len(rows) * 3, inherited_control_predictions=len(rows) * 2,
        compared_predictions=sum(map(len, cases.values())), results=results, winner_selected=False,
        analysis_scope='Follow-up on the same previously evaluated ablation rows; no new independent confirmation and no additional reserve consumption.',
        source_reference_recovered=True, probabilities_reconstructed=True, actual_feedback_weights_reconstructed=True,
        prior_and_grouped4_probabilities_bitwise_equal_to_parent=True,
        grouped4_feedback_weights_and_queries_bitwise_equal_to_parent=True,
        single_pass_and_grouped4_predictions_exact_to_parent=True,
        replay=dict(n=len(rows), target_residues=sum(len(r['mask_h']) for r in rows),
            maximum_prior_absolute_difference=0., maximum_grouped4_posterior_absolute_difference=0., changed_replayed_prediction_tokens=0),
        query_masks_audited=query_count, feedback_weight_residues_audited=weight_count,
        physical_sample_forwards=summary['physical_sample_forwards'], gate_cost_separate=True,
        posterior_sha256=arrays, parent_replayed_posterior_sha256=old_arrays, source_data_sha256=source_hashes,
        predictions_sha256=sha(folder / 'predictions.jsonl'), parent_predictions_sha256=sha(old_path),
        query_audit_sha256=sha(folder / 'query_audit.jsonl'), summary_sha256=sha(folder / 'summary.json'),
        identity_sha256=sha(folder / 'identity.json'), gate_sha256=sha(gate_path),
        vocabulary_sha256=sha(Path(plan['checkpoints'][args.checkpoint]) / 'vocabulary.json'),
        parent_audit_sha256=plan['parent_audit_sha256'][args.checkpoint], plan_sha256=sha(out / 'plan.json'),
        auditor_sha256=sha(Path(__file__)),
        bootstrap={**plan['bootstrap'], 'paired': True, 'interval': 'percentile',
            'per_comparison_adjusted_confidence': .975, 'nominal_family_coverage_at_least': .95},
        primary_interpretation='Grouped8 minus grouped4 on H3 at two checkpoints forms the fixed two-test family. Each Bonferroni-adjusted interval is 97.5%, targeting at least 95% nominal family coverage; percentile-bootstrap calibration is approximate.',
        limitations=[
            'Rows were already used in the preceding ablation; this is a budget follow-up, not fresh validation of a newly selected method.',
            'Only the partition count changes: prior plus three groups versus prior plus seven groups; this is not two successive four-forward passes.',
            'H1/H2 and comparisons against single pass or original4/8 are descriptive 95% intervals.',
            'The two-test correction covers this fixed local follow-up family, not all historical adaptive exploration.',
            'Modulo-three and modulo-seven partitions are not nested; the comparison changes masking geometry and feedback coverage along with query count.',
            'Original4/8 are inherited unchanged from hash-bound audited parent predictions, not newly executed.',
            'The strength=.5 embedding blend and immutable scored/context positions additionally rely on implementation gates.',
            'This comparison does not establish monotonic AAR for arbitrary sampling step counts or universal behavior on other datasets.',
        ])
    write(folder / 'independent_audit.json', report)
    print(json.dumps(dict(status='passed', checkpoint=args.checkpoint, n=len(rows),
        new_predictions=report['new_predictions'], compared_predictions=report['compared_predictions'],
        results=results), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
