"""Activate pllm; python -B FILE --out ABS --checkpoint 112000|50000.

Independently audit CDR component ablations and paired component-bootstrap
contrasts. All predeclared policies are reported; this script never selects one.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def paired_contrasts(rows, cases, contrasts, repeats, seed):
    """Paired sequence-macro AAR contrasts under connected-component resampling."""
    names = sorted(cases)
    index = {name: i for i, name in enumerate(names)}
    components = {group: i for i, group in enumerate(sorted({r['group'] for r in rows}))}
    counts = np.zeros(len(components))
    sums = np.zeros((len(components), len(names)))
    for row in rows:
        group = components[row['group']]
        counts[group] += 1
        for name, column in index.items():
            record = cases[name][row['id']]
            sums[group, column] += 100 * record['correct'] / record['length']
    means = sums.sum(0) / len(rows)
    coefficients = np.zeros((len(names), len(contrasts)))
    for column, contrast in enumerate(contrasts):
        assert abs(sum(contrast['coefficients'].values())) < 1e-10
        for name, weight in contrast['coefficients'].items():
            coefficients[index[name], column] = weight
    boot = np.empty((repeats, len(contrasts)))
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(components), 1 / len(components))
    for start in range(0, repeats, 256):
        weights = rng.multinomial(len(components), probabilities, size=min(256, repeats - start))
        resampled = (weights @ sums) / (weights @ counts)[:, None]
        boot[start:start + len(weights)] = resampled @ coefficients
    point = means @ coefficients
    estimates = {}
    for column, contrast in enumerate(contrasts):
        result = dict(delta_pp=float(point[column]), ci95_pp=np.percentile(boot[:, column], [2.5, 97.5]).tolist(),
            coefficients=contrast['coefficients'], role=contrast['role'])
        if contrast['role'] == 'primary':
            # Three prespecified H3 component tests at each of two checkpoints.
            alpha = .05 / 6
            interval = np.percentile(boot[:, column], [100 * alpha / 2, 100 * (1 - alpha / 2)]).tolist()
            result.update(bonferroni_family_tests=6, simultaneous_interval_confidence=1-alpha,
                ci_bonferroni_pp=interval, adjusted_interval_supports_positive_contribution=bool(interval[0] > 0))
        estimates[contrast['name']] = result
    return dict(n=len(rows), groups=len(components), mean_sequence_aar={name: float(means[i]) for i, name in enumerate(names)},
        logical_mean_nfe={name: float(np.mean([cases[name][r['id']]['logical_nfe'] for r in rows])) for name in names},
        contrasts=estimates)


def factorial_contrasts(cells):
    """Factor +/-: soft/hard, frozen/sequential, and prior0.5/prior0.0.

    Main effects average the two-level differences over other factors; two-way
    terms average difference-of-differences over the third factor; the three-way
    term is their difference. These contrasts are descriptive, without selection.
    """
    assert set(cells) == set(itertools.product((-1, 1), repeat=3))
    factors = ('soft_vs_hard', 'frozen_vs_sequential', 'prior0p5_vs_prior0')
    result = []
    for order in (1, 2, 3):
        for selected in itertools.combinations(range(3), order):
            weights = {policy: float(np.prod([levels[i] for i in selected])) / 2 ** (3-order)
                       for levels, policy in cells.items()}
            result.append(dict(name='factorial_' + '__'.join(factors[i] for i in selected), coefficients=weights,
                role='descriptive_factorial', factors=[factors[i] for i in selected], order=order))
    return result


def load_helper(path):
    spec = importlib.util.spec_from_file_location('frozen_block_audit_helper', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def verify_context(out):
    plan = read(out / 'plan.json')
    for field in ('source_sha256', 'external_source_sha256'):
        for path, digest in plan[field].items():
            assert sha(path) == digest, ('changed frozen file', path)
    for filename, field in (('rows.json', 'rows_sha256'), ('data_plan.json', 'data_plan_sha256'),
                            ('data_independent_audit.json', 'data_audit_sha256')):
        assert sha(out / filename) == plan[field]
    parent = Path(plan['parent'])
    helper_path = parent / 'source/audit_cdr_block_context.py'
    assert plan['external_source_sha256'][str(helper_path)] == sha(helper_path)
    helper = load_helper(helper_path)
    parent_plan = read(parent / 'plan.json')
    assert sha(parent / 'rows.json') == parent_plan['rows_sha256']
    assert sha(parent / 'data_plan.json') == parent_plan['data_plan_sha256']
    original = read(parent / 'rows.json')
    reserve = [row for row in original if row['split'] == 'reserve']
    groups = sorted({row['group'] for row in reserve})
    h3_groups, secondary_groups = set(groups[:4096]), set(groups[:512])
    expected = {r['id']: {**r, 'split': 'ablation', 'parent_split': 'reserve'} for r in reserve
                if (r['mode'] == 'cdrh3' and r['group'] in h3_groups)
                or (r['mode'] in ('cdrh1', 'cdrh2') and r['group'] in secondary_groups)}
    rows = read(out / 'rows.json')
    assert len(rows) == len(expected) == 5254
    assert {row['id']: row for row in rows} == expected
    assert not (set(expected) & {r['id'] for r in original if r['split'] == 'confirm'})
    assert plan['full_policy'] == 'soft_frozen_prior0.5'
    assert plan['bootstrap']['repeats'] == 50000 and plan['bootstrap']['seed'] == 20261016
    families = plan['families']
    names = {'soft_frozen', 'hard_frozen', 'soft_sequential', 'hard_sequential',
             'soft_frozen_strength1', 'soft_frozen_T1'}
    assert {f['name'] for f in families} == names and len(families) == 6
    configs = {}
    for family in families:
        name = family['name']
        assert family['groups'] == 3 and family['rule'] == 'interleaved'
        assert family['context'] == ('hard' if name.startswith('hard') else 'soft')
        assert family['feedback'] == ('sequential' if 'sequential' in name else 'frozen')
        assert family['strength'] == (1. if name.endswith('strength1') else .5)
        assert family['temperature'] == (1. if name.endswith('T1') else .5)
        weights = [.5] if name in ('soft_frozen_strength1', 'soft_frozen_T1') else [.5, 0.0]
        assert family['prior_weights'] == weights
        for weight in weights:
            configs[name + '_prior' + str(weight)] = {**family, 'prior_weight': weight}
    assert len(configs) == 10
    return plan, rows, families, configs, helper


def reconstruct_arrays(folder, rows, cases, families, configs, vocabulary_size, batch_size, helper):
    by_id = {row['id']: row for row in rows}
    expected = {'ids', 'lengths', 'positions', 'prior'}
    for family in families:
        name = family['name']
        expected.update((name, name + '__weights', name + '__queries'))
    seen, hashes = set(), {}
    audit_counts = dict(frozen_query_contexts=0, sequential_query_contexts=0,
                       hard_onehot_context_residues=0, soft_probability_context_residues=0,
                       sequential_previously_scored_feedback_residues=0)
    for path in sorted(folder.glob('batch_*.npz')):
        hashes[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as arrays:
            assert set(arrays.files) == expected
            ids, lengths = arrays['ids'].tolist(), arrays['lengths']
            assert 0 < len(ids) <= batch_size and lengths.shape == (len(ids),)
            assert np.issubdtype(lengths.dtype, np.integer) and (lengths > 0).all()
            width = int(lengths.max())
            shape = (len(ids), width, vocabulary_size)
            assert arrays['prior'].shape == shape and arrays['prior'].dtype == np.float32
            assert arrays['positions'].shape == shape[:2] and np.issubdtype(arrays['positions'].dtype, np.integer)
            for family in families:
                name = family['name']
                assert arrays[name].shape == shape and arrays[name].dtype == np.float32
                for suffix in ('__weights', '__queries'):
                    assert arrays[name + suffix].shape == (3,) + shape
                    assert arrays[name + suffix].dtype == np.float32
            for i, (key, length) in enumerate(zip(ids, lengths.tolist())):
                assert key in by_id and key not in seen
                seen.add(key)
                assert length == len(by_id[key]['mask_h'])
                assert arrays['positions'][i, :length].tolist() == [p + 1 for p in by_id[key]['mask_h']]
                assert (arrays['positions'][i, length:] == -1).all()
                prior = arrays['prior'][i, :length]
                helper.probability_check(prior)
                assert (arrays['prior'][i, length:] == 0).all()
                assert prior.argmax(-1).tolist() == cases['single_pass'][key]['prediction_ids']
                ordinal = np.arange(length)
                for family in families:
                    name = family['name']
                    posterior = arrays[name][i, :length]
                    helper.probability_check(posterior)
                    assert (arrays[name][i, length:] == 0).all()
                    bank = prior.copy()
                    merged = prior.copy()
                    already_scored = np.zeros(length, dtype=bool)
                    for query in range(3):
                        scored = ordinal % 3 == query
                        feedback = ~scored
                        actual = arrays[name + '__weights'][query, i, :length]
                        predictions = arrays[name + '__queries'][query, i, :length]
                        helper.probability_check(predictions)
                        assert np.isfinite(actual).all() and (actual >= 0).all()
                        assert (actual[scored] == 0).all()
                        assert (arrays[name + '__weights'][query, i, length:] == 0).all()
                        assert (arrays[name + '__queries'][query, i, length:] == 0).all()
                        if family['context'] == 'hard':
                            expected_weights = np.zeros_like(bank)
                            expected_weights[np.arange(length), bank.argmax(-1)] = 1.
                            audit_counts['hard_onehot_context_residues'] += int(feedback.sum())
                            assert np.array_equal(actual[feedback], expected_weights[feedback]), (key, name, query, 'hard weights')
                        else:
                            expected_weights = bank ** (1 / family['temperature'])
                            expected_weights /= expected_weights.sum(-1, keepdims=True)
                            audit_counts['soft_probability_context_residues'] += int(feedback.sum())
                            assert np.allclose(actual[feedback], expected_weights[feedback], atol=1e-6, rtol=1e-5), (key, name, query, 'soft weights')
                        assert np.allclose(actual[feedback].sum(-1), 1., atol=1e-5)
                        merged[scored] = predictions[scored]
                        if family['feedback'] == 'sequential':
                            audit_counts['sequential_query_contexts'] += 1
                            audit_counts['sequential_previously_scored_feedback_residues'] += int((feedback & already_scored).sum())
                            bank[scored] = predictions[scored]
                        else:
                            audit_counts['frozen_query_contexts'] += 1
                        already_scored |= scored
                    assert already_scored.all()
                    assert np.array_equal(merged, posterior), (key, name, 'posterior assembly')
                for policy, config in configs.items():
                    weight = config['prior_weight']
                    mixed = weight * prior + (1-weight) * arrays[config['name']][i, :length]
                    assert mixed.argmax(-1).tolist() == cases[policy][key]['prediction_ids'], (key, policy)
    assert seen == set(by_id)
    return hashes, audit_counts


def contrasts_for_mode(plan, configs, mode):
    full = plan['full_policy']
    primary = plan['primary_contrasts']
    secondary = plan['secondary_contrasts']
    assert len(primary) == 3 and len(secondary) == 2
    assert {c['negative'] for c in primary} == {'hard_frozen_prior0.5', 'soft_sequential_prior0.5', 'soft_frozen_prior0.0'}
    assert {c['negative'] for c in secondary} == {'soft_frozen_strength1_prior0.5', 'soft_frozen_T1_prior0.5'}
    contrasts = []
    for source, role in ((primary, 'primary' if mode == 'cdrh3' else 'descriptive_h1_h2'),
                         (secondary, 'secondary_descriptive')):
        for contrast in source:
            assert contrast['positive'] == full
            contrasts.append(dict(name=contrast['name'], coefficients={full: 1., contrast['negative']: -1.}, role=role))
    for policy in configs:
        for baseline in ('single_pass', 'original4', 'original8'):
            contrasts.append(dict(name=policy + '_vs_' + baseline, coefficients={policy: 1., baseline: -1.}, role='descriptive_baseline'))
    cells = {(1 if config['context'] == 'soft' else -1,
              1 if config['feedback'] == 'frozen' else -1,
              1 if config['prior_weight'] == .5 else -1): policy
             for policy, config in configs.items() if config['name'] not in ('soft_frozen_strength1', 'soft_frozen_T1')}
    contrasts.extend(factorial_contrasts(cells))
    assert len({c['name'] for c in contrasts}) == len(contrasts)
    return contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('112000', '50000'), required=True)
    args = parser.parse_args()
    out = args.out
    plan, rows, families, configs, helper = verify_context(out)
    folder = out / ('evaluation_' + args.checkpoint)
    summary = read(folder / 'summary.json')
    assert summary['status'] == 'passed' and summary['n'] == len(rows)
    assert summary['plan_sha256'] == sha(out / 'plan.json') and summary['gate_cost_separate']
    identity = read(folder / 'identity.json')
    assert identity['model']['checkpoint'] == plan['checkpoints'][args.checkpoint]
    assert identity['plan_sha256'] == sha(out / 'plan.json')
    vocabulary = read(Path(plan['checkpoints'][args.checkpoint]) / 'vocabulary.json')
    token = {aa: i for i, aa in enumerate(vocabulary)}
    assert len(token) == len(vocabulary)
    references, source_hashes = helper.recover_references(rows, token)
    pinned = read(Path(plan['parent']) / 'data_plan.json')['source_data_sha256']
    assert all(pinned.get(path) == digest for path, digest in source_hashes.items())
    by_id = {r['id']: r for r in rows}
    cases = helper.read_predictions(folder, by_id, references, configs, 'confirm')
    assert len(cases) == 13
    arrays, context_counts = reconstruct_arrays(folder, rows, cases, families, configs, len(vocabulary), plan['batch_size'], helper)
    query_count = helper.audit_queries(folder, by_id, families)
    assert query_count == len(rows) * 18
    assert summary['physical_sample_forwards'] == len(rows) * 31
    assert set(summary['mean_sequence_aar']) == set(cases)
    results = {}
    for mode in ('cdrh1', 'cdrh2', 'cdrh3'):
        subset = [r for r in rows if r['mode'] == mode]
        key = 'oas_new:' + mode
        assert all(r['dataset'] == 'oas_new' for r in subset)
        result = paired_contrasts(subset, cases, contrasts_for_mode(plan, configs, mode),
                                  plan['bootstrap']['repeats'], plan['bootstrap']['seed'])
        assert result['groups'] == (4096 if mode == 'cdrh3' else 512)
        for name, score in result['mean_sequence_aar'].items():
            assert abs(score - summary['mean_sequence_aar'][name][key]) < 1e-8
            assert result['logical_mean_nfe'][name] == (1 if name == 'single_pass' else 8 if name == 'original8' else 4)
        results[key] = result
    assert all(set(scores) == set(results) for scores in summary['mean_sequence_aar'].values())
    report = dict(status='passed', checkpoint=args.checkpoint, n=len(rows), policies=len(cases),
        predictions=sum(map(len, cases.values())), results=results, winner_selected=False,
        source_reference_recovered=True, probabilities_reconstructed=True,
        actual_feedback_weights_reconstructed=True, sequential_bank_reconstructed=True,
        query_masks_audited=query_count, context_audit_counts=context_counts,
        physical_sample_forwards=summary['physical_sample_forwards'],
        posterior_sha256=arrays, source_data_sha256=source_hashes,
        predictions_sha256=sha(folder / 'predictions.jsonl'), query_audit_sha256=sha(folder / 'query_audit.jsonl'),
        summary_sha256=sha(folder / 'summary.json'), identity_sha256=sha(folder / 'identity.json'),
        vocabulary_sha256=sha(Path(plan['checkpoints'][args.checkpoint]) / 'vocabulary.json'),
        plan_sha256=sha(out / 'plan.json'), auditor_sha256=sha(Path(__file__)),
        bootstrap={**plan['bootstrap'], 'paired': True, 'interval': 'percentile',
                   'primary_family_tests': 6, 'primary_adjusted_confidence': 1-.05/6},
        primary_interpretation='Positive full-minus-ablated H3 contrast supports a component only when its Bonferroni-adjusted interval lower bound exceeds zero. Three component comparisons across two checkpoints form the fixed six-test family.',
        limitations=[
            'All 13 fixed policies are reported; no development winner is selected from these ablation data.',
            'Additional strength/temperature contrasts, baseline contrasts, factorial effects, and H1/H2 contrasts are descriptive.',
            'Bootstrap adjustment uses six prespecified H3 comparisons; percentile bootstrap interval calibration is approximate.',
            'Independent saved-weight reconstruction verifies feedback provenance; embedding blend strength and scored/context immutability additionally rely on implementation gates.',
            'Original4/8 emitted tokens are audited without saved per-step trajectory reconstruction.',
            'This OAS supplement is distinct from official SAbDab benchmarks; exclusion keys do not establish universal homology or pretraining decontamination.',
        ])
    write(folder / 'independent_audit.json', report)
    print(json.dumps(dict(status='passed', checkpoint=args.checkpoint, predictions=report['predictions'],
        mean_sequence_aar={key: value['mean_sequence_aar'] for key, value in results.items()},
        primary_h3={c['name']: results['oas_new:cdrh3']['contrasts'][c['name']] for c in plan['primary_contrasts']}), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
