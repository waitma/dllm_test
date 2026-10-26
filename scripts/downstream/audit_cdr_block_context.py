"""Activate pllm; python -B FILE --out ABS --phase explore|confirm --checkpoint 112000.

Independently recover source references, reconstruct saved probability decisions,
audit group masks, and estimate paired component-bootstrap uncertainty. Only the
development audit freezes a selection; confirmation never chooses a new policy.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
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


def normalized(sequence):
    return sequence.replace('J', 'L')


def verify_context(out, phase, checkpoint):
    plan = read(out / 'plan.json')
    for field in ('source_sha256', 'external_source_sha256'):
        for path, digest in plan[field].items():
            assert sha(path) == digest, ('changed frozen file', path)
    assert sha(out / 'data_plan.json') == plan['data_plan_sha256']
    assert sha(out / 'rows.json') == plan['rows_sha256']
    parent = Path(plan['parent'])
    assert sha(parent / 'rows.json') == plan['development_rows_sha256']
    if phase == 'explore':
        assert checkpoint == '112000', 'The development checkpoint is frozen to 112000'
        rows = [r for r in read(parent / 'rows.json') if r['split'] == 'explore' and r['mode'] == 'cdrh3']
        assert len(rows) == 318
        families = plan['families']
        configs = {f['name'] + '_prior' + str(w): {**f, 'prior_weight': w}
                   for f in families for w in plan['prior_weights']}
        selection = None
    else:
        selection = read(out / 'selection.json')
        assert selection['eligible'] is True
        assert selection['plan_sha256'] == sha(out / 'plan.json')
        assert selection['development_audit_sha256'] == sha(out / 'explore_112000' / 'independent_audit.json')
        config = selection['config']
        families = [f for f in plan['families'] if f['name'] == config['name']]
        assert len(families) == 1
        assert all(config[k] == v for k, v in families[0].items())
        assert config['prior_weight'] in plan['prior_weights']
        name = config['name'] + '_prior' + str(config['prior_weight'])
        assert selection['selected_name'] == name
        configs = {name: config}
        rows = [r for r in read(out / 'rows.json') if r['split'] == 'confirm']
        assert rows and all(r['dataset'] == 'oas_new' for r in rows)
    assert len({r['id'] for r in rows}) == len(rows)
    assert len({f['name'] for f in families}) == len(families)
    assert configs and len(configs) == len(families) * (len(plan['prior_weights']) if phase == 'explore' else 1)
    for family in families:
        assert family['groups'] in (2, 3)
        assert family['rule'] in ('interleaved', 'contiguous')
        assert family['temperature'] > 0 and 0 <= family['strength'] <= 1
    return plan, rows, families, configs, selection


def recover_references(rows, token):
    """Recover every source row independently of the runner's prepared references."""
    required = defaultdict(dict)
    for row in rows:
        required[row['source']][row['row']] = None
    raw_hashes = {}
    for path, wanted in required.items():
        raw_hashes[path] = sha(path)
        with Path(path).open() as stream:
            for index, line in enumerate(stream):
                if index in wanted:
                    wanted[index] = json.loads(line)
        assert all(v is not None for v in wanted.values()), ('missing source row', path)
    references = {}
    for row in rows:
        raw = required[row['source']][row['row']]
        mode = row['mode']
        assert mode in ('cdrh1', 'cdrh2', 'cdrh3') and row['mask_l'] == []
        if row['dataset'] == 'oas_new':
            assert raw['chain_roles'] == ['antibody_heavy', 'antibody_light']
            heavy, light = map(normalized, raw['chains'])
            order = ('FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4')
            regions = raw['regions']['0']
            pieces = [normalized(regions[k]) for k in order]
            assert ''.join(pieces) == heavy
            target_region = 'CDR' + mode[-1]
            before = sum(len(regions[k]) for k in order[:order.index(target_region)])
            sequence = normalized(regions[target_region])
            positions = list(range(before, before + len(sequence)))
            labels = raw['chain_metadata']['0']['h_region_labels']
            assert len(labels) == len(heavy)
            assert positions == [j for j, label in enumerate(labels) if label == mode[-1]]
            assert row['h_cluster'] == raw['chain_metadata']['0']['H_cluster_id']
            assert row['key'] == hashlib.sha256((heavy + '|' + light).encode()).hexdigest()
        else:
            heavy = normalized(raw['heavy_chain_seq'])
            light = normalized(raw['light_chain_seq'])
            first, last = raw[mode + '_pos']
            positions = list(range(first, last + 1))
            sequence = normalized(raw[mode + '_seq'])
        assert heavy == row['gold_h'] == row['heavy']
        assert light == row['gold_l'] == row['light']
        assert positions == row['mask_h'] and positions
        assert ''.join(heavy[j] for j in positions) == sequence
        references[row['id']] = [token[aa] for aa in sequence]
    return references, raw_hashes


def read_predictions(folder, by_id, references, configs, phase):
    expected = {'single_pass'} | set(configs)
    if phase == 'confirm':
        expected |= {'original4', 'original8'}
    cases = defaultdict(dict)
    with (folder / 'predictions.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            key, policy = record['id'], record['policy']
            assert key in by_id and policy in expected
            row = by_id[key]
            assert key not in cases[policy], ('duplicate prediction', policy, key)
            for field in ('group', 'dataset', 'mode', 'fold'):
                assert record[field] == row[field], (field, key)
            reference = references[key]
            assert record['reference_ids'] == reference
            assert record['length'] == len(reference) == len(record['prediction_ids'])
            assert all(isinstance(x, int) for x in record['prediction_ids'])
            assert record['correct'] == sum(a == b for a, b in zip(record['prediction_ids'], reference))
            nfe = 1 if policy == 'single_pass' else (int(policy[-1]) if policy in ('original4', 'original8') else configs[policy]['groups'] + 1)
            assert record['logical_nfe'] == nfe, ('logical_nfe', policy, key)
            cases[policy][key] = record
    assert set(cases) == expected
    assert all(set(records) == set(by_id) for records in cases.values())
    return cases


def probability_check(values):
    assert values.dtype == np.float32
    assert np.isfinite(values).all() and (values >= 0).all()
    assert np.allclose(values.sum(-1), 1., atol=1e-5)


def reconstruct_arrays(folder, by_id, cases, families, configs, vocabulary_size, batch_size):
    seen, hashes = set(), {}
    expected_arrays = {'ids', 'lengths', 'positions', 'prior'} | {f['name'] for f in families}
    for path in sorted(folder.glob('batch_*.npz')):
        hashes[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as arrays:
            assert set(arrays.files) == expected_arrays, (path, arrays.files)
            ids = arrays['ids'].tolist()
            lengths = arrays['lengths']
            assert 0 < len(ids) <= batch_size and lengths.shape == (len(ids),)
            assert np.issubdtype(lengths.dtype, np.integer)
            assert (lengths > 0).all()
            width = int(lengths.max())
            assert arrays['positions'].shape == (len(ids), width)
            assert np.issubdtype(arrays['positions'].dtype, np.integer)
            for name in ['prior'] + [f['name'] for f in families]:
                assert arrays[name].shape == (len(ids), width, vocabulary_size)
                assert arrays[name].dtype == np.float32
            for index, (key, length) in enumerate(zip(ids, lengths.tolist())):
                assert key in by_id and key not in seen, ('unexpected or duplicate array id', key)
                seen.add(key)
                assert length == len(by_id[key]['mask_h'])
                assert arrays['positions'][index, :length].tolist() == [j + 1 for j in by_id[key]['mask_h']]
                assert (arrays['positions'][index, length:] == -1).all()
                prior = arrays['prior'][index, :length]
                probability_check(prior)
                assert prior.argmax(-1).tolist() == cases['single_pass'][key]['prediction_ids']
                for family in families:
                    probability_check(arrays[family['name']][index, :length])
                for name in ['prior'] + [f['name'] for f in families]:
                    assert (arrays[name][index, length:] == 0).all()
                for policy, config in configs.items():
                    posterior = arrays[config['name']][index, :length]
                    weight = config['prior_weight']
                    mixed = weight * prior + (1 - weight) * posterior
                    assert mixed.argmax(-1).tolist() == cases[policy][key]['prediction_ids'], (policy, key)
    assert seen == set(by_id)
    return hashes


def audit_queries(folder, by_id, families):
    configured = {f['name']: f for f in families}
    seen = set()
    with (folder / 'query_audit.jsonl').open() as stream:
        for line in stream:
            query = json.loads(line)
            key, name, index = query['id'], query['family'], query['query']
            assert key in by_id and name in configured
            family = configured[name]
            assert type(index) is int and 0 <= index < family['groups']
            signature = (key, name, index)
            assert signature not in seen
            seen.add(signature)
            length = len(by_id[key]['mask_h'])
            scored, feedback = query['scored'], query['feedback']
            assert len(scored) == len(feedback) == length
            assert all(type(value) is bool for value in scored + feedback)
            ordinal = np.arange(length)
            labels = ordinal % family['groups'] if family['rule'] == 'interleaved' else ordinal * family['groups'] // length
            expected = (labels == index).tolist()
            assert scored == expected
            assert feedback == [not value for value in expected]
    expected_count = len(by_id) * sum(f['groups'] for f in families)
    assert len(seen) == expected_count
    return expected_count


def bootstrap_results(rows, cases, summary, repeats, seed):
    """Resample connected components; each replicate retains sequence-macro weighting."""
    names = sorted(cases)
    results = {}
    for dataset_mode in sorted({r['dataset'] + ':' + r['mode'] for r in rows}):
        subset = [r for r in rows if r['dataset'] + ':' + r['mode'] == dataset_mode]
        group_index = {group: i for i, group in enumerate(sorted({r['group'] for r in subset}))}
        counts = np.zeros(len(group_index))
        sums = np.zeros((len(group_index), len(names)))
        for row in subset:
            group = group_index[row['group']]
            counts[group] += 1
            for column, name in enumerate(names):
                record = cases[name][row['id']]
                sums[group, column] += 100 * record['correct'] / record['length']
        means = sums.sum(0) / len(subset)
        for name, value in zip(names, means):
            assert abs(value - summary['mean_sequence_aar'][name][dataset_mode]) < 1e-8
        rng = np.random.default_rng(seed)
        boot = np.empty((repeats, len(names)))
        probabilities = np.full(len(group_index), 1 / len(group_index))
        for start in range(0, repeats, 512):
            weights = rng.multinomial(len(group_index), probabilities, size=min(512, repeats - start))
            boot[start:start + len(weights)] = (weights @ sums) / (weights @ counts)[:, None]
        comparisons = {}
        for base in ('single_pass', 'original4', 'original8'):
            if base not in names:
                continue
            column = names.index(base)
            comparisons[base] = {
                name: {'delta_pp': float(means[i] - means[column]),
                       'ci95_pp': np.percentile(boot[:, i] - boot[:, column], [2.5, 97.5]).tolist()}
                for i, name in enumerate(names)}
        results[dataset_mode] = dict(n=len(subset), groups=len(group_index),
            mean_sequence_aar={name: float(value) for name, value in zip(names, means)},
            logical_mean_nfe={name: float(np.mean([cases[name][r['id']]['logical_nfe'] for r in subset])) for name in names},
            comparisons=comparisons)
    assert set(summary['mean_sequence_aar']) == set(names)
    assert all(set(scores) == set(results) for scores in summary['mean_sequence_aar'].values())
    return results


def freeze_selection(out, plan, configs, report):
    result = report['results']['modern:cdrh3']
    best = min(configs, key=lambda name: (-result['mean_sequence_aar'][name], result['logical_mean_nfe'][name], name))
    delta = result['comparisons']['single_pass'][best]['delta_pp']
    eligible = delta > 0
    selected = dict(eligible=eligible, selected_name=best if eligible else None,
        config=configs[best] if eligible else None, strongest_explored_name=best,
        best_explore_delta_pp=delta, development_audit_sha256=sha(out / 'explore_112000' / 'independent_audit.json'),
        plan_sha256=sha(out / 'plan.json'),
        selection_rule='Maximum development sequence-macro AAR; ties use lower logical NFE, then lexical policy name; require strict improvement over single_pass.',
        warning='Development selection only; unadjusted intervals across the searched grid are not confirmation evidence.')
    path = out / 'selection.json'
    if path.exists():
        existing = read(path)
        assert all(existing.get(k) == value for k, value in selected.items()), 'Frozen selection differs; refusing to overwrite'
        return existing
    selected['frozen_utc'] = datetime.now(timezone.utc).isoformat()
    with path.open('x') as stream:
        stream.write(json.dumps(selected, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=('explore', 'confirm'), required=True)
    parser.add_argument('--checkpoint', choices=('112000', '50000'), required=True)
    args = parser.parse_args()
    out = args.out
    plan, rows, families, configs, selection = verify_context(out, args.phase, args.checkpoint)
    folder = out / (('explore_' if args.phase == 'explore' else 'confirmation_') + args.checkpoint)
    summary = read(folder / 'summary.json')
    assert summary['status'] == 'passed' and summary['n'] == len(rows)
    assert summary['plan_sha256'] == sha(out / 'plan.json')
    assert summary['gate_cost_separate'] is True
    checkpoint = Path(plan['checkpoints'][args.checkpoint])
    identity = read(folder / 'identity.json')
    assert Path(identity['model']['checkpoint']) == checkpoint
    assert identity['plan_sha256'] == sha(out / 'plan.json')
    assert identity['phase'] == args.phase
    vocabulary = read(checkpoint / 'vocabulary.json')
    token = {aa: index for index, aa in enumerate(vocabulary)}
    assert len(token) == len(vocabulary)
    by_id = {r['id']: r for r in rows}
    references, raw_hashes = recover_references(rows, token)
    pinned = read(Path(plan['parent']) / 'plan.json')['data_sha256'] if args.phase == 'explore' else read(out / 'data_plan.json')['source_data_sha256']
    assert all(pinned.get(path) == digest for path, digest in raw_hashes.items()), 'Source data differ from frozen provenance'
    cases = read_predictions(folder, by_id, references, configs, args.phase)
    hashes = reconstruct_arrays(folder, by_id, cases, families, configs, len(vocabulary), plan['batch_size'])
    queries = audit_queries(folder, by_id, families)
    physical = len(rows) + queries + (12 * len(rows) if args.phase == 'confirm' else 0)
    assert physical == summary['physical_sample_forwards'], ('physical_sample_forwards', physical, summary['physical_sample_forwards'])
    repeats, seed = 10000, 20261015
    results = bootstrap_results(rows, cases, summary, repeats, seed)
    report = dict(status='passed', phase=args.phase, checkpoint=args.checkpoint, n=len(rows),
        policies=len(cases), predictions=sum(map(len, cases.values())), results=results,
        query_masks_audited=queries, source_reference_recovered=True, probabilities_reconstructed=True,
        physical_sample_forwards=physical, posterior_sha256=hashes,
        source_data_sha256=raw_hashes, predictions_sha256=sha(folder / 'predictions.jsonl'),
        query_audit_sha256=sha(folder / 'query_audit.jsonl'), summary_sha256=sha(folder / 'summary.json'),
        identity_sha256=sha(folder / 'identity.json'),
        vocabulary_sha256=sha(checkpoint / 'vocabulary.json'), plan_sha256=sha(out / 'plan.json'),
        auditor_sha256=sha(Path(__file__)), bootstrap=dict(repeats=repeats, seed=seed, unit='connected component', interval='paired percentile 95%'),
        limitations=('Development grid selection; unadjusted intervals are descriptive, not evidence of confirmed improvement.' if args.phase == 'explore'
            else 'Fresh OAS supplement is separate from the official SAbDab benchmark. Original4/8 are audited from emitted tokens, without per-step trajectory reconstruction. Exact and supplied-cluster exclusions do not establish universal homology or pretraining decontamination.'))
    # Deterministic reports permit a read-only re-audit of an existing frozen selection.
    report_path = folder / 'independent_audit.json'
    if args.phase == 'explore' and (out / 'selection.json').exists():
        assert report_path.exists() and read(report_path) == report, 'Existing frozen development report differs; refusing to overwrite'
    else:
        write(report_path, report)
    if args.phase == 'explore':
        selection = freeze_selection(out, plan, configs, report)
    print(json.dumps(dict(status='passed', phase=args.phase, checkpoint=args.checkpoint,
        predictions=report['predictions'], results=results, selection=selection), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
