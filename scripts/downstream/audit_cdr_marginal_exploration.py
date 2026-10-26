"""Activate pllm; python -B FILE --out ABS --checkpoint 112000.

Independently recover references, audit actual query masks and reconstruct
every emitted decision from stored probability arrays before selecting a
candidate. Exploration selection is never a successful confirmation result.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', default='112000')
    args = parser.parse_args(); out = args.out; folder = out/f'explore_{args.checkpoint}'
    plan = json.loads((out/'plan.json').read_text())
    for field in ('source_sha256', 'data_sha256', 'external_source_sha256'):
        for path, digest in plan[field].items(): assert sha(path) == digest, path
    assert sha(out/'rows.json') == plan['rows_sha256']
    all_rows = json.loads((out/'rows.json').read_text())
    rows = {r['id']: r for r in all_rows if r['mode'] == 'cdrh3' and r['split'] == 'explore'}
    summary = json.loads((folder/'summary.json').read_text()); assert summary['status'] == 'passed'
    identity = json.loads((folder/'identity.json').read_text())
    vocab = json.loads((Path(identity['model']['checkpoint'])/'vocabulary.json').read_text())
    token = {aa: i for i, aa in enumerate(vocab)}; mask = token['<mask>']
    cases = defaultdict(dict); sources = {}
    for line in (folder/'predictions.jsonl').open():
        r = json.loads(line); source = rows[r['id']]
        if source['source'] not in sources:
            sources[source['source']] = [json.loads(x) for x in Path(source['source']).read_text().splitlines()]
        raw = sources[source['source']][source['row']]
        assert raw['heavy_chain_seq'].replace('J', 'L') == source['gold_h']
        start, end = raw['cdrh3_pos']; assert source['mask_h'] == list(range(start, end+1))
        ref = [token[aa] for aa in raw['cdrh3_seq'].replace('J', 'L')]
        assert r['reference_ids'] == ref
        assert len(r['prediction_ids']) == r['length'] == len(ref)
        assert r['correct'] == sum(a == b for a, b in zip(r['prediction_ids'], ref))
        assert r['group'] == source['group']
        assert r['id'] not in cases[r['policy']]
        cases[r['policy']][r['id']] = r
    expected_names = {'single_pass'}
    configurations = {}
    for f in plan['families']:
        prefixes = [f"{f['name']}_k{k}" for k in f['draws']] if f['kind'] == 'mc_context' else [f['name']]
        for prefix in prefixes:
            for weight in plan['prior_weights']:
                name = prefix+f'_prior{weight}'
                expected_names.add(name); configurations[name] = dict(family=f, posterior=prefix, prior_weight=weight)
    assert set(cases) == expected_names
    for values in cases.values(): assert set(values) == set(rows)
    reconstructed = set(); array_hashes = {}; lengths_by_id = {}
    for path in sorted(folder.glob('posterior_batch_*.npz')):
        array_hashes[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as arrays:
            for index, (key, length) in enumerate(zip(arrays['ids'].tolist(), arrays['lengths'].tolist())):
                assert key in rows and key not in reconstructed; reconstructed.add(key)
                lengths_by_id[key] = length
                prior = arrays['prior'][index, :length]
                assert np.isfinite(prior).all() and (prior >= 0).all()
                assert np.allclose(prior.sum(-1), 1., atol=1e-5)
                assert prior.argmax(-1).tolist() == cases['single_pass'][key]['prediction_ids']
                for name, cfg in configurations.items():
                    posterior = arrays[cfg['posterior']][index, :length]
                    assert np.isfinite(posterior).all() and (posterior >= 0).all()
                    assert np.allclose(posterior.sum(-1), 1., atol=1e-5)
                    w = cfg['prior_weight']; mixed = w*prior+(1-w)*posterior
                    assert mixed.argmax(-1).tolist() == cases[name][key]['prediction_ids'], (name, key)
                    family = cfg['family']
                    if family['kind'] == 'mc_context': nfe = int(cfg['posterior'].rsplit('_k', 1)[1])+1
                    elif family['kind'] == 'leave_one_out': nfe = int(arrays['lengths'].max())+1
                    else: nfe = family['anchors']*family['topk']+1
                    assert cases[name][key]['logical_forwards'] == nfe
    assert reconstructed == set(rows)
    query_count = 0; seen = set(); per_family = Counter()
    for line in (folder/'query_audit.jsonl').open():
        q = json.loads(line); key = q['id']
        assert key in rows and len(q['inputs']) == len(q['scored']) == lengths_by_id[key]
        assert all(not scored or x == mask for x, scored in zip(q['inputs'], q['scored']))
        signature = key, q['family'], q['query']; assert signature not in seen; seen.add(signature)
        query_count += 1; per_family[q['family']] += 1
    assert query_count+len(rows) == summary['actual_sample_forwards']
    assert summary['additional_equivalence_sample_forwards'] == len(rows)
    ids = sorted(rows); groups = {g: i for i, g in enumerate(sorted({r['group'] for r in rows.values()}))}
    counts = np.zeros(len(groups))
    for key in ids: counts[groups[rows[key]['group']]] += 1
    rng = np.random.default_rng(20261015)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=4000)
    denominators = weights @ counts
    comparisons = {}; base = cases['single_pass']
    for name, records in cases.items():
        scores = [records[k]['correct']/records[k]['length']*100 for k in ids]
        assert abs(float(np.mean(scores))-summary['scores'][name]) < 1e-8
        assert abs(np.mean([records[k]['logical_forwards'] for k in ids])-summary['logical_mean_forwards'][name]) < 1e-8
        sums = np.zeros(len(groups))
        for key in ids:
            sums[groups[rows[key]['group']]] += (records[key]['correct']-base[key]['correct'])/records[key]['length']*100
        replicate = weights @ sums/denominators
        comparisons[name] = dict(delta_vs_single_pass_pp=float(sums.sum()/len(rows)),
            paired_component_bootstrap95ci_pp=np.percentile(replicate, [2.5, 97.5]).tolist())
    report = dict(status='passed', checkpoint=args.checkpoint, n=len(rows), policies=len(cases),
        predictions=sum(map(len, cases.values())), query_masks_audited=query_count,
        source_reference_recovered=True, probabilities_reconstructed=True,
        mean_aar=summary['scores'], logical_mean_forwards=summary['logical_mean_forwards'], comparisons=comparisons,
        actual_sampler_sample_forwards=summary['actual_sample_forwards'],
        additional_equivalence_sample_forwards=summary['additional_equivalence_sample_forwards'],
        predictions_sha256=sha(folder/'predictions.jsonl'), query_audit_sha256=sha(folder/'query_audit.jsonl'),
        posterior_sha256=array_hashes, auditor_sha256=sha(Path(__file__)),
        limitations='Exploration only; 52-way sweep, unadjusted conditional-on-random-stream intervals. '
                    'No confirm scores inspected and no positive result established by this file.')
    write(folder/'independent_audit.json', report)
    best = min(configurations, key=lambda name: (-summary['scores'][name], summary['logical_mean_forwards'][name], name))
    improves = comparisons[best]['delta_vs_single_pass_pp'] > 0
    selected = dict(status='candidate_for_confirmation' if improves else 'no_positive_candidate',
        checkpoint=args.checkpoint, selected_name=best if improves else None,
        strongest_explored_name=best, best_explore_delta_pp=comparisons[best]['delta_vs_single_pass_pp'],
        selected_config=configurations[best] if improves else None,
        source_audit_sha256=sha(folder/'independent_audit.json'), frozen_utc=datetime.now(timezone.utc).isoformat(),
        warning='Exploration winner, not confirmed improvement. Goal remains active.')
    selection_path = out/f'selection_{args.checkpoint}.json'
    assert not selection_path.exists(), 'Preserve existing frozen selection'
    write(selection_path, selected)
    print(json.dumps(dict(status='passed', predictions=report['predictions'], selection=selected), ensure_ascii=False))


if __name__ == '__main__': main()
