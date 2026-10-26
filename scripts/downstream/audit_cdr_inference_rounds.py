"""Activate pllm; python -B FILE --out ABS --round 1|2|3.

Independent audit: recover references from frozen source rows, reconcile every
saved transition, recompute scores, and bootstrap whole H/L sequence groups
jointly across folds (duplicates across folds share their bootstrap weight).
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--round', type=int, required=True)
    args = parser.parse_args()
    out, folder = args.out, args.out/f'round{args.round}'
    plan = json.loads((out/'plan.json').read_text())
    for mapping in ('source_sha256', 'external_source_sha256', 'data_sha256'):
        for path, expected in plan[mapping].items(): assert sha(path) == expected, path
    assert sha(out/'rows.json') == plan['rows_sha256']
    rows = json.loads((out/'rows.json').read_text())
    split = 'explore' if args.round == 1 else 'confirm'
    selected = {r['id']: r for r in rows if r['split'] == split}
    groups = {s: {r['group'] for r in rows if r['split'] == s} for s in ('explore', 'confirm')}
    assert not groups['explore'] & groups['confirm']
    identity = json.loads((folder/'identity.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    vocab = json.loads((Path(identity['model']['checkpoint'])/'vocabulary.json').read_text())
    token = {c: i for i, c in enumerate(vocab)}
    mask = token['<mask>']
    cases, configs = defaultdict(dict), {p['name']: p for p in identity['configs']}
    for line in (folder/'predictions.jsonl').open():
        r = json.loads(line); source = selected[r['id']]
        ref = [token[source['gold_h'][j]] for j in source['mask_h']]
        ref += [token[source['gold_l'][j]] for j in source['mask_l']]
        assert r['reference_ids'] == ref
        assert r['length'] == len(ref) == len(r['prediction_ids'])
        assert r['correct'] == sum(a == b for a, b in zip(r['prediction_ids'], ref))
        assert mask not in r['prediction_ids']
        assert r['id'] not in cases[r['policy']]
        cases[r['policy']][r['id']] = r
    assert set(cases) == set(configs)
    for records in cases.values(): assert set(records) == set(selected)
    stats = {p: defaultdict(int) for p in configs}
    seen, baseline_commits = set(), {}
    for line in (folder/'traces.jsonl').open():
        tr = json.loads(line); key = tr['policy'], tr['id']
        assert key not in seen; seen.add(key)
        record = cases[key[0]][key[1]]; cfg = configs[key[0]]; gold = record['reference_ids']
        assert tr['steps'][-1]['after'] == record['prediction_ids']
        assert len(tr['steps']) == record['sample_forwards'] <= cfg['steps']
        assert tr['steps'][0]['before'] == [mask]*len(gold)
        previous = None
        for step, t in enumerate(tr['steps']):
            assert all(len(v) == len(gold) for v in t.values())
            if previous is not None: assert t['before'] == previous
            for j, (before, after, proposal, commit, remasked, pending, conf) in enumerate(zip(
                t['before'], t['after'], t['proposal'], t['commit'], t['remasked'], t['pending'], t['confidence'])):
                assert 0 <= conf <= 1
                expected = mask if remasked else before
                if commit: expected = proposal
                assert after == expected
                if cfg['kind'] in ('repair', 'random_repair') and step:
                    assert pending == remasked and commit == pending
                else:
                    assert pending == (before == mask)
                    assert not commit or pending
                    assert not remasked or not pending
                if cfg['kind'] not in ('repair', 'random_repair', 'remdm'):
                    assert not remasked
                if cfg['kind'] == 'remdm' and cfg['eta'] == 0: assert not remasked
                stats[key[0]]['remasked_positions'] += remasked
                stats[key[0]]['committed_positions'] += commit
                if remasked and commit:
                    stats[key[0]]['repair_right_to_wrong'] += before == gold[j] and after != gold[j]
                    stats[key[0]]['repair_wrong_to_right'] += before != gold[j] and after == gold[j]
            stats[key[0]]['no_commit_sample_forwards'] += not any(t['commit'])
            stats[key[0]]['sample_forwards'] += 1
            previous = t['after']
        if key[0] in ('original4', 'original8'):
            baseline_commits[key] = [t['commit'] for t in tr['steps']]
        if cfg['kind'] == 'oracle':
            assert [t['commit'] for t in tr['steps']] == baseline_commits[(f"original{cfg['steps']}", key[1])]
    assert len(seen) == len(selected)*len(configs)
    reports, comparisons = {}, {}
    bootstrap_cache = {}
    for name, records in cases.items():
        assert stats[name]['sample_forwards'] == summary['costs'][name]['sample_forwards']
        scores = defaultdict(lambda: defaultdict(list))
        for r in records.values(): scores[r['dataset']+':'+r['mode']][r['fold']].append(r['correct']/r['length']*100)
        reports[name] = {key: float(np.mean([np.mean(v) for v in folds.values()])) for key, folds in scores.items()}
        for key, score in reports[name].items():
            assert abs(score-summary['cases'][name][key]['fold_macro_aar']) < 1e-8
            subset = sorted([r for r in records.values() if r['dataset']+':'+r['mode'] == key], key=lambda r: r['id'])
            if key not in bootstrap_cache:
                gs = {g: i for i, g in enumerate(sorted({r['group'] for r in subset}))}
                fs = {f: i for i, f in enumerate(sorted({r['fold'] for r in subset}))}
                counts = np.zeros((len(gs), len(fs)))
                for r in subset: counts[gs[r['group']], fs[r['fold']]] += 1
                rng = np.random.default_rng(20261015)
                weights = rng.multinomial(len(gs), np.full(len(gs), 1/len(gs)), size=2000).astype(float)
                denominators = weights @ counts
                valid = (denominators > 0).all(1)
                bootstrap_cache[key] = gs, fs, counts, weights[valid], denominators[valid]
            gs, fs, counts, weights, denominators = bootstrap_cache[key]
            sums = np.zeros_like(counts)
            for r in subset:
                base = cases['original1'][r['id']]
                sums[gs[r['group']], fs[r['fold']]] += (r['correct']-base['correct'])/r['length']*100
            boot = ((weights @ sums)/denominators).mean(1)
            comparisons[name+':'+key] = dict(delta_vs_original1_pp=float((sums.sum(0)/counts.sum(0)).mean()),
                paired_HL_group_bootstrap95ci_pp=np.percentile(boot, [2.5, 97.5]).tolist(),
                bootstrap_replicates=len(boot), sequence_pairs=len(gs))
    report = dict(status='passed', round=args.round, predictions=len(seen),
        recovered_references_from_source=True, disjoint_sequence_pair_groups=True,
        transitions_audited=True, scores=reports, transitions=stats, comparisons=comparisons,
        source_sha256=sha(Path(__file__)), predictions_sha256=sha(folder/'predictions.jsonl'),
        traces_sha256=sha(folder/'traces.jsonl'),
        interval_method='Whole H/L group resampling across all folds; descriptive, no multiplicity adjustment',
        limitations='Benchmark-internal exploration/confirmation, not independent validation or guaranteed pretraining decontamination.')
    (folder/'independent_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed', predictions=len(seen), round=args.round)))


if __name__ == '__main__': main()
