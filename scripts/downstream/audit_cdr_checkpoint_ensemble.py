"""Activate pllm; python -B FILE --out ABS to audit the exploration ensemble.

Recover source references and reconstruct every decision from stored posteriors.
Report component-paired uncertainty against each individual, without treating
the development winner as a confirmed result.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); out = args.out; dest = out/'checkpoint_ensemble'
    plan = json.loads((out/'plan.json').read_text())
    assert sha(out/'rows.json') == plan['rows_sha256']
    for field in ('source_sha256', 'data_sha256', 'external_source_sha256'):
        for p, digest in plan[field].items(): assert sha(p) == digest
    rows = {r['id']: r for r in json.loads((out/'rows.json').read_text())
            if r['split'] == 'explore' and r['mode'] == 'cdrh3'}
    vocab = json.loads((Path(plan['checkpoints']['50000'])/'vocabulary.json').read_text())
    token = {aa: i for i, aa in enumerate(vocab)}
    sources = {}; refs = {}
    for key, r in rows.items():
        if r['source'] not in sources:
            sources[r['source']] = [json.loads(x) for x in Path(r['source']).read_text().splitlines()]
        raw = sources[r['source']][r['row']]; start, end = raw['cdrh3_pos']
        assert raw['heavy_chain_seq'][start:end+1] == raw['cdrh3_seq']
        assert r['mask_h'] == list(range(start, end+1))
        refs[key] = [token[aa] for aa in raw['cdrh3_seq'].replace('J', 'L')]
    parent = out/'explore_112000'
    parent_audit = json.loads((parent/'independent_audit.json').read_text())
    assert parent_audit['status'] == 'passed'
    p112 = {}; p50 = {}; hashes = {}
    for path in sorted(parent.glob('posterior_batch_*.npz')):
        assert sha(path) == parent_audit['posterior_sha256'][str(path)]
        with np.load(path, allow_pickle=False) as arrays:
            for i, (key, length) in enumerate(zip(arrays['ids'].tolist(), arrays['lengths'].tolist())):
                assert key not in p112; p112[key] = arrays['prior'][i, :length].copy()
    for path in sorted(dest.glob('prior50000_batch_*.npz')):
        hashes[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                assert key not in p50; p50[key] = arrays[key].copy()
    assert set(p50) == set(p112) == set(rows)
    cases = defaultdict(dict)
    for line in (dest/'predictions.jsonl').open():
        r = json.loads(line); key = r['id']; name = r['policy']
        assert key in rows and key not in cases[name]
        assert r['group'] == rows[key]['group'] and r['reference_ids'] == refs[key]
        assert r['length'] == len(refs[key]) == len(r['prediction_ids'])
        assert r['correct'] == sum(a == b for a, b in zip(r['prediction_ids'], refs[key]))
        if name == 'single50000': p = p50[key]
        elif name == 'single112000': p = p112[key]
        else:
            kind, weight = name.split('_w112_'); w = float(weight)
            assert w in (.1, .25, .5, .75, .9)
            if kind == 'arithmetic': p = w*p112[key]+(1-w)*p50[key]
            else:
                assert kind == 'geometric'
                p = w*np.log(p112[key].clip(1e-30))+(1-w)*np.log(p50[key].clip(1e-30))
                p = np.exp(p-p.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(-1), 1, atol=1e-5)
        assert p.argmax(-1).tolist() == r['prediction_ids']
        cases[name][key] = r
    expected = {'single50000', 'single112000'} | {f'{k}_w112_{w}' for k in ('arithmetic', 'geometric') for w in (.1, .25, .5, .75, .9)}
    assert set(cases) == expected
    for c in cases.values(): assert set(c) == set(rows)
    summary = json.loads((dest/'summary.json').read_text())
    means = {name: float(np.mean([r['correct']/r['length']*100 for r in c.values()])) for name, c in cases.items()}
    for name in means: assert abs(means[name]-summary['scores'][name]) < 1e-8
    groups = {g: i for i, g in enumerate(sorted({r['group'] for r in rows.values()}))}
    counts = np.zeros(len(groups))
    for r in rows.values(): counts[groups[r['group']]] += 1
    weights = np.random.default_rng(20261015).multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=4000)
    comparisons = {}
    for name, c in cases.items():
        comparisons[name] = {}
        for baseline in ('single50000', 'single112000'):
            sums = np.zeros(len(groups))
            for key, r in c.items():
                sums[groups[r['group']]] += (r['correct']-cases[baseline][key]['correct'])/r['length']*100
            comparisons[name][baseline] = dict(delta_pp=float(sums.sum()/len(rows)),
                paired_component_bootstrap95ci_pp=np.percentile(weights@sums/(weights@counts), [2.5, 97.5]).tolist())
    report = dict(status='passed', n=len(rows), predictions=len(rows)*len(cases), mean_aar=means,
        comparisons=comparisons, predictions_sha256=sha(dest/'predictions.jsonl'), posterior_sha256=hashes,
        auditor_sha256=sha(Path(__file__)), limitation='Exploration selection; unadjusted intervals, not confirmation.')
    (dest/'independent_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status='passed', scores=means, comparisons=comparisons)), flush=True)


if __name__ == '__main__': main()
