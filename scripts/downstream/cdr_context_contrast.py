"""Activate pllm/protenix_abtcr; python -B FILE run|audit --out ABS.

Inference-only context-strength exploration. GPU run must use eval_jobs YAML.
Audit reconstructs decisions and source references without model execution.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_runner(out):
    spec = importlib.util.spec_from_file_location('contrast_frozen_runner', out/'source/run_cdr_marginal_exploration.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    return runner


def guided(prior, ablated, alpha, forbidden):
    score = (1+alpha)*np.log(prior.clip(1e-30))-alpha*np.log(ablated.clip(1e-30))
    score[..., forbidden] = -np.inf
    return score.argmax(-1)


def run(args):
    import torch
    torch.set_num_threads(4)
    out = args.out; runner = load_runner(out); plan, all_rows = runner.verify(out)
    protocol = json.loads((out/'contrast_plan.json').read_text())
    assert sha(Path(__file__)) == protocol['source_sha256']
    rows = [r for r in all_rows if r['split'] == 'explore' and r['mode'] == 'cdrh3']
    dest = out/'context_contrast'; dest.mkdir(exist_ok=False)
    write(dest/'status.json', dict(status='running'))
    adapter, evaluator, sampler, model, identity = runner.setup(out, plan['checkpoints']['112000'], args.device)
    write(dest/'identity.json', dict(model=identity, source_sha256=sha(Path(__file__))))
    scores = defaultdict(list)
    with (dest/'predictions.jsonl').open('x') as stream, torch.inference_mode():
        for start in range(0, len(rows), plan['batch_size']):
            chunk = rows[start:start+plan['batch_size']]
            f, b, gold, target, maps = adapter.airgen_inputs(model, chunk)
            f = {k: v.to(args.device) for k, v in f.items()}; b = {k: v.to(args.device) for k, v in b.items()}
            gold, target = gold.to(args.device), target.to(args.device)
            bridge = evaluator.decoder(model, f, maps)
            predict, counter = runner.prediction_fn(bridge, b, target)
            prior = sampler.base_distribution(predict, b, target, bridge.mask_id)
            original, _ = bridge.generate(b, max_iter=1, temperature=1.,
                partial_masks=~target & b['input_ids'].ne(bridge.bos_id), sampling_strategy='argmax', cfg_scale=0.)
            assert torch.equal(original[target], prior.argmax(-1)[target])
            residues = torch.zeros_like(target)
            for i, mapping in enumerate(maps): residues[i, [v for v, n in mapping]] = True
            distributions = {'prior': prior.cpu().numpy()}; inputs = {}
            for kind in protocol['contexts']:
                hide = residues & ~target
                if kind == 'light': hide &= b['chain_ids'].eq(1)
                elif kind == 'heavy': hide &= b['chain_ids'].eq(0)
                else: assert kind == 'all'
                tokens = b['input_ids'].clone(); tokens[hide] = bridge.mask_id
                assert tokens[target].eq(bridge.mask_id).all()
                assert torch.equal(tokens[~hide], b['input_ids'][~hide])
                changed_batch = {**b, 'input_ids': tokens}
                alternative, _ = runner.prediction_fn(bridge, changed_batch, target)
                distributions[kind] = alternative(tokens).cpu().numpy()
                inputs[kind] = tokens.cpu().numpy()
            arrays = dict(ids=np.array([r['id'] for r in chunk]), lengths=target.sum(-1).cpu().numpy(),
                target=target.cpu().numpy(), original_inputs=b['input_ids'].cpu().numpy(),
                residues=residues.cpu().numpy(), forbidden=np.array(bridge.special_token_list), mask_id=np.array(bridge.mask_id))
            arrays.update(distributions); arrays.update({f'inputs_{k}': v for k, v in inputs.items()})
            np.savez_compressed(dest/f'batch_{start:05d}.npz', **arrays)
            for i, row in enumerate(chunk):
                mask = arrays['target'][i]; p = distributions['prior'][i, mask]
                predictions = {'single_pass': p.argmax(-1)}
                for kind in protocol['contexts']:
                    for alpha in protocol['alphas']:
                        predictions[f'{kind}_alpha{alpha}'] = guided(p, distributions[kind][i, mask], alpha, bridge.special_token_list)
                ref = gold[i, target[i]].tolist()
                for name, guess in predictions.items():
                    correct = sum(x == y for x, y in zip(guess.tolist(), ref))
                    stream.write(json.dumps(dict(id=row['id'], group=row['group'], policy=name, reference_ids=ref,
                        prediction_ids=guess.tolist(), correct=correct, length=len(ref)))+'\n')
                    scores[name].append(correct/len(ref)*100)
            stream.flush(); print('CONTEXT PROBE', start+len(chunk), '/', len(rows), flush=True)
    write(dest/'summary.json', dict(status='passed', n=len(rows), scores={n: float(np.mean(s)) for n, s in scores.items()},
        logical_nfe=dict(single_pass=1, candidate=2), physical_sample_forwards=5*len(rows),
        note='Four distinct context queries plus one original equivalence query per row. Explore only.'))
    write(dest/'status.json', dict(status='success'))


def audit(args):
    out = args.out; dest = out/'context_contrast'; plan, all_rows = load_runner(out).verify(out)
    protocol = json.loads((out/'contrast_plan.json').read_text())
    assert sha(Path(__file__)) == protocol['source_sha256']
    rows = {r['id']: r for r in all_rows if r['split'] == 'explore' and r['mode'] == 'cdrh3'}
    vocab = json.loads((Path(plan['checkpoints']['112000'])/'vocabulary.json').read_text())
    tok = {aa: i for i, aa in enumerate(vocab)}; sources = {}; refs = {}
    for key, row in rows.items():
        if row['source'] not in sources:
            sources[row['source']] = [json.loads(x) for x in Path(row['source']).read_text().splitlines()]
        raw = sources[row['source']][row['row']]; start, end = raw['cdrh3_pos']
        assert raw['heavy_chain_seq'][start:end+1] == raw['cdrh3_seq']
        assert row['mask_h'] == list(range(start, end+1))
        refs[key] = [tok[aa] for aa in raw['cdrh3_seq'].replace('J', 'L')]
    cases = defaultdict(dict)
    for line in (dest/'predictions.jsonl').open():
        r = json.loads(line); key = r['id']; assert key in rows and key not in cases[r['policy']]
        assert r['reference_ids'] == refs[key] and r['group'] == rows[key]['group']
        assert r['length'] == len(refs[key]) == len(r['prediction_ids'])
        assert r['correct'] == sum(a == b for a, b in zip(r['prediction_ids'], refs[key]))
        cases[r['policy']][key] = r
    expected = {'single_pass'} | {f'{k}_alpha{a}' for k in protocol['contexts'] for a in protocol['alphas']}
    assert set(cases) == expected
    for c in cases.values(): assert set(c) == set(rows)
    seen = set(); hashes = {}
    for path in sorted(dest.glob('batch_*.npz')):
        hashes[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            for i, key in enumerate(z['ids'].tolist()):
                assert key not in seen; seen.add(key); target = z['target'][i]
                assert target.nonzero()[0].tolist() == [j+1 for j in rows[key]['mask_h']]
                assert np.all(z['original_inputs'][i, target] == z['mask_id'])
                p = z['prior'][i, target]
                assert p.argmax(-1).tolist() == cases['single_pass'][key]['prediction_ids']
                for kind in protocol['contexts']:
                    hide = z['residues'][i] & ~target
                    if kind == 'light': hide &= np.arange(278) >= 150
                    elif kind == 'heavy': hide &= np.arange(278) < 150
                    expected_input = z['original_inputs'][i].copy(); expected_input[hide] = z['mask_id']
                    assert np.array_equal(expected_input, z[f'inputs_{kind}'][i])
                    q = z[kind][i, target]
                    for probs in (p, q):
                        assert np.isfinite(probs).all() and (probs >= 0).all() and np.allclose(probs.sum(-1), 1, atol=1e-5)
                    for alpha in protocol['alphas']:
                        score = np.log(p.clip(1e-30))+alpha*(np.log(p.clip(1e-30))-np.log(q.clip(1e-30)))
                        score[:, z['forbidden']] = -np.inf
                        assert score.argmax(-1).tolist() == cases[f'{kind}_alpha{alpha}'][key]['prediction_ids']
    assert seen == set(rows)
    summary = json.loads((dest/'summary.json').read_text()); base = cases['single_pass']
    groups = {g: i for i, g in enumerate(sorted({r['group'] for r in rows.values()}))}; counts = np.zeros(len(groups))
    for r in rows.values(): counts[groups[r['group']]] += 1
    weights = np.random.default_rng(20261015).multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=4000)
    comparisons = {}
    for name, c in cases.items():
        mean = np.mean([r['correct']/r['length']*100 for r in c.values()]); assert abs(mean-summary['scores'][name]) < 1e-8
        sums = np.zeros(len(groups))
        for key, r in c.items(): sums[groups[r['group']]] += (r['correct']-base[key]['correct'])/r['length']*100
        comparisons[name] = dict(delta_pp=float(sums.sum()/len(rows)), ci95_pp=np.percentile(weights@sums/(weights@counts), [2.5,97.5]).tolist())
    best = min(expected-{'single_pass'}, key=lambda n: (-summary['scores'][n], n))
    write(dest/'independent_audit.json', dict(status='passed', n=len(rows), predictions=len(rows)*len(cases),
        scores=summary['scores'], comparisons=comparisons, posterior_sha256=hashes, predictions_sha256=sha(dest/'predictions.jsonl'),
        warning='Exploration only; selection not adjusted, independent confirmation required.'))
    selection = dict(created_utc=datetime.now(timezone.utc).isoformat(), candidate=best,
        explore_aar=summary['scores'][best], comparison=comparisons[best],
        audit_sha256=sha(dest/'independent_audit.json'), status='candidate_for_confirmation' if comparisons[best]['delta_pp'] > 0 else 'no_positive_candidate')
    assert not (out/'contrast_selection.json').exists(); write(out/'contrast_selection.json', selection)
    print(json.dumps(selection), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=('run','audit'))
    p.add_argument('--out', type=Path, required=True); p.add_argument('--device', default='cuda', choices=('cpu','cuda'))
    args = p.parse_args(); globals()[args.command](args)
