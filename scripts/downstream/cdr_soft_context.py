"""Activate pllm/protenix_abtcr; python -B FILE run|audit --out ABS.

Soft context uses a hook on token embeddings only, with every scored residue
left at its original MASK embedding. Weights stay frozen. GPU via eval_jobs.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def run(args):
    import torch
    torch.set_num_threads(4)
    out = args.out; helper = module(out/'cdr_context_contrast_v1.py', 'soft_helper')
    runner = helper.load_runner(out); plan, all_rows = runner.verify(out)
    protocol = json.loads((out/'soft_plan.json').read_text()); assert helper.sha(Path(__file__)) == protocol['source_sha256']
    rows = [r for r in all_rows if r['split'] == 'explore' and r['mode'] == 'cdrh3']
    dest = out/'soft_context'; dest.mkdir(exist_ok=False)
    adapter, evaluator, sampler, model, identity = runner.setup(out, plan['checkpoints']['112000'], args.device)
    helper.write(dest/'identity.json', dict(model=identity, source_sha256=helper.sha(Path(__file__))))
    scores = defaultdict(list); forwards = 0; gate_results = []
    with (dest/'predictions.jsonl').open('x') as stream, torch.inference_mode():
        for start in range(0, len(rows), plan['batch_size']):
            chunk = rows[start:start+plan['batch_size']]
            f, b, gold, target, maps = adapter.airgen_inputs(model, chunk)
            f = {k: v.to(args.device) for k, v in f.items()}; b = {k: v.to(args.device) for k, v in b.items()}
            target, gold = target.to(args.device), gold.to(args.device)
            bridge = evaluator.decoder(model, f, maps); predict, counter = runner.prediction_fn(bridge, b, target)
            prior = sampler.base_distribution(predict, b, target, bridge.mask_id); forwards += len(chunk)
            embed = model.backbone.transformer.wte
            positions = [r.nonzero().flatten() for r in target]
            native = [{v: n for v, n in m} for m in maps]
            base_embedding = embed(f['input_ids'])
            replacement = base_embedding.clone(); active = torch.zeros_like(f['input_ids'], dtype=torch.bool)
            calls = [0]
            def hook(layer, inputs, output):
                calls[0] += 1
                assert torch.equal(inputs[0], f['input_ids'])
                return torch.where(active[..., None], replacement, output)
            handle = embed.register_forward_hook(hook)
            posteriors = {}
            try:
                # Empty hook must exactly preserve the real baseline.
                zero = predict(b['input_ids']); forwards += len(chunk)
                assert calls[0] == 1 and torch.equal(zero, prior)
                if start == 0:
                    # A one-hot expected embedding must match hard-token inference.
                    greedy = prior.argmax(-1); hard = b['input_ids'].clone()
                    for i, pos in enumerate(positions):
                        for v in pos[1:].tolist():
                            active[i, native[i][v]] = True
                            replacement[i, native[i][v]] = embed.weight[greedy[i, v]]
                            hard[i, v] = greedy[i, v]
                    soft_onehot = predict(b['input_ids']); forwards += len(chunk)
                    handle.remove(); handle = None
                    alternate, _ = runner.prediction_fn(bridge, {**b, 'input_ids': hard}, target)
                    hard_probs = alternate(hard); forwards += len(chunk)
                    assert torch.equal(soft_onehot, hard_probs)
                    gate_results.append(dict(empty_hook_exact=True, onehot_hook_exact=True, rows=len(chunk)))
                    active.zero_(); replacement.copy_(base_embedding)
                    handle = embed.register_forward_hook(hook)
                for temperature in protocol['temperatures']:
                    weights = prior.pow(1/temperature); weights /= weights.sum(-1, keepdim=True)
                    expected = weights @ embed.weight
                    for strength in protocol['strengths']:
                        name = f'soft_t{temperature}_s{strength}'
                        posterior = prior.clone(); counts = torch.zeros_like(target, dtype=torch.int)
                        for j in range(max(map(len, positions))):
                            active.zero_(); replacement.copy_(base_embedding)
                            for i, pos in enumerate(positions):
                                for k, v in enumerate(pos.tolist()):
                                    if k == j: continue
                                    n = native[i][v]; active[i, n] = True
                                    replacement[i, n] = (1-strength)*base_embedding[i, n]+strength*expected[i, v]
                                if j < len(pos): assert not active[i, native[i][int(pos[j])]]
                            old_calls = calls[0]; probs = predict(b['input_ids']); forwards += len(chunk)
                            assert calls[0] == old_calls+1
                            for i, pos in enumerate(positions):
                                if j < len(pos): posterior[i, pos[j]] = probs[i, pos[j]]; counts[i, pos[j]] += 1
                        assert counts[target].eq(1).all(); posteriors[name] = posterior.cpu().numpy()
            finally:
                if handle is not None: handle.remove()
            arrays = dict(ids=np.array([r['id'] for r in chunk]), target=target.cpu().numpy(), prior=prior.cpu().numpy(), **posteriors)
            np.savez_compressed(dest/f'batch_{start:05d}.npz', **arrays)
            for i, row in enumerate(chunk):
                mask = arrays['target'][i]; p = arrays['prior'][i, mask]
                predictions = {'single_pass': p.argmax(-1)}
                for name, posterior in posteriors.items():
                    for w in protocol['prior_weights']:
                        predictions[f'{name}_prior{w}'] = (w*p+(1-w)*posterior[i, mask]).argmax(-1)
                ref = gold[i, target[i]].tolist()
                for name, guess in predictions.items():
                    correct = sum(a == b for a,b in zip(guess.tolist(),ref)); scores[name].append(correct/len(ref)*100)
                    stream.write(json.dumps(dict(id=row['id'], group=row['group'], policy=name, reference_ids=ref,
                        prediction_ids=guess.tolist(), length=len(ref), correct=correct))+'\n')
            stream.flush(); print('SOFT CONTEXT', start+len(chunk), '/', len(rows), flush=True)
    helper.write(dest/'summary.json', dict(status='passed', n=len(rows), scores={n: float(np.mean(s)) for n,s in scores.items()},
        physical_sample_forwards=forwards, gates=gate_results,
        logical_nfe='Single pass 1; candidate 1+maximum CDR length in its batch, padded queries counted.'))


def audit(args):
    out=args.out; dest=out/'soft_context'; helper=module(out/'cdr_context_contrast_v1.py','soft_audit_helper')
    runner=helper.load_runner(out); plan, all_rows=runner.verify(out)
    protocol=json.loads((out/'soft_plan.json').read_text()); assert helper.sha(Path(__file__)) == protocol['source_sha256']
    rows={r['id']:r for r in all_rows if r['split']=='explore' and r['mode']=='cdrh3'}
    vocab=json.loads((Path(plan['checkpoints']['112000'])/'vocabulary.json').read_text()); token={aa:i for i,aa in enumerate(vocab)}
    cases=defaultdict(dict); sources={}
    for line in (dest/'predictions.jsonl').open():
        r=json.loads(line); key=r['id']; row=rows[key]
        if row['source'] not in sources: sources[row['source']]=[json.loads(x) for x in Path(row['source']).read_text().splitlines()]
        raw=sources[row['source']][row['row']]; a,b=raw['cdrh3_pos']; assert raw['heavy_chain_seq'][a:b+1]==raw['cdrh3_seq']
        assert r['reference_ids']==[token[x] for x in raw['cdrh3_seq'].replace('J','L')]
        assert r['group']==row['group'] and key not in cases[r['policy']]
        assert r['length']==len(r['reference_ids'])==len(r['prediction_ids'])
        assert r['correct']==sum(a==b for a,b in zip(r['prediction_ids'],r['reference_ids']))
        cases[r['policy']][key]=r
    names=[f'soft_t{t}_s{s}' for t in protocol['temperatures'] for s in protocol['strengths']]
    assert set(cases)=={'single_pass'}|{f'{n}_prior{w}' for n in names for w in protocol['prior_weights']}
    for c in cases.values(): assert set(c)==set(rows)
    seen=set(); hashes={}; physical=0; logical=[]
    for path in sorted(dest.glob('batch_*.npz')):
        hashes[str(path)]=helper.sha(path)
        with np.load(path,allow_pickle=False) as z:
            maxlen=int(z['target'].sum(-1).max()); n=len(z['ids']); physical+=n*(2+len(names)*maxlen); logical += [1+maxlen]*n
            if not seen: physical+=2*n
            for i,key in enumerate(z['ids'].tolist()):
                assert key not in seen; seen.add(key); mask=z['target'][i]
                assert mask.nonzero()[0].tolist()==[j+1 for j in rows[key]['mask_h']]
                p=z['prior'][i,mask]; assert p.argmax(-1).tolist()==cases['single_pass'][key]['prediction_ids']
                for name in names:
                    q=z[name][i,mask]
                    for dist in (p,q): assert np.isfinite(dist).all() and (dist>=0).all() and np.allclose(dist.sum(-1),1,atol=1e-5)
                    for w in protocol['prior_weights']:
                        assert (w*p+(1-w)*q).argmax(-1).tolist()==cases[f'{name}_prior{w}'][key]['prediction_ids']
    assert seen==set(rows); summary=json.loads((dest/'summary.json').read_text()); assert physical==summary['physical_sample_forwards']
    groups={g:i for i,g in enumerate(sorted({r['group'] for r in rows.values()}))}; counts=np.zeros(len(groups))
    for r in rows.values(): counts[groups[r['group']]]+=1
    weights=np.random.default_rng(20261015).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=4000)
    comparisons={}
    for name,c in cases.items():
        assert abs(np.mean([r['correct']/r['length']*100 for r in c.values()])-summary['scores'][name])<1e-8
        sums=np.zeros(len(groups))
        for key,r in c.items(): sums[groups[r['group']]]+=(r['correct']-cases['single_pass'][key]['correct'])/r['length']*100
        comparisons[name]=dict(delta_pp=float(sums.sum()/len(rows)),ci95_pp=np.percentile(weights@sums/(weights@counts),[2.5,97.5]).tolist())
    best=min(set(cases)-{'single_pass'},key=lambda n:(-summary['scores'][n],n))
    helper.write(dest/'independent_audit.json',dict(status='passed',n=len(rows),predictions=len(rows)*len(cases),scores=summary['scores'],
        comparisons=comparisons,posterior_sha256=hashes,predictions_sha256=helper.sha(dest/'predictions.jsonl'),
        mean_logical_candidate_nfe=float(np.mean(logical)),physical_sample_forwards=physical,
        limitation='Probability and reference reconstruction; hook isolation asserted at runtime, not independent forward replay. Exploration only.'))
    selection=dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate=best,explore_aar=summary['scores'][best],comparison=comparisons[best],
        audit_sha256=helper.sha(dest/'independent_audit.json'),status='candidate_for_confirmation' if comparisons[best]['delta_pp']>0 else 'no_positive_candidate')
    assert not (out/'soft_selection.json').exists();helper.write(out/'soft_selection.json',selection);print(json.dumps(selection),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('run','audit'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');args=p.parse_args();globals()[args.command](args)
