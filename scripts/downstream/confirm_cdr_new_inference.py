"""Activate pllm/protenix_abtcr; python -B FILE run|audit --out ABS --checkpoint 112000.

Read a frozen development selection, then evaluate reserved modern groups and
the established H3 benchmarks. GPU execution must use an eval_jobs YAML.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

import numpy as np


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def selected_posteriors(model, sampler, predict, batch, target, features, maps, prior, config):
    import torch
    mask=model.tokenizer.mask_token_id
    if config['kind']=='mc_context':
        outputs={}
        for seed in config['seeds']:
            family={**config['family'],'draws':[config['draws']],'seed':seed}
            _,posterior,_,nfe,_=next(sampler.aggregate(predict,batch,target,mask,prior,family))
            outputs[f'candidate_seed{seed}']=(posterior,nfe)
        return outputs
    assert config['kind']=='soft_context'
    embed=model.backbone.transformer.wte; base=embed(features['input_ids'])
    weights=prior.pow(1/config['temperature']);weights/=weights.sum(-1,keepdim=True);expected=weights@embed.weight
    positions=[r.nonzero().flatten() for r in target]; native=[dict(m) for m in maps]
    active=torch.zeros_like(features['input_ids'],dtype=torch.bool);replacement=base.clone()
    def hook(layer, inputs, output):
        assert torch.equal(inputs[0],features['input_ids'])
        return torch.where(active[...,None],replacement,output)
    handle=embed.register_forward_hook(hook); posterior=prior.clone();strength=config['strength']
    try:
        for j in range(max(map(len,positions))):
            active.zero_();replacement.copy_(base)
            for i,pos in enumerate(positions):
                for k,v in enumerate(pos.tolist()):
                    if k!=j:
                        n=native[i][v];active[i,n]=True
                        replacement[i,n]=(1-strength)*base[i,n]+strength*expected[i,v]
                if j<len(pos): assert not active[i,native[i][int(pos[j])]]
            probs=predict(batch['input_ids'])
            for i,pos in enumerate(positions):
                if j<len(pos):posterior[i,pos[j]]=probs[i,pos[j]]
    finally:handle.remove()
    return {'candidate':(posterior,1+max(map(len,positions)))}


def context(args):
    out=args.out;helper=module(out/'cdr_context_contrast_v1.py','confirmation_helper');runner=helper.load_runner(out)
    plan,rows=runner.verify(out); frozen=json.loads((out/'confirmation_plan.json').read_text())
    assert helper.sha(Path(__file__))==frozen['source_sha256']
    assert helper.sha(out/frozen['selection_file'])==frozen['selection_sha256']
    old=runner.OLD; oldplan=json.loads((old/'plan.json').read_text())
    assert helper.sha(old/'rows.json')==oldplan['rows_sha256']
    for path,digest in oldplan['data_sha256'].items(): assert helper.sha(path)==digest
    test=[r for r in rows if r['split']=='confirm']
    test += [r for r in json.loads((old/'rows.json').read_text()) if r['mode']=='cdrh3']
    assert len({r['id'] for r in test})==len(test)
    return helper,runner,plan,frozen,rows,test


def run(args):
    import torch
    torch.set_num_threads(4)
    helper,runner,plan,frozen,all_rows,rows=context(args);out=args.out;cfg=frozen['config']
    dest=out/f'confirmation_{args.checkpoint}';dest.mkdir(exist_ok=False)
    adapter,evaluator,sampler,model,identity=runner.setup(out,plan['checkpoints'][args.checkpoint],args.device)
    helper.write(dest/'identity.json',dict(model=identity,plan_sha256=helper.sha(out/'confirmation_plan.json')))
    scores=defaultdict(lambda:defaultdict(list));forwards=0
    def prepare(chunk):
        f,b,gold,target,maps=adapter.airgen_inputs(model,chunk)
        f={k:v.to(args.device) for k,v in f.items()};b={k:v.to(args.device) for k,v in b.items()}
        target,gold=target.to(args.device),gold.to(args.device);bridge=evaluator.decoder(model,f,maps)
        predict,counter=runner.prediction_fn(bridge,b,target)
        prior=sampler.base_distribution(predict,b,target,bridge.mask_id)
        post=selected_posteriors(model,sampler,predict,b,target,f,maps,prior,cfg)
        return f,b,gold,target,bridge,prior,post,counter
    with torch.inference_mode():
        if args.checkpoint=='112000':
            dev=[r for r in all_rows if r['split']=='explore' and r['mode']=='cdrh3'][:plan['batch_size']]
            f,b,g,t,bridge,p,post,counter=prepare(dev)
            path=out/('soft_context/batch_00000.npz' if cfg['kind']=='soft_context' else 'explore_112000/posterior_batch_00000.npz')
            with np.load(path,allow_pickle=False) as z:
                assert z['ids'].tolist()==[r['id'] for r in dev]
                current=post['candidate' if cfg['kind']=='soft_context' else 'candidate_seed42'][0]
                for i in range(len(dev)):
                    if cfg['kind']=='soft_context': expected=z[cfg['posterior']][i,z['target'][i]]
                    else:expected=z[cfg['posterior']][i,:int(t[i].sum())]
                    actual=current[i,t[i]].cpu().numpy();assert np.allclose(actual,expected,atol=1e-6)
                    assert np.array_equal(actual.argmax(-1),expected.argmax(-1))
            helper.write(dest/'implementation_gate.json',dict(status='passed',n=len(dev),source_posterior=str(path),sample_forwards=counter['calls']*len(dev)))
        with (dest/'predictions.jsonl').open('x') as stream:
            for start in range(0,len(rows),plan['batch_size']):
                chunk=rows[start:start+plan['batch_size']];f,b,gold,target,bridge,prior,post,counter=prepare(chunk)
                original,_=bridge.generate(b,max_iter=1,temperature=1.,partial_masks=~target & b['input_ids'].ne(bridge.bos_id),sampling_strategy='argmax',cfg_scale=0.)
                assert torch.equal(original[target],prior.argmax(-1)[target])
                originals={}
                for steps in (4,8):
                    pred,_=bridge.generate(b,max_iter=steps,temperature=1.,partial_masks=~target & b['input_ids'].ne(bridge.bos_id),sampling_strategy='argmax',cfg_scale=0.)
                    originals[f'original{steps}']=pred
                arrays=dict(ids=np.array([r['id'] for r in chunk]),target=target.cpu().numpy(),prior=prior.cpu().numpy(),
                    **{n:p.cpu().numpy() for n,(p,_) in post.items()})
                np.savez_compressed(dest/f'batch_{start:05d}.npz',**arrays)
                forwards+=(counter['calls']+13)*len(chunk)
                for i,row in enumerate(chunk):
                    ref=gold[i,target[i]].tolist();preds={'single_pass':(prior[i,target[i]].argmax(-1).tolist(),1)}
                    preds.update({n:(p[i,target[i]].tolist(),int(n[-1])) for n,p in originals.items()})
                    w=cfg['prior_weight']
                    preds.update({n:((w*prior[i,target[i]]+(1-w)*p[i,target[i]]).argmax(-1).tolist(),nfe) for n,(p,nfe) in post.items()})
                    for name,(guess,nfe) in preds.items():
                        correct=sum(a==b for a,b in zip(guess,ref));key=row['dataset']+':'+row['mode'];scores[name][key].append(correct/len(ref)*100)
                        stream.write(json.dumps(dict(id=row['id'],group=row['group'],dataset=row['dataset'],mode=row['mode'],fold=row['fold'],
                            policy=name,reference_ids=ref,prediction_ids=guess,correct=correct,length=len(ref),logical_nfe=nfe))+'\n')
                stream.flush();print('CONFIRM',args.checkpoint,start+len(chunk),'/',len(rows),flush=True)
    helper.write(dest/'summary.json',dict(status='passed',n=len(rows),physical_sample_forwards=forwards,
        mean_sequence_aar={n:{k:float(np.mean(v)) for k,v in s.items()} for n,s in scores.items()},
        gate_cost_separate=True,source_sha256=helper.sha(Path(__file__))))


def audit(args):
    helper,runner,plan,frozen,_,rows=context(args);out=args.out;dest=out/f'confirmation_{args.checkpoint}';cfg=frozen['config']
    byid={r['id']:r for r in rows};vocab=json.loads((Path(plan['checkpoints'][args.checkpoint])/'vocabulary.json').read_text());token={aa:i for i,aa in enumerate(vocab)}
    cases=defaultdict(dict)
    for line in (dest/'predictions.jsonl').open():
        r=json.loads(line);row=byid[r['id']];ref=[token[row['gold_h'][j]] for j in row['mask_h']]
        assert r['reference_ids']==ref and r['length']==len(ref)==len(r['prediction_ids'])
        assert r['correct']==sum(a==b for a,b in zip(ref,r['prediction_ids']))
        for field in ('group','dataset','mode','fold'):assert r[field]==row[field]
        assert r['id'] not in cases[r['policy']];cases[r['policy']][r['id']]=r
    names=['candidate'] if cfg['kind']=='soft_context' else [f'candidate_seed{s}' for s in cfg['seeds']]
    assert set(cases)=={'single_pass','original4','original8'}|set(names)
    for c in cases.values():assert set(c)==set(byid)
    seen=set();hashes={};physical=0
    for path in sorted(dest.glob('batch_*.npz')):
        hashes[str(path)]=helper.sha(path)
        with np.load(path,allow_pickle=False) as z:
            nfe=1+int(z['target'].sum(-1).max()) if cfg['kind']=='soft_context' else 1+cfg['draws']
            physical+=len(z['ids'])*(14+(nfe-1)*len(names))
            for i,key in enumerate(z['ids'].tolist()):
                assert key not in seen;seen.add(key);mask=z['target'][i]
                assert mask.nonzero()[0].tolist()==[j+1 for j in byid[key]['mask_h']]
                prior=z['prior'][i,mask];assert prior.argmax(-1).tolist()==cases['single_pass'][key]['prediction_ids']
                for name in names:
                    q=z[name][i,mask];w=cfg['prior_weight']
                    for p in (prior,q):assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(-1),1,atol=1e-5)
                    assert (w*prior+(1-w)*q).argmax(-1).tolist()==cases[name][key]['prediction_ids']
                    assert cases[name][key]['logical_nfe']==nfe
    assert seen==set(byid);summary=json.loads((dest/'summary.json').read_text());assert physical==summary['physical_sample_forwards']
    results={}
    for key in sorted({r['dataset']+':'+r['mode'] for r in rows}):
        subset=[r for r in rows if r['dataset']+':'+r['mode']==key]
        groups={g:i for i,g in enumerate(sorted({r['group'] for r in subset}))};folds={f:i for i,f in enumerate(sorted({r['fold'] for r in subset}))}
        counts=np.zeros((len(groups),len(folds)));sums={n:np.zeros_like(counts) for n in cases}
        for r in subset:
            g,f=groups[r['group']],folds[r['fold']];counts[g,f]+=1
            for name in cases:
                record=cases[name][r['id']];sums[name][g,f]+=100*record['correct']/record['length']
        means={n:float((s.sum(0)/counts.sum(0)).mean()) for n,s in sums.items()}
        sums['candidate_mean']=np.mean([sums[n] for n in names],axis=0)
        means['candidate_mean']=float((sums['candidate_mean'].sum(0)/counts.sum(0)).mean())
        weights=np.random.default_rng(20261015).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=4000)
        den=weights@counts;valid=(den>0).all(-1);weights=weights[valid];den=den[valid]
        comparisons={}
        for base in ('single_pass','original4','original8'):
            delta=sums['candidate_mean']-sums[base];boot=((weights@delta)/den).mean(-1)
            comparisons[base]=dict(delta_pp=means['candidate_mean']-means[base],ci95_pp=np.percentile(boot,[2.5,97.5]).tolist())
        results[key]=dict(n=len(subset),groups=len(groups),fold_macro_aar=means,comparisons=comparisons,
            candidate_mean_nfe=float(np.mean([cases[names[0]][r['id']]['logical_nfe'] for r in subset])))
    helper.write(dest/'independent_audit.json',dict(status='passed',n=len(rows),predictions=len(rows)*len(cases),results=results,
        posterior_sha256=hashes,predictions_sha256=helper.sha(dest/'predictions.jsonl'),physical_sample_forwards=physical,
        limitations='Modern component confirmation is primary; established benchmarks are historical checks, not independent new test data. MC reports mean across three fixed seeds. Original4/8 scores checked from predictions, no per-step trajectory replay here.'))
    print(json.dumps(dict(checkpoint=args.checkpoint,results=results)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('run','audit'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',choices=('112000','50000'),required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=p.parse_args();globals()[args.command](args)
