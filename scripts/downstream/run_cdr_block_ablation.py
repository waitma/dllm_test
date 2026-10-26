"""Activate pllm/protenix_abtcr; python -B FILE gate|run --out ABS --checkpoint 112000.

Frozen factorial ablation without model training or candidate selection. GPU
execution must use eval_jobs YAML. Save the actual feedback and query traces.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def context(out):
    plan = json.loads((out/'plan.json').read_text())
    for field in ('source_sha256', 'external_source_sha256'):
        for path, digest in plan[field].items(): assert sha(path) == digest, path
    for name, field in (('rows.json','rows_sha256'), ('data_plan.json','data_plan_sha256'), ('data_independent_audit.json','data_audit_sha256')):
        assert sha(out/name) == plan[field]
    parent = module(Path(plan['parent'])/'source/run_cdr_block_context.py', 'ablation_parent')
    parent_plan, old, reference, dev = parent.context(Path(plan['parent']))
    assert parent_plan['checkpoints'] == plan['checkpoints']
    sampler = module(out/'source/cdr_block_ablation.py', 'ablation_sampler')
    return plan, parent, old, reference, sampler, dev


def gate(args):
    import torch
    torch.set_num_threads(4)
    plan, parent, old, reference, sampler, dev = context(args.out)
    adapter, evaluator, _, model, identity = old.setup(Path(plan['marginal_parent']), plan['checkpoints'][args.checkpoint], args.device)
    with torch.inference_mode():
        f,b,gold,target,maps,bridge,predict,prior,physical = parent.prepare(adapter,evaluator,old,model,dev[:2],args.device)
        changed = [{**r,'gold_h':'G'*len(r['gold_h']),'gold_l':'G'*len(r['gold_l'])} for r in dev[:2]]
        f2,b2,_,t2,_ = adapter.airgen_inputs(model,changed)
        assert all(torch.equal(v.cpu(),f2[k]) for k,v in f.items())
        assert torch.equal(b['input_ids'].cpu(),b2['input_ids']) and torch.equal(target.cpu(),t2)
        native,_ = bridge.generate(b,max_iter=1,temperature=1.,partial_masks=~target & b['input_ids'].ne(bridge.bos_id),sampling_strategy='argmax',cfg_scale=0.)
        assert torch.equal(native[target],prior.argmax(-1)[target])
        common = dict(groups=3,rule='interleaved',temperature=.5,strength=.5)
        expected,_ = reference.posterior(model,predict,b,target,f,maps,prior,**common)
        actual,_ = sampler.posterior(model,predict,b,target,f,maps,prior,**common)
        assert torch.equal(actual,expected)
        for kind in ('soft','hard'):
            for feedback in ('frozen','sequential'):
                q,_ = sampler.posterior(model,predict,b,target,f,maps,prior,**{**common,'strength':0.},context=kind,feedback=feedback)
                assert torch.equal(q,prior)
        masks = sampler.partitions(target,3,'interleaved')
        traces = []
        def observe(index,scored,feedback,weights,probs):
            traces.append((scored.clone(),feedback.clone(),weights.clone(),probs.clone()))
        seq,_ = sampler.posterior(model,predict,b,target,f,maps,prior,**common,feedback='sequential',observer=observe)
        bank = prior.clone()
        for j,(scored,feedback,weights,probs) in enumerate(traces):
            assert torch.equal(scored,masks[j]) and torch.equal(feedback,target & ~scored)
            w = bank.pow(2); w /= w.sum(-1,keepdim=True)
            assert torch.equal(weights[feedback],w[feedback]) and weights[~feedback].eq(0).all()
            assert torch.equal(seq[scored],probs[scored])
            bank[scored] = probs[scored]
        assert torch.equal(predict(b['input_ids']),prior)
        assert not model.training and not any(p.requires_grad for p in model.parameters())
    record = dict(status='passed',checkpoint=args.checkpoint,device=args.device,identity=identity,n=2,
        no_reference_in_features=True,original_single_pass_exact=True,full_method_matches_frozen_parent_exact=True,
        zero_strength_all_factor_cells_exact=True,sequential_feedback_bank_reconstructed=True,hook_cleanup_exact=True,
        physical_sample_forwards=physical['calls']*2,plan_sha256=sha(args.out/'plan.json'))
    write(args.out/f'gate_{args.device}_{args.checkpoint}.json',record); print(json.dumps(record),flush=True)


def run(args):
    import torch
    torch.set_num_threads(4)
    plan,parent,old,_,sampler,_ = context(args.out)
    gate_record = json.loads((args.out/f'gate_{args.device}_{args.checkpoint}.json').read_text())
    assert gate_record['status']=='passed' and gate_record['plan_sha256']==sha(args.out/'plan.json')
    rows = json.loads((args.out/'rows.json').read_text())
    assert len(rows)==5254 and all(r['split']=='ablation' and r['parent_split']=='reserve' for r in rows)
    dest = args.out/f'evaluation_{args.checkpoint}'; dest.mkdir(exist_ok=False)
    write(dest/'status.json',dict(status='running',started_utc=datetime.now(timezone.utc).isoformat()))
    adapter,evaluator,_,model,identity = old.setup(Path(plan['marginal_parent']),plan['checkpoints'][args.checkpoint],args.device)
    write(dest/'identity.json',dict(model=identity,plan_sha256=sha(args.out/'plan.json'),phase='ablation'))
    scores = defaultdict(lambda:defaultdict(list)); forwards=0
    with (dest/'predictions.jsonl').open('x') as stream,(dest/'query_audit.jsonl').open('x') as queries,torch.inference_mode():
        for start in range(0,len(rows),plan['batch_size']):
            chunk=rows[start:start+plan['batch_size']]
            f,b,gold,target,maps,bridge,predict,prior,physical=parent.prepare(adapter,evaluator,old,model,chunk,args.device)
            positions=[x.nonzero().flatten() for x in target];lengths=np.array([len(p) for p in positions],dtype=np.int64)
            def pack(probs):
                packed=np.zeros((len(chunk),int(lengths.max()),probs.shape[-1]),dtype=np.float32)
                for i,pos in enumerate(positions):packed[i,:len(pos)]=probs[i,pos].cpu().numpy()
                return packed
            position_array=np.full((len(chunk),int(lengths.max())),-1,dtype=np.int64)
            for i,pos in enumerate(positions):position_array[i,:len(pos)]=pos.cpu().numpy()
            arrays=dict(ids=np.array([r['id'] for r in chunk]),lengths=lengths,positions=position_array,prior=pack(prior))
            predictions={'single_pass':(prior.argmax(-1),1)}
            for family in plan['families']:
                weights_trace=[];probs_trace=[]
                def observer(index,scored,feedback,weights,probs):
                    assert index==len(weights_trace)
                    weights_trace.append(pack(weights));probs_trace.append(pack(probs))
                    for i,row in enumerate(chunk):
                        queries.write(json.dumps(dict(id=row['id'],family=family['name'],query=index,
                            scored=scored[i,target[i]].cpu().tolist(),feedback=feedback[i,target[i]].cpu().tolist()))+'\n')
                config={k:v for k,v in family.items() if k not in ('name','prior_weights')}
                post,nfe=sampler.posterior(model,predict,b,target,f,maps,prior,**config,observer=observer)
                assert len(weights_trace)==family['groups']
                arrays[family['name']]=pack(post)
                arrays[family['name']+'__weights']=np.stack(weights_trace)
                arrays[family['name']+'__queries']=np.stack(probs_trace)
                for weight in family['prior_weights']:
                    predictions[family['name']+'_prior'+str(weight)]=((weight*prior+(1-weight)*post).argmax(-1),nfe)
            for steps in (4,8):
                before=physical['calls']
                pred,_=bridge.generate(b,max_iter=steps,temperature=1.,partial_masks=~target & b['input_ids'].ne(bridge.bos_id),sampling_strategy='argmax',cfg_scale=0.)
                assert physical['calls']-before==steps
                predictions[f'original{steps}']=(pred,steps)
            assert len(predictions)==13 and physical['calls']==31
            forwards+=physical['calls']*len(chunk)
            np.savez_compressed(dest/f'batch_{start:05d}.npz',**arrays)
            for i,row in enumerate(chunk):
                ref=gold[i,target[i]].tolist()
                for name,(pred,nfe) in predictions.items():
                    guess=pred[i,target[i]].tolist();correct=sum(a==b for a,b in zip(guess,ref))
                    scores[name][row['dataset']+':'+row['mode']].append(100*correct/len(ref))
                    stream.write(json.dumps(dict(id=row['id'],group=row['group'],dataset=row['dataset'],mode=row['mode'],fold=row['fold'],
                        policy=name,reference_ids=ref,prediction_ids=guess,correct=correct,length=len(ref),logical_nfe=nfe))+'\n')
            stream.flush();queries.flush();print('ABLATION',args.checkpoint,start+len(chunk),'/',len(rows),flush=True)
    write(dest/'summary.json',dict(status='passed',n=len(rows),physical_sample_forwards=forwards,
        mean_sequence_aar={name:{k:float(np.mean(s)) for k,s in by.items()} for name,by in scores.items()},
        gate_cost_separate=True,plan_sha256=sha(args.out/'plan.json')))
    write(dest/'status.json',dict(status='passed',finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('gate','run'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',choices=('112000','50000'),required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');args=p.parse_args();globals()[args.command](args)
