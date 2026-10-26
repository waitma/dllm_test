"""Evaluate frozen Stage1 models on native H/L partner retrieval.

Activate /project_c20250601/251105016/conda/envs/pllm, then invoke this absolute
file with prepare, gate, or run and --output ABS_DIRECTORY. prepare creates a
new immutable protocol. gate checks real inputs/weights on CPU/CUDA; run writes
append-only per-candidate scores and can resume completed candidates. No model
weights are updated. This grouped-mask score is not exact pseudo-likelihood.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
from evaluate_stage1_frozen_gdp import ROOT, sha, write

REGIONS = ['fwr1','cdr1','fwr2','cdr2','fwr3','cdr3','fwr4']
READOUTS = ['all', 'cdr1', 'cdr2', 'cdr3', 'framework']


def load(out):
    p = json.loads((out/'plan.json').read_text())
    assert sha(__file__) == p['script_sha256'], 'Source changed after protocol freeze'
    for name in ['records.json','candidates.json']:
        assert sha(out/name) == p['files'][name]
    return p, json.loads((out/'records.json').read_text()), json.loads((out/'candidates.json').read_text())


def prepare(args):
    if args.output.exists(): raise FileExistsError(args.output)
    source = ROOT/'data/downstream/comp_chain/test_data_oas_holdout.csv'
    df = pd.read_csv(source); assert len(df)==500
    records=[]
    for i, row in df.iterrows():
        record=dict(row=int(i),species=row.species,light_locus=row.l_locus,
                    ab_cluster=str(row.ab_cluster_id))
        for role, cap in [('h',166),('l',134)]:
            s=row[role+'_sequence'].replace('-','')
            pieces=[row[role+'_'+r].replace('-','') for r in REGIONS]
            assert s==row['cleaned_'+role+'_sequence']==''.join(pieces)
            assert set(s)<=set('ACDEFGHIKLMNPQRSTVWY') and len(s)<=cap
            record[role]=s; record[role+'_cluster']=row[role.upper()+'_cluster_id']
            match=re.match(r'(IG[HKL]V\d+)',row[role+'_v_call'])
            assert match, row[role+'_v_call']
            record[role+'_family']=match.group(1)
            record[role+'_regions']=[name for name,part in zip(REGIONS,pieces) for _ in part]
        records.append(record)
    # Audit only the actual selected training requests, not the broader parent corpus.
    gdp=ROOT/'output/stage1_downstream_9770_20261025/gdp_fixed_ridge/plan.json'
    bindings=json.loads(gdp.read_text())['arms']
    subset_path=ROOT/'output/stage1_ab500k_pilot_20261022/data/subset.json'
    subset=json.loads(subset_path.read_text())
    assert sha(subset_path)==bindings['mlm']['provenance']['subset_manifest_sha256']
    maps={kind:defaultdict(list) for kind in ['h','l','pair']}
    for r in records:
        for kind,key in [('h',r['h']),('l',r['l']),('pair',(r['h'],r['l']))]:maps[kind][key].append(r['row'])
    hits={k:Counter() for k in maps}; counts=Counter();digest=hashlib.sha256()
    with Path(subset['requests']).open('rb',buffering=8<<20) as stream:
        for raw in stream:
            digest.update(raw);row=json.loads(raw);counts[row['split']]+=1
            if row['split']!='train':continue
            c={x['role']:x['sequence'] for x in row['request']['chains']}
            h,l=c['antibody_heavy'],c['antibody_light']
            for kind,key in [('h',h),('l',l),('pair',(h,l))]:
                for index in maps[kind].get(key,[]):hits[kind][index]+=1
    assert digest.hexdigest()==subset['requests_sha256'] and counts['train']==500000
    for r in records:r['training_exact_counts']={k:int(hits[k][r['row']]) for k in hits}
    candidates=[]; groups=[]; rng=np.random.default_rng(20261025)
    for target,query in [('l','h'),('h','l')]:
        partners=defaultdict(set)
        for r in records:partners[r[query]].add(r[target])
        for r in records:
            pool=[];seen=set()
            for s in records:
                if s['species']!=r['species']:continue
                if target=='l' and s['light_locus']!=r['light_locus']:continue
                if s[target] in partners[r[query]] or s[target] in seen:continue
                if s[target+'_cluster']==r[target+'_cluster']:continue
                seen.add(s[target])
                pool.append((s[target+'_family']!=r[target+'_family'],abs(len(s[target])-len(r[target])),rng.random(),s['row']))
            selected=[r['row']]+[x[-1] for x in sorted(pool)[:15]]
            group=len(groups)
            groups.append(dict(group=group,target=target,query_row=r['row'],candidate_count=len(selected),
                               retrieval_eligible=len(selected)>1))
            for rank,j in enumerate(selected):
                s=records[j]
                candidates.append(dict(item=len(candidates),group=group,target=target,query_row=r['row'],
                    target_row=j,h_row=j if target=='h' else r['row'],l_row=j if target=='l' else r['row'],
                    native=rank==0,same_family=s[target+'_family']==r[target+'_family'],
                    length_difference=abs(len(s[target])-len(r[target]))))
    args.output.mkdir(parents=True)
    write(args.output/'records.json',records);write(args.output/'candidates.json',candidates)
    plan=dict(schema='stage1.native_partner.group_mask.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        data=str(source),data_sha256=sha(source),arms=bindings,script_sha256=sha(__file__),
        files={n:sha(args.output/n) for n in ['records.json','candidates.json']},groups=groups,
        checkpoint_step=9770,mask_groups=8,mask_rule='Zero-based target residue offset modulo 8; all target residues scored once',
        conditions=['observed_partner','masked_partner'],precision='float32; TF32 disabled',
        candidate_rule='All 500 pairs, both directions. Same species; L candidates same locus; '
        'exclude other observed partners and same target CDR3 cluster. Prefer same V family, then closest length, '
        'seeded random tiebreak. Up to 15 unique decoys; variable pool sizes disclosed.',
        primary='All-residue mean conditional log-probability retrieval, both directions; equal weight per query row',
        supporting=['context_gain = observed minus masked partner log-probability','CDR3-only retrieval',
                    'native-partner residue recovery/NLL by region','paired context effect by region'],
        metrics=['Recall@1','Recall@5','MRR'],ties='average over random ordering within exactly tied scores',
        exact_exposure={k:len(v) for k,v in hits.items()},request_sha256=digest.hexdigest(),
        limitations=['Grouped simultaneous masks, not single-site PLL or normalized joint probability',
            'Repaired chains are non-observed partners, not experimental negatives',
            'Single pretrained seed; residual homology and teacher pretraining exposure unknown',
            'Masked-partner control is out of training distribution and retains partner length',
            'Candidate matching uses native metadata only to define a controlled evaluation pool'])
    write(args.output/'plan.json',plan)
    print(json.dumps(dict(phase='prepared',records=len(records),groups=len(groups),items=len(candidates),
                          exact_exposure=plan['exact_exposure'],pool_sizes=dict(Counter(g['candidate_count'] for g in groups)))),flush=True)


def model_and_collator(plan,arm,device,threads):
    import torch
    binding=plan['arms'][arm]
    sys.path.insert(0,binding['source_root'])
    for source,digest in binding['provenance']['sources'].items():assert sha(source)==digest,source
    checkpoint=Path(binding['checkpoint'])
    assert sha(checkpoint/'model.safetensors')==binding['model_sha256']
    assert sha(checkpoint/'research_config.json')==binding['research_config_sha256']
    from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMModel,FullTokenMLMCollator
    from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel
    assert Path(sys.modules[StructureMLMModel.__module__].__file__).is_relative_to(binding['source_root'])
    torch.set_num_threads(threads);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=(FullTokenMLMModel if arm=='mlm' else StructureMLMModel).from_pretrained(checkpoint)
    model=model.to(device).float().eval().requires_grad_(False)
    return model,FullTokenMLMCollator(model.config,hide_relation=True)


def assemble(records,items,collator,model,device):
    import torch
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord,BioSeqChain
    seqs=[BioSeqRecord([BioSeqChain(records[x['h_row']]['h'],'antibody_heavy'),
                      BioSeqChain(records[x['l_row']]['l'],'antibody_light')],
                      task_type='antibody',source='stage1_frozen_partner_eval') for x in items]
    batch=collator(seqs)
    masks=[];regions=[]
    for i,x in enumerate(items):
        residue=batch['residue_mask'][i].bool() & batch['attention_mask'][i].bool()
        chain_ids=batch['chain_ids'][i][residue].unique(sorted=True)
        assert len(chain_ids)==2
        selected=residue & (batch['chain_ids'][i]==chain_ids[0 if x['target']=='h' else 1])
        assert int(selected.sum())==len(records[x['target_row']][x['target']])
        labels=records[x['target_row']][x['target']+'_regions']
        region=torch.zeros((len(READOUTS),selected.numel()),dtype=torch.bool)
        positions=selected.nonzero().flatten()
        region[0,positions]=True
        for j,name in enumerate(labels):
            region[READOUTS.index(name) if name.startswith('cdr') else 4,positions[j]]=True
        assert not (selected & batch['terminal_mask'][i]).any()
        masks.append(selected);regions.append(region)
    assert (batch['input_ids'][batch['relation_mask']]==model.tokenizer.mask_token_id).all()
    return {k:v.to(device) for k,v in model.feature_args(batch).items()},torch.stack(masks).to(device),torch.stack(regions).to(device),batch['residue_mask'].to(device).bool()


def score_batch(model,features,target_mask,regions,residue_mask,groups=8):
    import torch
    original=features['input_ids'].clone()
    offsets=target_mask.long().cumsum(1)-1
    totals=torch.zeros((len(original),2,len(READOUTS),3),device=original.device,dtype=torch.float64)
    with torch.inference_mode():
        for group in range(groups):
            selected=target_mask & ((offsets%groups)==group)
            for condition in range(2):
                masked=selected | ((residue_mask & ~target_mask) if condition else torch.zeros_like(selected))
                current=dict(features);current['input_ids']=original.masked_fill(masked,model.tokenizer.mask_token_id)
                logits,_=model.denoise(**current)
                pos=selected.nonzero()
                values=logits[pos[:,0],pos[:,1]].float()
                truth=original[pos[:,0],pos[:,1]]
                logp=values.log_softmax(-1).gather(1,truth[:,None]).squeeze(1)
                correct=values.argmax(-1).eq(truth).double()
                assert torch.isfinite(logp).all()
                for r in range(len(READOUTS)):
                    keep=regions[pos[:,0],r,pos[:,1]].double()
                    totals[:,condition,r,0].index_add_(0,pos[:,0],logp.double()*keep)
                    totals[:,condition,r,1].index_add_(0,pos[:,0],correct*keep)
                    totals[:,condition,r,2].index_add_(0,pos[:,0],keep)
        assert torch.equal(totals[:,0,:,2],regions.sum(-1).double())
        assert torch.equal(totals[:,0,:,2],totals[:,1,:,2])
    return totals.cpu().numpy()


def gate(args):
    import torch
    plan,records,items=load(args.output)
    model,collator=model_and_collator(plan,args.arm,'cpu',args.threads)
    chosen=[items[0],items[1],next(x for x in items if x['target']=='h')]
    features,target,regions,residue=assemble(records,chosen,collator,model,'cpu')
    start=time.monotonic()
    reference=score_batch(model,features,target,regions,residue)
    # Single-vs-batch check; neither labels nor region metadata enters model forward.
    single=score_batch(model,{k:v[:1] for k,v in features.items()},target[:1],regions[:1],residue[:1])
    np.testing.assert_allclose(single,reference[:1],atol=3e-4,rtol=3e-5)
    # If the scored/partner residues are changed before the same masks are applied,
    # the actual forward features are identical; fixed lengths/grammar are allowed.
    selected=target & (((target.long().cumsum(1)-1)%8)==0)
    hidden=residue & ~target | selected
    altered=features['input_ids'].clone();altered[hidden]=model.tokenizer.mask_token_id
    assert torch.equal(features['input_ids'].masked_fill(hidden,model.tokenizer.mask_token_id),
                       altered.masked_fill(hidden,model.tokenizer.mask_token_id))
    report=dict(status='passed',arm=args.arm,plan_sha256=sha(args.output/'plan.json'),
        single_batch_max_abs=float(np.abs(single-reference[:1]).max()),residue_counts_verified=True,
        scored_positions_hidden=True,relation_hidden=True,full_pair_forward=True,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),cpu_seconds=time.monotonic()-start)
    if args.device!='cpu':
        model=model.to(args.device)
        gpu=score_batch(model,{k:v.to(args.device) for k,v in features.items()},target.to(args.device),regions.to(args.device),residue.to(args.device))
        np.testing.assert_allclose(gpu,reference,atol=3e-3,rtol=3e-4)
        report['cpu_gpu_max_abs']=float(np.abs(gpu-reference).max())
        # Time a representative production batch, without publishing its task scores.
        b=assemble(records,items[:args.batch_size],collator,model,args.device)
        torch.cuda.synchronize();start=time.monotonic();score_batch(model,*b);torch.cuda.synchronize()
        report['profile_candidates']=len(b[0]['input_ids']);report['profile_seconds']=time.monotonic()-start
        report['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    write(args.output/(args.arm+'_gate.json'),report)
    print(json.dumps(report),flush=True)


def run(args):
    plan,records,items=load(args.output)
    receipt=json.loads((args.output/(args.arm+'_gate.json')).read_text())
    assert receipt['status']=='passed' and receipt['plan_sha256']==sha(args.output/'plan.json')
    path=args.output/(args.arm+'_scores.jsonl')
    done=set()
    if path.exists():
        with path.open() as f:
            for line in f:
                row=json.loads(line)
                assert row['item'] not in done and row['plan_sha256']==sha(args.output/'plan.json')
                done.add(row['item'])
    assert done<=set(range(len(items)))
    remaining=[x for x in items if x['item'] not in done]
    model,collator=model_and_collator(plan,args.arm,args.device,args.threads)
    start=time.monotonic();next_log=0;plan_sha=sha(args.output/'plan.json')
    with path.open('a',buffering=1) as out:
        for offset in range(0,len(remaining),args.batch_size):
            chunk=remaining[offset:offset+args.batch_size]
            b=assemble(records,chunk,collator,model,args.device)
            values=score_batch(model,*b)
            for item,value in zip(chunk,values):
                out.write(json.dumps(dict(item=item['item'],plan_sha256=plan_sha,statistics=value.tolist()),allow_nan=False)+'\n')
                done.add(item['item'])
            if time.monotonic()>=next_log or len(done)==len(items):
                status=dict(phase='running',arm=args.arm,done=len(done),total=len(items),
                            elapsed_seconds=time.monotonic()-start,updated_utc=datetime.now(timezone.utc).isoformat())
                write(args.output/(args.arm+'_status.json'),status);print(json.dumps(status),flush=True)
                next_log=time.monotonic()+60
    assert len(done)==len(items)
    assert sha(Path(plan['arms'][args.arm]['checkpoint'])/'model.safetensors')==plan['arms'][args.arm]['model_sha256']
    write(args.output/(args.arm+'_complete.json'),dict(status='complete',items=len(done),
        plan_sha256=plan_sha,scores_sha256=sha(path),seconds_this_attempt=time.monotonic()-start))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','gate','run']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['mlm','pair_kd']);p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--threads',type=int,default=8)
    args=p.parse_args();assert args.output.is_absolute()
    if args.action!='prepare' and args.arm is None:p.error('--arm required')
    globals()[args.action](args)


if __name__=='__main__':main()
