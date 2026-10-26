"""Frozen no-ESMC scoring and antigen-only paired generation.

Run: python -m downstream.benchmark.ab_antigen.sequence_eval --task binding --out-dir DIR
Use Volc GPU jobs; --preflight validates CPU data/config only. Scores are project
conditional pseudo-likelihood, never reported as author baseline implementations.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from downstream.benchmark.ab_antigen.sequence_data import ROOT, DATA, MAX_LENGTH, digest, write_json

CHECKPOINT = ROOT/'output/ab_eval_checkpoints/v2_noesmc_immune_v6_150000_20260928'
EXPECTED_SHA = 'a6ce5c2c22b2e938f4573eb552b6a8ce1c2f42a3647a345adc98e6a5a052b048'
AA='ACDEFGHIKLMNPQRSTVWY'


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def make_record(antigen, heavy, light):
    from dllm.pipelines.immune_llada.data import BioSeqChain,BioSeqRecord
    return BioSeqRecord([BioSeqChain(antigen,'antigen'),BioSeqChain(heavy,'antibody_heavy'),
        BioSeqChain(light,'antibody_light')],task_type='antibody_antigen',source='ab_ag_sequence_v1')


def make_generation_record(antigen):
    # No reference or reference length is accepted at this interface.
    return make_record(antigen,'A'*166,'A'*134)


def validate_batch(batch, antigen, heavy, light):
    chains=batch.get('position_ids_chain',batch.get('chain_ids'))
    residue=batch['residue_mask'].bool() & batch['attention_mask'].bool()
    counts=[int((residue & chains.eq(c)).sum()) for c in range(3)]
    if counts != [len(antigen),len(heavy),len(light)]:
        raise ValueError(f'Truncated or misassigned input: {counts}')
    return chains,residue


def move(batch,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


@torch.no_grad()
def score_pair(bundle, ag, h, l, batch_size=2, features=True, device="cuda"):
    batch=bundle.collator([make_record(ag,h,l)])
    chains,residue=validate_batch(batch,ag,h,l)
    batch=move(batch,device);chains=chains.to(device);residue=residue.to(device)
    model=bundle.model
    pooled=None
    if features:
        hidden=model.last_hidden_state(**batch).float()
        pooled=torch.cat([hidden[(residue & chains.eq(c))].mean(0) for c in range(3)]).cpu().tolist()
        if not np.isfinite(pooled).all():raise ValueError('Nonfinite features')
        del hidden
    positions=torch.where((residue & chains.ge(1))[0])[0]
    values=[]
    for start in range(0,len(positions),batch_size):
        pos=positions[start:start+batch_size];n=len(pos)
        b={k:(v.repeat((n,)+(1,)*(v.ndim-1)) if torch.is_tensor(v) and v.ndim and v.shape[0]==1 else v) for k,v in batch.items()}
        b['input_ids']=b['input_ids'].clone()
        target=b['input_ids'][torch.arange(n,device=device),pos].clone()
        b['input_ids'][torch.arange(n,device=device),pos]=model.config.mask_token_id
        # _denoise is inference; model.forward would sample a training corruption.
        out=model._denoise(**b).logits
        selected=out[torch.arange(n,device=device),pos].float()
        lp=selected.log_softmax(-1).gather(1,target[:,None]).squeeze(1)
        values.extend(lp.cpu().tolist());del out,selected,b
    if len(values)!=len(h)+len(l) or not np.isfinite(values).all():raise ValueError('Invalid PLL')
    return float(np.mean(values)),pooled


@torch.no_grad()
def generate_one(bundle, antigen, seed):
    from downstream.grammar.common import run_grammar_generate
    batch=bundle.collator([make_generation_record(antigen)])
    chains,_=validate_batch(batch,antigen,'A'*166,'A'*134)
    slots=batch['chain_slot_mask'].bool() & batch['attention_mask'].bool()
    target=slots & chains.ge(1)
    if int(target.sum())!=302:raise ValueError('Both complete EOS canvases must be hidden')
    partial=batch['attention_mask'].bool() & ~target
    batch=move(batch,'cuda');partial=partial.cuda()
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    tokens,_=run_grammar_generate(bundle.model,batch,partial_mask=partial,max_iter=32,
        temperature=1.,cfg_scale=0.)
    tokenizer=bundle.grammar_tokenizer
    lookup={tokenizer.encode_residues(a)[0]:a for a in AA}
    result={}
    for c,name in [(1,'heavy'),(2,'light')]:
        ids=tokens[0][(slots & chains.eq(c))[0].cuda()].tolist()
        terminated=tokenizer.eos_token_id in ids
        end=ids.index(tokenizer.eos_token_id) if terminated else len(ids)
        seq=''.join(lookup.get(i,'X') for i in ids[:end])
        result[name]=seq;result[name+'_eos']=terminated
    result['token_valid']=all(result[n+'_eos'] and result[n] and set(result[n])<=set(AA) for n in ['heavy','light'])
    result['failure_reason']=None if result['token_valid'] else 'missing_eos_or_invalid_residue'
    return result


def group_metrics(rows,score_key='score',group_key='antibody_key',classification=False):
    from sklearn.metrics import roc_auc_score,average_precision_score
    from scipy.stats import spearmanr
    groups=defaultdict(list)
    for r in rows:groups[r[group_key]].append(r)
    values=[];excluded=[]
    for key,items in sorted(groups.items()):
        y=np.array([r['label'] for r in items]);s=np.array([r[score_key] for r in items])
        if len(y)<(2 if classification else 3) or len(np.unique(y))<2 or not np.isfinite(s).all():
            excluded.append(dict(group=key,n=len(y),reason='insufficient_or_constant_label'));continue
        if classification:
            metrics=dict(roc_auc=float(roc_auc_score(y,s)),auprc=float(average_precision_score(y,s)))
        else:
            if len(np.unique(s))<2:
                excluded.append(dict(group=key,n=len(y),reason='constant_prediction'));continue
            metrics=dict(spearman=float(spearmanr(y,s).statistic))
        values.append(dict(group=key,n=len(y),**metrics))
    summary={}
    for key in (['roc_auc','auprc'] if classification else ['spearman']):
        a=np.array([r[key] for r in values]);rng=np.random.default_rng(42)
        ci=np.quantile([rng.choice(a,len(a),replace=True).mean() for _ in range(1000)],[.025,.975]).tolist() if len(a)>1 else None
        summary[key]=dict(mean=float(a.mean()) if len(a) else None,ci95=ci)
    return dict(groups=values,excluded=excluded,macro=summary,
        protocol='project_group_macro_v1_not_official_covunibind_evaluator')


def fit_probes(rows,out_dir,task):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression,Ridge
    from sklearn.metrics import mean_squared_error,mean_absolute_error
    import joblib
    result={}
    if task=='spr' and any(r.get('pkd') is None for r in rows):
        return {'status':'blocked_unit_and_censoring_no_pkd_fit'}
    splits=DATA/f'{task}_splits.json'
    if not splits.exists():return {'status':'blocked_missing_homology_checked_splits'}
    definitions=json.loads(splits.read_text())
    x=np.array([r['features'] for r in rows]);y=np.array([r['label'] if task=='binding' else r['pkd'] for r in rows])
    for axis,membership in definitions.items():
        masks={s:np.array([membership.get(r['id'])==s for r in rows]) for s in ['train','val','test']}
        if any(not m.any() for m in masks.values()) or (task=='binding' and any(len(np.unique(y[m]))<2 for m in masks.values())):
            result[axis]={'status':'insufficient_independent_split'};continue
        candidates=[]
        for reg in [100,10,1,.1,.01]:
            head=LogisticRegression(C=1/reg,max_iter=2000,random_state=42) if task=='binding' else Ridge(alpha=reg)
            model=make_pipeline(StandardScaler(),head);model.fit(x[masks['train']],y[masks['train']])
            pred=model.predict_proba(x[masks['val']])[:,1] if task=='binding' else model.predict(x[masks['val']])
            if task=='binding':
                scored=[{**r,'pred':float(p)} for r,p in zip([r for r,m in zip(rows,masks['val']) if m],pred)]
                # Evaluate both ranking directions; use the direction retaining validation groups.
                key='lineage' if axis=='antibody' else 'antibody_key'
                v=group_metrics(scored,'pred',key,True)['macro']['auprc']['mean']
                if v is None:continue
                loss=-v
            else:loss=float(mean_squared_error(y[masks['val']],pred))
            candidates.append((loss,reg,model))
        if not candidates:
            result[axis]={'status':'no_eligible_validation_groups'};continue
        loss,reg,model=min(candidates,key=lambda c:c[0])
        joblib.dump(model,out_dir/f'{axis}_probe.joblib')
        pred=model.predict_proba(x[masks['test']])[:,1] if task=='binding' else model.predict(x[masks['test']])
        selected=[r for r,m in zip(rows,masks['test']) if m]
        scored=[{k:v for k,v in r.items() if k!='features'}|{'probe_score':float(p)} for r,p in zip(selected,pred)]
        write_json(out_dir/f'{axis}_test_predictions.json',scored)
        result[axis]=dict(status='complete',regularization=reg,validation_loss=loss,
            split_counts={s:int(m.sum()) for s,m in masks.items()},
            probe={k:group_metrics(scored,'probe_score',g,task=='binding') for k,g in [('fixed_antibody','antibody_key'),('fixed_antigen','lineage')]},
            zero_shot_same_test={k:group_metrics(scored,'score',g,task=='binding') for k,g in [('fixed_antibody','antibody_key'),('fixed_antigen','lineage')]})
        if task=='spr':result[axis].update(mae=float(mean_absolute_error(y[masks['test']],pred)),rmse=float(mean_squared_error(y[masks['test']],pred)**.5))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['binding','spr','abbibench','chimera'],required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    p.add_argument('--limit',type=int);p.add_argument('--assay');p.add_argument('--pll-batch-size',type=int,default=2)
    p.add_argument('--include-extra-assays',action='store_true',help='Explicit supplementary AbBiBench evaluation')
    p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.task=='chimera':
        p.error('CHIMERA full-VH/VL generation is retired. Use run_v5_suite_task.py --task ag-chimera for epitope_group H3; prepare data with python -m downstream.benchmark.ab_antigen.chimera_h3 prepare')
    config=json.loads((args.checkpoint/'fusion_config.json').read_text())
    assert config['residue_cond_mode']=='token' and config['condition_hidden_size']==0
    manifest=json.loads((args.checkpoint/'eval_snapshot_manifest.json').read_text())
    assert manifest['global_step']==150000 and sha(args.checkpoint/'model.safetensors')==EXPECTED_SHA
    source=DATA/f'{args.task}.jsonl';rows=read_rows(source)
    if args.task=='abbibench':
        from downstream.benchmark.ab_antigen.abbibench_scope import select_rows
        rows=select_rows(rows,assay=args.assay,include_extra_assays=args.include_extra_assays)
    elif args.assay:rows=[r for r in rows if r.get('assay')==args.assay]
    if args.task=='chimera':rows=[r for r in rows if r['split']=='test']
    if args.limit:rows=rows[:args.limit]
    if not rows:raise ValueError('No eligible rows')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    provenance=dict(protocol='ab_ag_sequence_v2_raw_rows',task=args.task,assay=args.assay,limit=args.limit,include_extra_assays=args.include_extra_assays,
        input_sha256=sha(source),checkpoint_sha256=EXPECTED_SHA,step=150000,max_length=MAX_LENGTH,
        checkpoint_config_sha256=sha(args.checkpoint/'fusion_config.json'),tokenizer_sha256=sha(args.checkpoint/'tokenizer.json'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('sequence_data.py'),Path(__file__).with_name('sequence_metrics.py'),ROOT/'examples/llada/protein_fusion_model.py',ROOT/'examples/llada/load_fusion_checkpoint.py',ROOT/'dllm/pipelines/immune_llada/data/grammar.py']},
        pll='single_residue_mask_HL_mean_log_probability',code_sha256=sha(__file__),
        generation=dict(candidates=20,max_iter=32,temperature=1,cfg_scale=0,seed=42),
        scope='smoke' if args.limit else 'full',preflight=args.preflight)
    provenance_path=args.out_dir/'provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text())!=provenance:
        raise ValueError('Output provenance mismatch; use a new run directory')
    write_json(provenance_path,provenance)
    if args.preflight:
        write_json(args.out_dir/'preflight.json',dict(status='passed_no_model_inference',rows=len(rows)));return
    if not torch.cuda.is_available():raise RuntimeError('GPU execution requires Volc job')
    from examples.llada.load_fusion_checkpoint import load_fusion_for_eval
    bundle=load_fusion_for_eval(args.checkpoint,device='cuda',max_length=MAX_LENGTH)
    bundle.model.eval().requires_grad_(False)
    output=args.out_dir/'predictions.jsonl'
    done=read_rows(output) if output.exists() else []
    if args.task=='chimera':
        targets={r['antigen_key']:r['antigen'] for r in rows};keys=sorted(targets)
        if len(keys)<2:raise ValueError('Condition ablation needs two distinct antigens')
        complete={(r['antigen_key'],r['condition'],r['candidate']) for r in done}
        with output.open('a',buffering=1) as f:
            for i,key in enumerate(keys):
                for condition,ag in [('correct',targets[key]),('replaced',targets[keys[(i+1)%len(keys)]])]:
                    for n in range(20):
                        if (key,condition,n) in complete:continue
                        seed=42+int(key[:8],16)%1000000+n
                        r=dict(antigen_key=key,condition=condition,condition_antigen_key=digest(ag),candidate=n,seed=seed,
                            **generate_one(bundle,ag,seed))
                        f.write(json.dumps(r)+'\n')
                print('generated_target',i+1,len(keys),flush=True)
        write_json(args.out_dir/'generation_complete.json',dict(targets=len(keys),attempts=len(keys)*40,metrics_status='requires_sequence_metrics'))
        return
    cache={(r['antigen_key'],r['antibody_key']):r for r in done};scored=[];wt_cache={}
    with output.open('a',buffering=1) as f:
        for i,r in enumerate(rows):
            key=(r['antigen_key'],r['antibody_key'])
            if key not in cache:
                score,features=score_pair(bundle,r['antigen'],r['heavy'],r['light'],args.pll_batch_size,args.task!='abbibench')
                new={**r,'score':score,'features':features}
                if args.task=='abbibench':
                    wtkey=(r['antigen_key'],digest([r['wt_heavy'],r['wt_light']]))
                    if wtkey not in wt_cache:
                        wt,_=score_pair(bundle,r['antigen'],r['wt_heavy'],r['wt_light'],args.pll_batch_size,False)
                        wt_cache[wtkey]=wt
                    new['wt_score']=wt_cache[wtkey];new['score']=score-new['wt_score'];new['raw_pll']=score
                cache[key]=new;f.write(json.dumps(new)+'\n')
            saved=cache[key]
            scored.append({**r,'score':saved['score'],'features':saved['features']})
            if i%10==0:print('scored',i+1,len(rows),flush=True)
    metrics={'status':'complete','scope':provenance['scope'],'n':len(scored)}
    if args.task=='abbibench':
        from downstream.benchmark.ab_antigen.abbibench_scope import summarize_groups
        metrics['per_assay']=group_metrics(scored,group_key='assay')
        metrics['benchmark16']=summarize_groups(metrics['per_assay'])
    else:
        for row in scored:
            context='|'+row['source_doi']+'|'+row['assay'] if args.task=='spr' else ''
            row['ranking_antibody_group']=row['antibody_key']+context
            row['ranking_antigen_group']=row['lineage']+context
        metrics['fixed_antibody']=group_metrics(scored,group_key='ranking_antibody_group',classification=args.task=='binding')
        metrics['fixed_antigen']=group_metrics(scored,group_key='ranking_antigen_group',classification=args.task=='binding')
        metrics['probe']=fit_probes(scored,args.out_dir,args.task) if not args.limit else {'status':'smoke_no_probe_fit'}
    overlap_path=DATA/'audit/training/overlap_rows.json'
    if overlap_path.exists():
        overlaps={r['id']:r for r in json.loads(overlap_path.read_text())}
        keys=['antigen','heavy','light','h3_overlap','pair_overlap']
        metrics['training_exact_overlap']={k:sum(bool(overlaps.get(r['id'],{}).get(k)) for r in scored) for k in keys}
        metrics['overlap_audit_missing_rows']=sum(r['id'] not in overlaps for r in scored)
        metrics['generalization_claim']='mixed_seen_status; no clean or unseen-family claim'
    else:metrics['generalization_claim']='overlap_audit_pending; no clean claim'
    write_json(args.out_dir/'metrics.json',metrics)


if __name__=='__main__':main()
