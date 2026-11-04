"""Audit frozen metrics without training or changing V8 records.

Run after activating pllm, with USE_TF=0 and the exploration Python:
  output/arch70m_env/bin/python metric_audit_20261104.py --cpu
  CUDA_VISIBLE_DEVICES=0 output/arch70m_env/bin/python metric_audit_20261104.py --gpu
Outputs: docs/arch70m_results/metric_audit_20261104/. GPU inference uses the
existing final checkpoints only. Diagnostic controls are not new benchmarks.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dllm.pipelines.immune_llada.data.records import BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig, FixedChainModel
from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyCollator
from experiments.arch70m import evaluate as ev
from experiments.arch70m.data import REGIONS, RECEPTORS

ROOT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore')
OUT = ROOT / 'docs/arch70m_results/metric_audit_20261104'
RUNS = {42: ROOT/'output/arch70m_attention_seed42_20261103',
        43: ROOT/'output/arch70m_attention_20261102'}
EVIDENCE = {}


def read(path):
    raw = Path(path).read_bytes()
    EVIDENCE[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def save(name, value):
    with (ROOT/'output/arch70m_publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def inputs():
    settings = read(ROOT/'experiments/arch70m/base.json')
    index = read(ROOT/'output/arch70m_v8_20261028/data/index.json')
    refs = index['files']['screen']
    samples = {k: ev.rows(v) for k,v in refs.items()}
    EVIDENCE.update({v['path']:v['sha256'] for v in refs.values()})
    fake = SimpleNamespace(config=FixedChainConfig(**settings['model']),
        family_spec=settings['family_prediction'], feature_args=FixedChainModel.feature_args)
    collator = VFamilyCollator(fake.config, family_spec=fake.family_spec, hide_relation=True)
    fake.tokenizer = collator.tokenizer
    tasks, excluded = ev.make_tasks(fake, samples['task_valid'])
    return settings, samples, fake, collator, tasks, excluded


def task_id(t):
    return t['source'],t['key'],t['role']


def average_groups(items, field, sources=None):
    groups=defaultdict(list)
    for x in items:
        if sources is None or x['source'] in sources: groups[x['source']].append(x[field])
    return float(np.mean([np.mean(v) for v in groups.values()]))


def actual_chains(sample):
    r=sample['record']; result={}
    for i,(role,sequence) in enumerate(zip(r['chain_roles'], r['chains'])):
        mask=r.get('chain_metadata',{}).get(str(i),{}).get('synthetic_residue_mask',[0]*len(sequence))
        observed=''.join(a for a,m in zip(sequence,mask) if not m)
        result[role]=(sequence,observed,mask)
    return result


def baseline_features(sample, mode):
    # Never read relation, metadata.source, identifiers or validation labels.
    result={}
    for role,(seq,obs,mask) in actual_chains(sample).items():
        receptor=role in RECEPTORS
        if mode=='receptor_composition' and not receptor: continue
        if mode=='context_composition' and receptor: continue
        result[role+':present']=1.
        result[role+':length']=len(seq)
        result[role+':observed']=len(obs)
        result[role+':synthetic']=sum(mask)
        if mode!='layout':
            for aa,n in Counter(obs).items():result[role+':aa:'+aa]=n/max(len(obs),1)
    return result


def cpu_audit():
    settings,samples,fake,collator,tasks,excluded=inputs()
    tok=fake.tokenizer
    decode=lambda ids:''.join(tok.token(int(x)) for x in ids)
    lookup={ (s['source_group'],s['receptor_key'],role):(s['record'],i)
        for s in samples['task_valid'] for i,role in enumerate(s['record']['chain_roles'])}
    checked=Counter(); task_lookup={}
    for task in tasks:
        record,i=lookup[task_id(task)]
        if task['kind']=='conditional':
            assert decode(task['truth'][task['score_mask']])==record['chains'][i]
            task_lookup[task_id(task)]=task
        else:
            regions=record['regions'][str(i)]
            assert decode(task['truth'][task['target']])==''.join(regions[k] for k in REGIONS if k.startswith('CDR'))
            assert torch.equal(task['target'],task['score_mask'])
            assert torch.all(task['clean']['residue_mask'][0,task['score_mask']])
        features,target,score,truth=ev.task_batch(fake,[task],torch.device('cpu'))
        assert torch.all(features['input_ids'][score]==tok.mask_token_id)
        assert torch.equal(features['input_ids'][~target],truth[~target])
        assert 'residue_mask' not in features and 'labels' not in features
        checked[task['kind']+':'+task['source']]+=1
    relation_checks=0
    for s in samples['probe_train']+samples['probe_valid']:
        altered=deepcopy(s['record'])
        altered['labels']['relation']='nonbinding' if altered['labels']['relation']=='binding' else 'binding'
        a,b=collator([s['record']]),collator([altered])
        assert all(torch.equal(a[k],b[k]) for k in fake.feature_args(a))
        assert torch.equal(a['residue_mask'],b['residue_mask'])
        relation_checks+=1
    overlaps={}; baselines={}; composition={}
    for source in sorted({s['source_group'] for s in samples['probe_valid']}):
        split={p:[s for s in samples['probe_'+p] if s['source_group']==source] for p in ('train','valid')}
        keys={p:{s['receptor_key'] for s in split[p]} for p in split}
        info={'full_receptor_overlap':len(keys['train']&keys['valid'])}
        for role in ['tcr_beta','tcr_alpha']:
            sets={p:{actual_chains(s)[role][1] for s in split[p] if role in actual_chains(s)} for p in split}
            hits=sum(actual_chains(s).get(role,('',None,[]))[1] in sets['train'] for s in split['valid'])
            real=sets['train']&sets['valid']-{''}
            info[role]={'unique_nonempty_observed_chain_overlap':len(real),'valid_records_with_seen_nonempty_chain':sum(actual_chains(s).get(role,('',None,[]))[1] in real for s in split['valid']),
                'unique_observed_chain_overlap_including_empty':len(sets['train']&sets['valid']),
                'valid_records_with_seen_chain':hits,'valid_unique_chains':len(sets['valid'])}
        info['alpha_observation_by_label']={p:{label:dict(Counter('observed' if actual_chains(s).get('tcr_alpha',('', '', []))[1] else 'missing' for s in split[p] if s['record']['labels']['relation']==label)) for label in ['binding','nonbinding']} for p in split}
        info['metadata_subsources']={p:dict(Counter(s['record'].get('metadata',{}).get('source','unspecified')+':'+s['record']['labels']['relation'] for s in split[p])) for p in split}
        overlaps[source]=info
        y={p:np.array([s['record']['labels']['relation']=='binding' for s in split[p]],dtype=int) for p in split}
        missing=np.array([not actual_chains(s).get('tcr_alpha',('', '', []))[1] for s in split['valid']],dtype=int)
        baselines[source]={'missing_alpha_only':{'auroc':float(roc_auc_score(y['valid'],missing)),'auprc':float(average_precision_score(y['valid'],missing))}}
        for mode in ['layout','receptor_composition','context_composition','all_composition']:
            x={p:[baseline_features(s,mode) for s in split[p]] for p in split}
            clf=make_pipeline(DictVectorizer(sparse=False), StandardScaler(),
                LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=42))
            clf.fit(x['train'],y['train']); pred=clf.predict_proba(x['valid'])[:,1]
            baselines[source][mode]={'auroc':float(roc_auc_score(y['valid'],pred)),
                'auprc':float(average_precision_score(y['valid'],pred))}
        composition[source]={p:dict(Counter(s['record']['labels']['relation'] for s in split[p])) for p in split}
    endpoints=[]; coverage=[]; paired_probe=[]; probability_checks=0
    cdr_sources={t['source'] for t in tasks if t['kind']=='cdr'}
    for seed in [42,43]:
        for variant in ['A0','C0','C1','C2','C3']:
            t=RUNS[seed]/'trials'/f'screen_{variant}_s{seed}'/'metrics'
            initial,final=[read(t/f'screen_{step:06d}.json') for step in [0,1276]]
            for step,metrics in [(0,initial),(1276,final)]:
                probabilities=read(t/f'screen_{step:06d}_observations.json')['representation']
                for source in metrics['representation']['sources']:
                    expected_rows=[x for x in samples['probe_valid'] if x['source_group']==source]
                    observed=[x for x in probabilities if x['source']==source]
                    assert [(x['key'],x['label']) for x in observed]==[(x['receptor_key'],int(x['record']['labels']['relation']=='binding')) for x in expected_rows]
                    y=[x['label'] for x in observed];p=[x['probability'] for x in observed]
                    assert abs(roc_auc_score(y,p)-metrics['representation']['sources'][source]['auroc'])<1e-12
                    assert abs(average_precision_score(y,p)-metrics['representation']['sources'][source]['auprc'])<1e-12
                    probability_checks+=1
                    paired=[o for o,row in zip(observed,expected_rows) if actual_chains(row).get('tcr_alpha',('', '', []))[1]]
                    py=[x['label'] for x in paired];pp=[x['probability'] for x in paired]
                    paired_probe.append(dict(seed=seed,variant=variant,step=step,source=source,records=len(py),positives=sum(py),
                        auroc=float(roc_auc_score(py,pp)),auprc=float(average_precision_score(py,pp))))
            obs=read(t/'screen_001276_observations.json')['generation']
            generated=[]
            for o in obs:
                if o['kind']!='conditional':continue
                truth=decode(task_lookup[task_id(o)]['truth'][task_lookup[task_id(o)]['score_mask']])
                seq=o['sequence'] if o['valid'] else ''
                correct=sum(a==b for a,b in zip(seq,truth))
                assert abs(o['aar']-correct/len(truth))<1e-12
                n=min(len(seq),len(truth))
                generated.append(dict(o,reference_length=len(truth),predicted_length=len(seq),
                    coverage=n/len(truth),covered_accuracy=correct/n if n else 0.,
                    shorter_than_half=float(len(seq)<len(truth)/2)))
            coverage.append({'seed':seed,'variant':variant,'records':len(generated),
                **{k:average_groups(generated,k) for k in ['coverage','covered_accuracy','shorter_than_half','predicted_length','reference_length']},
                'same_cdr_source_full_aar':average_groups(generated,'aar',cdr_sources)})
            endpoints.append({'seed':seed,'variant':variant,'initial_representation':initial['representation'],
                'final_representation':final['representation'],'generation':final['generation']})
    result=dict(task_counts=dict(checked),exclusions=excluded,
        checks={'chain_alignment':True,'cdr_alignment':True,'score_only_masked_residues':True,
                'unmasked_conditions_unchanged':True,'relation_flip_invariant_records':relation_checks,'probability_metric_checks':probability_checks},
        probe_counts=composition,probe_overlap=overlaps,baselines=baselines,
        endpoints=endpoints,coverage=coverage,observed_alpha_probe_subset=paired_probe,evidence=EVIDENCE)
    save('cpu_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['evidence','endpoints','observed_alpha_probe_subset']},indent=2),flush=True)


def cdr_parts(task, record, tokenizer):
    i=record['chain_roles'].index(task['role']); regions=record['regions'][str(i)]
    # Independent reference indices: the first receptor residue, not a grammar marker.
    positions=(task['clean']['role_ids'][0].eq(task['clean']['role_ids'][0,task['target']][0]) &
               task['clean']['residue_mask'][0]).nonzero().flatten()
    offset=0; parts={}
    for name in REGIONS:
        end=offset+len(regions[name])
        if name.startswith('CDR'):parts[name]=positions[offset:end]
        offset=end
    return parts


def gpu_audit():
    from safetensors.torch import load_file
    from architecture_attention.models import build
    from experiments.arch70m.train import setup
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    settings,samples,_,_,_,_=inputs()
    records={(s['source_group'],s['receptor_key']):s['record'] for s in samples['task_valid']}
    results=[]
    with (ROOT/'output/arch70m_local_gpu0.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=torch.device('cuda:0');torch.cuda.set_device(device)
        for seed,variant in [(42,'A0'),(43,'A0'),(42,'C0')]:
            trial=RUNS[seed]/'trials'/f'screen_{variant}_s{seed}'
            checkpoint=trial/'checkpoints/step_001276/model.safetensors'
            # Binding + receipt record the exact immutable checkpoint loaded here.
            receipt=read(trial/'checkpoints/step_001276/receipt.json') if (trial/'checkpoints/step_001276/receipt.json').exists() else read(trial/'receipt.json')
            weight_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            assert weight_hash==receipt['files']['model.safetensors']
            EVIDENCE[str(checkpoint)]=weight_hash
            setup(seed)
            model=build(settings,variant,seed)
            model.load_state_dict(load_file(str(checkpoint),device='cpu'),strict=True)
            model.to(device).eval()
            # Keep the exact original token-validation prelude (including rotary caches).
            refs=read(ROOT/'output/arch70m_v8_20261028/data/index.json')['files']['screen']
            ev.token_validation(model, {'files':refs}, device, batch_size=8)
            captures=[]; original=ev.task_batch
            def capture(*args,**kwargs):
                output=original(*args,**kwargs);captures.append((args[1],output));return output
            ev.task_batch=capture
            try: summary,observations,excluded=ev.generation(model,
                    read(ROOT/'output/arch70m_v8_20261028/data/index.json')['files']['screen']['task_valid'],device)
            finally:ev.task_batch=original
            expected=read(trial/'metrics/screen_001276_observations.json')['generation']
            save(f'replay_{variant}_{seed}.json', {'observations':observations,'summary':summary})
            differences=[{'identity':task_id(a),'kind':a['kind'],'original':b['aar'],'replay':a['aar']} for a,b in zip(observations,expected) if a!=b]
            print(json.dumps({'replay_differences':len(differences),'first':differences[:3]}),flush=True)
            exact=observations==expected
            assert len(expected)==len(observations)
            assert all(task_id(a)==task_id(b) and a['kind']==b['kind'] and (a['kind']!='cdr' or abs(a['aar']-b['aar'])<1e-7)
                       for a,b in zip(observations,expected)), 'Frozen CDR AAR did not replay'
            details=[]
            for tasks, (features,targets,scores,truth) in captures:
                for i,task in enumerate(tasks):
                    if task['kind']!='cdr':continue
                    parts=cdr_parts(task,records[task['source'],task['key']],model.tokenizer)
                    pred=features['input_ids'][i].cpu(); true=truth[i].cpu()
                    for name,indices in parts.items():
                        details.append({'seed':seed,'variant':variant,'source':task['source'],
                            'key':task['key'],'role':task['role'],'region':name,'tokens':len(indices),
                            'correct':int(pred[indices].eq(true[indices]).sum()),
                            'aar':float(pred[indices].eq(true[indices]).float().mean()),
                            'prediction':''.join(model.tokenizer.token(int(x)) for x in pred[indices]),
                            'reference':''.join(model.tokenizer.token(int(x)) for x in true[indices])})
            region_summary={name:{'source_macro_chain_aar':average_groups([x for x in details if x['region']==name],'aar'),
                'token_micro_aar':sum(x['correct'] for x in details if x['region']==name)/sum(x['tokens'] for x in details if x['region']==name)}
                for name in ['CDR1','CDR2','CDR3']}
            value=dict(seed=seed,variant=variant,exact_frozen_replay=exact,
                frozen_summary=summary,cdr_regions=region_summary,details=details,replay_differences=differences)
            results.append(value);save('gpu_audit.json',{'results':results,'evidence':EVIDENCE})
            print(json.dumps({k:v for k,v in value.items() if k not in ['details','frozen_summary']}),flush=True)
            del captures,model;torch.cuda.empty_cache()


def training_overlap():
    """Read exactly the selected 10% offsets, without resampling or training."""
    try:
        import orjson
        loads=orjson.loads
    except ImportError:
        loads=json.loads
    from experiments.arch70m.data import identity
    settings,samples,_,_,tasks,_=inputs()
    index=read(ROOT/'output/arch70m_v8_20261028/data/index.json')
    refs={(s['source_group'],s['receptor_key']):s['record'] for s in samples['task_valid']}
    chains={}; cdr3={}
    for t in tasks:
        rec=refs[t['source'],t['key']]; i=rec['chain_roles'].index(t['role'])
        chains[task_id(t)]=(t['role'],rec['chains'][i])
        if t['kind']=='cdr':cdr3[task_id(t)]=(t['role'],rec['regions'][str(i)]['CDR3'])
    needed_chains=set(chains.values());needed_cdr3=set(cdr3.values())
    probe_keys={s['receptor_key'] for s in samples['probe_valid']}
    task_keys={s['receptor_key'] for s in samples['task_valid']}
    seen_chains=set();seen_cdr3=set();receptor_overlap=set();count=0
    for shard in index['shards']:
        if shard['split']!='train':continue
        assert hashlib.sha256(Path(shard['index']).read_bytes()).hexdigest()==shard['index_sha256']
        offsets=np.load(shard['index'])
        with Path(shard['path']).open('rb') as stream:
            for offset in offsets:
                stream.seek(int(offset));record=loads(stream.readline());count+=1
                key=identity(record)
                if key in probe_keys or key in task_keys:receptor_overlap.add(key)
                for i,(role,sequence) in enumerate(zip(record['chain_roles'],record['chains'])):
                    if role not in RECEPTORS:continue
                    mask=record.get('chain_metadata',{}).get(str(i),{}).get('synthetic_residue_mask',[0]*len(sequence))
                    observed=''.join(a for a,m in zip(sequence,mask) if not m)
                    if (role,observed) in needed_chains:seen_chains.add((role,observed))
                    region=record.get('regions',{}).get(str(i),{}).get('CDR3')
                    if region and 'X' not in region and (role,region) in needed_cdr3:seen_cdr3.add((role,region))
    assert count==index['records']==653288
    results={'scanned_selected_training_records':count,'full_receptor_overlap':len(receptor_overlap),
        'task_receptor_overlap':len(task_keys&receptor_overlap),'probe_receptor_overlap':len(probe_keys&receptor_overlap),
        'exact_full_chain_matches':[{'source':k[0],'key':k[1],'role':k[2]} for k,v in chains.items() if v in seen_chains],
        'exact_cdr3_matches':[{'source':k[0],'key':k[1],'role':k[2]} for k,v in cdr3.items() if v in seen_cdr3],
        'conditional_chain_tasks':len(chains),'cdr_chain_tasks':len(cdr3),
        'limitation':'Exact identity audit only; homologous/germline/CDR1-2 similarity is not excluded.',
        'evidence':EVIDENCE}
    save('training_overlap.json',results)
    print(json.dumps({k:(len(v) if isinstance(v,list) else v) for k,v in results.items() if k!='evidence'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cpu',action='store_true');parser.add_argument('--gpu',action='store_true');parser.add_argument('--overlap',action='store_true')
    args=parser.parse_args()
    if args.cpu:cpu_audit()
    if args.gpu:gpu_audit()
    if args.overlap:training_overlap()
