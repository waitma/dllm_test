"""Audit the current three AB frozen probes with fresh official Ophiuchus weights.

Activate pllm, then protenix_abtcr. Run this absolute script path with
--out ABS_NEW_RUN --prepare, then --out ABS_RUN --task TASK --device cuda
on an allocated Volc worker. --verify rechecks saved predictions on CPU.
No old embeddings, metrics, production entrypoints, or splits are overwritten.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace, ModuleType

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
AIR = ROOT.parent / 'airgen/AirGen-Dev'
sys.path.insert(0, str(ROOT))
TASKS = ('gdp_a1', 'specificity', 'm396')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def prepare(out):
    from downstream.grammar.ab_probes import prepare_task
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / 'model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt'
    tokenizer = ROOT / 'model_weights/esm2/esm2_t33_650M_UR50D'
    files = {Path(__file__).resolve(), ROOT / 'downstream/common.py', ROOT / 'downstream/embeddings.py'}
    for folder in ('downstream/grammar', 'downstream/ophiuchus_eval', 'dllm/pipelines/bioseq'):
        files.update((ROOT / folder).rglob('*.py'))
    files.update((AIR / 'src/byprot/models/lm/modules/mint').rglob('*.py'))
    for rel in ('dev/finetune_dev.py', 'specificity/HD_Flu_Cov-paired.py', 'in_silico/finetune_in_silico.py'):
        files.add(AIR / 'downstream' / rel)
    files.update(tokenizer.glob('*.json'))
    files.add(tokenizer / 'vocab.txt')
    tasks = {}
    for task in TASKS:
        df, pairs, paths, folds = prepare_task(task)
        files.update(paths)
        tasks[task] = dict(n_rows=len(df), n_inputs=len(pairs), data=[str(p) for p in paths],
                           pairs_sha256=hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
                           overlength_rows=sum(len(h)>148 or len(l)>126 for h,l in pairs),
                           max_chain_lengths=[max(len(p[k]) for p in pairs) for k in (0,1)])
        if folds is not None:
            tasks[task]['folds'] = [dict(train=tr.tolist(), test=te.tolist()) for tr,te in folds]
    plan = dict(schema='ophiuchus.current_probe_audit.v1', created_utc=stamp(),
                checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), tokenizer=str(tokenizer),
                source_and_data_sha256={str(p):sha(p) for p in sorted(files)}, tasks=tasks,
                precision='float32', embedding_batch_size=16, seed=42, epochs=100,
                head_batch_size=8, lr=5e-5, split_seed=2023, ridge_alpha=0.01,
                gdp='existing legacy_reference; full-label transform and same-CV alpha selection',
                specificity='current production fixed-last-100 head; seed42+fold; no test selection',
                m396='current row_reference plus separately labelled sequence_grouped; WT-minus-mut',
                feature_protocol='current Ophiuchus 150/128 EOS layout; H148/L126 truncation; sep-chain mean2560',
                scope='current evaluator validation with official weights, not blanket exact paper reproduction',
                fresh_features=True)
    write(out / 'plan.json', plan)
    shutil.copy2(__file__, out / 'runner_snapshot.py')
    print(json.dumps({'status':'prepared','tasks':{k:{a:b for a,b in v.items() if a!='folds'} for k,v in tasks.items()},'plan':str(out/'plan.json')}, indent=2))


def validate_plan(out):
    plan=read(out/'plan.json')
    assert sha(plan['checkpoint']) == plan['checkpoint_sha256'], 'Checkpoint drift'
    for path,digest in plan['source_and_data_sha256'].items():
        assert sha(path)==digest, f'Source/data drift: {path}'
    return plan


def official_env(path, names):
    import numpy as np
    import pandas as pd
    import scipy
    import scipy.stats
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    from sklearn.preprocessing import PowerTransformer
    from transformers import EsmTokenizer
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==names
    env=dict(np=np,pd=pd,scipy=scipy,torch=torch,nn=torch.nn,Dataset=torch.utils.data.Dataset,
             EsmTokenizer=EsmTokenizer, Ridge=Ridge, GridSearchCV=GridSearchCV,
             PredefinedSplit=PredefinedSplit, make_scorer=make_scorer, PowerTransformer=PowerTransformer)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env


def source_gate(model, pairs, plan, folder, device):
    """Compare independent original backbone and pooling on identical token inputs."""
    import numpy as np
    import torch
    from downstream.common import ChainPaddingCollator
    from dllm.pipelines.bioseq import Esm2ProteinTokenizer
    pkg=ModuleType('probe_audit_original_mint')
    pkg.__path__=[str(AIR/'src/byprot/models/lm/modules/mint')]
    sys.modules[pkg.__name__]=pkg
    ESM2=importlib.import_module(pkg.__name__+'.model.esm2').ESM2
    original=ESM2(num_layers=33,embed_dim=1280,attention_heads=20,token_dropout=True,use_multimer=True)
    original.load_state_dict(model.model.state_dict(),strict=True)
    env=official_env(AIR/'downstream/dev/finetune_dev.py',{'FlabWrapper','MyCollateFn'})
    wrapper=env['FlabWrapper'].__new__(env['FlabWrapper'])
    torch.nn.Module.__init__(wrapper)
    wrapper.model,wrapper.sep_chains=original,True
    wrapper.to(device).eval().requires_grad_(False)
    chosen=np.unique(np.linspace(0,len(pairs)-1,min(8,len(pairs)),dtype=int))
    sample=[pairs[i] for i in chosen]
    collator=ChainPaddingCollator(Esm2ProteinTokenizer())
    ids,chain_ids,_=collator.stack_chains([p[0] for p in sample],[p[1] for p in sample])
    native_collator=env['MyCollateFn'](tokenizer_path=plan['tokenizer'])
    # Use the exact visible sequence for the original collator, which itself has no safe truncation.
    visible=[(h[:148],l[:126],0.,0) for h,l in sample]
    original_ids,original_chains,_,_=native_collator(visible)
    assert torch.equal(ids,original_ids) and torch.equal(chain_ids,original_chains)
    with torch.inference_mode():
        current=model(ids.to(device),chain_ids.to(device)).cpu()
        reference=wrapper(ids.to(device),chain_ids.to(device)).cpu()
    np.testing.assert_allclose(current.numpy(),reference.numpy(),atol=2e-5,rtol=2e-5)
    write(folder/'original_forward_check.json',dict(status='passed',rows=chosen.tolist(),
          same_visible_input_tokens=True,original_backbone_and_pooling=True,
          max_abs_error=float((current-reference).abs().max()),atol=2e-5,rtol=2e-5,
          limitation='Truncation is the current harness adaptation, not performed by the original collator.'))
    del wrapper,original
    gc.collect()
    if device.startswith('cuda'): torch.cuda.empty_cache()


def extract(model,pairs,plan,folder,device):
    import numpy as np
    import torch
    from downstream.common import ChainPaddingCollator
    from dllm.pipelines.bioseq import Esm2ProteinTokenizer
    from transformers import EsmTokenizer
    collator=ChainPaddingCollator(Esm2ProteinTokenizer())
    independent=EsmTokenizer.from_pretrained(plan['tokenizer'],local_files_only=True)
    features=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype=np.float32,shape=(len(pairs),2560))
    start=time.time()
    batch=plan['embedding_batch_size']
    for offset in range(0,len(pairs),batch):
        sample=pairs[offset:offset+batch]
        ids,chains,_=collator.stack_chains([p[0] for p in sample],[p[1] for p in sample])
        for row,(h,l) in enumerate(sample):
            for seq,pos,width in ((h,0,150),(l,150,128)):
                expected=independent.encode(seq[:width-2].replace('J','L'))
                expected += [independent.eos_token_id]*(width-len(expected))
                assert ids[row,pos:pos+width].tolist()==expected, (offset+row,pos)
        assert (chains[:,:150]==0).all() and (chains[:,150:]==1).all()
        with torch.inference_mode():
            value=model(ids.to(device),chains.to(device)).cpu()
        assert value.shape==(len(sample),2560) and torch.isfinite(value).all()
        features[offset:offset+len(sample)]=value.numpy()
        if offset==0:
            torch.save(dict(input_ids=ids,chain_ids=chains,features=value),folder/'first_batch.pt')
            with torch.inference_mode():
                single=model(ids[:1].to(device),chains[:1].to(device)).cpu()
            np.testing.assert_allclose(single.numpy(),value[:1].numpy(),atol=3e-5,rtol=3e-5)
        if offset==0 or (offset//batch)%50==0 or offset+batch>=len(pairs):
            write(folder/'status.json',dict(status='running',phase='features',completed=offset+len(sample),
                  total=len(pairs),elapsed_seconds=time.time()-start,updated_utc=stamp()))
            print(f'{folder.name}: features {offset+len(sample)}/{len(pairs)} {time.time()-start:.1f}s',flush=True)
    features.flush()
    write(folder/'feature_receipt.json',dict(status='passed',shape=list(features.shape),
          sha256=sha(folder/'features.npy'),plan_sha256=sha(folder.parent/'plan.json'),
          all_input_tokens_checked=True,fresh_features=True,single_batch_check=True,device=device,
          overlength_rows=plan['tasks'][folder.name]['overlength_rows']))
    return torch.from_numpy(np.asarray(features).copy())


def run_task(args):
    import numpy as np
    import torch
    from downstream.grammar.ab_probes import prepare_task,run_gdp,run_specificity,run_m396
    from downstream.ophiuchus_eval.probe_common import load_embedder
    plan=validate_plan(args.out)
    folder=args.out/args.task
    folder.mkdir(exist_ok=False)
    write(folder/'status.json',dict(status='running',phase='loading',updated_utc=stamp()))
    torch.set_num_threads(8)
    if args.device.startswith('cuda'):
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    df,pairs,paths,folds=prepare_task(args.task)
    assert hashlib.sha256(json.dumps(pairs).encode()).hexdigest()==plan['tasks'][args.task]['pairs_sha256']
    model=load_embedder(plan['checkpoint'],torch.device(args.device),sep_chains=True).eval().requires_grad_(False)
    state=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['state_dict']
    actual=model.model.state_dict()
    assert set(state)==set(actual), 'Official checkpoint key mismatch'
    assert all(torch.equal(value,actual[key].cpu()) for key,value in state.items()), 'Loaded weight mismatch'
    write(folder/'checkpoint_check.json',dict(status='passed',all_tensors_equal=True,keys=len(state)))
    del state,actual
    source_gate(model,pairs,plan,folder,args.device)
    x=extract(model,pairs,plan,folder,args.device)
    del model
    gc.collect()
    if args.device.startswith('cuda'): torch.cuda.empty_cache()
    cfg=SimpleNamespace(**{k:plan[k] for k in ('epochs','head_batch_size','lr','seed','split_seed','ridge_alpha')},
                        out_dir=folder,device=args.device,log_epoch_metrics=False)
    write(folder/'status.json',dict(status='running',phase='readout',updated_utc=stamp()))
    if args.task=='gdp_a1':
        result=run_gdp(df,x,cfg)
        searches=[]
        env=official_env(AIR/'downstream/dev/finetune_dev.py',{'gaussian_transform','spearman_corr','cross_validate'})
        constructor=env['GridSearchCV']
        def capture(*a,**kw):
            search=constructor(*a,**kw);searches.append(search);return search
        env['GridSearchCV']=capture
        for prop,item in result['properties'].items():
            valid=df[prop].notna().to_numpy()
            with redirect_stdout(io.StringIO()):
                env['cross_validate'](x[valid].numpy(),df.loc[valid,prop].to_numpy(float),
                    df.loc[valid,'hierarchical_cluster_IgG_isotype_stratified_fold'].to_numpy(),scale_all=True)
            selected=item['legacy_reference']
            assert abs(searches[-1].best_score_-selected['cv_mean_spearman'])<1e-10
            assert searches[-1].best_params_['alpha']==selected['alpha']
        write(folder/'original_ridge_check.json',dict(status='passed',all_five_scores_and_alphas_equal=True))
    elif args.task=='specificity': result=run_specificity(df,x,folds,cfg)
    else: result=run_m396(df,x,cfg)
    result.update(model='Ophiuchus-Ab official checkpoint',feature_protocol=plan['feature_protocol'],
                  evidence_type='local_reimplementation',plan_sha256=sha(args.out/'plan.json'),
                  feature_sha256=sha(folder/'features.npy'))
    write(folder/'metrics.json',result)
    validate_plan(args.out)
    verification=verify_task(args.out,args.task)
    write(folder/'verification.json',verification)
    write(folder/'status.json',dict(status='success',verified=True,updated_utc=stamp()))
    print(json.dumps({'task':args.task,'status':'success','verification':verification},indent=2),flush=True)


def rank_corr(y,p):
    import numpy as np
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(y),rankdata(p))[0,1])


def verify_task(out,task):
    import joblib
    import numpy as np
    import pandas as pd
    from downstream.grammar.ab_probes import prepare_task
    folder=out/task
    plan=read(out/'plan.json'); result=read(folder/'metrics.json'); receipt=read(folder/'feature_receipt.json')
    accepted={sha(out/'plan.json')}
    for previous in plan.get('prior_plans',[]):
        assert sha(previous['path'])==previous['sha256']
        old=read(previous['path'])
        for key in ('checkpoint_sha256','tasks','precision','embedding_batch_size','seed','epochs',
                    'head_batch_size','lr','split_seed','ridge_alpha','gdp','specificity','m396','feature_protocol'):
            assert old[key]==plan[key], f'Previous plan changed evaluation setting: {key}'
        accepted.add(previous['sha256'])
    assert result['plan_sha256']==receipt['plan_sha256'] and result['plan_sha256'] in accepted
    assert result['feature_sha256']==receipt['sha256']==sha(folder/'features.npy')
    df,pairs,_,folds=prepare_task(task)
    x=np.load(folder/'features.npy',mmap_mode='r')
    checks=[]
    if task=='gdp_a1':
        for prop,item in result['properties'].items():
            a=np.load(folder/prop/'legacy_oof.npz')
            rows=np.flatnonzero(df[prop].notna().to_numpy())
            np.testing.assert_array_equal(rows,a['row_index'])
            np.testing.assert_array_equal(df.loc[rows,prop].to_numpy(),a['y_true'])
            np.testing.assert_array_equal(df.loc[rows,'hierarchical_cluster_IgG_isotype_stratified_fold'].to_numpy(),a['fold'])
            # GridSearchCV scores transformed labels; finite precision can change ties (HIC).
            # pooled_oof_spearman intentionally uses raw labels, per the existing production recipe.
            saved=joblib.load(folder/prop/'legacy_reference.joblib')
            transformed=saved['transformer_all_labels'].transform(a['y_true'][:,None]).ravel()
            np.testing.assert_allclose(transformed,a['y_transformed'],rtol=1e-12,atol=1e-12)
            scores=[rank_corr(a['y_transformed'][a['fold']==k],a['prediction'][a['fold']==k]) for k in range(5)]
            assert abs(np.mean(scores)-item['legacy_reference']['cv_mean_spearman'])<1e-10
            assert abs(rank_corr(a['y_true'],a['prediction'])-item['legacy_reference']['pooled_oof_spearman'])<1e-10
            checks.append(dict(property=prop,mean_recomputed=float(np.mean(scores))))
    elif task=='specificity':
        all_ids=[]; recomputed=[]
        for k,(tr,te) in enumerate(folds):
            a=pd.read_csv(folder/f'fold_{k}'/'predictions.csv')
            np.testing.assert_array_equal(a.sample_id.astype(str),df.name.iloc[te].astype(str))
            np.testing.assert_array_equal(a.y_true,df.label.iloc[te])
            np.testing.assert_array_equal(a.y_pred,a[[f'logit_{c}' for c in range(3)]].to_numpy().argmax(1))
            cm=np.zeros((3,3),dtype=np.int64)
            np.add.at(cm,(a.y_true.to_numpy(),a.y_pred.to_numpy()),1)
            n=cm.sum(); true=cm.sum(1); pred=cm.sum(0); correct=np.trace(cm)
            values=dict(accuracy=float(correct/n),f1_macro=float(np.mean(2*np.diag(cm)/(true+pred))),
                mcc=float((correct*n-true@pred)/np.sqrt((n*n-true@true)*(n*n-pred@pred))))
            for key,val in values.items(): assert abs(val-result['folds'][k][key])<1e-10
            selection=read(folder/f'fold_{k}'/'selection.json')
            assert selection['selected_epoch']==plan['epochs'] and selection['rule']=='fixed_last_no_test_selection'
            assert selection['train_ids']==df.name.iloc[tr].tolist() and selection['test_ids']==df.name.iloc[te].tolist()
            all_ids.extend(a.sample_id.astype(str));recomputed.append(values)
        assert len(set(all_ids))==len(df)==len(all_ids)
        for key,val in result['mean'].items(): assert abs(np.mean([r[key] for r in recomputed])-val)<1e-10
        checks=recomputed
    else:
        from downstream.ophiuchus_eval.affinity_m396 import SCORE_COLS
        from sklearn.model_selection import GroupShuffleSplit
        y=df[SCORE_COLS].to_numpy(float)
        # Reconstruct group identity independently as full H/L strings, then preserve production hash ordering.
        from downstream.grammar.ab_features import json_hash
        groups=np.array([json_hash(pair) for pair in pairs[1:]])
        delta=x[:1]-x[1:]
        for row in result['splits']:
            frac=row['train_fraction'];mode=row['split_protocol']
            path=folder/f'{mode}_{frac:g}'
            a=np.load(path/'predictions.npz');tr,te=a['train_index'],a['test_index']
            if mode=='row_reference':
                expected=df.sample(frac=frac,random_state=plan['split_seed']).index.to_numpy()
                np.testing.assert_array_equal(tr,expected)
                np.testing.assert_array_equal(te,df.drop(expected).index.to_numpy())
                target=(y[te]-y[te].mean(0))/np.clip(y[te].std(0),1e-8,None)
            else:
                etr,ete=next(GroupShuffleSplit(n_splits=1,train_size=frac,random_state=plan['split_seed']).split(x[1:],groups=groups))
                np.testing.assert_array_equal(tr,etr);np.testing.assert_array_equal(te,ete)
                assert not set(groups[tr])&set(groups[te])
                target=y[te]
            np.testing.assert_allclose(a['y_true'],target,rtol=1e-12,atol=1e-12)
            scores=[rank_corr(target[:,j],a['prediction'][:,j]) for j in range(5)]
            np.testing.assert_allclose(scores,row['spearman_per_target'],atol=1e-10,rtol=0)
            assert abs(np.mean(scores)-row['spearman_mean'])<1e-10
            saved=joblib.load(path/'selected_ridge.joblib')
            prediction=saved['ridge'].predict(delta[te])
            if mode=='sequence_grouped': prediction=prediction*saved['train_std']+saved['train_mean']
            np.testing.assert_allclose(prediction,a['prediction'],atol=1e-6,rtol=1e-6)
            checks.append(dict(protocol=mode,fraction=frac,mean_recomputed=float(np.mean(scores))))
    return dict(status='passed',task=task,checked_utc=stamp(),checks=checks,
                metrics_sha256=sha(folder/'metrics.json'),saved_predictions_independently_rescored=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--task',choices=TASKS)
    mode.add_argument('--verify',action='store_true')
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    assert args.out.is_absolute()
    if args.prepare: prepare(args.out)
    elif args.verify:
        validate_plan(args.out)
        results={task:verify_task(args.out,task) for task in TASKS}
        write(args.out/'verification.json',dict(status='passed',tasks=results,checked_utc=stamp()))
        print(json.dumps(results,indent=2))
    else:
        try: run_task(args)
        except Exception as error:
            folder=args.out/args.task
            if folder.exists():
                write(folder/'status.json',dict(status='failed',error=repr(error),updated_utc=stamp()))
            raise


if __name__=='__main__':
    main()
