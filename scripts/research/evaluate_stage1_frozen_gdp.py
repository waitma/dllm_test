"""Compare completed Stage1 checkpoints on GDPa1 with frozen features.

Activate /project_c20250601/251105016/conda/envs/pllm. Run this absolute file:
  python -B SCRIPT prepare --output ABS_NEW_OUTPUT
  python -B SCRIPT features --output ABS_OUTPUT --arm mlm --device cpu
  python -B SCRIPT features --output ABS_OUTPUT --arm pair_kd --device cpu
  python -B SCRIPT score --output ABS_OUTPUT
No pretrained weights, existing protocols, or shared datasets are modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
FOLD = 'hierarchical_cluster_IgG_isotype_stratified_fold'
TARGETS = ['Titer', 'HIC', 'PR_CHO', 'Tm2', 'AC-SINS_pH7.4']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def announce(**data):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def prepare(args):
    import pandas as pd
    if args.output.exists():
        raise FileExistsError(args.output)
    arms = {}
    source_data = ROOT/'data/downstream/dev/GDPa1_v1.2_20250814.csv'
    df = pd.read_csv(source_data)
    assert len(df) == 246 and set(df[FOLD]) == set(range(5))
    seen = {}
    for h, l, fold in zip(df.vh_protein_sequence, df.vl_protein_sequence, df[FOLD]):
        assert len(h) <= 166 and len(l) <= 134
        assert set(h+l) <= set('ACDEFGHIKLMNPQRSTVWY')
        if (h,l) in seen:
            assert seen[h,l] == fold
        seen[h,l] = fold
    for arm, run in [('mlm','stage1_ab500k_scratch_20261024'),
                     ('pair_kd','stage1_ab500k_scratch_fast_20261024')]:
        base = ROOT/'output'/run
        checkpoint = base/'train'/arm/'checkpoint-9770'
        complete = json.loads((checkpoint.parent/'COMPLETE.json').read_text())
        assert complete['status']=='success' and complete['step']==9770
        receipt = json.loads((checkpoint/'checkpoint_receipt.json').read_text())
        digest = sha(checkpoint/'model.safetensors')
        assert receipt['files']['model.safetensors']==digest
        provenance = json.loads((checkpoint/'run_provenance.json').read_text())
        arms[arm] = dict(checkpoint=str(checkpoint), model_sha256=digest,
                         source_root=str(base/'launch/source'), provenance=provenance,
                         research_config_sha256=sha(checkpoint/'research_config.json'))
    for key in ['initialization','training','subset_manifest_sha256','recipe','world_size']:
        assert arms['mlm']['provenance'][key] == arms['pair_kd']['provenance'][key], key
    plan = dict(schema='stage1.frozen.gdp.fixed_ridge.v1', created_utc=datetime.now(timezone.utc).isoformat(),
                arms=arms, data=str(source_data), data_sha256=sha(source_data), targets=TARGETS, fold_column=FOLD,
                checkpoint_step=9770, pooling='last_hidden_real_residue_global_mean',
                hide_relation=True, precision='float32', fixed_ridge_alpha=1.0,
                feature_scaling='StandardScaler fit on training rows only',
                target_scaling='none', head_selection='none; fixed before inspecting scores',
                fractions=[.1,.25,.5,1.0], subset_seeds=[11,22,33,44,55],
                primary='mean of five held-out-fold Spearman correlations, fraction=1',
                scope='Frozen downstream diagnostic; independent new protocol, not legacy_reference replication',
                pretraining_exposure='pending exact and homology audit',
                source_script_sha256=sha(__file__))
    args.output.mkdir(parents=True)
    df.to_csv(args.output/'data_snapshot.csv',index=False)
    plan['data_snapshot_sha256']=sha(args.output/'data_snapshot.csv')
    write(args.output/'plan.json',plan)
    announce(phase='prepared',output=str(args.output),records=len(df))


def load_plan(out):
    p = json.loads((out/'plan.json').read_text())
    assert sha(p['data'])==p['data_sha256']
    assert sha(out/'data_snapshot.csv')==p['data_snapshot_sha256']
    assert sha(__file__)==p['source_script_sha256'], 'Script changed after protocol freeze'
    return p


def features(args):
    import numpy as np
    import pandas as pd
    import torch
    plan = load_plan(args.output)
    binding = plan['arms'][args.arm]
    destination = args.output/(args.arm+'_features.npz')
    if destination.exists():
        raise FileExistsError(destination)
    sys.path.insert(0,binding['source_root'])
    for source, digest in binding['provenance']['sources'].items():
        assert sha(source)==digest, source
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
    from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMModel, FullTokenMLMCollator
    from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel
    module = sys.modules[StructureMLMModel.__module__]
    assert Path(module.__file__).is_relative_to(binding['source_root'])
    checkpoint = Path(binding['checkpoint'])
    assert sha(checkpoint/'model.safetensors')==binding['model_sha256']
    assert sha(checkpoint/'research_config.json')==binding['research_config_sha256']
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    model = (FullTokenMLMModel if args.arm=='mlm' else StructureMLMModel).from_pretrained(checkpoint)
    model = model.to(args.device).float().eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    collator = FullTokenMLMCollator(model.config,hide_relation=True)
    df = pd.read_csv(args.output/'data_snapshot.csv')
    records = [BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],
                           task_type='antibody',source='stage1_frozen_gdp_eval')
               for h,l in zip(df.vh_protein_sequence,df.vl_protein_sequence)]
    start = time.monotonic()
    outputs=[]; shapes=[]; next_log=0
    with torch.inference_mode():
        for i in range(0,len(records),args.batch_size):
            batch = collator(records[i:i+args.batch_size])
            mask = batch['residue_mask'].bool() & batch['attention_mask'].bool()
            counts = mask.sum(1).tolist()
            assert counts==[sum(len(c.sequence) for c in r.chains) for r in records[i:i+args.batch_size]]
            assert (batch['input_ids'][batch['relation_mask']]==model.tokenizer.mask_token_id).all()
            assert not (mask & batch['terminal_mask'].bool()).any()
            batch = {k:v.to(args.device) for k,v in batch.items()}
            _, hidden = model.denoise(**model.feature_args(batch),output_hidden_states=True)
            mask=mask.to(args.device)
            pooled=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
            assert torch.isfinite(pooled).all()
            if i==0:
                single={k:v[:1] for k,v in batch.items()}
                _, one=model.denoise(**model.feature_args(single),output_hidden_states=True)
                one=(one*mask[:1].unsqueeze(-1)).sum(1)/mask[:1].sum(1,keepdim=True)
                torch.testing.assert_close(one,pooled[:1],atol=2e-5,rtol=2e-4)
                batch_error=float((one-pooled[:1]).abs().max())
            outputs.append(pooled.cpu().numpy());shapes.append(list(hidden.shape))
            if time.monotonic()>=next_log or i+args.batch_size>=len(records):
                announce(phase='features',arm=args.arm,done=min(i+args.batch_size,len(records)),
                         total=len(records),elapsed_seconds=round(time.monotonic()-start,2))
                next_log=time.monotonic()+60
    values=np.concatenate(outputs)
    assert values.shape==(len(df),model.config.d_model)
    assert sha(checkpoint/'model.safetensors')==binding['model_sha256']
    np.savez_compressed(destination,features=values,row_ids=df.antibody_id.astype(str).to_numpy(dtype=str))
    write(args.output/(args.arm+'_features_receipt.json'),dict(status='complete',shape=list(values.shape),
        features_sha256=sha(destination),plan_sha256=sha(args.output/'plan.json'),
        model_sha256=binding['model_sha256'],frozen_module=str(module.__file__),
        frozen_module_sha256=sha(module.__file__),batch_invariance_max_abs=batch_error,
        input_rows=len(records),residue_counts_verified=True,relation_hidden=True,
        parameter_grad_enabled=False,device=args.device,precision='float32',
        seconds=time.monotonic()-start))


def score(args):
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from threadpoolctl import threadpool_limits
    plan=load_plan(args.output)
    if (args.output/'metrics.json').exists():
        raise FileExistsError(args.output/'metrics.json')
    df=pd.read_csv(args.output/'data_snapshot.csv')
    predictions=[]; metrics=[]
    for arm in plan['arms']:
        receipt=json.loads((args.output/(arm+'_features_receipt.json')).read_text())
        f=args.output/(arm+'_features.npz')
        assert sha(f)==receipt['features_sha256'] and receipt['plan_sha256']==sha(args.output/'plan.json')
        with np.load(f) as values:
            x=values['features'].astype(np.float64)
            assert np.array_equal(values['row_ids'],df.antibody_id.astype(str).to_numpy())
        for target in plan['targets']:
            y=pd.to_numeric(df[target],errors='raise').to_numpy(dtype=float)
            valid=np.isfinite(y)
            for fraction in plan['fractions']:
                seeds=plan['subset_seeds'] if fraction<1 else [11]
                for seed in seeds:
                    for fold in range(5):
                        train=np.flatnonzero(valid & (df[FOLD].to_numpy()!=fold))
                        test=np.flatnonzero(valid & (df[FOLD].to_numpy()==fold))
                        rng=np.random.default_rng(seed+fold*100)
                        train=np.sort(rng.permutation(train)[:max(2,int(len(train)*fraction))])
                        head=make_pipeline(StandardScaler(),Ridge(alpha=plan['fixed_ridge_alpha']))
                        with threadpool_limits(limits=args.threads):
                            head.fit(x[train],y[train]); pred=head.predict(x[test])
                        rho=float(spearmanr(y[test],pred).statistic)
                        assert np.isfinite(rho) and np.isfinite(pred).all()
                        metrics.append(dict(arm=arm,target=target,fraction=fraction,seed=seed,fold=fold,
                            train_n=len(train),test_n=len(test),spearman=rho,
                            rmse=float(np.sqrt(mean_squared_error(y[test],pred)))))
                        for row,p in zip(test,pred):
                            predictions.append(dict(arm=arm,target=target,fraction=fraction,seed=seed,
                                fold=fold,row=int(row),id=str(df.antibody_id.iloc[row]),truth=float(y[row]),prediction=float(p)))
            announce(phase='scored',arm=arm,target=target)
    frame=pd.DataFrame(metrics)
    summary=frame.groupby(['arm','target','fraction']).agg(spearman=('spearman','mean'),
        fold_repeat_sd=('spearman','std'),mean_train_n=('train_n','mean')).reset_index()
    pd.DataFrame(predictions).to_csv(args.output/'predictions.csv',index=False)
    frame.to_csv(args.output/'fold_metrics.csv',index=False)
    summary.to_csv(args.output/'summary.csv',index=False)
    write(args.output/'metrics.json',dict(status='complete',protocol=plan['schema'],
        plan_sha256=sha(args.output/'plan.json'),summary=summary.to_dict('records'),
        predictions_sha256=sha(args.output/'predictions.csv'),
        limitations=['single pretrained seed','pretraining exposure audit pending',
                    'new fixed-alpha diagnostic; not historical legacy_reference scores']))
    announce(phase='score_complete',output=str(args.output))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','features','score'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['mlm','pair_kd'])
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=4)
    args=p.parse_args()
    assert args.output.is_absolute()
    if args.action=='features' and args.arm is None: p.error('features requires --arm')
    globals()[args.action](args)


if __name__=='__main__':
    main()
