"""Verify paired retention continuations and compare their fixed6000 endpoints.

Activate pllm/protenix_abtcr; run from repository root:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/audit_final_retention_pilot.py --out ABS_NEW_DIR
Requires both arms terminal. Does not turn replay-monitor improvements into
claims about actual TCR few-shot classification or multi-step CDR generation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    root=ROOT/'output/diagnostics'
    run=root/'final_layer_retention_pilot_20261018'
    hashes={}
    def read(path):
        hashes[str(path)]=sha(path);return json.loads(path.read_text())
    def array(path):
        hashes[str(path)]=sha(path);return np.load(path)
    assert read(run/'status.json')['status']=='success'
    plan=read(run/'plan.json')
    assert not plan['preflight_only']
    assert plan['arms']==['ab_control','tcr_cdr_replay']
    assert plan['initial_steps']==5000 and plan['total_steps']==6000 and plan['additional_steps']==1000
    for key in ['source_sha256','control_artifact_sha256','replay_source_sha256']:
        for path,digest in plan[key].items():assert sha(Path(path))==digest,path
    for filename,key in [('schedule.json','schedule_sha256'),('tcr_schedule.json','tcr_schedule_sha256')]:
        assert sha(run/filename)==plan[key]
        assert len(read(run/filename))==1000
    initial=Path(plan['initial_checkpoint'])
    start=read(initial/'checkpoint.json')
    assert start['sha256']==plan['initial_parameter_sha256']
    assert sha(Path(start['parameter_file']))==start['sha256']
    assert sha(initial/'optimizer.pt')==plan['initial_optimizer_sha256']
    preflight=read(run/'replay_preflight.json')
    assert preflight['status']=='passed' and preflight['parameters_unchanged'] and preflight['gradient_norm']>0
    normalization=read(run/'replay_normalization.json')
    assert normalization['train_rows']==128 and normalization['all_three_cdrs']
    for key,weight in normalization['weights'].items():
        assert abs(weight*normalization['initial_means'][key]-.1)<1e-12
    results=read(run/'results.json');assert set(results)==set(plan['arms'])
    checkpoints,validation,replay,first={},{},{},{}
    for name in plan['arms']:
        folder=run/name
        side=read(folder/'checkpoint.json');checkpoints[name]=side
        assert side['steps']==6000 and side['initial_adaptation_sha256']==start['sha256']
        assert sha(Path(side['parameter_file']))==side['sha256']
        assert side['frozen_parameters_bitwise_equal']
        verification=read(folder/'verification.json')
        assert verification['status']=='passed' and len(verification['oof_checks'])==10
        assert verification['reloaded_features_bitwise_equal'] and verification['frozen_parameters_bitwise_equal']
        resume=read(folder/'resume_verification.json')
        assert resume['global_step']==5000 and resume['optimizer_steps']==[4700]
        assert resume['optimizer_sha256']==plan['initial_optimizer_sha256']
        assert sha(folder/'optimizer.pt')==side['optimizer_sha256']
        optimizer=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=False)
        assert optimizer['global_step']==6000
        assert set(optimizer['ordered_parameter_names'])==set(side['trainable_names'])
        assert {int(state['step'].item()) for state in optimizer['optimizer']['state'].values()}=={5700}
        del optimizer
        history=read(folder/'validation.json')
        assert [v['step'] for v in history]==list(range(5000,6001,100))
        initial_validation=read(run/'initial_validation.json')
        assert history[0].keys()==initial_validation.keys()
        for key,value in initial_validation.items():
            assert abs(history[0][key]-value)<2e-6,(name,key)
        validation[name]=history[-1]
        path=folder/'replay_validation.jsonl';hashes[str(path)]=sha(path)
        observations=[json.loads(line) for line in path.read_text().splitlines()]
        assert [v['step'] for v in observations]==list(range(5100,6001,100))
        replay[name]=observations[-1]
        path=folder/'train.jsonl';hashes[str(path)]=sha(path)
        records=[json.loads(line) for line in path.read_text().splitlines()]
        assert [v['step'] for v in records]==[5001]+list(range(5020,6001,20))
        first[name]=read(folder/'first_continued_batch.json')
    assert first['ab_control']==first['tcr_cdr_replay']
    assert checkpoints['ab_control']['trainable_names']==checkpoints['tcr_cdr_replay']['trainable_names']
    teacher_dir=root/'fixed50000_depth_20261018/native_block2'
    teacher=read(teacher_dir/'metrics.json')
    initial_results=read(initial.parent/'results.json')['token_all_direction']
    comparison,checks={},[]
    for mode,dirname in [('native','gdp_a1'),('no_mask_keys','gdp_no_mask_keys')]:
        comparison[mode]={}
        for prop,reference in teacher['properties'].items():
            paths={'block2':teacher_dir/prop/'legacy_oof.npz','ce1_5000':initial/dirname/prop/'legacy_oof.npz'}
            paths.update({name:run/name/dirname/prop/'legacy_oof.npz' for name in plan['arms']})
            data={name:array(path) for name,path in paths.items()};scores={}
            for name,saved in data.items():
                for key in ['row_index','fold','y_true','y_transformed']:
                    np.testing.assert_array_equal(saved[key],data['block2'][key])
                fold=saved['fold']
                values=[float(spearmanr(saved['y_transformed'][fold==k],saved['prediction'][fold==k]).statistic) for k in range(5)]
                if name=='block2':reported=reference
                elif name=='ce1_5000':reported=initial_results[mode]['properties'][prop]
                else:reported=results[name][mode]['properties'][prop]
                assert abs(np.mean(values)-reported['legacy_reference']['cv_mean_spearman'])<1e-12
                scores[name]=dict(mean=float(np.mean(values)),folds=values)
                checks.append(dict(mode=mode,model=name,property=prop,rows=len(fold)))
            comparison[mode][prop]=dict(scores=scores,replay_minus_control=(np.array(scores['tcr_cdr_replay']['folds'])-np.array(scores['ab_control']['folds'])).tolist())
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'results.json',dict(GDP=comparison,validation=validation,replay_screen=replay,
        initial_replay_screen=read(run/'initial_replay_screen.json'),normalization=normalization,
        limitations='Exploratory single seed. Replay monitors are not actual TCR/CDR task scores; full cross-task validation and stability remain necessary.'))
    hashes[str(Path(__file__))]=sha(Path(__file__))
    write(out/'verification.json',dict(status='passed',fixed_endpoint=6000,matched_first_batch=True,
        initial_optimizer_and_schedule_verified=True,final_optimizer_internal_steps=5700,
        actual_parameter_sha_verified=True,oof_checks=checks,source_sha256=hashes))
    write(out/'status.json',dict(status='success',finished_utc=datetime.now(timezone.utc).isoformat()))
    for mode,properties in comparison.items():
        print(mode,{p:{n:round(s['mean'],6) for n,s in entry['scores'].items()} for p,entry in properties.items()},flush=True)
    print('replay screen',replay,flush=True)


if __name__=='__main__':main()
