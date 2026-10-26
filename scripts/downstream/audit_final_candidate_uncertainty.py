"""Conditional paired bootstrap of existing GDP out-of-fold predictions.

Activate pllm/protenix_abtcr and run from repository root:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/audit_final_candidate_uncertainty.py --arm ABS_COMPLETED_ARM --out ABS_NEW_DIR
Resamples rows within each original fold; no refitting or hyperparameter search.
Intervals exclude training-seed and alpha-selection uncertainty, so are NOT
independent confirmation or a noninferiority test.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha


def main():
    import numpy as np
    from scipy.stats import rankdata
    depth=ROOT/'output/diagnostics/fixed50000_depth_20261018'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();arm=args.arm.resolve();out=args.out.resolve()
    assert json.loads((arm.parent/'status.json').read_text())['status']=='success'
    verification=json.loads((arm/'verification.json').read_text())
    assert verification['status']=='passed' and verification['reloaded_features_bitwise_equal']
    side=json.loads((arm/'checkpoint.json').read_text())
    assert sha(Path(side['parameter_file']))==side['sha256']
    out.mkdir(parents=True,exist_ok=False)
    methods={'native':arm/'gdp_a1','no_mask_keys':arm/'gdp_no_mask_keys'}
    teacher=depth/'native_block2'
    plan=dict(seed=20261018,replicates=2000,method='paired row resampling within each fixed fold, then mean of five correlations',
              limitations='fixed saved OOF and selected alphas; excludes parameter, training-seed and selection uncertainty; exploratory, not confirmation',
              methods={k:str(v) for k,v in methods.items()},reference=str(teacher),script_sha256=sha(Path(__file__)))
    write(out/'plan.json',plan)
    rng=np.random.default_rng(plan['seed']);results={};sources={str(arm/'checkpoint.json'):sha(arm/'checkpoint.json'),str(Path(__file__)):sha(Path(__file__))}
    def corr(a,b):
        a=rankdata(a,axis=-1);b=rankdata(b,axis=-1)
        a-=a.mean(axis=-1,keepdims=True);b-=b.mean(axis=-1,keepdims=True)
        return (a*b).sum(-1)/np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
    for prop in ['AC-SINS_pH7.4','HIC','PR_CHO','Titer','Tm2']:
        baseline=np.load(teacher/prop/'legacy_oof.npz')
        actual={name:np.load(folder/prop/'legacy_oof.npz') for name,folder in methods.items()}
        for name,data in actual.items():
            for key in ['row_index','fold','y_true','y_transformed']:
                np.testing.assert_array_equal(data[key],baseline[key])
            sources[str(methods[name]/prop/'legacy_oof.npz')]=sha(methods[name]/prop/'legacy_oof.npz')
        boot={name:np.zeros(plan['replicates']) for name in methods}
        point={name:[] for name in methods}
        for fold in range(5):
            m=baseline['fold']==fold;y=baseline['y_transformed'][m];t=baseline['prediction'][m]
            draws=rng.integers(0,len(y),size=(plan['replicates'],len(y)))
            ref=corr(y[draws],t[draws]);ref_point=float(corr(y,t))
            for name,data in actual.items():
                pred=data['prediction'][m]
                boot[name]+=(corr(y[draws],pred[draws])-ref)/5
                point[name].append(float(corr(y,pred))-ref_point)
        results[prop]={}
        for name in methods:
            assert np.isfinite(boot[name]).all()
            results[prop][name]=dict(mean_delta=float(np.mean(point[name])),
                conditional_paired_95_interval=np.quantile(boot[name],[.025,.975]).tolist(),fold_deltas=point[name])
        sources[str(teacher/prop/'legacy_oof.npz')]=sha(teacher/prop/'legacy_oof.npz')
    for path,digest in sources.items():assert sha(Path(path))==digest,path
    write(out/'results.json',results);write(out/'source_sha256.json',sources)
    write(out/'status.json',dict(status='success'))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
