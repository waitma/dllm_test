"""Diagnose linear recoverability of block2 from original final features.

Activate pllm/protenix_abtcr; from repository root:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/audit_final_linear_recoverability.py --out ABS_NEW_DIR
Fit maps on existing unlabeled OAS features; choose alpha on a fixed OAS-only
holdout. This diagnostic projection is NOT an improved backbone checkpoint.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from downstream.grammar.ab_probes import prepare_task,run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS,FOLD_COL
    torch.set_num_threads(4)
    source=ROOT/'output/diagnostics/final_layer_coordinate_pilot_20261018/calibration.npz'
    depth=ROOT/'output/diagnostics/fixed50000_depth_20261018'
    cal=np.load(source);x,y=cal['student'],cal['teacher']
    assert x.shape==y.shape==(4096,768)
    order=np.random.default_rng(20261018).permutation(len(x));train,test=order[:3072],order[3072:]
    alphas=[0.,1e-5,1e-4,1e-3,1e-2,.1,1.]
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),alphas=alphas,
              selection='fixed3072/1024 OAS-only holdout; separate minimum uniform and bounded-spectral error; final map refit4096',
              seed=20261018,scope='diagnostic affine map, NOT actual final-layer backbone improvement',
              source_sha256={str(p):sha(p) for p in [Path(__file__),source,depth/'features.npz',ROOT/'downstream/grammar/ab_probes.py']})
    write(out/'plan.json',plan);np.savez_compressed(out/'oas_selection_split.npz',train=train,validation=test)
    write(out/'status.json',dict(status='running',stage='OAS_map_selection'))
    def fit(xs,ys,alpha):
        mx,my=xs.mean(0),ys.mean(0)
        sx=np.sqrt(np.mean((xs-mx)**2));sy=np.sqrt(np.mean((ys-my)**2))
        a,b=(xs-mx)/sx,(ys-my)/sy
        u,s,vh=np.linalg.svd(a,full_matrices=False)
        gain=np.divide(s,s*s+len(a)*alpha,out=np.zeros_like(s),where=s>1e-12)
        w=(vh.T*gain)@(u.T@b)
        return mx,my,sx,sy,w
    def predict(model,values):
        mx,my,sx,sy,w=model
        return ((values-mx)/sx)@w*sy+my
    center=y[train].mean(0);_,singular,vh=np.linalg.svd(y[train]-center,full_matrices=False)
    eigen=singular**2/len(train);weights=1/(eigen+eigen[63])
    selection=[]
    for alpha in alphas:
        prediction=predict(fit(x[train],y[train],alpha),x[test])
        error=prediction-y[test]
        selection.append(dict(alpha=alpha,uniform=float(np.mean(error**2)/np.mean((y[test]-center)**2)),
                              spectral=float(np.mean(((error@vh.T)**2)*weights)/np.mean((((y[test]-center)@vh.T)**2)*weights))))
    chosen={objective:min(selection,key=lambda row:row[objective])['alpha'] for objective in ['uniform','spectral']}
    write(out/'selection.json',dict(validation=selection,chosen=chosen,labels_used=False))
    features=np.load(depth/'features.npz');df,_,_,_=prepare_task('gdp_a1')
    results,checks={},[]
    for objective,alpha in chosen.items():
        model=fit(x,y,alpha)
        mapped=predict(model,features['native_final_norm'].astype(np.float64))
        folder=out/objective;folder.mkdir()
        np.savez_compressed(folder/'map.npz',mu_x=model[0],mu_y=model[1],scale_x=model[2],scale_y=model[3],weight=model[4])
        # Persist/reload the actual map and check prediction equality before fitting heads.
        saved=np.load(folder/'map.npz')
        restored=((features['native_final_norm'].astype(np.float64)-saved['mu_x'])/saved['scale_x'])@saved['weight']*saved['scale_y']+saved['mu_y']
        np.testing.assert_array_equal(restored,mapped)
        np.save(folder/'features.npy',mapped)
        metrics=run_gdp(df,torch.from_numpy(mapped),SimpleNamespace(out_dir=folder));write(folder/'metrics.json',metrics)
        for col in ALL_COLS:
            oof=np.load(folder/col/'legacy_oof.npz');valid=df[col].notna().to_numpy()
            np.testing.assert_array_equal(oof['row_index'],np.flatnonzero(valid));np.testing.assert_array_equal(oof['fold'],df.loc[valid,FOLD_COL]);np.testing.assert_array_equal(oof['y_true'],df.loc[valid,col])
            scores=[float(spearmanr(oof['y_transformed'][oof['fold']==i],oof['prediction'][oof['fold']==i]).statistic) for i in range(5)]
            assert abs(np.mean(scores)-metrics['properties'][col]['legacy_reference']['cv_mean_spearman'])<1e-12
            checks.append(dict(objective=objective,property=col,fold_scores=scores))
        results[objective]=metrics
        write(out/'metrics.json',results)
    for path,digest in plan['source_sha256'].items():assert sha(Path(path))==digest
    write(out/'verification.json',dict(status='passed',map_roundtrip_exact=True,oof_checks=checks))
    write(out/'status.json',dict(status='success',finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(chosen=chosen,scores={k:{p:v['legacy_reference']['cv_mean_spearman'] for p,v in m['properties'].items()} for k,m in results.items()}),indent=2))


if __name__=='__main__':main()
