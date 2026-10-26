"""Map candidate teacher targets into the expected final-hidden coordinates.

Activate pllm/protenix_abtcr and run from repository root:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/calibrate_multilayer_teacher_targets.py
Only OAS fits means, scales and semi-orthogonal Procrustes rotations. These
teacher diagnostics are not newly trained final-layer model results.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha


def main():
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from types import SimpleNamespace
    from downstream.grammar.ab_probes import prepare_task,run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS,FOLD_COL
    torch.set_num_threads(4)
    prior=ROOT/'output/diagnostics/multilayer_teacher_targets_20261018'
    assert json.loads((prior/'status.json').read_text())['status']=='success'
    out=ROOT/'output/diagnostics/multilayer_teacher_calibrated_20261018'
    out.mkdir(exist_ok=False)
    calpath=ROOT/'output/diagnostics/final_layer_coordinate_pilot_20261018/calibration.npz'
    cal=np.load(calpath);train=np.load(prior/'oas_targets.npz');gdp=np.load(prior/'gdp_target_features.npz')
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),targets=list(train.files),
              transform='OAS-only thin Procrustes; mu_parent_final + rms_parent_final*sqrt(768/d)*target@R',
              reason='Previous unit-variance target probes hit alpha1000 grid ceiling; evaluate actual planned final-hidden scale while retaining original Ridge grid',
              scope='teacher target calibration, NOT trained actual final features or independent confirmation',
              source_sha256={str(p):sha(p) for p in [Path(__file__),calpath,prior/'oas_targets.npz',prior/'gdp_target_features.npz',ROOT/'downstream/grammar/ab_probes.py']})
    write(out/'plan.json',plan);write(out/'status.json',dict(status='running'))
    parent=(cal['student']-cal['mu_s'])/cal['rms_s']
    df,_,_,_=prepare_task('gdp_a1');results,checks,geometry={},{},{}
    for name in train.files:
        folder=out/name;folder.mkdir()
        d=train[name].shape[1];u,_,vh=np.linalg.svd(train[name].T@parent,full_matrices=False)
        rotation=u@vh;scale=float(cal['rms_s']*np.sqrt(768/d))
        np.testing.assert_allclose(rotation@rotation.T,np.eye(d),atol=1e-12,rtol=1e-12)
        transformed_train=cal['mu_s']+train[name]@rotation*scale
        transformed_gdp=cal['mu_s']+gdp[name]@rotation*scale
        a=transformed_gdp-transformed_gdp.mean(0);b=(gdp[name]-gdp[name].mean(0))*scale
        np.testing.assert_allclose(a@a.T,b@b.T,atol=1e-10,rtol=1e-10)
        np.savez_compressed(folder/'target_calibration.npz',rotation=rotation,scale=scale,mean=cal['mu_s'])
        np.save(folder/'oas_targets.npy',transformed_train);np.save(folder/'gdp_targets.npy',transformed_gdp)
        geometry[name]=dict(input_dimension=d,output_dimension=768,scale=scale,
              train_centered_rms=float(np.sqrt(np.mean((transformed_train-transformed_train.mean(0))**2))),
              parent_centered_rms=float(cal['rms_s']),rotation_orthogonality_max_error=float(np.max(np.abs(rotation@rotation.T-np.eye(d)))))
        assert abs(geometry[name]['train_centered_rms']-float(cal['rms_s']))<1e-10
        metrics=run_gdp(df,torch.from_numpy(transformed_gdp.copy()),SimpleNamespace(out_dir=folder))
        write(folder/'metrics.json',metrics);results[name]=metrics;write(out/'metrics.json',results)
        checks[name]=[]
        for col in ALL_COLS:
            oof=np.load(folder/col/'legacy_oof.npz');valid=df[col].notna().to_numpy()
            np.testing.assert_array_equal(oof['row_index'],np.flatnonzero(valid));np.testing.assert_array_equal(oof['fold'],df.loc[valid,FOLD_COL]);np.testing.assert_array_equal(oof['y_true'],df.loc[valid,col])
            scores=[float(spearmanr(oof['y_transformed'][oof['fold']==i],oof['prediction'][oof['fold']==i]).statistic) for i in range(5)]
            assert abs(np.mean(scores)-metrics['properties'][col]['legacy_reference']['cv_mean_spearman'])<1e-12
            checks[name].append(dict(property=col,fold_scores=scores))
    for path,digest in plan['source_sha256'].items():assert sha(Path(path))==digest
    write(out/'verification.json',dict(status='passed',oof_checks=checks,geometry=geometry,orthogonal_GDP_gram_preserved=True))
    write(out/'status.json',dict(status='success',finished_utc=datetime.now(timezone.utc).isoformat()))
    print({name:{p:(v['legacy_reference']['cv_mean_spearman'],v['legacy_reference']['alpha']) for p,v in m['properties'].items()} for name,m in results.items()})


if __name__=='__main__':main()
