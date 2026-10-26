"""Evaluate candidate training targets built only from frozen parent features.

Activate pllm/protenix_abtcr; from repository root:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/audit_multilayer_teacher_targets.py --out ABS_NEW_DIR
PCA/scales fit4096 OAS training pairs only. This is a teacher-target diagnostic,
not a backbone checkpoint, inference feature proposal, or target achievement.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from types import SimpleNamespace
    from downstream.grammar.ab_probes import prepare_task,run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS,FOLD_COL
    torch.set_num_threads(4)
    source=ROOT/'output/diagnostics/final_layer_coordinate_pilot_20261018/calibration.npz'
    depth=ROOT/'output/diagnostics/fixed50000_depth_20261018/features.npz'
    cal=np.load(source);gdp=np.load(depth)
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),
              variants={'early256':'256 block2 PCs','early512':'512 block2 PCs, dimension control',
                        'early256_late256':'256 block2 PCs plus256 original-final PCs, equal branch mean energy'},
              normalization='Each PCA fitted4096 OAS; scores divided by sqrt(lambda_i+lambda_64); each branch has unit mean squared coordinate on train',
              selection='all three targets and all five properties; no GDP label fitting of PCA/scales',
              scope='diagnostic target features, NOT new actual final-layer scores or independent confirmation',
              source_sha256={str(p):sha(p) for p in [Path(__file__),source,depth,ROOT/'downstream/grammar/ab_probes.py']})
    write(out/'plan.json',plan);write(out/'status.json',dict(status='running'))
    fitted={};artifacts={}
    for name,train,test in [('early',cal['teacher'],gdp['native_block2'].astype(np.float64)),
                            ('late',cal['student'],gdp['native_final_norm'].astype(np.float64))]:
        mean=train.mean(0);_,s,vh=np.linalg.svd(train-mean,full_matrices=False)
        eigen=s*s/len(train);denominator=np.sqrt(eigen+eigen[63])
        train_scores=(train-mean)@vh.T/denominator
        scores=(test-mean)@vh.T/denominator
        fitted[name]=(train_scores,scores)
        artifacts.update({name+'_mean':mean,name+'_axes':vh,name+'_eigenvalues':eigen,name+'_denominator':denominator})
    features={};normalized_train={}
    for name,branches in [('early256',[('early',256)]),('early512',[('early',512)]),('early256_late256',[('early',256),('late',256)])]:
        train_chunks,test_chunks=[],[]
        for branch,rank in branches:
            train,test=fitted[branch]
            scale=float(np.sqrt(np.mean(train[:,:rank]**2)))
            artifacts[name+'_'+branch+'_scale']=np.array(scale)
            train_chunks.append(train[:,:rank]/scale);test_chunks.append(test[:,:rank]/scale)
        normalized_train[name]=np.concatenate(train_chunks,axis=1)
        features[name]=np.concatenate(test_chunks,axis=1)
        assert abs(np.mean(normalized_train[name]**2)-1)<1e-10
    np.savez_compressed(out/'oas_fitted_targets.npz',**artifacts)
    np.savez_compressed(out/'oas_targets.npz',**normalized_train)
    np.savez_compressed(out/'gdp_target_features.npz',**features)
    df,_,_,_=prepare_task('gdp_a1');results,checks={},[]
    for name,values in features.items():
        folder=out/name;folder.mkdir()
        metrics=run_gdp(df,torch.from_numpy(values.copy()),SimpleNamespace(out_dir=folder))
        write(folder/'metrics.json',metrics);results[name]=metrics;write(out/'metrics.json',results)
        for col in ALL_COLS:
            saved=np.load(folder/col/'legacy_oof.npz');valid=df[col].notna().to_numpy()
            np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid));np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL]);np.testing.assert_array_equal(saved['y_true'],df.loc[valid,col])
            scores=[float(spearmanr(saved['y_transformed'][saved['fold']==i],saved['prediction'][saved['fold']==i]).statistic) for i in range(5)]
            assert abs(np.mean(scores)-metrics['properties'][col]['legacy_reference']['cv_mean_spearman'])<1e-12
            checks.append(dict(target=name,property=col,fold_scores=scores))
    for path,digest in plan['source_sha256'].items():assert sha(Path(path))==digest
    write(out/'verification.json',dict(status='passed',oof_checks=checks,normalization_train_only=True))
    write(out/'status.json',dict(status='success',finished_utc=datetime.now(timezone.utc).isoformat()))
    print({name:{p:v['legacy_reference']['cv_mean_spearman'] for p,v in result['properties'].items()} for name,result in results.items()})


if __name__=='__main__':main()
