"""Test whether low-variance teacher directions matter to the original probes.

Activate pllm/protenix_abtcr; from repository root:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -B \
  scripts/downstream/audit_teacher_spectral_readout.py --out ABS_NEW_DIRECTORY
PCA is fitted only to the existing 4096 OAS training feature matrix. No GPU.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

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
    from downstream.grammar.ab_probes import prepare_task,run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS,FOLD_COL
    torch.set_num_threads(4)
    calibrated=ROOT/'output/diagnostics/final_layer_coordinate_pilot_20261018/calibration.npz'
    features=ROOT/'output/diagnostics/fixed50000_depth_20261018/features.npz'
    previous=ROOT/'output/diagnostics/fixed50000_depth_20261018/metrics.json'
    sources=[Path(__file__),calibrated,features,previous,ROOT/'downstream/grammar/ab_probes.py',ROOT/'data/downstream/dev/GDPa1_v1.2_20250814.csv']
    plan=dict(ranks=[16,64,256,768],PCA_fit='4096 unlabeled OAS training sequences; never GDP labels',
              method='truncate orthogonal teacher PCs, original float64 Ridge grid/folds/label transform',
              interpretation='exploratory readout ablation, not an adapted final-layer result',
              source_sha256={str(p):sha(p) for p in sources})
    write(out/'plan.json',plan)
    cal=np.load(calibrated)
    _,singular,vh=np.linalg.svd(cal['teacher']-cal['mu_t'],full_matrices=False)
    teacher=np.load(features)['native_block2'].astype(np.float64)
    projected=(teacher-cal['mu_t'])@vh.T
    df,_,_,_=prepare_task('gdp_a1')
    results,checks={},[]
    for rank in plan['ranks']:
        folder=out/f'pc{rank}';folder.mkdir()
        result=run_gdp(df,torch.from_numpy(projected[:,:rank].copy()),SimpleNamespace(out_dir=folder))
        write(folder/'metrics.json',result);results[str(rank)]=result;write(out/'metrics.json',results)
        for col in ALL_COLS:
            saved=np.load(folder/col/'legacy_oof.npz');valid=df[col].notna().to_numpy()
            np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid));np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL]);np.testing.assert_array_equal(saved['y_true'],df.loc[valid,col])
            scores=[float(spearmanr(saved['y_transformed'][saved['fold']==i],saved['prediction'][saved['fold']==i]).statistic) for i in range(5)]
            assert abs(np.mean(scores)-result['properties'][col]['legacy_reference']['cv_mean_spearman'])<1e-12
            checks.append(dict(rank=rank,property=col,fold_scores=scores))
    original=json.loads(previous.read_text())['native_block2']['properties']
    full=results['768']['properties']
    deltas={col:full[col]['legacy_reference']['cv_mean_spearman']-original[col]['legacy_reference']['cv_mean_spearman'] for col in ALL_COLS}
    assert abs(deltas['HIC'])<1e-12,deltas
    for p,digest in plan['source_sha256'].items():assert sha(p)==digest,p
    write(out/'verification.json',dict(status='passed',oof_checks=checks,full_rotation_score_deltas=deltas,
          retained_training_variance={str(k):float(np.sum(singular[:k]**2)/np.sum(singular**2)) for k in plan['ranks']}))
    write(out/'status.json',dict(status='success'))


if __name__=='__main__':main()
