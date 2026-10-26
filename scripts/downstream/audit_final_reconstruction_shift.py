"""Compare OAS-heldout and GDP teacher reconstruction, without GDP labels.

Activate pllm/protenix_abtcr; run from repository root:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/audit_final_reconstruction_shift.py
Reuses the OAS-selected linear diagnostic; applies the same3072-fit map to
OAS1024 and GDP246, and records descriptive teacher-PC distances and errors.
"""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha


def main():
    import numpy as np
    run=ROOT/'output/diagnostics/final_layer_linear_recoverability_20261018'
    assert json.loads((run/'status.json').read_text())['status']=='success'
    cal_path=ROOT/'output/diagnostics/final_layer_coordinate_pilot_20261018/calibration.npz'
    features_path=ROOT/'output/diagnostics/fixed50000_depth_20261018/features.npz'
    cal=np.load(cal_path);split=np.load(run/'oas_selection_split.npz');features=np.load(features_path)
    train,test=split['train'],split['validation'];x,y=cal['student'],cal['teacher']
    alpha=json.loads((run/'selection.json').read_text())['chosen']['uniform']
    mx,my=x[train].mean(0),y[train].mean(0)
    sx=np.sqrt(np.mean((x[train]-mx)**2));sy=np.sqrt(np.mean((y[train]-my)**2))
    a,b=(x[train]-mx)/sx,(y[train]-my)/sy
    u,s,vh=np.linalg.svd(a,full_matrices=False)
    weights=(vh.T*(s/(s*s+len(train)*alpha)))@(u.T@b)
    _,values,axes=np.linalg.svd(y[train]-my,full_matrices=False)
    eigen=values**2/len(train)
    result=dict(method='Same OAS3072-fit affine map, selected alpha; no GDP labels. PC geometry fitted on OAS3072 only.',
                alpha=alpha,sets={})
    for name,xx,yy in [('OAS_train',x[train],y[train]),('OAS_holdout',x[test],y[test]),
                       ('GDP',features['native_final_norm'].astype(np.float64),features['native_block2'].astype(np.float64))]:
        predicted=((xx-mx)/sx)@weights*sy+my
        error=predicted-yy
        projected=(yy-my)@axes.T
        err_pc=error@axes.T
        distance=np.mean(projected[:,:64]**2/(eigen[:64]+eigen[63]),axis=1)
        item=dict(n=len(xx),mse_over_OAS_training_variance=float(np.mean(error**2)/sy**2),
                  mse_over_own_centered_variance=float(np.mean(error**2)/np.mean((yy-yy.mean(0))**2)),
                  teacher_PC64_distance_quantiles={str(q):float(np.quantile(distance,q)) for q in [0,.25,.5,.75,.9,1.]},bands={})
        for start,stop in [(0,16),(16,64),(64,256),(256,768)]:
            item['bands'][f'PC{start+1}-{stop}']=dict(mse_over_OAS_band_variance=float(np.mean(err_pc[:,start:stop]**2)/np.mean(eigen[start:stop])),
                     teacher_mean_shift_over_band_variance=float(np.mean(projected[:,start:stop].mean(0)**2)/np.mean(eigen[start:stop])))
        result['sets'][name]=item
    result['source_sha256']={str(p):sha(p) for p in [Path(__file__),cal_path,features_path,run/'selection.json',run/'oas_selection_split.npz']}
    write(run/'distribution_shift.json',result)
    print(json.dumps(result['sets'],indent=2))


if __name__=='__main__':main()
