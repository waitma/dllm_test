"""Check necessary RMSNorm bounds for pooled teacher targets, without labels.

Activate pllm/protenix_abtcr and run after chain calibration exists:
python -B scripts/downstream/audit_preservation_target_feasibility.py
This necessary bound is not proof of model realizability. Uses starting1000
RMSNorm gains; those gains can subsequently train. No GPU or probe fitting.
"""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha


def main():
    import numpy as np
    from safetensors import safe_open
    run=ROOT/'output/diagnostics/final_layer_chain_preservation_20261018'
    cal_path=ROOT/'output/diagnostics/final_layer_coordinate_pilot_20261018/calibration.npz'
    chain_path=run/'chain_calibration.npz'
    delta_path=ROOT/'output/diagnostics/final_layer_spectral_pilot_20261018/spectral_continue/trained_parameters.safetensors'
    source=ROOT/'output/downstream_generation/fixed50000_gdpa1_pooling_20261018'
    cal=np.load(cal_path);chain=np.load(chain_path);gdp=np.load(source/'features.npz')
    with safe_open(str(delta_path),framework='pt',device='cpu') as f:
        gain=f.get_tensor('backbone.transformer.ln_f.weight').numpy().astype(np.float64)
    assert np.min(np.abs(gain))>.1
    train_path=ROOT/'output/diagnostics/final_layer_preservation_pilot_20261018/train.jsonl'
    rows=[json.loads(line) for line in train_path.read_text().splitlines()]
    lengths=np.array([[len(dict(zip(row['chain_roles'],row['chains']))[role]) for role in ['antibody_heavy','antibody_light']] for row in rows])
    token_audit=json.loads((source/'token_audit.json').read_text())
    gdp_lengths=np.array([[row['heavy'],row['light']] for row in token_audit['rows']])
    teacher_hl=gdp['block2_residue_hl_concat'].astype(np.float64)
    contexts=[('OAS_train',cal['teacher'],chain['teacher_difference'],lengths),
              ('GDP',gdp['block2_residue_global'].astype(np.float64),teacher_hl[:,:768]-teacher_hl[:,768:],gdp_lengths)]
    result=dict(method='For z=gamma*x/RMS(x), every token has ||z/gamma||<=sqrt(d). A mean satisfies the same bound. Targets checked at starting gamma only.',
                limitation='Necessary, not sufficient; trainable gamma can change the bound. No GDP labels used.',sets={})
    for name,global_teacher,difference_teacher,lens in contexts:
        global_target=cal['mu_s']+((global_teacher-cal['mu_t'])/cal['rms_t'])@cal['rotation']*cal['rms_s']
        difference_target=chain['mu_ds']+((difference_teacher-chain['mu_dt'])/cal['rms_t'])@cal['rotation']*cal['rms_s']
        wh=lens[:,0:1]/lens.sum(1,keepdims=True);wl=1-wh
        targets={'global':global_target,'heavy':global_target+wl*difference_target,'light':global_target-wh*difference_target}
        np.testing.assert_allclose(wh*targets['heavy']+wl*targets['light'],global_target,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(targets['heavy']-targets['light'],difference_target,atol=1e-12,rtol=1e-12)
        result['sets'][name]={}
        for target_name,target in targets.items():
            radius=np.sqrt(np.mean((target/gain)**2,axis=1))
            result['sets'][name][target_name]=dict(n=len(radius),above_bound=int(np.sum(radius>1)),
                radius_quantiles={str(q):float(np.quantile(radius,q)) for q in [0,.5,.9,.99,1.]})
    result['source_sha256']={str(p):sha(p) for p in [Path(__file__),cal_path,chain_path,delta_path,train_path,source/'features.npz',source/'token_audit.json']}
    write(run/'target_feasibility.json',result)
    print(json.dumps(result['sets'],indent=2))


if __name__=='__main__':main()
