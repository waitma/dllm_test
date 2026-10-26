"""Test OAS-fitted complementary targets in the frozen LM-head nullspace.

Activate pllm/protenix_abtcr; run from repository root:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B scripts/downstream/audit_head_nullspace_targets.py --out ABS_NEW_DIR
CPU-only target diagnostic, not student training or generation preservation.
All transforms fit OAS train; GDP labels are used only by the existing probes.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args();out = args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    state = dict(status='running',stage='prepare',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',state)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    torch.set_num_threads(4)
    root = ROOT/'output/diagnostics'
    mixed = root/'global_teacher_targets_20261018/early256_global256'
    assert json.loads((mixed.parent/'status.json').read_text())['status'] == 'success'
    assert json.loads((mixed.parent/'verification.json').read_text())['status'] == 'passed'
    hashes = {}
    def pin(path):
        hashes[str(path)] = sha(path);return path
    cal = np.load(pin(root/'final_layer_coordinate_pilot_20261018/calibration.npz'))
    old = np.load(pin(mixed/'calibration.npz'))
    target = {split:np.load(pin(mixed/file)).astype(np.float64)
              for split,file in [('train','oas_targets.npy'),('gdp','gdp_targets.npy')]}
    parent = {'train':cal['student'].astype(np.float64),
              'gdp':np.load(pin(root/'fixed50000_depth_20261018/features.npz'))['native_final_norm'].astype(np.float64)}
    from downstream.fixed_chain_eval import load
    checkpoint = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
    model, identity = load(checkpoint,device='cpu')
    head = model.backbone.transformer.ff_out
    assert getattr(head,'bias',None) is None
    w = head.weight.detach().double().numpy().copy()
    gamma = model.backbone.transformer.ln_f.weight.detach().double().numpy().copy()
    assert w.shape == (50,768) and np.min(np.abs(gamma)) > 0
    _,singular,vh = np.linalg.svd(w,full_matrices=True)
    tolerance = np.max(w.shape)*np.finfo(np.float64).eps*singular[0]
    rank = int(np.sum(singular>tolerance))
    row = vh[:rank].T;null = vh[rank:].T
    assert null.shape[1] >= 512
    np.testing.assert_allclose(w@null,0,atol=1e-12,rtol=0)
    np.testing.assert_allclose(row@row.T+null@null.T,np.eye(768),atol=1e-12,rtol=0)
    mean = cal['mu_s'].astype(np.float64)
    z = {split:(x-old['mean'])@old['rotation'].T/old['scale'] for split,x in target.items()}
    for split in z:
        np.testing.assert_allclose(old['mean']+z[split]@old['rotation']*old['scale'],target[split],atol=1e-9,rtol=1e-9)
    np.testing.assert_allclose(z['train'].mean(0),0,atol=1e-10)
    centered = parent['train']-mean
    u,_,v = np.linalg.svd(z['train'].T@(centered@null),full_matrices=False)
    rotation = (u@v)@null.T
    np.testing.assert_allclose(rotation@rotation.T,np.eye(512),atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(rotation@w.T,0,atol=1e-12,rtol=0)
    null_energy = float(np.mean(np.sum((centered@null)**2,axis=1)))
    z_energy = float(np.mean(np.sum(z['train']**2,axis=1)))
    null_scale = float(np.sqrt(null_energy/z_energy))
    original_scale = float(old['scale'])
    np.savez_compressed(out/'calibration.npz',head=w,gamma=gamma,row_basis=row,null_basis=null,
        rotation=rotation,mean=mean,null_scale=null_scale,original_scale=original_scale,singular_values=singular)
    plan = dict(created_utc=state['started_utc'],parent_identity=identity,
        fit='OAS4096 only; existing512 complementary semantic coordinates; thin Procrustes into complete50-class head nullspace',
        rank=rank,null_dimension=null.shape[1],semantic_dimension=512,
        variants=['null_mixed512_isometry','null_mixed512_parent_rowspace'],
        isometry_control='same semantic scale as original mixed target; preserves centered sample Gram',
        rowspace_target='parent native pooled rowspace + OAS-mean null component + semantic null component scaled to original train null energy',
        limitation='Target-only. Equal mean raw head outputs on clean full sequences do NOT prove tokenwise predictions, masked-context behavior, actual student learnability, or CDR retention. No inference projection is proposed as a completed model.',
        labels_used_for_transform=False)
    write(out/'plan.json',plan)
    from downstream.grammar.ab_probes import prepare_task, run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS, FOLD_COL
    df,_,_,_ = prepare_task('gdp_a1')
    del model,head
    results, checks, geometry = {}, [], {}
    for name in plan['variants']:
        folder = out/name;folder.mkdir();arrays = {}
        for split in z:
            if name == 'null_mixed512_isometry':
                value = mean+z[split]@rotation*original_scale
                a = value-value.mean(0);b = target[split]-target[split].mean(0)
                np.testing.assert_allclose(a@a.T,b@b.T,atol=1e-8,rtol=1e-10)
            else:
                value = mean+((parent[split]-mean)@row)@row.T+z[split]@rotation*null_scale
                np.testing.assert_allclose(value@w.T,parent[split]@w.T,atol=1e-10,rtol=1e-10)
            weighted_rms = np.sqrt(np.mean((value/gamma)**2,axis=1))
            geometry[name+'/'+split] = dict(rows=len(value),centered_rms=float(np.sqrt(np.mean((value-value.mean(0))**2))),
                mean_raw_head_difference_max=float(np.max(np.abs(value@w.T-parent[split]@w.T))),
                necessary_mean_norm_bound_fraction_exceeded=float(np.mean(weighted_rms>1+1e-7)),
                necessary_mean_norm_bound_max=float(weighted_rms.max()))
            arrays[split] = value
            np.save(folder/('oas_targets.npy' if split=='train' else 'gdp_targets.npy'),value)
        state.update(stage='ridge',variant=name);write(out/'status.json',state)
        metric = run_gdp(df,torch.from_numpy(arrays['gdp']),SimpleNamespace(out_dir=folder))
        results[name] = metric;write(folder/'metrics.json',metric);write(out/'results.json',results)
        for prop in ALL_COLS:
            saved = np.load(folder/prop/'legacy_oof.npz');reference = np.load(pin(mixed/prop/'legacy_oof.npz'))
            valid = df[prop].notna().to_numpy()
            np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid))
            np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL])
            for key in ['row_index','fold','y_true','y_transformed']:
                np.testing.assert_array_equal(saved[key],reference[key])
            scores = [float(spearmanr(saved['y_transformed'][saved['fold']==k],saved['prediction'][saved['fold']==k]).statistic) for k in range(5)]
            assert abs(np.mean(scores)-metric['properties'][prop]['legacy_reference']['cv_mean_spearman'])<1e-12
            checks.append(dict(variant=name,property=prop,fold_scores=scores,
                max_oof_difference_from_original_mixed=float(np.max(np.abs(saved['prediction']-reference['prediction'])))))
        print(name,{p:v['legacy_reference']['cv_mean_spearman'] for p,v in metric['properties'].items()},flush=True)
    for path in [Path(__file__),ROOT/'downstream/fixed_chain_eval.py',ROOT/'downstream/grammar/ab_probes.py',ROOT/'data/downstream/dev/GDPa1_v1.2_20250814.csv']:
        pin(path)
    for path,digest in hashes.items():assert sha(Path(path))==digest,path
    write(out/'geometry.json',geometry)
    write(out/'verification.json',dict(status='passed',source_sha256=hashes,oof_checks=checks,
        full_head_nullspace_verified=True,control_centered_gram_preserved=True,
        rowspace_target_clean_mean_raw_head_outputs_preserved=True,
        actual_student_trained=False,actual_generation_preservation_tested=False))
    state.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',state)


if __name__ == '__main__':
    main()
