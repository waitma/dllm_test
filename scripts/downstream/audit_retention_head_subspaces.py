"""Locate GDP changes between retention arms using cached head-subspace swaps.

Activate pllm/protenix_abtcr; run from repository root:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/audit_retention_head_subspaces.py --out ABS_NEW
CPU diagnostic only. Exchanges the frozen50-class head row/null components of
saved final features; these hybrids are not model outputs or deployment options.
The head basis uses no GDP labels. Original folds and Ridge protocol are reused.
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
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    status = dict(status='running',stage='prepare',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',status)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    torch.set_num_threads(4)
    run = ROOT/'output/diagnostics/final_layer_retention_pilot_20261018'
    basis_root = ROOT/'output/diagnostics/head_nullspace_targets_20261018'
    hashes = {}
    def pin(path):
        hashes[str(path)] = sha(path); return path
    def read(path): return json.loads(pin(path).read_text())
    assert read(run/'status.json')['status']=='success'
    assert read(basis_root/'verification.json')['status']=='passed'
    basis = np.load(pin(basis_root/'calibration.npz'))
    w,row,null = [basis[name] for name in ['head','row_basis','null_basis']]
    np.testing.assert_allclose(w@null,0,atol=1e-12,rtol=0)
    np.testing.assert_allclose(row@row.T+null@null.T,np.eye(768),atol=1e-12,rtol=0)
    assert w.shape==(50,768) and row.shape==(768,50)
    identities = {}
    for arm in ['ab_control','tcr_cdr_replay']:
        verification = read(run/arm/'verification.json')
        assert verification['status']=='passed' and verification['reloaded_features_bitwise_equal']
        side = read(run/arm/'checkpoint.json')
        assert sha(side['parameter_file'])==side['sha256'] and side['frozen_parameters_bitwise_equal']
        identities[arm] = side['sha256']
    from downstream.grammar.ab_probes import prepare_task,run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS,FOLD_COL
    from sklearn.linear_model import Ridge
    df,_,_,_ = prepare_task('gdp_a1')
    results,checks,geometry,transfers = {},[],{},{}
    for mode,filename,dirname in [('native','final_features.npy','gdp_a1'),('no_mask_keys','final_no_mask_features.npy','gdp_no_mask_keys')]:
        control = np.load(pin(run/'ab_control'/filename)).astype(np.float64)
        replay = np.load(pin(run/'tcr_cdr_replay'/filename)).astype(np.float64)
        delta = replay-control
        drow = (delta@row)@row.T; dnull = (delta@null)@null.T
        np.testing.assert_allclose(drow+dnull,delta,atol=1e-12,rtol=1e-12)
        geometry[mode] = dict(delta_energy=float(np.mean(delta**2)),row_energy=float(np.mean(drow**2)),
            null_energy=float(np.mean(dnull**2)),row_fraction=float(np.sum(drow**2)/np.sum(delta**2)),
            mean_raw_head_change_rms=float(np.sqrt(np.mean((delta@w.T)**2))))
        variants = {'replay_row_control_null':control+drow,'control_row_replay_null':control+dnull}
        np.testing.assert_allclose(variants['replay_row_control_null']@w.T,replay@w.T,atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(variants['control_row_replay_null']@w.T,control@w.T,atol=1e-10,rtol=1e-10)
        for name,x in variants.items():
            label=mode+'/'+name;folder=out/mode/name;folder.mkdir(parents=True)
            status.update(stage='ridge',variant=label);write(out/'status.json',status)
            np.save(folder/'features.npy',x)
            metrics=run_gdp(df,torch.from_numpy(x),SimpleNamespace(out_dir=folder));write(folder/'metrics.json',metrics)
            results[label]=metrics;transfers[label]={}
            for prop in ALL_COLS:
                saved=np.load(folder/prop/'legacy_oof.npz')
                original=np.load(pin(run/'ab_control'/dirname/prop/'legacy_oof.npz'))
                valid=df[prop].notna().to_numpy()
                for key in ['row_index','fold','y_true','y_transformed']:
                    np.testing.assert_array_equal(saved[key],original[key])
                np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid))
                np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL])
                fold=saved['fold'];target=saved['y_transformed']
                scores=[float(spearmanr(target[fold==k],saved['prediction'][fold==k]).statistic) for k in range(5)]
                assert abs(np.mean(scores)-metrics['properties'][prop]['legacy_reference']['cv_mean_spearman'])<1e-12
                reference=read(run/'ab_control'/dirname/'metrics.json')['properties'][prop]['legacy_reference']
                fixed=np.zeros(len(fold));reproduced=np.zeros(len(fold))
                for k in range(5):
                    train=fold!=k;test=fold==k
                    probe=Ridge(alpha=reference['alpha']).fit(control[valid][train],target[train])
                    fixed[test]=probe.predict(x[valid][test]);reproduced[test]=probe.predict(control[valid][test])
                np.testing.assert_allclose(reproduced,original['prediction'],atol=1e-8,rtol=1e-8)
                fixed_scores=[float(spearmanr(target[fold==k],fixed[fold==k]).statistic) for k in range(5)]
                np.savez_compressed(folder/prop/'control_probe_transfer.npz',prediction=fixed,fold=fold,y_transformed=target)
                transfers[label][prop]=dict(control_alpha=reference['alpha'],fixed_control_probe_mean=float(np.mean(fixed_scores)),refit_hybrid_mean=float(np.mean(scores)))
                checks.append(dict(variant=label,property=prop,rows=len(fold),control_probe_reproduced=True))
            write(out/'results.json',results)
            print(label,{p:v['legacy_reference']['cv_mean_spearman'] for p,v in metrics['properties'].items()},flush=True)
    for path in [Path(__file__),ROOT/'downstream/grammar/ab_probes.py',ROOT/'data/downstream/dev/GDPa1_v1.2_20250814.csv']:pin(path)
    for path,digest in hashes.items():assert sha(path)==digest,path
    write(out/'geometry.json',geometry);write(out/'fixed_probe_transfer.json',transfers)
    write(out/'verification.json',dict(status='passed',source_sha256=hashes,actual_arm_sha256=identities,oof_checks=checks,
        head_null_and_completeness_verified=True,hybrid_mean_raw_logits_verified=True,
        limitations='Off-model cached-feature counterfactuals; cannot prove tokenwise/masked generation retention or separate TCR versus CDR loss causality. Single-seed exploratory same-CV Ridge. Full50-class head basis is conservative;24 classes are masked in generation.'))
    status.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',status)


if __name__=='__main__':main()
