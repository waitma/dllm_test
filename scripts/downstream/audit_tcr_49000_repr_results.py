"""Audit 49000 T2/T3 saved artifacts without rerunning a backbone or clustering.

Activate pllm then protenix_abtcr. Example (CPU only):
CUDA_VISIBLE_DEVICES='' python scripts/downstream/audit_tcr_49000_repr_results.py \
  --tasks t2a t2b t3broad --report /abs/new_report.json
Checks identity, coverage, saved sweeps/episodes and aggregates; it cannot
recompute individual clustering/NN scores without their unsaved embeddings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.downstream import run_tcr_native_repr as rep
from scripts.downstream.run_tcr_local_short import now, sha


def close(a,b,tol=1e-12):
    np.testing.assert_allclose(a,b,rtol=0,atol=tol)


def audit(task):
    output = rep.BENCH / 'outputs' / rep.DATASETS[task][0] / ('ours_'+rep.TAG+'_runtime_v2')
    manifest = json.loads((output/'run_manifest.json').read_text())
    assert manifest['status'] == 'success' and manifest['task'] == task
    assert manifest['identity'] == rep.identity(task)
    assert manifest['identity']['global_step'] == 49000
    report = {'status':'artifact_consistency_passed', 'output':str(output), 'checks':[]}
    if task in ('t2a','t2b'):
        result = json.loads((output/'metrics.json').read_text())
        curve = pd.read_csv(output/'curve.csv', float_precision='round_trip')
        numeric = curve.select_dtypes('number')
        assert np.isfinite(numeric.to_numpy()).all()
        assert result['n_units'] == (4779 if task=='t2a' else 9033)
        if task=='t2a':
            from downstream.benchmark.tcr_clustering.run import PAPER_RETENTIONS
            assert curve.tau.tolist() == result['taus'] and len(curve) == len(set(result['taus']))
            close(curve.retention,curve.n_clustered / 4779)
            assert curve.retention.between(0,1).all() and curve.purity.between(0,1).all()
            assert curve.retention.is_monotonic_decreasing
            sorted_curve = curve.sort_values('retention')
            close(np.trapezoid(sorted_curve.purity,sorted_curve.retention), result['auc_purity_retention'])
            assert len(result['aligned_points']) == len(PAPER_RETENTIONS) == 9
            assert sorted(x['target_retention'] for x in result['aligned_points'].values()) == sorted(PAPER_RETENTIONS.values())
            for point in result['aligned_points'].values():
                row = curve.iloc[(curve.retention-point['target_retention']).abs().argmin()]
                for field in ('tau','retention','purity'):
                    close(row[field], point[field])
                close(point['retention_gap'],abs(point['retention']-point['target_retention']))
                assert point['comparison_allowed'] == (point['retention_gap'] <= .02)
            report['checks'] += ['exact saved tau', '4779 denominator and monotonic retention',
                'purity-retention AUC integration', 'nine nearest-retention anchors and gap flags']
        else:
            assert curve.algo.eq('kmeans').all() and curve.k.tolist() == list(range(10,101,5)) == result['k_sweep']
            assert (curve.n_clusters == curve.k).all() and result['seed'] == 0
            for field in ('ari','nmi','purity'):
                close(round(curve[field].mean(),4), result[f'kmeans_{field}_mean'])
                close(round(curve[field].max(),4), result[f'kmeans_{field}_best'])
            report['checks'] += ['complete 19-point K sweep', '9033 denominator', 'mean and best recomputed from curve']
    else:
        result = json.loads((output/'fewshot.json').read_text())
        frames,_,_ = rep.inputs(task)
        if task=='t3broad':
            train,test = [rep.receptor_groups(f) for f in frames]
            assert result['n_train']==len(train)==6885 and result['n_test']==len(test)==1721
            assert result['shots']==[1,2,5,10,20,50,100] and result['seeds']==5
            seen = set()
            for episode in result['episodes']:
                k,ep,seed = episode['k'],episode['epitope'],episode['seed']
                key = (k,ep,seed)
                assert key not in seen and 0 <= seed < 5 and k in result['shots']
                seen.add(key)
                pool = np.array([i for i,labels in enumerate(train.peptide) if ep in labels])
                expected = np.random.default_rng(seed+1000*k).choice(pool,k,replace=False)
                assert episode['support_indices']==expected.tolist()
                assert 0 <= episode['auroc'] <= 1
            expected_count = 0
            for k in result['shots']:
                per_ep = result['per_epitope_by_shot'][str(k)]
                for ep,value in per_ep.items():
                    rows = [e for e in result['episodes'] if e['k']==k and e['epitope']==ep]
                    assert len(rows)==5
                    close(np.mean([e['auroc'] for e in rows]),value)
                expected_count += 5*len(per_ep)
                assert result['n_epitopes_by_shot'][str(k)]==len(per_ep)
                close(np.mean(list(per_ep.values())),result['auroc_by_shot'][str(k)]['mean'])
                close(np.std(list(per_ep.values())),result['auroc_by_shot'][str(k)]['std'])
            assert expected_count==len(result['episodes'])
            report['episodes_checked']=expected_count
            report['checks'] += ['unique multilabel receptor counts', 'all saved support identities and seeds',
                'per-epitope episode aggregates and macro mean/std']
        else:
            assert result['n_universe']==sum(map(len,frames))==25816
            assert result['columns']==result['input_fields']==['cdr3b','cdr3a']
            assert result['shots']==[1,2,5,10,20,50,100,200] and result['seeds']==100
            assert result['n_background']==21875 and sum(result['per_target_binders'].values())==3941
            for k in result['shots']:
                entries = result['per_epitope_by_shot'][str(k)]
                assert len(entries)==6
                assert all(e['n_seeds']==100 and 0 <= e['mean'] <= 1 and np.isfinite(e['std']) for e in entries.values())
                values = [e['mean'] for e in entries.values()]
                close(np.mean(values),result['auroc_by_shot'][str(k)]['mean'])
                close(np.std(values),result['auroc_by_shot'][str(k)]['std'])
            report['checks'] += ['25816 universe and six target counts', '8 shot sizes x 100 seeds x 6 targets',
                'macro mean/std from per-target statistics']
    report['artifact_sha256']={str(p):sha(p) for p in output.iterdir() if p.suffix in ('.json','.csv')}
    report['checks'] += ['current checkpoint/data/protocol/source manifest identity']
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',nargs='+',choices=['t2a','t2b','t3broad','t3deep'],required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():
        raise FileExistsError('choose a new report path')
    result={'status':'running','start_utc':now(),'global_step':49000,'auditor_sha256':sha(__file__),'tasks':{}}
    try:
        for task in args.tasks:
            result['tasks'][task]=audit(task)
            print('PASS',task,flush=True)
        result['status']='artifact_consistency_passed'
    except Exception as exc:
        result.update(status='failed',error=repr(exc))
        raise
    finally:
        result['end_utc']=now()
        result['limits']=['no new model forward or reclustering/NN distance recomputation',
            'saved aggregate checks do not establish pretraining decontamination or paper-exact parity']
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open('x') as handle:
            json.dump(result,handle,indent=2)


if __name__=='__main__':
    main()
