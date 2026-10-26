"""Audit the parent/adapted by native/no-MASK T3-broad comparison.

Activate pllm/protenix_abtcr; from repository root run:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/audit_tcr_attention_controls.py --out ABS_NEW_DIRECTORY
Requires all three new T3 runs completed. Recompute cosine scores and rank-based
AUROC for both newly cached control runs, then report all seven shot settings.
Original parent/native and first adapted/no-MASK runs have episode artifacts
but no retained feature matrices; their distance layer is not re-audited here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from scipy.stats import rankdata
    from scripts.downstream.run_tcr_native_repr import inputs, receptor_groups

    cross = ROOT/'output/diagnostics/final_layer_joint_cross_task_20261018'
    folders = dict(parent_native=ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/t3broad',
                   parent_no_mask=cross/'t3broad_parent_no_mask_keys',
                   joint_native=cross/'t3broad_joint_native',
                   joint_no_mask=cross/'t3broad_no_mask_keys')
    sources = {}

    def read(path):
        sources[str(path)] = sha(path)
        return json.loads(path.read_text())

    frames, paths, _ = inputs('t3broad')
    train,test = [receptor_groups(f) for f in frames]
    data = {str(p):sha(p) for p in paths}
    results = {}
    for name, folder in folders.items():
        assert read(folder/'status.json')['status']=='success'
        results[name] = read(folder/'fewshot.json')
        if name != 'parent_native':
            assert read(folder/'comparison_verification.json')['status']=='passed'
            assert read(folder/'data_identity.json') == data
    ref = results['parent_native']
    for name, result in results.items():
        assert result['n_train']==len(train) and result['n_test']==len(test)
        assert len(result['episodes'])==len(ref['episodes'])==775
        for a,b in zip(ref['episodes'],result['episodes']):
            assert {k:v for k,v in a.items() if k!='auroc'} == {k:v for k,v in b.items() if k!='auroc'}
    checks = []
    for name in ['parent_no_mask','joint_native']:
        folder=folders[name]
        matrices=[]
        for i,frame in enumerate([train,test]):
            saved=read(folder/f'tcr_inputs_{i}.json')
            assert saved==dict(cdr3b=frame.cdr3b.tolist(),cdr3a=frame.cdr3a.tolist())
            path=folder/f'tcr_features_{i}.npy';sources[str(path)]=sha(path)
            x=np.load(path).astype(np.float64)
            assert x.shape==(len(frame),768) and np.isfinite(x).all()
            assert (np.linalg.norm(x,axis=1)>1e-8).all()
            matrices.append(x/np.linalg.norm(x,axis=1,keepdims=True))
        distance=(1.-matrices[1]@matrices[0].T).astype(np.float32)
        max_error=0.
        for episode in results[name]['episodes']:
            labels=np.array([episode['epitope'] in p for p in test.peptide],dtype=bool)
            scores=-distance[:,episode['support_indices']].min(axis=1)
            ranks=rankdata(scores,method='average')
            positive=int(labels.sum());negative=len(labels)-positive
            assert positive>0 and negative>0
            auc=(float(ranks[labels].sum())-positive*(positive+1)/2)/(positive*negative)
            error=abs(auc-episode['auroc']);max_error=max(max_error,error)
            assert error<1e-12,(name,episode['k'],episode['epitope'],error)
        checks.append(dict(model=name,episodes=775,source_input_order_verified=True,
                           cosine_scores_and_rank_auroc_recomputed=True,max_absolute_auroc_error=max_error))
    rows=[]
    for shot in ref['shots']:
        values={name:result['auroc_by_shot'][str(shot)]['mean'] for name,result in results.items()}
        rows.append(dict(shot=shot,**values,
                         training_delta_native=values['joint_native']-values['parent_native'],
                         training_delta_no_mask=values['joint_no_mask']-values['parent_no_mask'],
                         mask_delta_parent=values['parent_no_mask']-values['parent_native'],
                         mask_delta_joint=values['joint_no_mask']-values['joint_native'],
                         interaction=(values['joint_no_mask']-values['parent_no_mask'])-(values['joint_native']-values['parent_native'])))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    sources[str(Path(__file__))]=sha(Path(__file__))
    write(out/'results.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
          scope='Fixed5000 CE1 weights by readout factorial control. Single training seed; local TCR task, no independent-seed significance claim.'))
    write(out/'verification.json',dict(status='passed',checks=checks,matched_episodes=775,
          distance_layer_not_reaudited=['parent_native','joint_no_mask'],source_sha256=sources))
    write(out/'status.json',dict(status='success',finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
