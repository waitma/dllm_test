"""Create scalar comparison artifacts from completed native teacher receipts.

Activate pllm and run this absolute file after validation. Partial or failed
models remain visible; engineering cases are never called a held-out benchmark.
"""
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from run_general_teacher_validation import quality

ROOT=Path('/project_c20250601/251105016/project/dllm_test/output/general_teacher_validation_20261024')


def chain_symmetry(example, scores):
    """Report reference-assisted AP over exactly sequence-identical chains.

    This is an evaluation-only oracle, never a reordering of training targets.
    All permitted permutations are retained; one common order is chosen by AP19.
    """
    if 'reference_coordinates' not in example or len(example['sequences'])<2:
        return None
    sequences=example['sequences']
    spans=np.split(np.arange(sum(example['chain_lengths'])),np.cumsum(example['chain_lengths'])[:-1])
    rows=[]
    for permutation in itertools.permutations(range(len(sequences))):
        if any(sequences[i]!=sequences[j] for i,j in enumerate(permutation)):
            continue
        order=np.concatenate([spans[i] for i in permutation])
        reordered={int(k):x[np.ix_(order,order)] for k,x in scores.items()}
        inter=quality(example,reordered).get('inter',{})
        ap=inter.get('ap',{})
        if ap.get('19') is not None:
            rows.append(dict(predicted_chain_order=list(permutation),inter_ap=ap))
    if not rows:return None
    best=max(rows,key=lambda x:x['inter_ap']['19'])
    return dict(permutations=rows,best_by_ap19=best,
        note='Reference-assisted evaluation; exact sequence identity only; zero-based predicted chain order for reference axes; not used for teacher inputs or stored targets.')


def main():
    models={}
    cases=[]
    symmetry={}
    validated=0
    examples={x['id']:x for x in json.loads((ROOT/'examples.json').read_text())['examples']}
    for model in ['esmfold2','protenix','boltz2']:
        path=ROOT/model/'summary.json'
        if not path.exists():
            models[model]=dict(status='not_started');continue
        d=json.loads(path.read_text())
        for row in d['results']:
            scores_path=ROOT/model/(row['id']+'_short_distance_scores.npz')
            with np.load(scores_path) as a:
                for k in ['18','19','20']:
                    x=a[k]
                    assert x.shape==(row['length'],row['length']) and np.isfinite(x).all()
                    assert x.min()>=0 and x.max()<=1.000001
                assert np.all(a['18']<=a['19']+1e-6) and np.all(a['19']<=a['20']+1e-6)
                permutations=chain_symmetry(examples[row['id']],{k:a[k] for k in a.files})
            if permutations:
                symmetry.setdefault(model,{})[row['id']]=permutations
            validated+=1
            q=row.get('quality') or {}
            cases.append(dict(model=model,id=row['id'],length=row['length'],
                chains=len(row['chain_lengths']),median_seconds=row['median_forward_seconds'],
                preprocessing_seconds=row['preprocessing_seconds'],allocated_gib=row['peak_allocated_gib'],
                intra_ap19=q.get('intra_long_range',{}).get('ap',{}).get('19'),
                inter_ap19=q.get('inter',{}).get('ap',{}).get('19'),
                inter_ap19_equivalent_chains=permutations['best_by_ap19']['inter_ap']['19'] if permutations else None,
                permitted_chain_permutations=len(permutations['permutations']) if permutations else None,
                scores_sha256=hashlib.sha256(scores_path.read_bytes()).hexdigest()))
        fab=[x for x in cases if x['model']==model and x['id'] in ['pdb_1jhl','pdb_1mlc','pdb_1vfb']]
        paired=[x for x in cases if x['model']==model and x['id'] in ['valid_01','pdb_1jhl','pdb_1mlc','pdb_1vfb']]
        models[model]=dict(status=d['status'],completed=len(d['results']),error=d.get('error'),
            gpu=d.get('gpu'),torch=d.get('torch_version'),cuda=d.get('cuda_version'),contract=d.get('contract'),
            mean_paired_forward_seconds=float(np.mean([x['median_seconds'] for x in paired])) if len(paired)==4 else None,
            three_fab_inter_ap19=float(np.mean([x['inter_ap19'] for x in fab])) if len(fab)==3 else None,
            receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    summary=dict(captured_utc=datetime.now(timezone.utc).isoformat(),models=models,
        artifacts_checked=validated,cases=cases,
        limits=['Eight engineering cases, not held-out generalization.',
                'Batch-one latency includes host-to-device transfer and online PLM where required; not optimized throughput.',
                'Boltz CPU preprocessing can reuse previously parsed records, so its prep+forward is not cold end-to-end latency.',
                'Native bins differ; AP19 is a short-distance score, with 18/20-bin sensitivity retained in raw receipts.',
                'Equivalent-chain AP is reference-assisted and permits only identical sequences; raw chain-ordered AP is retained.',
                'Native environments, precision policies and portable torch kernels differ; compare configured pipelines.'])
    (ROOT/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (ROOT/'equivalent_chain_evaluation.json').write_text(json.dumps(symmetry,indent=2,allow_nan=False)+'\n')
    if cases:
        with (ROOT/'comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    print(json.dumps(dict(models=models,artifacts_checked=validated),ensure_ascii=False))


if __name__=='__main__':main()
