"""Activate pllm; python -B FILE --out ABS --checkpoint 112000|50000.

Increase only bootstrap Monte Carlo precision for the unchanged primary H3
estimand. Keep both the preregistered 4000 and fixed 100000 replicate intervals.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',required=True,choices=('112000','50000'));args=p.parse_args();folder=args.out/f'confirmation_{args.checkpoint}'
    audit=json.loads((folder/'independent_audit.json').read_text());assert audit['status']=='passed'
    cases=defaultdict(dict)
    for line in (folder/'predictions.jsonl').open():
        r=json.loads(line)
        if r['dataset']=='modern' and r['mode']=='cdrh3' and r['policy'] in ('single_pass','candidate'):
            cases[r['policy']][r['id']]=r
    assert set(cases['single_pass'])==set(cases['candidate']) and len(cases['single_pass'])==1156
    groups={g:i for i,g in enumerate(sorted({r['group'] for r in cases['single_pass'].values()}))}
    counts=np.zeros(len(groups));sums=np.zeros(len(groups))
    for key,r in cases['single_pass'].items():
        i=groups[r['group']];counts[i]+=1;sums[i]+=(cases['candidate'][key]['correct']-r['correct'])/r['length']*100
    rng=np.random.default_rng(20261015);boots=[]
    for _ in range(20):
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=5000)
        boots.append(weights@sums/(weights@counts))
    boot=np.concatenate(boots);initial=np.percentile(boot[:4000],[2.5,97.5])
    registered=audit['results']['modern:cdrh3']['comparisons']['single_pass']['ci95_pp']
    assert np.allclose(initial,registered,atol=1e-10)
    result=dict(status='passed',checkpoint=args.checkpoint,estimand='modern reserved H3 mean-sequence AAR delta vs single_pass',
        delta_pp=float(sums.sum()/counts.sum()),preregistered_replicates=4000,preregistered_ci95_pp=registered,
        precision_replicates=100000,precision_ci95_pp=np.percentile(boot,[2.5,97.5]).tolist(),
        bootstrap_mass_nonpositive=float((boot<=0).mean()),seed=20261015,
        reason='Primary interval lower bound is close to zero; increase simulation precision without changing method, cohort, or estimand.',
        caveat='Percentile cluster bootstrap uncertainty, not proof of universal benefit; nonpositive bootstrap mass is not a randomization-test p-value.')
    (folder/'bootstrap_precision.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
