"""Activate pllm; python -B FILE --out ABS after both confirmation audits finish.

Aggregate audited outcomes, compare historical baseline tokens, and report
changed predictions and forward cost. This script does not select parameters.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out
    old=out.parent/'cdr_inference_rounds_20261015';report={'status':'passed','confirmation':{},'history_equivalence':{},'changes':{},'audit_sha256':{}}
    source=json.loads((out/'confirmation_source_audit.json').read_text());assert source['status']=='passed'
    for path,digest in source['source_sha256'].items():assert sha(path)==digest
    for cp,n in [('112000',3),('50000',2)]:
        folder=out/f'confirmation_{cp}';audit=json.loads((folder/'independent_audit.json').read_text());assert audit['status']=='passed'
        assert sha(folder/'predictions.jsonl')==audit['predictions_sha256']
        for path,digest in audit['posterior_sha256'].items():assert sha(path)==digest
        report['audit_sha256'][str(folder/'independent_audit.json')]=sha(folder/'independent_audit.json')
        report['confirmation'][cp]=audit['results'];cases=defaultdict(dict)
        for line in (folder/'predictions.jsonl').open():
            r=json.loads(line);cases[r['policy']][r['id']]=r
        checked=0;mismatches=[]
        for line in (old/f'round{n}/predictions.jsonl').open():
            r=json.loads(line);name={'original1':'single_pass','original4':'original4','original8':'original8'}.get(r['policy'])
            if name is None or r['id'] not in cases[name]:continue
            checked+=1
            if r['prediction_ids']!=cases[name][r['id']]['prediction_ids']:mismatches.append(dict(id=r['id'],policy=name))
        report['history_equivalence'][cp]=dict(checked=checked,mismatches=mismatches)
        changes=defaultdict(lambda:dict(sequences=0,changed_sequences=0,changed_residues=0,right_to_wrong=0,wrong_to_right=0))
        for key,r in cases['candidate'].items():
            base=cases['single_pass'][key];d=changes[r['dataset']+':'+r['mode']];d['sequences']+=1
            d['changed_sequences']+=r['prediction_ids']!=base['prediction_ids']
            for a,b,gold in zip(base['prediction_ids'],r['prediction_ids'],r['reference_ids']):
                d['changed_residues']+=a!=b;d['right_to_wrong']+=a==gold and b!=gold;d['wrong_to_right']+=a!=gold and b==gold
        report['changes'][cp]=dict(changes)
    primary=report['confirmation']['112000']['modern:cdrh3']['comparisons']['single_pass']
    secondary=report['confirmation']['50000']['modern:cdrh3']['comparisons']['single_pass']
    report['primary_confirmation_passed']=primary['ci95_pp'][0]>0
    report['cross_checkpoint_direction_positive']=secondary['delta_pp']>0
    report['preregistered_primary_and_cross_direction_checks_passed']=report['primary_confirmation_passed'] and report['cross_checkpoint_direction_positive']
    report['cross_checkpoint_ci_excludes_zero']=secondary['ci95_pp'][0]>0
    report['all_secondary_point_estimates_nonnegative']=all(
        r['comparisons']['single_pass']['delta_pp']>=0 for cp in report['confirmation'].values() for r in cp.values())
    report['interpretation']='A primary endpoint pass and positive replication direction do not establish useful or universal improvement; report cost, replication interval, and all regressions.'
    report['scope']='Frozen soft-context candidate only; no confirmation retuning. Historical checks do not replace reserved primary endpoint.'
    report['source_sha256']=sha(Path(__file__))
    (out/'confirmation_delivery.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('audit_sha256','changes')},indent=2),flush=True)


if __name__=='__main__':main()
