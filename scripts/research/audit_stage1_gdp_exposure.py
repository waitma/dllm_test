"""Audit exact GDPa1 exposure to the actual scratch 500k training requests.

Activate pllm, then run this absolute file with --output ABS_GDP_EVALUATION_DIR.
This scan verifies the request manifest hash and does not establish absence of
near homologs or exposure in the structural teacher's original training data.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pandas as pd
from evaluate_stage1_frozen_gdp import ROOT, sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert args.output.is_absolute()
    destination=args.output/'exact_exposure.json'
    if destination.exists(): raise FileExistsError(destination)
    plan=json.loads((args.output/'plan.json').read_text())
    assert sha(plan['data'])==plan['data_sha256']
    df=pd.read_csv(args.output/'data_snapshot.csv')
    hmap=defaultdict(list);lmap=defaultdict(list);pairs=defaultdict(list)
    for i,row in df.iterrows():
        hmap[row.vh_protein_sequence].append(i);lmap[row.vl_protein_sequence].append(i)
        pairs[row.vh_protein_sequence,row.vl_protein_sequence].append(i)
    hits={kind:Counter() for kind in ['pair','heavy','light']}
    manifest=ROOT/'output/stage1_ab500k_pilot_20261022/data/subset.json'
    subset=json.loads(manifest.read_text())
    assert sha(manifest)==plan['arms']['mlm']['provenance']['subset_manifest_sha256']
    digest=hashlib.sha256();counts=Counter()
    try:
        import orjson
        loads=orjson.loads
    except ImportError:
        loads=json.loads
    with Path(subset['requests']).open('rb',buffering=8<<20) as f:
        for raw in f:
            digest.update(raw);row=loads(raw);counts[row['split']]+=1
            if row['split']!='train':continue
            chains={c['role']:c['sequence'] for c in row['request']['chains']}
            h,l=chains['antibody_heavy'],chains['antibody_light']
            for kind,key,mapping in [('pair',(h,l),pairs),('heavy',h,hmap),('light',l,lmap)]:
                for i in mapping.get(key,[]):hits[kind][i]+=1
    assert digest.hexdigest()==subset['requests_sha256']
    assert counts['train']==500000 and counts['valid']==4096
    rows=[dict(row=i,id=str(df.antibody_id.iloc[i]),
               **{kind:int(hits[kind][i]) for kind in hits}) for i in range(len(df))]
    result=dict(status='complete',scope='Exact sequence identity only; full selected scratch training set',
        request_sha256=digest.hexdigest(),subset_manifest_sha256=sha(manifest),counts=dict(counts),
        exact_overlap_samples={kind:len(hits[kind]) for kind in hits},rows=rows,
        near_homology='not established by this audit',teacher_training_exposure='unknown')
    write(destination,result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
