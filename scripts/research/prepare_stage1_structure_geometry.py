"""Freeze experimental structure query/gallery pairs and RMSD labels.

Activate pllm, then run this absolute file with --output ABS_STRUCTURE_DIRECTORY
after prepare_stage1_structure_retrieval.py finishes. Only NumPy is required.
The selection uses no model predictions. Existing outputs are never overwritten.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from evaluate_stage1_frozen_gdp import ROOT,sha,write


def arrays(r):
    keys=r['h_keys']+r['l_keys'];xyz=np.array(r['h_ca']+r['l_ca'],dtype=float)
    return dict(xyz=xyz,index={k:i for i,k in enumerate(keys)},
                aa=dict(zip(keys,r['h']+r['l'])),
                framework={k for k,f in zip(keys,r['h_framework']+r['l_framework']) if f},
                h3=np.array(r['h_ca'],dtype=float)[r['h_cdr3_indices']])


def geometry(a,b):
    framework=sorted(a['framework']&b['framework'])
    common=sorted(a['index'].keys()&b['index'].keys())
    if len(framework)<120 or len(common)<160:return None
    x=np.array([a['xyz'][a['index'][k]] for k in framework])
    y=np.array([b['xyz'][b['index'][k]] for k in framework])
    xc=x.mean(0);yc=y.mean(0)
    u,_,vt=np.linalg.svd((x-xc).T@(y-yc))
    correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
    rotation=u@correction@vt
    assert np.isclose(np.linalg.det(rotation),1.,atol=1e-8)
    transform=lambda v:(v-xc)@rotation+yc
    all_x=np.array([a['xyz'][a['index'][k]] for k in common])
    all_y=np.array([b['xyz'][b['index'][k]] for k in common])
    rmsd=lambda x,y:float(np.sqrt(np.mean(np.sum((x-y)**2,axis=1))))
    return dict(fv_rmsd=rmsd(transform(all_x),all_y),h3_rmsd=rmsd(transform(a['h3']),b['h3']),
                framework_rmsd=rmsd(transform(x),y),common_fv=len(common),common_framework=len(framework))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;assert out.is_absolute()
    if (out/'geometry_plan.json').exists():raise FileExistsError(out/'geometry_plan.json')
    inventory=json.loads((out/'inventory_plan.json').read_text())
    assert inventory['status']=='complete' and sha(out/'structures.json')==inventory['structures_sha256']
    records=json.loads((out/'structures.json').read_text())
    assert len({r['cluster'] for r in records})==len(records)
    by_length=defaultdict(list)
    for r in records:by_length[len(r['h_regions']['CDR3'])].append(r['row'])
    queries=[];gallery=[];excluded=[]
    for length,indices in sorted(by_length.items()):
        ordered=sorted(indices,key=lambda i:hashlib.sha256(('structure_split_v1:'+records[i]['instance']).encode()).hexdigest())
        if len(ordered)<8:
            excluded.extend(dict(row=i,reason='H3_length_group_under_8') for i in ordered);continue
        queries.extend(ordered[::2]);gallery.extend(ordered[1::2])
    assert not set(queries)&set(gallery)
    prepared={r['row']:arrays(r) for r in records};pairs=[];rejections=Counter();checks=0
    rng=np.random.default_rng(20261025)
    for q in queries:
        r=records[q];h3=r['h_regions']['CDR3'];a=prepared[q]
        for g in gallery:
            s=records[g];other=s['h_regions']['CDR3']
            if len(other)!=len(h3):continue
            if r['pdb']==s['pdb']:rejections['same_PDB']+=1;continue
            if r['cluster']==s['cluster']:raise AssertionError('Cluster leak')
            if other==h3:rejections['identical_H3']+=1;continue
            b=prepared[g]
            identity=sum(a['aa'][key]==b['aa'][key] for key in a['aa'].keys()&b['aa'].keys())/max(len(a['aa']),len(b['aa']))
            if identity>=.9:rejections['Fv_identity_ge_90']+=1;continue
            geo=geometry(a,b)
            if geo is None:rejections['insufficient_common_positions']+=1;continue
            if checks<20:
                reverse=geometry(b,a)
                for name in ['fv_rmsd','h3_rmsd','framework_rmsd']:np.testing.assert_allclose(geo[name],reverse[name],atol=1e-8)
                rot,_=np.linalg.qr(rng.normal(size=(3,3)))
                if np.linalg.det(rot)<0:rot[:,0]*=-1
                moved=dict(a,xyz=a['xyz']@rot+17.,h3=a['h3']@rot+17.)
                invariant=geometry(moved,b)
                for name in ['fv_rmsd','h3_rmsd','framework_rmsd']:np.testing.assert_allclose(geo[name],invariant[name],atol=1e-8)
                checks+=1
            pairs.append(dict(query=q,gallery=g,h3_length=len(h3),fv_sequence_identity=identity,
                              h3_sequence_identity=sum(x==y for x,y in zip(h3,other))/len(h3),**geo))
    count=Counter(p['query'] for p in pairs)
    valid={q for q in queries if count[q]>=4}
    excluded.extend(dict(row=q,reason='fewer_than_4_eligible_gallery') for q in queries if q not in valid)
    pairs=[p for p in pairs if p['query'] in valid]
    assert pairs and checks==20
    # Exact exposure is audited against both actual arms' shared 500k records.
    gdp=json.loads((ROOT/'output/stage1_downstream_9770_20261025/gdp_fixed_ridge/plan.json').read_text())
    subset_path=ROOT/'output/stage1_ab500k_pilot_20261022/data/subset.json'
    assert sha(subset_path)==gdp['arms']['mlm']['provenance']['subset_manifest_sha256']
    subset=json.loads(subset_path.read_text());maps={k:defaultdict(list) for k in ['h','l','pair']}
    for r in records:
        for k,key in [('h',r['h']),('l',r['l']),('pair',(r['h'],r['l']))]:maps[k][key].append(r['row'])
    hits={k:Counter() for k in maps};digest=hashlib.sha256();train_count=0
    with Path(subset['requests']).open('rb',buffering=8<<20) as f:
        for raw in f:
            digest.update(raw);r=json.loads(raw)
            if r['split']!='train':continue
            train_count+=1;c={v['role']:v['sequence'] for v in r['request']['chains']}
            h,l=c['antibody_heavy'],c['antibody_light']
            for k,key in [('h',h),('l',l),('pair',(h,l))]:
                for i in maps[k].get(key,[]):hits[k][i]+=1
    assert train_count==500000 and digest.hexdigest()==subset['requests_sha256']
    pd.DataFrame(pairs).to_csv(out/'geometry_pairs.csv',index=False)
    write(out/'geometry_exclusions.json',excluded)
    write(out/'training_exact_exposure.json',dict(request_sha256=digest.hexdigest(),
        overlap={k:len(v) for k,v in hits.items()},rows=[dict(row=r['row'],**{k:int(hits[k][r['row']]) for k in hits}) for r in records],
        limitation='Exact sequence identity only; near homologs and teacher training exposure remain unknown'))
    plan=dict(schema='stage1.experimental_structure.retrieval.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        arms=gdp['arms'],inventory_plan_sha256=sha(out/'inventory_plan.json'),script_sha256=sha(__file__),
        files={n:sha(out/n) for n in ['structures.json','geometry_pairs.csv','geometry_exclusions.json','training_exact_exposure.json']},
        queries=sorted(valid),gallery=sorted({p['gallery'] for p in pairs}),pairs=len(pairs),
        pair_rejections=dict(rejections),split='Sort SHA256(structure_split_v1:instance) within same-H3-length groups; even=query, odd=gallery',
        distances='Rigid Kabsch fit on >=120 common IMGT H+L framework CA; Fv RMSD on >=160 common CA; H3 RMSD in sequence order',
        primary='Per-query cosine embedding ranking vs experimental H3 RMSD; mean query Spearman, nearest-neighbor RMSD and Recall@5 of true nearest structure',
        supporting=['Same metrics for Fv RMSD','IMGT Fv identity and H3 sequence identity baselines'],
        readout='Final hidden global real-residue mean, both full models, clean sequence input, relation hidden, float32 CPU',
        tie_policy='Expected metrics under uniform ordering within exactly tied representation or baseline scores; structural nearest ties are all relevant',
        validation=dict(symmetry_pairs=checks,rigid_motion_invariance_pairs=checks,query_gallery_disjoint=True,one_per_released_cluster=True),
        model_features_pending=True,limitations=['Residue CA geometry, not full-atom structural similarity',
            'Only equal-H3-length comparisons; canonical source split is not student pretraining split',
            'Single model seed; no exhaustive near-homology or teacher exposure exclusion'])
    write(out/'geometry_plan.json',plan)
    print(json.dumps(dict(status='complete',queries=len(valid),gallery=len(plan['gallery']),pairs=len(pairs),
                         geometry_checks=plan['validation'],exact_overlap={k:len(v) for k,v in hits.items()})),flush=True)


if __name__=='__main__':main()
