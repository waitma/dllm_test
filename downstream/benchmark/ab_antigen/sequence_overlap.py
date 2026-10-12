"""Reproduce exact Ag/H/L/H3/pair overlap against the checkpoint training set.

Run: python -m downstream.benchmark.ab_antigen.sequence_overlap
Does not label rows clean merely because they lack an exact match.
"""
from collections import Counter
import json
from downstream.benchmark.ab_antigen.sequence_data import DATA,ROOT,digest,write_json
from downstream.benchmark.ab_antigen.sequence_eval import read_rows
from downstream.benchmark.ab_antigen.sequence_metrics import numbering


def main():
    folder=DATA/'audit/training';folder.mkdir(parents=True,exist_ok=True);allrows=[]
    for name in ['binding','spr','abbibench','chimera']:allrows.extend(read_rows(DATA/f'{name}.jsonl'))
    nd=DATA/'audit/cov_h3_numbering';nd.mkdir(exist_ok=True)
    nums=numbering([r['heavy'] for r in allrows if r['dataset'] in ['binding','spr']],nd)
    wt={v['fv']:v for v in json.loads((DATA/'audit/abbibench_wt_numbering_case_resolved/numbering.json').read_text()).values() if v and v['chain_type']=='H'}
    queries={k:{r[k]:digest(r[k]) for r in allrows} for k in ['antigen','heavy','light']}
    queries['h3']={};row_h3={};unknown=[];pairs={digest([r['heavy'],r['light']]) for r in allrows}
    for r in allrows:
        if r['dataset']=='chimera':num=r['numbering']['heavy']
        elif r['dataset'] in ['binding','spr']:num=(nums.get(digest(r['heavy'])) or {}).get('numbering')
        else:num=(wt.get(r['wt_heavy']) or {}).get('numbering')
        if not num or len(num)!=len(r['heavy']):unknown.append(r['id']);continue
        h3=''.join(r['heavy'][i] for i,(p,_,_) in enumerate(num) if 105<=p<=117)
        if not h3:unknown.append(r['id']);continue
        queries['h3'][h3]=digest(h3);row_h3[r['id']]=digest(h3)
    counts={k:Counter() for k in queries};paircounts=Counter();missing=0
    for p in sorted((ROOT/'data/prepared/immune_v6_binding_only/train').glob('*.jsonl')):
        if p.name.startswith(('ots-','tcr_','trait-')):continue
        with p.open() as f:
            for line in f:
                r=json.loads(line);roles=dict(zip(r['chain_roles'],r['chains']))
                vals={'antigen':roles.get('antigen'),'heavy':roles.get('antibody_heavy'),'light':roles.get('antibody_light'),'h3':r.get('identifiers',{}).get('heavy_h3_core')}
                if vals['heavy'] and not vals['h3']:missing+=1
                for k,seq in vals.items():
                    if seq in queries[k]:counts[k][queries[k][seq]]+=1
                key=digest([vals['heavy'],vals['light']])
                if key in pairs:paircounts[key]+=1
    status=[]
    for r in allrows:
        status.append(dict(id=r['id'],dataset=r['dataset'],**{k:bool(counts[k].get(digest(r[k]))) for k in ['antigen','heavy','light']},h3_overlap=bool(counts['h3'].get(row_h3[r['id']])) if r['id'] in row_h3 else None,pair_overlap=bool(paircounts.get(digest([r['heavy'],r['light']])))))
    write_json(folder/'exact_final.json',dict(exact_matches=counts,pair_matches=paircounts,unknown_benchmark_h3=unknown,training_missing_h3=missing,scope='actual_prepared_training_only; upstream corpus unknown',query_rows=len(allrows)))
    write_json(folder/'overlap_rows.json',status)
    print('exact_final',len(allrows),{k:len(v) for k,v in counts.items()},'pairs',len(paircounts),'unknown_h3',len(unknown))


if __name__=='__main__':main()
