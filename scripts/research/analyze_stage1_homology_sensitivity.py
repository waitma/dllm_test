"""Post-hoc sensitivity to observed training near-homology, without inference.

Activate pllm, then use protenix_abtcr Python with this absolute script:
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/analyze_stage1_homology_sensitivity.py --root ABS_EVALUATION_ROOT --task structure
Use --task partner only after both partner arms have completed and been scored.
Both 90% and 95% thresholds are always reported. Original results are read only.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_stage1_frozen_gdp import sha,write
from score_stage1_structure_retrieval import measures
from score_stage1_partner_retrieval import grouped_ci


def exposure(root):
    folder=root/'shared_homology';receipt=json.loads((folder/'receipt.json').read_text())
    assert receipt['status']=='complete'
    mapping=[]
    for scope,source in receipt['sources'].items():
        assert sha(source['path'])==source['sha256']
        for row in json.loads(Path(source['path']).read_text()):
            mapping.append(dict(query='q'+str(len(mapping)),scope=scope,row=row['row']))
    frame=pd.DataFrame(mapping)
    for role in ['h','l']:
        path=folder/(role+'_alignments.tsv');assert sha(path)==receipt[role+'_alignments_sha256']
        raw=pd.read_csv(path,sep='\t',names=['query','target','identity','qcov','tcov','alnlen','qlen','tlen','evalue','bits'])
        best=raw.groupby('query').identity.max().rename(role+'_identity')
        frame=frame.merge(best,on='query',how='left',validate='one_to_one')
    assert frame[['h_identity','l_identity']].notna().all().all()
    return frame


def structure(root,out,identity):
    folder=root/'experimental_structure';plan=json.loads((folder/'geometry_plan.json').read_text())
    metrics=json.loads((folder/'metrics.json').read_text());assert metrics['status']=='complete'
    for name,digest in plan['files'].items():assert sha(folder/name)==digest
    records=json.loads((folder/'structures.json').read_text());pairs=pd.read_csv(folder/'geometry_pairs.csv')
    features={}
    for arm in ['mlm','pair_kd']:
        receipt=json.loads((folder/(arm+'_features_receipt.json')).read_text())
        assert sha(folder/(arm+'_features.npz'))==receipt['features_sha256']
        assert receipt['plan_sha256']==sha(folder/'geometry_plan.json')
        with np.load(folder/(arm+'_features.npz')) as data:
            assert np.array_equal(data['row_ids'],[r['instance'] for r in records])
            x=data['features'].astype(float)
        features[arm]=x/np.linalg.norm(x,axis=1,keepdims=True)
    selected=identity[identity.scope=='experimental_structure'];rows=[];coverage=[]
    for cutoff in [.90,.95]:
        keep=set(selected.loc[(selected.h_identity<cutoff)&(selected.l_identity<cutoff),'row'])
        sub=pairs[pairs['query'].isin(keep)&pairs.gallery.isin(keep)]
        counts=sub.groupby('query').size();queries=set(counts[counts>=4].index)
        sub=sub[sub['query'].isin(queries)].reset_index(drop=True)
        assert not set(sub['query'])&set(sub.gallery)
        assert set(sub['query'])<=set(plan['queries']) and set(sub.gallery)<=set(plan['gallery'])
        coverage.append(dict(cutoff=cutoff,low_exposure_records=len(keep),queries=len(queries),
                             gallery=sub.gallery.nunique(),pairs=len(sub),candidate_min=int(sub.groupby('query').size().min()),
                             candidate_median=float(sub.groupby('query').size().median())))
        values={'Fv_sequence':sub.fv_sequence_identity.to_numpy(),'H3_sequence':sub.h3_sequence_identity.to_numpy()}
        for arm,x in features.items():values[arm]=np.einsum('ij,ij->i',x[sub['query'].to_numpy()],x[sub.gallery.to_numpy()])
        for query,g in sub.groupby('query'):
            index=g.index.to_numpy()
            for endpoint in ['h3_rmsd','fv_rmsd']:
                distances=g[endpoint].to_numpy()
                for method,scores in values.items():
                    rows.append(dict(cutoff=cutoff,query=int(query),endpoint=endpoint,method=method,
                                     candidates=len(g),**measures(scores[index],distances)))
                rows.append(dict(cutoff=cutoff,query=int(query),endpoint=endpoint,method='random',candidates=len(g),
                                 spearman=0.,top1_rmsd=float(distances.mean()),recall5=min(5/len(g),1.)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_query.csv',index=False)
    summaries=[];differences=[];rng=np.random.default_rng(20261025)
    for (cutoff,endpoint),group in frame.groupby(['cutoff','endpoint']):
        pivot=group.pivot(index='query',columns='method',values='spearman')
        common=set(pivot.index[np.isfinite(pivot).all(axis=1)])
        for method,g in group.groupby('method'):
            summaries.append(dict(cutoff=cutoff,endpoint=endpoint,method=method,queries=len(g),
                spearman_queries=len(common),mean_spearman=float(g[g['query'].isin(common)].spearman.mean()),
                mean_top1_rmsd=float(g.top1_rmsd.mean()),mean_recall5=float(g.recall5.mean())))
        a=group[group.method=='mlm'].sort_values('query');b=group[group.method=='pair_kd'].sort_values('query')
        assert np.array_equal(a['query'],b['query'])
        for metric in ['spearman','top1_rmsd','recall5']:
            use=a['query'].isin(common).to_numpy() if metric=='spearman' else np.ones(len(a),dtype=bool)
            delta=b[metric].to_numpy()[use]-a[metric].to_numpy()[use]
            lo,hi=np.quantile(delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(1),[.025,.975])
            differences.append(dict(cutoff=cutoff,endpoint=endpoint,metric=metric,queries=len(delta),
                                    delta=float(delta.mean()),ci95_low=float(lo),ci95_high=float(hi)))
    return dict(coverage=coverage,summaries=summaries,paired_differences=differences,
                interpretation='Post-hoc query AND gallery filtering, original partition and labels retained; not a new blind benchmark')


def partner(root,out,identity):
    folder=root/'native_partner';metrics=json.loads((folder/'metrics.json').read_text());assert metrics['status']=='complete'
    path=folder/'retrieval_per_query.csv';assert sha(path)==metrics['outputs']['retrieval_per_query.csv']
    frame=pd.read_csv(path);selected=identity[identity.scope=='native_partner']
    summaries=[];deltas=[];coverage=[];rows=[];rng=np.random.default_rng(20261025)
    for cutoff in [.90,.95]:
        keep=set(selected.loc[(selected.h_identity<cutoff)&(selected.l_identity<cutoff),'row'])
        # Preserve the original candidate pool; this is query-stratified sensitivity.
        sub=frame[frame.query_row.isin(keep)].copy();sub['cutoff']=cutoff;rows.append(sub)
        coverage.append(dict(cutoff=cutoff,native_pairs=len(keep),retrieval_groups=sub.group.nunique()))
        for key,g in sub.groupby(['arm','target','region','score']):
            summaries.append(dict(zip(['arm','target','region','score'],key),cutoff=cutoff,queries=len(g),
                **{m:float(g[m].mean()) for m in ['recall1','recall5','mrr','random_recall1','random_recall5','random_mrr']}))
        for key,g in sub.groupby(['target','region','score']):
            a=g[g.arm=='mlm'].sort_values('group');b=g[g.arm=='pair_kd'].sort_values('group')
            assert np.array_equal(a.group,b.group)
            for metric in ['recall1','recall5','mrr']:
                delta=b[metric].to_numpy()-a[metric].to_numpy();ci,clusters=grouped_ci(delta,a.query_cluster.to_numpy(),rng)
                deltas.append(dict(zip(['target','region','score'],key),cutoff=cutoff,metric=metric,queries=len(a),
                    clusters=clusters,delta=float(delta.mean()),ci95_low=ci[0],ci95_high=ci[1]))
    pd.concat(rows).to_csv(out/'per_query.csv',index=False)
    return dict(coverage=coverage,summaries=summaries,paired_differences=deltas,
                interpretation='Post-hoc native-query stratification only; original candidates retain their exposure distribution')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',choices=['structure','partner'],required=True);args=p.parse_args()
    root=args.root;assert root.is_absolute()
    out=root/'homology_sensitivity'/args.task
    if out.exists():raise FileExistsError(out)
    scope='experimental_structure' if args.task=='structure' else 'native_partner'
    original=root/scope/'metrics.json';assert json.loads(original.read_text())['status']=='complete'
    identity=exposure(root);out.mkdir(parents=True)
    plan=dict(schema='stage1.posthoc.homology_sensitivity.v1',task=args.task,created_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(__file__),original_metrics=dict(path=str(original),sha256=sha(original)),
        homology_receipt_sha256=sha(root/'shared_homology/receipt.json'),thresholds=[.90,.95],
        criterion='Both chains have best observed local training alignment identity strictly below threshold',
        role='Post-hoc diagnostic; retain and report both thresholds; original results are not replaced',
        inference=False,training=False,
        scoring_sources={name:sha(Path(__file__).with_name(name)) for name in ['score_stage1_structure_retrieval.py','score_stage1_partner_retrieval.py']})
    write(out/'plan.json',plan)
    result=globals()[args.task](root,out,identity)
    pd.DataFrame(result['summaries']).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(result['paired_differences']).to_csv(out/'paired_differences.csv',index=False)
    write(out/'metrics.json',dict(status='complete',plan_sha256=sha(out/'plan.json'),**result,
        outputs={n:sha(out/n) for n in ['per_query.csv','summary.csv','paired_differences.csv']},
        limitations=['Heuristic search; no exhaustive absence guarantee and no teacher exposure exclusion',
                    'Filtering changes the antibody distribution and candidate difficulty',
                    'Single pretrained seed; fixed-model bootstrap is descriptive, with residual dependence',
                    'Post-hoc thresholds are not selected to optimize model differences']))
    assert sha(original)==plan['original_metrics']['sha256']
    print(json.dumps(dict(status='complete',task=args.task,coverage=result['coverage'],paired_differences=result['paired_differences'])),flush=True)


if __name__=='__main__':main()
