"""Score frozen representations against experimental structure distances.

Activate pllm, then invoke with protenix_abtcr Python:
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/score_stage1_structure_retrieval.py --output ABS_STRUCTURE_DIRECTORY
Requires completed geometry and both feature receipts. Produces per-query
metrics, all sequence baselines, paired bootstrap and large-font figures.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr,rankdata
from Bio.SVDSuperimposer import SVDSuperimposer
from evaluate_stage1_frozen_gdp import sha,write
from prepare_stage1_structure_geometry import arrays


def measures(scores,distances):
    best=np.flatnonzero(scores==scores.max())
    nearest=float(distances[best].mean())
    relevant=np.flatnonzero(distances==distances.min())
    recall=[]
    for index in relevant:
        above=np.count_nonzero(scores>scores[index]);ties=np.count_nonzero(scores==scores[index])
        recall.append(float(np.clip((5-above)/ties,0,1)))
    rho=float(spearmanr(scores,-distances).statistic) if np.ptp(scores)>0 and np.ptp(distances)>0 else float('nan')
    if np.isfinite(rho):
        alternate=np.corrcoef(rankdata(scores),rankdata(-distances))[0,1]
        np.testing.assert_allclose(rho,alternate,atol=1e-12)
    return dict(spearman=rho,top1_rmsd=nearest,recall5=float(np.mean(recall)),
                nearest_structure_count=len(relevant),top_score_ties=len(best))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;assert out.is_absolute()
    if (out/'metrics.json').exists():raise FileExistsError(out/'metrics.json')
    plan=json.loads((out/'geometry_plan.json').read_text());plan_sha=sha(out/'geometry_plan.json')
    for name,digest in plan['files'].items():assert sha(out/name)==digest
    records=json.loads((out/'structures.json').read_text());pairs=pd.read_csv(out/'geometry_pairs.csv')
    assert len(pairs)==plan['pairs'] and not pairs.duplicated(['query','gallery']).any()
    assert set(pairs['query'])==set(plan['queries']) and set(pairs.gallery)==set(plan['gallery'])
    assert not set(plan['queries'])&set(plan['gallery'])
    assert len({records[i]['cluster'] for i in plan['queries']+plan['gallery']})==len(plan['queries'])+len(plan['gallery'])
    for row in pairs.itertuples():
        a,b=records[row.query],records[row.gallery]
        assert a['pdb']!=b['pdb'] and a['cluster']!=b['cluster']
        assert len(a['h_regions']['CDR3'])==len(b['h_regions']['CDR3'])
        assert a['h_regions']['CDR3']!=b['h_regions']['CDR3']
        assert row.fv_sequence_identity<.9 and row.common_framework>=120 and row.common_fv>=160
    assert np.isfinite(pairs[['fv_rmsd','h3_rmsd','framework_rmsd']]).all().all()
    assert (pairs[['fv_rmsd','h3_rmsd','framework_rmsd']]>=0).all().all()
    # Independent library recomputation of actual stored geometry, not merely hashes.
    max_error=0.
    for row in pairs.iloc[np.linspace(0,len(pairs)-1,20,dtype=int)].itertuples():
        a,b=arrays(records[row.query]),arrays(records[row.gallery])
        framework=sorted(a['framework']&b['framework']);common=sorted(a['index'].keys()&b['index'].keys())
        x=np.array([a['xyz'][a['index'][k]] for k in framework]);y=np.array([b['xyz'][b['index'][k]] for k in framework])
        fitter=SVDSuperimposer();fitter.set(y,x);fitter.run();rotation,translation=fitter.get_rotran()
        fvx=np.array([a['xyz'][a['index'][k]] for k in common]);fvy=np.array([b['xyz'][b['index'][k]] for k in common])
        for value,xx,yy in [(row.fv_rmsd,fvx,fvy),(row.h3_rmsd,a['h3'],b['h3'])]:
            actual=float(np.sqrt(np.mean(np.sum((xx@rotation+translation-yy)**2,axis=1))))
            max_error=max(max_error,abs(actual-value));np.testing.assert_allclose(actual,value,atol=1e-8)
    scores={'Fv_sequence':pairs.fv_sequence_identity.to_numpy(),'H3_sequence':pairs.h3_sequence_identity.to_numpy()}
    feature_receipts={}
    for arm in plan['arms']:
        receipt=json.loads((out/(arm+'_features_receipt.json')).read_text())
        assert receipt['status']=='complete' and receipt['plan_sha256']==plan_sha and receipt['trainable_parameters']==0
        assert receipt['features_sha256']==sha(out/(arm+'_features.npz'))
        assert receipt['model_sha256']==plan['arms'][arm]['model_sha256']
        feature_receipts[arm]=sha(out/(arm+'_features_receipt.json'))
        with np.load(out/(arm+'_features.npz')) as data:
            x=data['features'].astype(np.float64)
            assert np.array_equal(data['row_ids'],[r['instance'] for r in records])
        norms=np.linalg.norm(x,axis=1,keepdims=True);assert np.isfinite(x).all() and (norms>0).all()
        x=x/norms
        scores[arm]=np.einsum('ij,ij->i',x[pairs['query'].to_numpy()],x[pairs.gallery.to_numpy()])
        assert np.isfinite(scores[arm]).all()
    rows=[]
    for q,sub in pairs.groupby('query',sort=True):
        idx=sub.index.to_numpy()
        for endpoint in ['h3_rmsd','fv_rmsd']:
            distances=sub[endpoint].to_numpy()
            for method,values in scores.items():
                rows.append(dict(query=int(q),instance=records[q]['instance'],method=method,endpoint=endpoint,
                    gallery_count=len(sub),h3_length=len(records[q]['h_regions']['CDR3']),**measures(values[idx],distances)))
            rows.append(dict(query=int(q),instance=records[q]['instance'],method='random',endpoint=endpoint,
                gallery_count=len(sub),h3_length=len(records[q]['h_regions']['CDR3']),spearman=0.,
                top1_rmsd=float(distances.mean()),recall5=min(5/len(sub),1.),nearest_structure_count=int((distances==distances.min()).sum()),
                top_score_ties=len(sub)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'structure_per_query.csv',index=False)
    summaries=[];deltas=[];rng=np.random.default_rng(20261025)
    for endpoint,sub in frame.groupby('endpoint'):
        pivot=sub.pivot(index='query',columns='method',values='spearman')
        common=pivot.index[np.isfinite(pivot).all(axis=1)].to_numpy()
        for method,g in sub.groupby('method'):
            summaries.append(dict(endpoint=endpoint,method=method,queries=len(g),spearman_common_queries=len(common),
                mean_spearman=float(g[g['query'].isin(common)].spearman.mean()),
                mean_top1_rmsd=float(g.top1_rmsd.mean()),median_top1_rmsd=float(g.top1_rmsd.median()),mean_recall5=float(g.recall5.mean())))
        a=sub[sub.method=='mlm'].sort_values('query');b=sub[sub.method=='pair_kd'].sort_values('query')
        assert np.array_equal(a['query'],b['query'])
        for metric in ['spearman','top1_rmsd','recall5']:
            mask=a['query'].isin(common).to_numpy() if metric=='spearman' else np.ones(len(a),dtype=bool)
            difference=b[metric].to_numpy()[mask]-a[metric].to_numpy()[mask]
            sampled=difference[rng.integers(len(difference),size=(2000,len(difference)))].mean(1)
            lo,hi=np.quantile(sampled,[.025,.975])
            deltas.append(dict(endpoint=endpoint,metric=metric,queries=len(difference),delta=float(difference.mean()),
                               ci95_low=float(lo),ci95_high=float(hi)))
    pd.DataFrame(summaries).to_csv(out/'structure_summary.csv',index=False)
    pd.DataFrame(deltas).to_csv(out/'structure_paired_differences.csv',index=False)
    write(out/'metrics.json',dict(status='complete',protocol=plan['schema'],plan_sha256=plan_sha,
        scorer_sha256=sha(__file__),queries=len(plan['queries']),gallery=len(plan['gallery']),pairs=len(pairs),
        summaries=summaries,paired_differences=deltas,feature_receipt_sha256=feature_receipts,
        outputs={n:sha(out/n) for n in ['structure_per_query.csv','structure_summary.csv','structure_paired_differences.csv']},
        verification=dict(all_pair_filters_checked=True,query_gallery_cluster_disjoint=True,
            independent_biopython_geometry_pairs=20,geometry_max_abs_error=max_error,rank_correlation_recomputed=True),
        limitations=plan['limitations']+['Only common finite queries used for Spearman across all methods; constant-score cases excluded explicitly',
            'Bootstrap over unique query antibody clusters with fixed gallery; shared-gallery dependence and training uncertainty not covered',
            'Confidence intervals descriptive; no multiple-comparison correction']))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':16,
                        'axes.titlesize':19,'axes.unicode_minus':False,'pdf.fonttype':42})
    methods=['random','Fv_sequence','H3_sequence','mlm','pair_kd']
    labels=['随机','Fv序列','H3序列','纯MLM','Pair+KD'];colors=['#cbd5e1','#64748b','#94a3b8','#2563eb','#ea580c']
    summary=pd.DataFrame(summaries);fig,axes=plt.subplots(2,3,figsize=(20,11))
    for row,endpoint in enumerate(['h3_rmsd','fv_rmsd']):
        s=summary[summary.endpoint==endpoint].set_index('method').loc[methods]
        for col,(metric,title) in enumerate([('mean_spearman','相似性 Spearman ↑'),('mean_top1_rmsd','Top1 实验 RMSD（Å）↓'),('mean_recall5','结构最近邻 Recall@5 ↑')]):
            ax=axes[row,col];ax.bar(range(5),s[metric],color=colors)
            ax.set_xticks(range(5),labels,rotation=25);ax.set_title(('CDRH3' if row==0 else 'Fv')+'：'+title)
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('冻结序列表征能否检索实验结构近邻？',fontsize=27)
    fig.text(.5,.01,'相同H3长度、相同候选集合；H/L框架联合对齐。无结构预测、无下游参数拟合。',ha='center',fontsize=17)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(out/'experimental_structure_retrieval_zh.png',dpi=150)
    fig.savefig(out/'experimental_structure_retrieval_zh.pdf');plt.close(fig)
    print(json.dumps(dict(status='complete',queries=len(plan['queries']),summaries=summaries,paired_differences=deltas)),flush=True)


if __name__=='__main__':main()
