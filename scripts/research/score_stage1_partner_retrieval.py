"""Score and verify completed frozen native-partner inference.

Activate pllm, then invoke this absolute script using protenix_abtcr Python:
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/score_stage1_partner_retrieval.py --output ABS_NATIVE_PARTNER_DIRECTORY
Requires both complete receipts; never selects a checkpoint, score, or mask
based on results. Produces all predefined readouts, paired cluster bootstrap,
native residue-recovery summaries, and large-font PNG/PDF plots.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_stage1_partner_retrieval import load, READOUTS
from evaluate_stage1_frozen_gdp import sha, write


def ranking(scores, positive):
    above=int(np.sum(scores>scores[positive])); tied=int(np.sum(scores==scores[positive]))
    ranks=np.arange(above+1,above+tied+1,dtype=float)
    result=dict(recall1=float(np.mean(ranks<=1)),recall5=float(np.mean(ranks<=5)),
                mrr=float(np.mean(1/ranks)),above=above,tied=tied)
    # Independently locate the score's block in a sorted list.
    ordered=np.sort(scores)[::-1]
    locations=np.flatnonzero(ordered==scores[positive])+1
    assert np.array_equal(locations,ranks)
    assert np.isclose(result['mrr'],sum(1./int(r) for r in locations)/len(locations))
    return result


def grouped_ci(values, labels, rng):
    groups=defaultdict(list)
    for value,label in zip(values,labels):groups[label].append(value)
    sums=np.array([np.sum(x) for x in groups.values()]);counts=np.array([len(x) for x in groups.values()])
    selected=rng.integers(len(sums),size=(2000,len(sums)))
    sampled=sums[selected].sum(1)/counts[selected].sum(1)
    return np.quantile(sampled,[.025,.975]).tolist(),len(groups)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;assert out.is_absolute()
    if (out/'metrics.json').exists():raise FileExistsError(out/'metrics.json')
    plan,records,items=load(out);plan_sha=sha(out/'plan.json')
    for arm in plan['arms']:
        receipt=json.loads((out/(arm+'_complete.json')).read_text())
        assert receipt['status']=='complete' and receipt['items']==len(items) and receipt['plan_sha256']==plan_sha
        assert receipt['scores_sha256']==sha(out/(arm+'_scores.jsonl'))
    grouped=defaultdict(list)
    for item in items:grouped[item['group']].append(item['item'])
    for group,indices in grouped.items():
        meta=plan['groups'][group];target=meta['target'];query='h' if target=='l' else 'l'
        native=records[meta['query_row']]
        observed={r[target] for r in records if r[query]==native[query]}
        sequences=[]
        for index in indices:
            item=items[index];candidate=records[item['target_row']]
            assert item['query_row']==native['row'] and item['target']==target
            assert records[item[query+'_row']][query]==native[query]
            assert records[item[target+'_row']][target]==candidate[target]
            assert candidate['species']==native['species']
            if target=='l':assert candidate['light_locus']==native['light_locus']
            if item['native']:assert item['target_row']==native['row']
            else:
                assert candidate[target] not in observed
                assert candidate[target+'_cluster']!=native[target+'_cluster']
            sequences.append(candidate[target])
        assert len(sequences)==len(set(sequences))==meta['candidate_count']
    retrieval=[];recovery=[];verified=0;arrs={}
    for arm in plan['arms']:
        values=np.full((len(items),2,len(READOUTS),3),np.nan);seen=set()
        with (out/(arm+'_scores.jsonl')).open() as f:
            for line in f:
                row=json.loads(line);i=row['item']
                assert i not in seen and row['plan_sha256']==plan_sha
                seen.add(i);values[i]=row['statistics']
        assert seen==set(range(len(items))) and np.isfinite(values).all()
        for item in items:
            i=item['item'];r=records[item['target_row']];names=r[item['target']+'_regions']
            expected=np.array([len(names),names.count('cdr1'),names.count('cdr2'),names.count('cdr3'),sum(x.startswith('fwr') for x in names)])
            assert (values[i,:, :,2]==expected[None,:]).all()
        assert (values[:,:,:,0]<=1e-8).all()
        assert (values[:,:,:,1]>=0).all() and (values[:,:,:,1]<=values[:,:,:,2]).all()
        assert (values[:,:,:,1]==np.rint(values[:,:,:,1])).all()
        np.testing.assert_allclose(values[:,:,0,:],values[:,:,1:,:].sum(2),atol=1e-8,rtol=1e-10)
        arrs[arm]=values
        for group,indices in grouped.items():
            meta=plan['groups'][group];query=records[meta['query_row']]
            positives=[j for j,i in enumerate(indices) if items[i]['native']]
            assert len(positives)==1 and len(indices)==meta['candidate_count']
            positive=positives[0];native=indices[positive]
            common=dict(arm=arm,group=group,target=meta['target'],query_row=query['row'],species=query['species'],
                query_cluster=query[('h' if meta['target']=='l' else 'l')+'_cluster'],candidate_count=len(indices))
            for reg,region in enumerate(READOUTS):
                for condition,label in enumerate(plan['conditions']):
                    v=values[native,condition,reg]
                    assert v[2]>0
                    recovery.append(dict(**common,region=region,condition=label,nll=-v[0]/v[2],accuracy=v[1]/v[2],residues=int(v[2])))
            if len(indices)<2:continue
            for region in ['all','cdr3']:
                reg=READOUTS.index(region)
                mean=values[indices,:,reg,0]/values[indices,:,reg,2]
                for label,scores in [('observed_partner',mean[:,0]),('masked_partner',mean[:,1]),('context_gain',mean[:,0]-mean[:,1])]:
                    rank=ranking(scores,positive)
                    chance=dict(random_recall1=1/len(indices),random_recall5=min(5/len(indices),1),
                                random_mrr=float(np.mean(1/np.arange(1,len(indices)+1))))
                    retrieval.append(dict(**common,region=region,score=label,**rank,**chance))
            verified+=len(indices)
    r=pd.DataFrame(retrieval);n=pd.DataFrame(recovery)
    r.to_csv(out/'retrieval_per_query.csv',index=False);n.to_csv(out/'native_recovery_per_query.csv',index=False)
    summaries=[];recoveries=[];deltas=[];rng=np.random.default_rng(20261025)
    for cohort in ['all_eligible','human_16']:
        selected=r if cohort=='all_eligible' else r[(r.species=='human')&(r.candidate_count==16)]
        for key,sub in selected.groupby(['arm','target','region','score']):
            summaries.append(dict(zip(['arm','target','region','score'],key),cohort=cohort,queries=len(sub),
                **{metric:float(sub[metric].mean()) for metric in ['recall1','recall5','mrr','random_recall1','random_recall5','random_mrr']}))
        for key,sub in selected.groupby(['target','region','score']):
            a=sub[sub.arm=='mlm'].sort_values('group');b=sub[sub.arm=='pair_kd'].sort_values('group')
            assert np.array_equal(a.group,b.group) and np.array_equal(a.query_cluster,b.query_cluster)
            for metric in ['recall1','recall5','mrr']:
                difference=b[metric].to_numpy()-a[metric].to_numpy()
                ci,clusters=grouped_ci(difference,a.query_cluster.to_numpy(),rng)
                deltas.append(dict(zip(['target','region','score'],key),cohort=cohort,metric=metric,queries=len(a),
                    clusters=clusters,delta=float(difference.mean()),ci95_low=ci[0],ci95_high=ci[1]))
        native=n if cohort=='all_eligible' else n[(n.species=='human')&(n.candidate_count==16)]
        # Native recovery includes the singleton group; label this difference explicitly.
        for key,sub in native.groupby(['arm','target','region','condition']):
            recoveries.append(dict(zip(['arm','target','region','condition'],key),cohort='all_native' if cohort=='all_eligible' else cohort,
                queries=len(sub),mean_nll=float(sub.nll.mean()),mean_accuracy=float(sub.accuracy.mean()),
                token_accuracy=float(np.average(sub.accuracy,weights=sub.residues))))
    pd.DataFrame(summaries).to_csv(out/'retrieval_summary.csv',index=False)
    pd.DataFrame(recoveries).to_csv(out/'recovery_summary.csv',index=False)
    pd.DataFrame(deltas).to_csv(out/'paired_differences.csv',index=False)
    # One candidate with no decoy remains in recovery, explicitly excluded from retrieval.
    write(out/'metrics.json',dict(status='complete',protocol=plan['schema'],plan_sha256=plan_sha,
        scorer_sha256=sha(__file__),candidate_count=len(items),query_groups=len(grouped),
        retrieval_groups=sum(len(v)>1 for v in grouped.values()),native_groups=len(grouped),
        summaries=summaries,recovery=recoveries,paired_differences=deltas,
        original_scores={arm:sha(out/(arm+'_scores.jsonl')) for arm in plan['arms']},
        outputs={name:sha(out/name) for name in ['retrieval_per_query.csv','native_recovery_per_query.csv','retrieval_summary.csv','recovery_summary.csv','paired_differences.csv']},
        verification=dict(all_candidates_complete=True,candidate_contract_verified=True,all_residue_counts=True,region_additivity=True,
            finite_log_probabilities=True,integer_correct_counts=True,rank_recomputed=True,paired_query_identity=True),
        limitations=plan['limitations']+['Cluster bootstrap uses query-chain CDR3 cluster; shared candidate dependencies remain',
            'Intervals are descriptive for fixed models, not multi-seed training uncertainty; multiple comparisons unadjusted']))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':16,
                        'axes.titlesize':19,'axes.unicode_minus':False,'pdf.fonttype':42})
    summary=pd.DataFrame(summaries);fig,axes=plt.subplots(2,3,figsize=(19,11))
    for row,target in enumerate(['l','h']):
        for col,score in enumerate(['observed_partner','masked_partner','context_gain']):
            ax=axes[row,col]
            sub=summary[(summary.cohort=='human_16')&(summary.target==target)&(summary.region=='all')&(summary.score==score)]
            for offset,arm,color,label in [(-.18,'mlm','#2563eb','纯 MLM'),(.18,'pair_kd','#ea580c','Pair + 链内 KD')]:
                s=sub[sub.arm==arm].iloc[0];ax.bar(np.arange(3)+offset,[s.recall1,s.recall5,s.mrr],width=.34,color=color,label=label)
            chance=sub.iloc[0]
            for x,metric in enumerate(['recall1','recall5','mrr']):ax.plot([x-.4,x+.4],[chance['random_'+metric]]*2,'k--',linewidth=1.5)
            names={'observed_partner':'天然伙伴条件评分','masked_partner':'遮蔽伙伴评分','context_gain':'伙伴上下文增益评分'}
            ax.set(title=('H → L' if target=='l' else 'L → H')+'：'+names[score],xticks=range(3),xticklabels=['Recall@1','Recall@5','MRR'],ylim=(0,1))
            ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.945),frameon=False)
    fig.suptitle('冻结模型天然伙伴检索：human / 每组16候选',fontsize=26)
    fig.text(.5,.015,'虚线为随机排名期望；八组mask覆盖全部目标残基。重配候选不是实验阴性。',ha='center',fontsize=16)
    fig.tight_layout(rect=(0,.055,1,.9));fig.savefig(out/'partner_retrieval_zh.png',dpi=150);fig.savefig(out/'partner_retrieval_zh.pdf');plt.close(fig)
    print(json.dumps(dict(status='complete',groups=len(grouped),candidates=len(items),summary_rows=len(summaries))))


if __name__=='__main__':main()
