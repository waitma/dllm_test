"""Paired hierarchical confirmation intervals; called only by run.sh --stage report."""
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from .common import read, write, digest

def align(base,candidate,kind):
    fields=('source','key','kind','role') if kind=='generation' else ('source','key','label')
    if [tuple(x[k] for k in fields) for x in base]!=[tuple(x[k] for k in fields) for x in candidate]:
        raise ValueError('Confirmation observations are not paired')
    groups=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(base):
        group=(row['source'],row['kind']) if kind=='generation' else row['source']
        groups[group][row['key']].append(i)
    return [(g,list(clusters.values())) for g,clusters in sorted(groups.items())]

def statistic(base,candidate,groups,kind,rng=None):
    values=[]
    for _,clusters in groups:
        chosen=range(len(clusters)) if rng is None else rng.integers(len(clusters),size=len(clusters))
        ids=[i for k in chosen for i in clusters[k]]
        if kind=='generation':
            a=np.mean([base[i]['ce'] for i in ids]); b=np.mean([candidate[i]['ce'] for i in ids]); values.append((a,b))
        else:
            y=[base[i]['label'] for i in ids]
            if len(set(y))<2: return None
            a=roc_auc_score(y,[base[i]['probability'] for i in ids]); b=roc_auc_score(y,[candidate[i]['probability'] for i in ids]); values.append(b-a)
    if kind=='generation':
        # Equal weight per task, then per source, matching the selection rule.
        by_kind=defaultdict(list)
        for (g,_),value in zip(groups,values): by_kind[g[1]].append(value)
        means=[np.mean(x,axis=0) for x in by_kind.values()]
        return float(np.mean([(a-b)/max(a,1e-12) for a,b in means]))
    return float(np.mean(values))

def confirmation(root,repetitions=5000):
    choice=read(root/'selection.json'); observations={}; metrics={}
    for v in choice['confirm_variants']:
        for seed in choice['seeds']:
            trial=root/'trials'/f'confirm_{v}_s{seed}'
            if not (trial/'status.json').exists() or read(trial/'status.json')['status']!='completed':
                return dict(status='pending',reason='All nine confirmation trials must finish before inference')
            step=read(trial/'status.json')['step']
            observations[v,seed]=read(trial/'metrics'/f'confirm_{step:06d}_observations.json')
            metrics[v,seed]=read(trial/'metrics'/f'confirm_{step:06d}.json')
    signature=digest({str(k):v for k,v in observations.items()}); cache=root/'confirmation_statistics.json'
    if cache.exists() and read(cache).get('observations_sha256')==signature: return read(cache)
    result=dict(status='complete',observations_sha256=signature,repetitions=repetitions,
        familywise_alpha=.05,comparisons=4,interval='98.75% two-sided percentile; Bonferroni over two candidates × two endpoints',
        resampling='Training seeds then exact receptor clusters within each source, paired against A0',
        limitation='Three training seeds and small protected relation probes; not a homology-independent or universal downstream claim.',candidates={})
    for v in choice['confirm_variants'][1:]:
        endpoints={}
        for kind in ('generation','representation'):
            pairs=[]
            for seed in choice['seeds']:
                a=observations['A0',seed][kind]; b=observations[v,seed][kind]
                pairs.append((a,b,align(a,b,kind)))
            gains=[statistic(a,b,g,kind) for a,b,g in pairs]; draws=[]; rng=np.random.default_rng(1729)
            for _ in range(repetitions):
                values=[]
                for j in rng.integers(len(pairs),size=len(pairs)):
                    a,b,g=pairs[j]; value=None
                    for _attempt in range(100):
                        value=statistic(a,b,g,kind,rng)
                        if value is not None: break
                    if value is None: raise ValueError('Insufficient two-class clusters for bootstrap')
                    values.append(value)
                draws.append(float(np.mean(values)))
            low,high=np.quantile(draws,[.00625,.99375]); consistent=sum(x>0 for x in gains)>=2
            endpoints[kind]=dict(mean_gain=float(np.mean(gains)),sample_sd=float(np.std(gains,ddof=1)),
                per_seed_gain=gains,adjusted_ci=[float(low),float(high)],at_least_two_seeds_improve=consistent,
                statistically_supported=bool(low>0 and consistent))
        guards=[]
        for seed in choice['seeds']:
            a=metrics['A0',seed]['generation']; b=metrics[v,seed]['generation']
            length_ok=(a.get('conditional_length_mae') is None and b.get('conditional_length_mae') is None) or (
                a.get('conditional_length_mae') is not None and b.get('conditional_length_mae') is not None and
                b['conditional_length_mae']<=1.1*a['conditional_length_mae'])
            guards.append(b['conditional_validity']>=a['conditional_validity']-.02 and length_ok and
                          b['conditional_aar']>=a['conditional_aar'] and b['cdr_aar']>=a['cdr_aar'])
        endpoints['generation']['decoder_guard_passes']=sum(guards)
        endpoints['generation']['supported_generation_improvement']=bool(endpoints['generation']['statistically_supported'] and sum(guards)>=2)
        result['candidates'][v]=endpoints
    write(cache,result); return result
