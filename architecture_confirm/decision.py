"""Frozen paired three-seed decisions; run python -B -m architecture_confirm.tests."""
import numpy as np
from experiments.arch70m.statistics import align,statistic
from .common import ROOT,VARIANTS,SEEDS,JOBS,STEPS,read,write,digest,trial


def quality_guard(base,candidate):
    a,b=base['generation'],candidate['generation']
    length_ok=(a['conditional_length_mae'] is None and b['conditional_length_mae'] is None) or (
        a['conditional_length_mae'] is not None and b['conditional_length_mae'] is not None and
        b['conditional_length_mae']<=1.1*a['conditional_length_mae'])
    return bool(length_ok and b['conditional_validity']>=a['conditional_validity']-.02 and
                b['conditional_aar']>=a['conditional_aar'] and b['cdr_aar']>=a['cdr_aar'])


def supported(endpoint,kind,guard_count=0,auprc_gain=0.):
    consistent=all(x>0 for x in endpoint['per_seed_gain'])
    significant=endpoint['adjusted_ci'][0]>0
    if kind=='generation':return bool(consistent and significant and endpoint['mean_gain']>=.02 and guard_count>=2)
    return bool(consistent and significant and endpoint['mean_gain']>=.005 and auprc_gain>=0)


def calculate(observations,metrics,repetitions=5000):
    result=dict(status='complete',repetitions=repetitions,familywise_alpha=.05,comparisons=6,
        interval='99.1667% two-sided paired hierarchical percentile interval; Bonferroni over 3 pairs x 2 endpoints',
        resampling='training seeds, then exact receptor clusters within each source; paired observations',pairs={},
        limitation='Three seeds; fixed small protected probes; exact receptor clustering is not homology-independent validation.')
    for pair_number,(candidate,control) in enumerate((('B0','A0'),('B1','A0'),('B1','B0'))):
        outcomes={}
        for kind_number,kind in enumerate(('generation','representation')):
            pairs=[]
            for seed in SEEDS:
                a=observations[control,seed][kind];b=observations[candidate,seed][kind]
                pairs.append((a,b,align(a,b,kind)))
            gains=[statistic(a,b,g,kind) for a,b,g in pairs]
            if any(x is None for x in gains):raise ValueError('Insufficient paired two-class observations.')
            rng=np.random.default_rng(1729+10*pair_number+kind_number);draws=[]
            for _ in range(repetitions):
                sample=[]
                for index in rng.integers(len(pairs),size=len(pairs)):
                    a,b,g=pairs[index]
                    for _attempt in range(100):
                        value=statistic(a,b,g,kind,rng)
                        if value is not None:break
                    if value is None:raise ValueError('Insufficient two-class bootstrap clusters.')
                    sample.append(value)
                draws.append(float(np.mean(sample)))
            tail=.05/(2*6);low,high=np.quantile(draws,[tail,1-tail])
            endpoint=dict(mean_gain=float(np.mean(gains)),sample_sd=float(np.std(gains,ddof=1)),
                per_seed_gain=gains,adjusted_ci=[float(low),float(high)],all_three_seeds_improve=all(x>0 for x in gains))
            if kind=='generation':
                guards=sum(quality_guard(metrics[control,s],metrics[candidate,s]) for s in SEEDS)
                endpoint.update(decoder_guard_passes=guards,stable_improvement=supported(endpoint,kind,guards))
            else:
                pr=float(np.mean([metrics[candidate,s]['representation']['macro_auprc']-metrics[control,s]['representation']['macro_auprc'] for s in SEEDS]))
                endpoint.update(mean_auprc_gain=pr,stable_improvement=supported(endpoint,kind,auprc_gain=pr))
            outcomes[kind]=endpoint
        result['pairs'][candidate+'_vs_'+control]=outcomes
    accepted={v:[k for k,x in result['pairs'][v+'_vs_A0'].items() if x['stable_improvement']] for v in ('B0','B1')}
    result['accepted_vs_original']={v:k for v,k in accepted.items() if k}
    result['decision']='stable_improvement_supported' if result['accepted_vs_original'] else 'not_yet_stable'
    result['next_action']=('Root review the final files/plots and report precisely which endpoint improves.' if result['accepted_vs_original'] else
        'Goal remains unfinished: inspect failure directions and preregister a new architecture batch; do not lower thresholds or claim success.')
    return result


def decision(out):
    pending=[f'{v}_s{s}' for v,s in JOBS if not (trial(out,v,s)/'status.json').exists() or read(trial(out,v,s)/'status.json')['status']!='completed']
    if pending:return dict(status='pending',remaining=pending,decision='not_assessed')
    observations={};metrics={}
    for v,s in JOBS:
        t=trial(out,v,s);observations[v,s]=read(t/f'metrics/confirm_{STEPS:06d}_observations.json')
        metrics[v,s]=read(t/f'metrics/confirm_{STEPS:06d}.json')
    signature=digest({str(k):v for k,v in observations.items()});path=out/'decision.json'
    if path.exists() and read(path).get('observations_sha256')==signature:return read(path)
    protocol=read(ROOT/'architecture_confirm/protocol.json')
    result=calculate(observations,metrics,protocol['decision']['bootstrap_repetitions']);result['observations_sha256']=signature
    write(path,result);return result
