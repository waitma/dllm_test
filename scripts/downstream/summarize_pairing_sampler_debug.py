"""Activate pllm; python -B SCRIPT to audit completed sampler counterfactuals.

Incomplete runs remain pending. Bootstrap resamples the 500 heavy chains,
keeping their eight candidates together. It measures benchmark variation for
this fixed seed allocation, not uncertainty across training runs or seeds.
"""
import hashlib
import json
import re
from collections import Counter
from math import sqrt
from pathlib import Path

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
GEN = ROOT/'output/downstream_generation'
DEBUG = GEN/'pairing_sampler_debug_20261014'


def read(path):
    return json.loads(path.read_text())


def independent_diversity(sequences):
    """Recompute the published 3-mer cosine distance without its vectorizer."""
    sequences=[s.strip().lower() for s in sequences if s.strip()]
    if len(sequences)<2:return 0.
    # Match the scorer's declared word-token vocabulary, including unusual
    # generated strings, without imposing an additional alphabet filter.
    vectors=[Counter(re.findall(r'(?u)\b\w\w+\b',' '.join(s[i:i+3] for i in range(len(s)-2))))
             for s in sequences]
    norms=[sqrt(sum(v*v for v in c.values())) for c in vectors]
    distances=[]
    for i in range(len(vectors)):
        for j in range(i+1,len(vectors)):
            dot=sum(v*vectors[j].get(k,0) for k,v in vectors[i].items())
            cosine=dot/(norms[i]*norms[j]) if norms[i]*norms[j] else 0.
            distances.append(1-cosine)
    return sum(distances)/len(distances)


def main():
    import numpy as np
    import pandas as pd
    candidates = [
        ('fixed50000','legacy','fixed50000_legacy_dedicated'),
        ('fixed50000','gumbel_prob','fixed50000_gumbel_prob'),
        ('fixed112000','legacy','fixed112000_legacy'),
        ('v7','native','v7_native'),
    ]
    benchmark = pd.read_csv(ROOT/'data/oas_previous_clean/splits/holdout_select_500_eval_OAS.csv')
    report={'status':'pending','verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runs':{},'comparisons':{},'pending':[]}
    arrays={}
    for model, sampler, folder in candidates:
        run=DEBUG/folder
        if not (run/'status.json').exists() or read(run/'status.json')['status']!='success':
            report['pending'].append(folder);continue
        manifest=read(run/'manifest.json');metrics=read(run/'metrics.json')
        checks=read(run/'checks.json');scores=read(run/'immunomatch_scores.json')
        assert checks['status']=='passed' and checks['historical_gpu_sequence_exact']
        assert manifest['model']==model and manifest['sampler']==sampler and manifest['count']==4000
        assert manifest['protocol']==dict(prefix=0,reference_length=True,steps=16,temperature=1.,cfg=0.,
                   seed='42+8*original_row+variant',finish_all_reference_positions=True)
        frame=pd.read_csv(run/'generated.csv',keep_default_na=False)
        assert len(frame)==4000 and frame.row_index.tolist()==np.repeat(np.arange(500),8).tolist()
        assert frame.variant_idx.tolist()==list(range(8))*500
        for c,source in [('h_sequence','h_sequence'),('raw_l_sequence','l_sequence')]:
            assert frame[c].tolist()==benchmark[source].str.replace('-','',regex=False).repeat(8).tolist()
        predictions=[json.loads(s) for s in (run/'predictions.jsonl').read_text().splitlines()]
        assert frame.gen_l_sequence.tolist()==[r['gen_l_sequence'] for r in predictions]
        values=np.asarray(scores['gen_l_sequence'],dtype=float)
        assert values.shape==(4000,) and np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
        assert metrics['total_pairs']==4000 and abs(values.mean()-metrics['gen_immunomatch_mean'])<1e-10
        diversity=float(np.mean([independent_diversity(g.gen_l_sequence.tolist())
                                 for _,g in frame.groupby('h_sequence')]))
        assert abs(diversity-metrics['diversity_mean'])<1e-10
        details=metrics['detailed_results']
        assert len(details)==4000 and [r['index'] for r in details]==list(range(4000))
        assert abs(sum(bool(r['gen_valid']) for r in details)/4000-metrics['gen_valid_rate'])<1e-10
        assert abs(sum(bool(r['chain_match']) for r in details)/4000-metrics['overall_chain_match_rate'])<1e-10
        assert [r['gen_length'] for r in details]==frame.gen_l_sequence.str.len().tolist()
        report['runs'][folder]={k:metrics[k] for k in ['gen_immunomatch_mean','ref_immunomatch_mean',
            'diversity_mean','gen_valid_rate','overall_chain_match_rate']}
        report['runs'][folder]['independent_sequence_and_aggregation_checks']='passed'
        report['runs'][folder]['predictions_sha256']=hashlib.sha256((run/'predictions.jsonl').read_bytes()).hexdigest()
        arrays[folder]=values
        if model=='v7':
            baseline=read(GEN/'v7_f011_step50082_pairing_scores_local_gpu_20261006/immunomatch_scores.json')['gen_l_sequence']
        else:
            path=DEBUG/'baseline_cpu_scores'/f'{model}.json'
            if not path.exists():report['pending'].append(str(path));continue
            baseline_report=read(path)
            assert baseline_report['status']=='passed' and abs(baseline_report['mean_difference'])<1e-4
            baseline=baseline_report['gen_l_sequence']
        arrays[model+'_baseline']=np.asarray(baseline)
        difference=(values-np.asarray(baseline)).reshape(500,8).mean(1)
        rng=np.random.default_rng(20261014)
        boot=difference[rng.integers(0,500,size=(10000,500))].mean(1)
        report['comparisons'][folder]=dict(delta_im=float(difference.mean()),
            paired_heavy_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
            heavies_improved=int((difference>0).sum()),heavies=500,
            baseline_device='cuda' if model=='v7' else 'cpu',
            caveat='Single generation seed allocation; CPU baseline mean checked against saved GPU mean.')
    a,b='fixed50000_legacy_dedicated','fixed50000_gumbel_prob'
    if a in arrays and b in arrays:
        difference=(arrays[a]-arrays[b]).reshape(500,8).mean(1)
        rng=np.random.default_rng(20261014)
        boot=difference[rng.integers(0,500,size=(10000,500))].mean(1)
        report['comparisons']['fixed50000_confidence_rule_only']=dict(delta_im=float(difference.mean()),
            paired_heavy_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist())
    for label,a,b in [
        ('fixed50000_vs_v7_same_legacy','fixed50000_legacy_dedicated','v7_baseline'),
        ('fixed50000_vs_v7_same_native','fixed50000_baseline','v7_native'),
        ('fixed112000_vs_v7_same_legacy','fixed112000_legacy','v7_baseline'),
    ]:
        if a in arrays and b in arrays:
            difference=(arrays[a]-arrays[b]).reshape(500,8).mean(1)
            rng=np.random.default_rng(20261014)
            boot=difference[rng.integers(0,500,size=(10000,500))].mean(1)
            report['comparisons'][label]=dict(delta_im=float(difference.mean()),
                paired_heavy_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
                interpretation='Observed mean difference at matched sampler; not a formal equivalence test.')
    if not report['pending']:report['status']='audited_complete'
    (DEBUG/'comparison_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
