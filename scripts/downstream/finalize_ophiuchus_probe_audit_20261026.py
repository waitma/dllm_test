"""Publish completed official-weight/current-evaluator checks without changing scores.

After activating pllm then protenix_abtcr, run this absolute script path with
--check-specificity for an independent float64 head replay, or --publish after
all three tasks and audit_ophiuchus_probes.py --verify have passed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path('/project_c20250601/251105016/project/dllm_test')
RUN=ROOT/'output/downstream_generation/ophiuchus_probe_audit_20261026'
sys.path.insert(0,str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def check_specificity():
    import numpy as np
    import pandas as pd
    import torch
    from downstream.grammar.ab_probes import prepare_task
    torch.set_num_threads(8)
    df,_,_,folds=prepare_task('specificity')
    folder=RUN/'specificity'
    x=np.load(folder/'features.npy',mmap_mode='r')
    result=[]
    for k,(_,test) in enumerate(folds):
        file=folder/f'fold_{k}'/'selected_head.pt'
        saved=torch.load(file,map_location='cpu',weights_only=False)
        assert saved['selected_epoch']==100 and saved['selection_rule']=='fixed_last'
        state={key:value.numpy().astype(np.float64) for key,value in saved['state_dict'].items()}
        # Independent NumPy equations, without either production head class.
        hidden=np.tanh(x[test].astype(np.float64)@state['dense.weight'].T+state['dense.bias'])
        logits=hidden@state['out_proj.weight'].T+state['out_proj.bias']
        pred=pd.read_csv(folder/f'fold_{k}'/'predictions.csv')
        values=pred[[f'logit_{c}' for c in range(3)]].to_numpy()
        np.testing.assert_allclose(logits,values,atol=1e-4,rtol=3e-5)
        np.testing.assert_array_equal(logits.argmax(1),pred.y_pred.to_numpy())
        assert len(pred)==len(test)
        result.append(dict(fold=k,rows=len(test),max_abs_logit_error=float(np.abs(logits-values).max()),
                           argmax_disagreements=0,head_sha256=sha(file)))
    report=dict(status='passed',method='independent NumPy float64 equations from persisted heads',
                rows=len(df),folds=result,atol=1e-4,rtol=3e-5,checked_utc=datetime.now(timezone.utc).isoformat())
    write(RUN/'specificity/independent_head_replay.json',report)
    print(json.dumps(report,indent=2))


def insert_after_row(text,needle,row):
    assert row not in text, 'Already published'
    lines=text.splitlines()
    locations=[i for i,line in enumerate(lines) if needle in line]
    assert len(locations)==1,(needle,len(locations))
    lines.insert(locations[0]+1,row)
    return '\n'.join(lines)+'\n'


def publish():
    verification=read(RUN/'verification.json')
    assert verification['status']=='passed'
    assert read(RUN/'specificity/independent_head_replay.json')['status']=='passed'
    for task in ('specificity','gdp_a1','m396'):
        assert read(RUN/task/'status.json')['status']=='success'
        assert verification['tasks'][task]['metrics_sha256']==sha(RUN/task/'metrics.json')
        assert read(RUN/f'{task}_exit.json')['exit_code']==0 if task!='gdp_a1' else True
    spec=read(RUN/'specificity/metrics.json')
    gdp=read(RUN/'gdp_a1/metrics.json')
    m396=read(RUN/'m396/metrics.json')
    historic_spec=read(ROOT/'output/downstream_generation/ophiuchus_ab/specificity_esm_ep100/specificity_hd_flu_cov_metrics.json')
    historic_m396=read(ROOT/'output/downstream_generation/ophiuchus_ab/affinity_m396_metrics.json')
    row_results=[r for r in m396['splits'] if r['split_protocol']=='row_reference']
    grouped=[r for r in m396['splits'] if r['split_protocol']=='sequence_grouped']
    assert [r['train_fraction'] for r in row_results]==[.005,.01,.02,.05,.1,.2]
    assert [r['train_fraction'] for r in grouped]==[.005,.01,.02,.05,.1,.2]
    assert len(m396['splits'])==12
    numeric=read(RUN/'m396/float64_readout_diagnostic.json')
    assert numeric['status']=='complete' and numeric['metrics_sha256']==sha(RUN/'m396/metrics.json')
    historical_by_fraction={r['train_frac']:r for r in historic_m396['splits']}
    differences=[dict(train_fraction=r['train_fraction'],
                      current_minus_historical=r['spearman_mean']-historical_by_fraction[r['train_fraction']]['spearman_mean'])
                 for r in row_results]
    report=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
        specificity=dict(current=spec['mean'],paper=dict(accuracy=.6796,f1_macro=.6790,mcc=.5203),
                         historical=historic_spec['mean'],rows=4398,folds=5,epochs=100),
        gdp_a1={key:value['legacy_reference'] for key,value in gdp['properties'].items()},
        m396=dict(row_reference=row_results,sequence_grouped=grouped,
                  historical=historic_m396['splits'],historical_differences=differences,
                  float64_readout_diagnostic=str(RUN/'m396/float64_readout_diagnostic.json'),
                  float64_max_abs_score_difference=numeric['max_abs_mean_difference'],paper_exact_values=None),
        evidence={task:dict(metrics=str(RUN/task/'metrics.json'),sha256=sha(RUN/task/'metrics.json'))
                  for task in ('specificity','gdp_a1','m396')},
        scope=dict(fresh_official_checkpoint_features=True,shared_production_readouts_checked=True,
                   labels_splits_predictions_metrics_checked=True,ours_specific_feature_adapter_checked=False,
                   exact_paper_protocol_for_all_three=False),
        limitations=['Specificity: two long input rows use current H148/L126 window; fixed-last reporting differs from original source epoch averaging.',
                     'GDPa1: existing full-label transform and same-CV alpha selection retained.',
                     'm396: current fixed window and separate-chain pooling differ from original active input/global-pooling code; exact Figure4 values unavailable.',
                     'All actual m396 variant positions are inside the visible window, but constant-region context is truncated.'])
    write(RUN/'summary.json',report)
    spec_row=f"| [Ophiuchus-Ab 官方权重／当前eval 2026-10-26 L]({RUN}/specificity/metrics.json) | 100（固定末轮） | "+' | '.join(f"{spec['mean'][k]:.6f}" for k in ('accuracy','f1_macro','mcc'))+' |'
    m_row=f"| [Ophiuchus-Ab 官方权重／当前eval 2026-10-26 L]({RUN}/m396/metrics.json) | "+' | '.join(f"{r['spearman_mean']:.6f}" for r in row_results)+' |'
    group_row=f"| [Ophiuchus-Ab 官方权重／当前eval 2026-10-26 L]({RUN}/m396/metrics.json) | {grouped[-1]['spearman_mean']:.6f} |"
    for name,spec_anchor,m_anchor in (
        ('RESULTS.md','| Ophiuchus-Ab [R] | 历史独立 100 轮 |','| Ophiuchus-Ab [R] | 0.930065 |'),
        ('RESULTS_SUMMARY.md','| Ophiuchus-Ab 复跑 | 100 |','| Ophiuchus-Ab 复跑 | 0.930065 |')):
        path=ROOT/'downstream/benchmark'/name
        text=path.read_text()
        text=insert_after_row(text,spec_anchor,spec_row)
        # The summary's historical m396 row may use a longer descriptive label.
        if m_anchor not in text:
            candidates=[line for line in text.splitlines() if 'Ophiuchus' in line and '| 0.930065 |' in line]
            assert len(candidates)==1
            m_anchor=candidates[0]
        text=insert_after_row(text,m_anchor,m_row)
        if name=='RESULTS.md':
            text=text.replace('| Ophiuchus-Ab [R] | 0.930065 |','| Ophiuchus-Ab 历史本地适配 [L] | 0.930065 |',1)
        section='#### m396：按序列分组划分补充结果'
        start=text.index(section)
        tail=text[start:]
        first_ours=next(line for line in tail.splitlines() if line.startswith('| [Ours'))
        tail=tail.replace(first_ours,group_row+'\n'+first_ours,1)
        text=text[:start]+tail
        text=text.replace('本轮新m396评测尚未运行。',f'本轮新m396评测已完成并通过保存预测重评分，六档原划分及分组结果分别报告。[完整核验]({RUN}/verification.json)。')
        path.write_text(text)
    # Read back every added numerical row exactly; retain hashes for this publication.
    published={}
    for name in ('RESULTS.md','RESULTS_SUMMARY.md'):
        path=ROOT/'downstream/benchmark'/name;text=path.read_text()
        assert all(row in text for row in (spec_row,m_row,group_row))
        published[str(path)]=sha(path)
    write(RUN/'publication_verification.json',dict(status='passed',documents_sha256=published,
          source_metrics_sha256={k:v['sha256'] for k,v in report['evidence'].items()},
          rows_verified=[spec_row,m_row,group_row],updated_utc=report['completed_utc']))
    print(json.dumps(report,indent=2,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-specificity',action='store_true')
    group.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    if args.check_specificity:check_specificity()
    else:publish()


if __name__=='__main__':main()
