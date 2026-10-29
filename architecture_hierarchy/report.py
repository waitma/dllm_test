"""Export matched-micro curves; run python -B -m architecture_hierarchy.report."""
from pathlib import Path
from .common import ROOT,DEFAULT_OUT,PREVIOUS_OUT,BASE_OUT,VARIANTS,LABELS,MICRO,SEED,read,write,stamp
from experiments.arch70m.report import records,flatten,export_csv,plot_curves


def report(out=DEFAULT_OUT):
    out=Path(out);dest=ROOT/'docs/arch70m_results/hierarchy_m32';dest.mkdir(parents=True,exist_ok=True)
    trials={};rows=[];curves=[];endpoints={}
    for variant in VARIANTS:
        trial=out/'trials'/f'screen_{variant}_s{SEED}'
        status=read(trial/'status.json') if (trial/'status.json').exists() else dict(status='pending',step=0)
        train=records(trial/'train.jsonl');evals=records(trial/'eval.jsonl')
        name=f"{variant}_{'A5_control_' if variant=='B0' else ''}s43_m32"
        trials[name]=dict(status=status,train=train,eval=evals)
        row=dict(variant=variant,label=LABELS[variant],status=status['status'],step=status.get('step',0),
                 seed=SEED,micro_batch=MICRO,global_batch=512,epochs=1,path=str(trial))
        if status['status']=='completed':
            endpoints[variant]=flatten(read(trial/f"metrics/screen_{status['step']:06d}.json"))
            row.update(endpoints[variant]);row['parameters']=status.get('parameters')
        if train:
            row['training_gpu_hours']=train[-1]['gpu_hours']
            row['peak_allocated_gib']=max(x['allocated_bytes'] for x in train)/2**30
        if (trial/'profile_attempts.json').exists():
            p=read(trial/'profile_attempts.json')['candidates'][0]
            row.update(profile_safe=p['safe'],profile_allocated_gib=p['allocated_bytes']/2**30,
                       profile_reserved_gib=p['reserved_bytes']/2**30)
        rows.append(row)
        for kind,values in (('train',train),('eval',evals)):
            curves.extend(dict(trial=name,curve=kind,**flatten(x)) for x in values)
    differences=[]
    fields=('token.all.loss','generation.conditional_ce','generation.conditional_aar','generation.cdr_aar',
            'generation.conditional_length_mae','representation.macro_auroc','representation.macro_auprc')
    for candidate,control in (('B2','B0'),('B3','B2'),('B3','B1'),('B1','B0')):
        if candidate in endpoints and control in endpoints:
            for field in fields:
                differences.append(dict(candidate=candidate,control=control,metric=field,
                    candidate_value=endpoints[candidate][field],control_value=endpoints[control][field],
                    difference=endpoints[candidate][field]-endpoints[control][field],micro_batch=MICRO,seed=SEED))
    historical=[]
    for variant,folder in (('A0',BASE_OUT),('A5',BASE_OUT),('B1',PREVIOUS_OUT)):
        trial=folder/'trials'/f'screen_{variant}_s43';status=read(trial/'status.json')
        if status['status']=='completed':
            m=flatten(read(trial/'metrics/screen_001276.json'))
            historical.append(dict(variant=variant,seed=43,micro_batch=56,comparison='historical_not_matched_micro',
                                   **{k:m[k] for k in fields}))
    write(dest/'summary.json',dict(updated=stamp(),output_root=str(out),trials=rows,historical=historical))
    export_csv(dest/'summary.csv',rows);export_csv(dest/'all_curves.csv',curves)
    export_csv(dest/'comparisons.csv',differences);export_csv(dest/'historical_micro56.csv',historical)
    plot_curves(trials,dest)
    done=sum(r['status']=='completed' for r in rows)
    lines=['# 层次注意力探索：共同micro32','',f'更新时间：{stamp()} UTC；完成{done}/4。','',
        '统一seed43、1epoch/1276步、global512=16×micro32，原V8形式及653,288条固定10%子集。所有变体先通过micro32/maxlen2048的3次真实更新与原85%显存门槛，再由全新进程正式训练。',
        'B0是与A5架构相同的共同micro32对照；B1是全局＋可学习缩放；B2每层链内→全局→FFN2048；B3为B2＋可学习缩放。四组复用之前冻结的模型实现，不修改原目标/数据/评估。',
        '主要比较仅使用本轮micro32终点。旧micro56结果独立保留为历史参考，不与本轮混作严格配对；单种子不作显著性/结合特异性结论。B2/B3同时重分配attention和FFN预算。','',
        '| 变体 | 状态 | step | 整链AAR↑ | CDR AAR↑ | AUROC↑ | AUPRC↑ | profile allocated/reserved GiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        def fmt(k):return f'{row[k]:.4f}' if row.get(k) is not None else '—'
        memory=(f"{row['profile_allocated_gib']:.2f}/{row['profile_reserved_gib']:.2f}" if 'profile_allocated_gib' in row else '—')
        label=row['variant']+(' (A5 control)' if row['variant']=='B0' else '')
        lines.append(f"| {label} | {row['status']} | {row['step']} | {fmt('generation.conditional_aar')} | {fmt('generation.cdr_aar')} | {fmt('representation.macro_auroc')} | {fmt('representation.macro_auprc')} | {memory} |")
    lines += ['',f'![完整曲线]({dest}/curves.png)','',f'[逐步原始CSV]({dest}/all_curves.csv) · [同micro终点差值]({dest}/comparisons.csv) · [历史micro56参考]({dest}/historical_micro56.csv)','',
        f'原micro56失败及B1完成证据保持在 {PREVIOUS_OUT}。本轮输出 {out}。','']
    (dest/'REPORT.md').write_text('\n'.join(lines))
    return done


if __name__=='__main__':report()
