"""Export local follow-up curves: python -B -m architecture_round2.report."""
from pathlib import Path
from .common import ROOT, BASE_OUT, DEFAULT_OUT, VARIANTS, read, write, stamp
from experiments.arch70m.report import records, flatten, export_csv, plot_curves


def report(out=DEFAULT_OUT):
    folder=ROOT/'docs/arch70m_results/round2'; folder.mkdir(parents=True,exist_ok=True)
    trials={}; summary=[]; curves=[]; endpoints={}
    entries=[(v,BASE_OUT/'trials'/f'screen_{v}_s43','reference') for v in ('A0','A4','A5')]
    entries += [(v,Path(out)/'trials'/f'screen_{v}_s43','new') for v in VARIANTS]
    for variant,path,kind in entries:
        status=read(path/'status.json') if (path/'status.json').exists() else dict(status='pending',step=0)
        train=records(path/'train.jsonl'); evals=records(path/'eval.jsonl')
        name=f'{kind}_{variant}_s43'; trials[name]=dict(status=status,train=train,eval=evals)
        row=dict(variant=variant,kind=kind,seed=43,status=status['status'],step=status.get('step',0),path=str(path))
        if status['status']=='completed':
            metrics=read(path/'metrics'/f"screen_{status['step']:06d}.json")
            endpoints[variant]=flatten(metrics); row.update(endpoints[variant])
            row['parameters']=status.get('parameters')
        if train:
            row['training_gpu_hours']=train[-1]['gpu_hours']; row['peak_allocated_gib']=max(x['allocated_bytes'] for x in train)/2**30
        summary.append(row)
        for curve,values in (('train',train),('eval',evals)):
            curves.extend(dict(trial=name,curve=curve,**flatten(v)) for v in values)
    differences=[]
    fields=('token.all.loss','generation.conditional_ce','generation.conditional_aar',
            'generation.conditional_validity','generation.conditional_length_mae','generation.cdr_aar',
            'representation.macro_auroc','representation.macro_auprc')
    for candidate,control in (('B1','A5'),('B1','A4'),('B2','A5'),('B3','B2'),('B3','B1')):
        if candidate in endpoints and control in endpoints:
            for field in fields:
                differences.append(dict(candidate=candidate,control=control,metric=field,
                    candidate_value=endpoints[candidate][field],control_value=endpoints[control][field],
                    difference=endpoints[candidate][field]-endpoints[control][field]))
    write(folder/'summary.json',dict(updated=stamp(),output_root=str(out),trials=summary))
    export_csv(folder/'summary.csv',summary); export_csv(folder/'all_curves.csv',curves)
    export_csv(folder/'comparisons.csv',differences); plot_curves(trials,folder)
    done=sum(x['kind']=='new' and x['status']=='completed' for x in summary)
    lines=['# 本地第二轮架构探索', '',f'更新时间：{stamp()}；新试验完成 {done}/3。','',
        '仅seed43；原V8形式与固定10%子集；global512、micro56、1epoch/1276步，冻结原MLM配方和评估代码。',
        'A0/A4/A5复用八卡侧同seed、同预算结果；B1全局＋可学习残差，B2每层链内→全局→FFN2048，B3为B2＋可学习残差。',
        'B2/B3重新分配attention/FFN参数预算，不能将差异只归于attention顺序；不复现PoET-2全套训练。',
        '单种子筛选结果只用于继续探索；整链AAR不是结合特异性证据。', '',
        '| 变体 | 类型 | 状态 | step | 整链AAR | CDR AAR | AUROC | AUPRC |','|---|---|---|---:|---:|---:|---:|---:|']
    for row in summary:
        def fmt(key):return f'{row[key]:.4f}' if row.get(key) is not None else '—'
        lines.append(f"| {row['variant']} | {row['kind']} | {row['status']} | {row['step']} | {fmt('generation.conditional_aar')} | {fmt('generation.cdr_aar')} | {fmt('representation.macro_auroc')} | {fmt('representation.macro_auprc')} |")
    lines += ['',f'![完整曲线]({folder}/curves.png)','',f'[原始CSV]({folder}/all_curves.csv) · [同seed对照差值]({folder}/comparisons.csv)','']
    (folder/'REPORT.md').write_text('\n'.join(lines))
    return done


if __name__=='__main__': report()
