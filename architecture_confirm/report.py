"""Confirmation curves and decisions; run python -B -m architecture_confirm.report."""
from pathlib import Path
from .common import ROOT,OUT,JOBS,STEPS,read,write,stamp,trial
from .decision import decision
from experiments.arch70m.report import records,flatten,export_csv,plot_curves


def report(out=OUT):
    out=Path(out);dest=ROOT/'docs/arch70m_results/confirmation_20261031';dest.mkdir(parents=True,exist_ok=True)
    trials={};rows=[];curves=[]
    for v,s in JOBS:
        t=trial(out,v,s);name=f'confirm_{v}_s{s}'
        status=read(t/'status.json') if (t/'status.json').exists() else dict(status='pending',step=0)
        train=records(t/'train.jsonl');evals=records(t/'eval.jsonl');trials[name]=dict(status=status,train=train,eval=evals)
        row=dict(variant=v,seed=s,status=status['status'],step=status.get('step',0),total_steps=STEPS,epochs=3,micro_batch=32,global_batch=512)
        if status['status']=='completed':
            row.update(flatten(read(t/f'metrics/confirm_{STEPS:06d}.json')))
            row.update({'ab_specificity.'+k:value for k,value in flatten(read(t/'ab_specificity/metrics.json')).items()})
        if train:row.update(training_gpu_hours=train[-1]['gpu_hours'],last_train_ce=train[-1]['loss'])
        if (t/'profile.json').exists():
            q=read(t/'profile.json')['candidates'][0];row.update(profile_allocated_gib=q['allocated_bytes']/2**30,profile_reserved_gib=q['reserved_bytes']/2**30)
        rows.append(row)
        for kind,values in (('train',train),('eval',evals)):
            curves.extend(dict(trial=name,curve=kind,**flatten(x)) for x in values)
    result=decision(out);write(dest/'decision.json',result)
    write(dest/'summary.json',dict(updated=stamp(),output_root=str(out),trials=rows,decision=result))
    export_csv(dest/'summary.csv',rows);export_csv(dest/'all_curves.csv',curves);plot_curves(trials,dest)
    done=sum(r['status']=='completed' for r in rows)
    lines=['# 三种子完整架构确认','',f'更新时间：{stamp()} UTC；完成{done}/9；decision={result["decision"]}。','',
        'A0原架构、B0=A5全局注意力、B1=全局＋可学习残差。配对seed42/43/44，各随机初始化3epoch/3828步；共同micro32/global512，原V8形式与固定10%数据。',
        '候选来自既有screen筛选；所有九组完成后才对confirm留出终点作配对统计。训练过程中只用screen曲线观察，不按confirm结果选checkpoint。原抗体特异性五折评估一并保留，属于辅助验证。',
        '主要门槛：三种子增益全同向，5000次seed→来源内受体cluster配对bootstrap，三对模型×两终点共六项比较使用Bonferroni(99.1667%双侧区间)，区间下界>0；生成平均相对CE增益≥2%并通过原解码质量guard≥2/3；表征平均AUROC增益≥0.005且平均AUPRC不回退。只宣告通过的方向，不声称全面提升。','',
        '| 变体 | seed | 状态 | step | confirm整链AAR | confirm CDR AAR | confirm AUROC | confirm AUPRC |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        def fmt(k):return f'{r[k]:.4f}' if r.get(k) is not None else '—'
        lines.append(f"| {r['variant']} | {r['seed']} | {r['status']} | {r['step']} | {fmt('generation.conditional_aar')} | {fmt('generation.cdr_aar')} | {fmt('representation.macro_auroc')} | {fmt('representation.macro_auprc')} |")
    lines+=['',f'![完整screen训练/验证曲线]({dest}/curves.png)','',f'[逐步原始CSV]({dest}/all_curves.csv) · [终点汇总]({dest}/summary.csv) · [冻结统计判定]({dest}/decision.json)','',
        '尚未完成全部配对不作稳定性结论；门槛不通过则目标仍未完成，保留负结果并继续设计下一轮，不降低阈值、不挑最好seed。三种子与小样本受体聚类限制保留；不是同源独立或普适下游能力证明。',
        '训练GPU小时不含评估和checkpoint保存；模型按实验最新/最佳生成/最佳表征及epoch末节点保存完整恢复状态。','']
    if result['status']=='complete':lines+=['## 终点统计','', '```json',__import__('json').dumps(result,indent=2,ensure_ascii=False),'```','']
    (dest/'REPORT.md').write_text('\n'.join(lines));return done,result


if __name__=='__main__':report()
