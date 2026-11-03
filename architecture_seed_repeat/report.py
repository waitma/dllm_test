"""Two-seed curves and paired comparisons: python -B -m architecture_seed_repeat.report."""
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT, OLD, OUT, SEED, VARIANTS, LABELS, DATA_SHA, PREVIOUS_SHA, read, write, stamp, digest, source_binding, trial
from experiments.arch70m.report import records, flatten, export_csv, plot_curves, save_figure

FIELDS = ('generation.conditional_aar', 'generation.cdr_aar', 'generation.conditional_ce', 'generation.cdr_ce',
          'generation.conditional_length_mae', 'representation.macro_auroc', 'representation.macro_auprc')


def matched_binding(binding, variant, seed, signature):
    expected = dict(variant=variant, seed=seed, epochs=1, total_steps=1276, micro_batch=32,
                    source_sha256=signature, data_sha256=DATA_SHA)
    if any(binding.get(k) != v for k,v in expected.items()): raise ValueError('Unmatched two-seed endpoint binding.')
    if binding['settings'] != read(ROOT / 'experiments/arch70m/base.json'): raise ValueError('Training settings differ.')


def paired(endpoints):
    """Require the whole matrix before reporting paired effects or seed averages."""
    if set(endpoints) != {(v,s) for v in VARIANTS for s in (42,43)}: return [], []
    differences = []; means = []
    for seed in (42,43):
        for control in ('C0','A0'):
            for variant in VARIANTS:
                if variant == control: continue
                base, value = endpoints[control,seed], endpoints[variant,seed]
                for key in FIELDS:
                    differences.append(dict(seed=seed,variant=variant,control=control,metric=key,
                        value=value[key],baseline=base[key],difference=value[key]-base[key]))
    for variant in VARIANTS:
        for key in FIELDS:
            values = [endpoints[variant,s][key] for s in (42,43)]
            gains = [endpoints[variant,s][key]-endpoints['C0',s][key] for s in (42,43)]
            means.append(dict(variant=variant,metric=key,mean=statistics.mean(values),sample_sd=statistics.stdev(values),
                seed42=values[0],seed43=values[1],difference_from_C0_seed42=gains[0],difference_from_C0_seed43=gains[1],
                both_seeds_positive=all(x>0 for x in gains),inference='Two seeds only; descriptive replication, no stable significance claim.'))
    return differences, means


def report(out=OUT):
    out = Path(out); dest = ROOT / 'docs/arch70m_results/attention_seed42_20261103'; dest.mkdir(parents=True, exist_ok=True)
    rows = []; curves = []; trials = {}; endpoints = {}
    signatures = {42:digest(source_binding()),43:PREVIOUS_SHA}
    for seed, folder in ((42,out),(43,OLD)):
        for variant in VARIANTS:
            t = trial(folder,variant,seed)
            status = read(t/'status.json') if (t/'status.json').exists() else dict(status='pending',step=0)
            train, evals = records(t/'train.jsonl'), records(t/'eval.jsonl')
            trials[t.name] = dict(status=status,train=train,eval=evals)
            row = dict(variant=variant,seed=seed,status=status['status'],step=status.get('step',0),total_steps=1276,
                       micro_batch=32,global_batch=512,epochs=1)
            if status['status']=='completed':
                matched_binding(read(t/'binding.json'),variant,seed,signatures[seed])
                metrics = read(t/'metrics/screen_001276.json')
                if metrics['partition']!='screen' or metrics['step']!=1276: raise ValueError('Wrong evaluation endpoint.')
                endpoints[variant,seed] = flatten(metrics); row.update(endpoints[variant,seed])
            if train: row.update(training_gpu_hours=train[-1]['gpu_hours'],last_train_ce=train[-1]['loss'])
            rows.append(row)
            for kind, values in (('train',train),('eval',evals)):
                curves.extend(dict(trial=t.name,curve=kind,**flatten(x)) for x in values)
    differences, means = paired(endpoints)
    done = sum(r['seed']==42 and r['status']=='completed' for r in rows)
    old_done = sum(r['seed']==43 and r['status']=='completed' for r in rows)
    execution = read(out/'execution.json') if (out/'execution.json').exists() else dict(phase='not_started')
    write(dest/'summary.json',dict(updated=stamp(),output_root=str(out),seed42_completed=done,seed43_completed=old_done,
        execution=execution,trials=rows,stable_improvement_assessed=False))
    export_csv(dest/'summary.csv',rows);export_csv(dest/'all_curves.csv',curves)
    export_csv(dest/'paired_comparisons.csv',differences);export_csv(dest/'two_seed_summary.csv',means)
    plot_curves(trials,dest)
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    targets = ('generation.cdr_aar','representation.macro_auroc','representation.macro_auprc','generation.conditional_length_mae')
    for name, item in trials.items():
        for ax, key in zip(axes.flat,targets):
            points = [flatten(x) for x in item['eval'] if x['partition']=='screen']
            points = [p for p in points if p.get(key) is not None]
            if points: ax.plot([p['step'] for p in points],[p[key] for p in points],
                label=name,linestyle='--' if name.endswith('s43') else '-',linewidth=1)
    for ax, key in zip(axes.flat,targets):
        ax.set(title=key,xlabel='Optimizer step');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=6,ncol=2)
    fig.suptitle('Matched one-epoch screen: seed42 solid, seed43 dashed; no smoothing')
    save_figure(fig,dest,'target_curves');plt.close(fig)
    lines = ['# 相同五架构：仅新增seed42配对复核','',
        f'更新时间：{stamp()} UTC；seed42完成{done}/5，seed43完成{old_done}/5；队列状态{execution["phase"]}。','',
        '同一冻结模型/原V8形式/653,288条固定10%数据，均随机初始化1epoch/1276步、micro32/global512。只新增seed42；先等待原seed43队列完整结束，再做五组新GPU资格与训练。',
        '只用screen固定最终checkpoint；两种子用于复核方向是否重复，不等同于三种子稳定性确认或新的未使用留出。全部10组齐全后才生成配对差值和两种子均值/样本标准差，不挑最好seed。','',
        '| 模型 | seed | 状态 | step | 整链AAR | CDR AAR | AUROC | AUPRC |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        def fmt(k):return f'{r[k]:.4f}' if r.get(k) is not None else '—'
        lines.append(f"| {r['variant']} | {r['seed']} | {r['status']} | {r['step']} | {fmt('generation.conditional_aar')} | {fmt('generation.cdr_aar')} | {fmt('representation.macro_auroc')} | {fmt('representation.macro_auprc')} |")
    lines += ['',f'![完整loss/生成/表征曲线]({dest}/curves.png)',f'![重点任务双种子曲线]({dest}/target_curves.png)','',
        f'[原始逐步CSV]({dest}/all_curves.csv) · [同seed配对差值]({dest}/paired_comparisons.csv) · [两seed汇总]({dest}/two_seed_summary.csv)','']
    (dest/'REPORT.md').write_text('\n'.join(lines));return done


if __name__=='__main__':report()
