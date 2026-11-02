"""Full curves and factorial effects: python -B -m architecture_attention.report."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT, OUT, VARIANTS, LABELS, COUNTS, SEED, STEPS, trial, read, write, stamp
from experiments.arch70m.report import records, flatten, export_csv, plot_curves, save_figure


def report(out=OUT):
    out = Path(out); dest = ROOT / 'docs/arch70m_results/attention_20261102'; dest.mkdir(parents=True, exist_ok=True)
    trials = {}; rows = []; curves = []; endpoints = {}
    for v in VARIANTS:
        t = trial(out, v); status = read(t / 'status.json') if (t / 'status.json').exists() else dict(status='pending', step=0)
        train, evals = records(t / 'train.jsonl'), records(t / 'eval.jsonl')
        trials[t.name] = dict(status=status, train=train, eval=evals)
        row = dict(variant=v, label=LABELS[v], parameters=COUNTS[v], status=status['status'],
                   step=status.get('step', 0), total_steps=STEPS, seed=SEED, micro_batch=32, global_batch=512)
        if status['status'] == 'completed':
            endpoints[v] = flatten(read(t / 'metrics/screen_001276.json')); row.update(endpoints[v])
        if train: row.update(training_gpu_hours=train[-1]['gpu_hours'], last_train_ce=train[-1]['loss'])
        if (t / 'profile_attempts.json').exists():
            p = read(t / 'profile_attempts.json')['candidates'][0]
            row.update(profile_safe=p['safe'], profile_allocated_gib=p['allocated_bytes']/2**30, profile_reserved_gib=p['reserved_bytes']/2**30)
        rows.append(row)
        for kind, values in (('train', train), ('eval', evals)):
            curves.extend(dict(trial=t.name, curve=kind, **flatten(x)) for x in values)
    effects = []; fields = ('token.all.loss', 'generation.conditional_ce', 'generation.cdr_ce',
        'generation.conditional_aar', 'generation.cdr_aar', 'generation.conditional_length_mae',
        'representation.macro_auroc', 'representation.macro_auprc')
    if len(endpoints) == len(VARIANTS):
        for field in fields:
            for a, b in (('C1','C0'),('C2','C0'),('C3','C2'),('C3','C1')):
                effects.append(dict(metric=field, comparison=a+'-'+b, difference=endpoints[a][field]-endpoints[b][field]))
            effects.append(dict(metric=field, comparison='interaction C3-C2-C1+C0',
                difference=endpoints['C3'][field]-endpoints['C2'][field]-endpoints['C1'][field]+endpoints['C0'][field]))
            for v in ('C0','C1','C2','C3'):
                effects.append(dict(metric=field, comparison=v+'-A0', difference=endpoints[v][field]-endpoints['A0'][field]))
    done = sum(r['status']=='completed' for r in rows)
    write(dest / 'summary.json', dict(updated=stamp(), output_root=str(out), completed=done, trials=rows,
          stable_improvement_assessed=False, inference='Exploratory single-seed screen; no confirm data or stable claim.'))
    export_csv(dest / 'summary.csv', rows); export_csv(dest / 'all_curves.csv', curves); export_csv(dest / 'factorial_effects.csv', effects)
    plot_curves(trials, dest)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    targets = ('generation.cdr_aar', 'representation.macro_auroc', 'representation.macro_auprc', 'generation.conditional_length_mae')
    for name, values in trials.items():
        for ax, field in zip(axes.flat, targets):
            points = [flatten(x) for x in values['eval'] if x['partition']=='screen']
            points = [x for x in points if x.get(field) is not None]
            if points: ax.plot([x['step'] for x in points], [x[field] for x in points], marker='.', label=name)
    for ax, field in zip(axes.flat, targets):
        ax.set(title=field, xlabel='Optimizer step'); ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=7)
    fig.suptitle('Screen tasks: unsmoothed, one seed, matched final epoch')
    save_figure(fig, dest, 'target_curves'); plt.close(fig)
    lines = ['# A5全局注意力：QK归一化×单位初始门控，加原始A0对照', '', f'更新时间：{stamp()} UTC；完成{done}/5。', '',
        '五组均随机初始化，seed43/1epoch/1276步、micro32/global512；原V8形式与固定653,288条10%子集。',
        'C0为全新A5对照；C1仅Q/K每头RMS归一化；C2仅SDPA输出2sigmoid门控，零初始化使初始输出与A5相同；C3组合。',
        'C0-C3的8层全局注意力、链内RoPE、chain bias、FFN3072、目标/数据/解码均保持；参数差异最大约0.1%，不改变FFN预算。另从头训练原始A0以作同micro/同epoch配方对照。',
        '只用screen评估，固定epoch末终点；五组齐全才计算因子差值/交互。一种子不能证明稳定提升；已查看的旧confirm不是新候选的全新留出。', '',
        '| 模型 | 状态 | step | 整链AAR | CDR AAR | AUROC | AUPRC |', '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        def fmt(k): return f'{r[k]:.4f}' if r.get(k) is not None else '—'
        lines.append(f"| {r['variant']} | {r['status']} | {r['step']} | {fmt('generation.conditional_aar')} | {fmt('generation.cdr_aar')} | {fmt('representation.macro_auroc')} | {fmt('representation.macro_auprc')} |")
    lines += ['', f'![完整loss/生成/表征曲线]({dest}/curves.png)', f'![重点任务曲线]({dest}/target_curves.png)', '',
        f'[逐步原始CSV]({dest}/all_curves.csv) · [因子差值与交互]({dest}/factorial_effects.csv)', '',
        '方法与参考文献见 docs/ARCH70M_ATTENTION_FACTORIAL.md；曲线训练CE同时含原始值和25步尾均值，任务验证不平滑。']
    (dest / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    return done


if __name__ == '__main__': report()
