"""Read-only experiment audit and figures; activate pllm, then run output/arch70m_env/bin/python -B /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/review_arch70m_20261029.py."""
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.arch70m.common import ROOT, read, write, sha, stamp, lock
from experiments.arch70m.report import records, flatten, export_csv, save_figure

BASE = ROOT/'output/arch70m_v8_20261028'
SECOND = ROOT/'output/arch70m_round2_20261029'
DEST = ROOT/'docs/arch70m_results/review_20261029'
DATA = '4beaac6b54a1f46492533aeb77c2de2c6ac2a0bd8266fd57a6bb312679435c51'
OLD_SOURCE = '053297983e0f2ece2aaad807032695fb5aebd1e0f221138a9bf0d202cc9b9ac4'
NEW_SOURCE = '93518fa2a14bc029e808cefae7aef8c38d79804683bb5274af27ff5f4e67d65e'
FIELDS = ['token.all.loss', 'generation.conditional_ce', 'generation.conditional_aar',
          'generation.cdr_aar', 'generation.conditional_length_mae',
          'representation.macro_auroc', 'representation.macro_auprc']


def collect():
    rows, trials, audits = [], {}, []
    assert sha(BASE/'data/index.json') == sha(SECOND/'data/index.json') == DATA
    for seed in (42, 43):
        variants = [f'A{i}' for i in range(8)] + (['B1'] if seed == 43 else [])
        for variant in variants:
            trial = (SECOND if variant == 'B1' else BASE)/'trials'/f'screen_{variant}_s{seed}'
            status, binding = read(trial/'status.json'), read(trial/'binding.json')
            assert status['status'] == 'completed' and status['step'] == binding['total_steps'] == 1276
            assert binding['seed'] == seed and binding['epochs'] == 1 and binding['data_sha256'] == DATA
            assert binding['source_sha256'] == (NEW_SOURCE if variant == 'B1' else OLD_SOURCE)
            assert binding['settings'] == read(ROOT/'experiments/arch70m/base.json')
            assert binding['micro_batch'] == (96 if variant == 'A5' and seed == 42 else 56)
            train, evals = records(trial/'train.jsonl'), records(trial/'eval.jsonl')
            assert [x['step'] for x in train] == list(range(1, 1277))
            assert train[-1]['seen_records'] == 653288 and train[-1]['seen_tokens'] == 214482921
            assert all(math.isfinite(x[k]) for x in train for k in ('loss','total_loss','grad_norm'))
            metric_path = trial/'metrics/screen_001276.json'
            metrics = read(metric_path)
            assert metrics == [x for x in evals if x['partition'] == 'screen' and x['step'] == 1276][-1]
            flat = flatten(metrics)
            row = dict(variant=variant, seed=seed, micro_batch=binding['micro_batch'],
                       step=1276, records=653288, train_gpu_hours=train[-1]['gpu_hours'],
                       **{k:flat[k] for k in FIELDS})
            rows.append(row); trials[(variant, seed)] = dict(train=train, eval=evals, metrics=metrics)
            audit = dict(variant=variant, seed=seed, binding_sha256=sha(trial/'binding.json'),
                         metrics_sha256=sha(metric_path), training_steps=1276,
                         finite_training=True, metrics_match_eval=True,
                         source_sha256=binding['source_sha256'], data_sha256=DATA)
            if seed == 43:
                cp = trial/'checkpoints/step_001276'; receipt = read(cp/'receipt.json')
                assert receipt['status'] == 'complete' and len(receipt['files']) == 6
                assert all(sha(cp/name) == checksum for name,checksum in receipt['files'].items())
                state = read(cp/'state.json')
                assert state['step'] == 1276 and state['seen_records'] == 653288
                assert state['epoch'] == 1 and state['next_batch'] == 0
                assert read(cp/'binding.json') == binding
                audit.update(checkpoint_files_checked=6, checkpoint_sha256=receipt['files'])
            audits.append(audit)
    return rows, trials, audits


def figures(rows, trials):
    plt.rcParams.update({'font.size': 10})
    variants = [f'A{i}' for i in range(8)]+['B1']; x = np.arange(len(variants))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [('generation.conditional_aar', 'Full-chain AAR (%)', 100),
              ('generation.cdr_aar', 'CDR infilling AAR (%)', 100),
              ('representation.macro_auroc', 'Frozen-probe macro AUROC', 1),
              ('representation.macro_auprc', 'Frozen-probe macro AUPRC', 1)]
    for ax, (field, title, scale) in zip(axes.flat, panels):
        for seed, offset, color in [(42, -.13, '#9c9c9c'), (43, .13, '#1673b9')]:
            values = [next((r[field]*scale for r in rows if r['seed']==seed and r['variant']==v), np.nan) for v in variants]
            ax.scatter(x+offset, values, color=color, s=38, label=f'seed {seed}', zorder=3)
        if field == 'generation.conditional_aar':
            ax.set_ylim(0, 43)
            for i in (5, 8):
                value = next(r[field]*scale for r in rows if r['seed']==43 and r['variant']==variants[i])
                ax.annotate(f'{value:.2f}%', (i+.13, value), xytext=(0, 9), textcoords='offset points', ha='center')
        ax.set(title=title, xticks=x, xticklabels=variants)
        ax.grid(axis='y', alpha=.25); ax.set_axisbelow(True)
    axes.flat[0].legend(loc='upper left')
    fig.suptitle('Matched one-epoch endpoints | original V8 / fixed 10% subset / global batch 512\nA5 seed42 uses micro96; all seed43 trials use micro56. B1 has seed43 only; B2/B3 have no trained endpoints.', fontsize=11)
    save_figure(fig, DEST, 'endpoints'); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels = [('train', 'Training token CE (25-step mean)'),
              ('token.all.loss', 'Validation token CE'),
              ('generation.conditional_aar', 'Full-chain AAR (%)'),
              ('generation.cdr_aar', 'CDR infilling AAR (%)'),
              ('generation.conditional_ce', 'Conditional generation CE'),
              ('generation.conditional_length_mae', 'Generated-length MAE'),
              ('representation.macro_auroc', 'Frozen-probe macro AUROC'),
              ('representation.macro_auprc', 'Frozen-probe macro AUPRC')]
    for variant, color in [('A0','#1673b9'),('A4','#d57a00'),('A5','#23983b'),('B1','#943fa9')]:
        trial = trials[(variant,43)]
        for ax, (field,title) in zip(axes.flat,panels):
            if field == 'train':
                values = trial['train']; y = [r['loss'] for r in values]; steps = [r['step'] for r in values]
                ax.plot(steps,y,color=color,alpha=.12,lw=.4)
                ax.plot(steps,[np.mean(y[max(0,i-24):i+1]) for i in range(len(y))],color=color,label=variant,lw=1.3)
            else:
                values = [flatten(r) for r in trial['eval'] if r['partition']=='screen']
                values = [r for r in values if field in r]
                scale = 100 if field.endswith('_aar') else 1
                ax.plot([r['step'] for r in values],[r[field]*scale for r in values],color=color,label=variant,marker='.',lw=1.2)
            ax.set(title=title,xlabel='Optimizer step'); ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4)
    fig.suptitle('Complete seed43 curves | same data, evaluation and one-epoch budget\nValidation/task values unsmoothed; training raw values faint. No interpolation beyond measured task evaluations.',fontsize=12)
    save_figure(fig,DEST,'curves'); plt.close(fig)


def main():
    rows, trials, audits = collect()
    blocked=[]
    for variant in ('B2','B3'):
        trial=SECOND/'trials'/f'screen_{variant}_s43';status=read(trial/'status.json')
        profile=read(trial/'profile_attempts.json')['candidates'][0]
        assert status['status']=='failed' and status['step']==0 and profile['status']=='passed' and not profile['safe']
        blocked.append(dict(variant=variant,status='memory_gate_before_formal_training',profile=profile,
                            gate_bytes=.85*profile['total_bytes'],formal_steps=0))
    with lock(ROOT/'output/arch70m_publish.lock'):
        DEST.mkdir(parents=True,exist_ok=True)
        write(DEST/'audit.json',dict(checked=stamp(),completed_trials=17,seed43_checkpoint_files_checked=54,
              all_step_sequences_finite_and_complete=True,trials=audits,blocked_trials=blocked,
              scope='File/binding/log integrity only; no new training, inference or GPU numerical replay.'))
        export_csv(DEST/'endpoints.csv',rows); figures(rows,trials)
        lines=['# 架构探索结果复核（2026-10-29）','',f'核查时间：{stamp()} UTC。',
            'A0–A7：seed42与seed43各8组均完成1epoch/1276步；本地第二轮B1完成，B2/B3止于训练前显存门槛。GPU当前空闲，本次只审计、作图和整理结果，没有启动新任务。',
            '', '## 同预算 seed43 终点', '',
            '| 架构 | 验证token CE↓ | 整链AAR↑ | 长度MAE↓ | CDR AAR↑ | AUROC↑ | AUPRC↑ |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for r in [r for r in rows if r['seed']==43]:
            lines.append(f"| {r['variant']} | {r['token.all.loss']:.4f} | {100*r['generation.conditional_aar']:.2f}% | {r['generation.conditional_length_mae']:.2f} | {100*r['generation.cdr_aar']:.2f}% | {r['representation.macro_auroc']:.4f} | {r['representation.macro_auprc']:.4f} |")
        lines += ['', '## 目前能得出的结论', '',
            '1. A5是目前整链生成最有重复性的候选：seed42/43整链AAR为37.02%/36.51%，A0为12.18%/3.42%；长度MAE分别5.61/6.57，对照76.50/99.03。A5在seed42用了micro96，seed43已经统一micro56，优势依然存在。',
            '2. B1相对A5（同seed43）整链AAR下降1.16个百分点，CDR AAR上升0.22个百分点；AUROC增加0.0042、AUPRC增加0.0176。它提供一个可能更平衡的候选，尚无全面胜出的证据。B1的AUPRC在本次seed43九组中最高，AUROC仍低于A0。',
            '3. A4在seed42的AUROC优势（0.8478对0.8252）未在seed43重现（0.8338对0.8354），不能延续“表征稳定最好”的说法。表征暂无稳定赢家。',
            '4. A6在两个种子上的验证token CE最低（0.7160/0.7143），但整链AAR远低于A5。较低MLM loss不能直接推断更好的生成。',
            '5. 两个种子只提供重复性线索；B1仅一个种子。当前是原screen划分上的探索，完整留出确认尚未针对这些候选重新完成。整链生成117条、CDR填补64条，表征两个验证来源共106条（33+73）；小幅差异不能视为统计显著或泛化改善。AAR也不证明抗原/结合特异性。',
            '', '## B2/B3为何没有结果', '',
            '| 变体 | 最大长度profile allocated | reserved | 外部 | reserved＋外部/总量 | 正式步数 |',
            '|---|---:|---:|---:|---:|---:|']
        for b in blocked:
            p=b['profile']; ratio=(p['reserved_bytes']+p['external_bytes'])/p['total_bytes']
            lines.append(f"| {b['variant']} | {p['allocated_bytes']/2**30:.2f}GiB | {p['reserved_bytes']/2**30:.2f}GiB | {p['external_bytes']/2**30:.2f}GiB | {ratio:.2%} | 0 |")
        lines += ['',
            '两组均完成3次micro56/maxlen2048试运行更新，随后因reserved＋外部显存超过总量85%主动停止。总量94.97GiB，门槛80.73GiB；日志没有OOM或NCCL失败。这是显存预算未通过，不能据此评价架构效果。若继续，应另行记录统一较小micro的协议并先实测，global512和原数据仍可保持；本次没有修改配方或重新启动。',
            '', '## 曲线与审计', '',
            f'![各架构终点]({DEST}/endpoints.png)', '',f'![完整同seed曲线]({DEST}/curves.png)', '',
            f'[终点CSV]({DEST}/endpoints.csv) · [审计回执]({DEST}/audit.json)', '',
            '17份终点指标已与eval日志匹配；每组1276条训练记录连续，loss/总loss/梯度均有限，消费653,288条、214,482,921个nonpad tokens。原数据index、settings、source、seed、epoch和micro绑定检查通过；额外逐一重算seed43九组最终checkpoint共54文件SHA并核验epoch/数据游标。未重跑推理或GPU数值重放。',
            f'原八卡[报告]({ROOT}/docs/arch70m_results/seed43_validation/REPORT.md)与[逐步原始CSV]({ROOT}/docs/arch70m_results/seed43_validation/all_curves.csv)。本地第二轮[报告]({ROOT}/docs/arch70m_results/round2/REPORT.md)与[逐步原始CSV]({ROOT}/docs/arch70m_results/round2/all_curves.csv)。','']
        (DEST/'REPORT.md').write_text('\n'.join(lines))
        status=f'> Last updated: {stamp()} (UTC)。结果复核：A0–A7 seed42/43各8组完成；本地B1完成1276步，B2/B3因micro56/2048 profile reserved＋外部显存约91%超过85%门槛止于正式step0，非OOM。A5两种子整链AAR37.02%/36.51%保持优势；B1 seed43整链AAR35.35%、CDR80.50%、AUROC0.8321/AUPRC0.6895，较A5有取舍；A4表征优势未在第二种子重现。17组日志与绑定及seed43共54份checkpoint文件SHA核验；曲线/完整结论 {DEST}/REPORT.md。本次未启动或修改训练，GPU空闲。'
        p=ROOT/'PROJECT_PROCESS.md';p.write_text(status+'\n\n'+p.read_text())
        p=ROOT/'BIOSEQ_MODEL_PLAN.md';p.write_text(f'> 2026-10-29（UTC）同预算终点复核支持继续保留A5整链生成候选；A4在seed42的表征优势没有在seed43重复。B1作为全局＋可学习残差，仅seed43显示CDR/AUPRC小幅改善、整链AAR小幅回落，不认定全面胜出或显著改善。B2/B3尚无正式训练结果，85%显存门槛失败不构成效果消融。未改变原配方/数据或追加种子。完整证据 {DEST}/REPORT.md。\n\n'+p.read_text())
        p=ROOT/'docs/ARCH70M_ROUND2.md';p.write_text(f'> 当前状态复核：{stamp()} UTC。B1已完成1276步；B2/B3在正式step0因显存安全门槛停止，队列已结束。下文启动时进度是历史记录。结果与完整曲线见[本次复核]({DEST}/REPORT.md)。\n\n'+p.read_text())
        p=ROOT/'docs/ARCH70M_EXPLORATION.md';text=p.read_text();head='# V8 约 70M 架构探索实验记录\n'
        assert text.startswith(head)
        p.write_text(head+f'\n## 已完成结果复核（{stamp()} UTC）\n\n八架构seed42/43均完成，第二轮B1完成，B2/B3因显存门槛止于正式step0。A5整链生成优势在两个种子重复；A4表征优势未重复；B1呈现生成/表征取舍。17组连续训练日志与绑定、seed43九组最终checkpoint共54文件SHA通过核验；没有新增训练或改变数据。[完整比较与曲线]({DEST}/REPORT.md)。\n'+text[len(head):])
        print(dict(report=str(DEST/'REPORT.md'),completed=17,seed43_checkpoint_files_checked=54,blocked=['B2','B3']),flush=True)


if __name__ == '__main__':
    main()
