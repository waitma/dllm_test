"""Plot the scratch MLM / optimized Pair runs without modifying training.

Activate conda env /project_c20250601/251105016/conda/envs/pllm, then run:
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/plot_stage1_scratch_progress.py --output ABS_NEW_DIRECTORY
The output contains PNG/PDF curves, CSV data and a frozen provenance snapshot.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
from matplotlib.ticker import ScalarFormatter
import numpy as np

from plot_stage1_ab500k_progress import read_arm, write_csv, train_line, save_figure

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
COLORS = {'mlm': '#2563eb', 'pair_kd': '#ea580c'}
LABELS = {'mlm': '纯 MLM', 'pair_kd': 'Pair + 链内 KD'}


def decorate(ax, title, ylabel):
    ax.set(title=title, xlabel='训练步数', ylabel=ylabel)
    ax.grid(alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        raise ValueError('Use an absolute output path')
    roots = {'mlm': ROOT/'output/stage1_ab500k_scratch_20261024/train',
             'pair_kd': ROOT/'output/stage1_ab500k_scratch_fast_20261024/train'}
    arms = {arm: read_arm(root, arm) for arm, root in roots.items()}
    # Plans differ intentionally in execution backend and output paths.
    # Check the scientific comparison contract, including scratch initialization.
    contract = ['subset_manifest_sha256', 'parent_model_sha256', 'initialization',
                'training', 'noise_recipe', 'noise_parent_records', 'recipe',
                'validation_seeds', 'world_size', 'micro_batch']
    for key in contract:
        if arms['mlm']['provenance'][key] != arms['pair_kd']['provenance'][key]:
            raise ValueError(f'Comparison contract differs: {key}')
    for arm, data in arms.items():
        plan_path = roots[arm].parent/'launch/plan.json'
        if hashlib.sha256(plan_path.read_bytes()).hexdigest() != data['provenance']['plan_sha256']:
            raise ValueError(f'Plan hash mismatch: {arm}')
        data['status'] = json.loads((roots[arm]/arm/'status.json').read_text())
        data['latest_checkpoint'] = json.loads((roots[arm]/arm/'latest.json').read_text())
        data['runtime'] = json.loads((roots[arm]/arm/'runtime_rank0.json').read_text())
        raw = [json.loads(s) for s in (roots[arm]/arm/'metrics.jsonl').read_text().splitlines() if s.strip()]
        data['replay_boundaries'] = [row['step'] for prev, row in zip(raw, raw[1:]) if row['step'] <= prev['step']]
        data['discarded_replayed_rows'] = len(raw)-len(data['training'])
    validation = {arm: {v['step']: v for v in data['validation']} for arm, data in arms.items()}
    shared = sorted(set(validation['mlm']) & set(validation['pair_kd']))
    for step in shared:
        for key in ['records', 'masks', 'seeds', 'selected_tokens', 'aa_tokens']:
            if validation['mlm'][step][key] != validation['pair_kd'][step][key]:
                raise ValueError(f'Validation contract differs: {step}, {key}')
    comparisons = [dict(step=s, mlm=validation['mlm'][s]['aa_ce'], pair=validation['pair_kd'][s]['aa_ce'],
                        improvement_percent=100*(validation['mlm'][s]['aa_ce']-validation['pair_kd'][s]['aa_ce'])/validation['mlm'][s]['aa_ce']) for s in shared]
    # Verify per-record identities and selected-token counts at the matched endpoint.
    paired = []
    for arm in arms:
        ref = validation[arm][shared[-1]]['record_statistics']
        if hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('Record-statistics hash mismatch')
        with np.load(ref['path']) as values:
            paired.append((values['row_ids'].copy(), values['statistics'][:, [1, 3, 5]].copy()))
    if not all(np.array_equal(a, b) for a, b in zip(*paired)):
        raise ValueError('Matched validation identities or token counts differ')
    args.output.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc).isoformat()
    snapshot = dict(observed_utc=now, arms=arms, shared_validation=comparisons,
                    checked_contract_fields=contract, matched_record_identity_check=True,
                    replay_policy='Discard superseded training trajectory; keep newest replay. Validation files are latest per step.',
                    limitation='Single seed, unfinished Pair arm; intra-chain KD only; no cross-chain conclusion.')
    (args.output/'snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n')
    for kind, columns in [('training', ['step', 'mlm', 'objective', 'intra_kl', 'kd_scale', 'seen_records']),
                          ('validation', ['step', 'aa_ce', 'full_token_ce', 'mask_accuracy', 'intra_kl'])]:
        write_csv(args.output/f'{kind}.csv', arms, kind, columns)
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'font.size': 16, 'axes.titlesize': 19, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    max_step = arms['mlm']['provenance']['training']['max_steps']
    caption = f"MLM {arms['mlm']['last_step']}/{max_step} 步 · Pair {arms['pair_kd']['last_step']}/{max_step} 步 · {now[:16].replace('T',' ')} UTC"
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('从零预训练：MLM 与优化版 Pair 的整体曲线', fontsize=25, y=.98)
    fig.text(.5, .935, caption, ha='center', fontsize=16)
    for arm, data in arms.items():
        train_line(axes[0, 0], data['training'], 'mlm', COLORS[arm], LABELS[arm], '--' if arm=='pair_kd' else '-')
        vals = data['validation']
        for ax, lower in [(axes[0, 1], 0), (axes[1, 0], 5000)]:
            rows = [v for v in vals if v['step'] >= lower]
            ax.plot([v['step'] for v in rows], [v['aa_ce'] for v in rows],
                    color=COLORS[arm], label=LABELS[arm], marker='o', markersize=4,
                    linestyle='--' if arm=='pair_kd' else '-', linewidth=2)
    decorate(axes[0, 0], '① 训练 MLM loss：完整过程', '交叉熵（对数纵轴）')
    decorate(axes[0, 1], '② 固定验证 AA loss：含初始模型', '交叉熵（对数纵轴）')
    for ax in axes[0]:
        ax.set_yscale('log')
        ax.set_yticks([.3, .5, 1, 2, 4])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlim(0, max_step)
    decorate(axes[1, 0], '③ 固定验证：后期放大', 'AA 交叉熵（越低越好）')
    axes[1, 0].set_xlim(5000, max_step)
    axes[1, 0].ticklabel_format(axis='y', style='plain', useOffset=False)
    ax = axes[1, 1]
    ax.axhline(0, color='#64748b', linewidth=1)
    ax.plot([v['step'] for v in comparisons], [v['improvement_percent'] for v in comparisons], color=COLORS['pair_kd'], marker='o', markersize=4)
    decorate(ax, '④ 同一步验证：Pair 的相对改善', '改善幅度（%）；正值表示 Pair 更低')
    ax.set_xlim(0, max_step)
    last = comparisons[-1]
    ax.text(.04, .92, f"共同 {last['step']} 步：{last['improvement_percent']:+.3f}%", transform=ax.transAxes)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.legend(frameon=False, fontsize=14)
    fig.text(.07, .065, '训练：浅线为已记录批次，粗线为最近 5 个日志点均值。验证：固定 4096 条 × 3 组 MASK，不平滑。', fontsize=15)
    fig.text(.07, .025, '横轴均为 optimizer step；恢复重放保留最新轨迹。Pair 尚无完成回执；微小差异不代表稳定收益。', fontsize=15)
    fig.subplots_adjust(top=.86, bottom=.16, left=.08, right=.97, wspace=.27, hspace=.40)
    save_figure(fig, args.output, 'scratch_overview_zh')

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Pair 蒸馏：学生与教师的链内距离分布逐渐接近', fontsize=24, y=.98)
    fig.text(.5, .91, caption, ha='center', fontsize=16)
    pair = arms['pair_kd']
    for ax, lower, title in zip(axes, [0, 5000], ['完整训练过程', '后期放大：5000 步以后']):
        rows = [v for v in pair['training'] if v['step'] >= lower]
        vals = [v for v in pair['validation'] if v['step'] >= lower]
        train_line(ax, rows, 'intra_kl', COLORS['pair_kd'], '训练 KL：5 点均值')
        ax.plot([v['step'] for v in vals], [v['intra_kl'] for v in vals], color='#334155', marker='o', markersize=5, label='固定验证 KL')
        decorate(ax, title, '链内 KL（未乘权重；越低越好）')
        ax.set_xlim(lower, max_step)
        ax.legend(frameon=False, fontsize=15)
    fig.text(.08, .065, '这里只监督重链内部、轻链内部的残基对；KL 下降不能证明跨链接触关系已学好。', fontsize=17)
    fig.subplots_adjust(top=.79, bottom=.22, left=.08, right=.97, wspace=.23)
    save_figure(fig, args.output, 'scratch_kd_zh')
    summary = dict(observed_utc=now, last_step={a:d['last_step'] for a,d in arms.items()},
                   latest_common=last, validation_kd_initial=pair['validation'][0]['intra_kl'],
                   validation_kd_latest=pair['validation'][-1]['intra_kl'],
                   discarded_replayed_rows={a:d['discarded_replayed_rows'] for a,d in arms.items()},
                   replay_boundaries={a:d['replay_boundaries'] for a,d in arms.items()})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
