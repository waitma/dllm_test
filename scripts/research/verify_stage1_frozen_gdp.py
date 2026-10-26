"""Verify saved GDP predictions and plot the frozen Stage1 comparison.

After activating pllm, invoke with the protenix_abtcr environment's Python:
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/verify_stage1_frozen_gdp.py --output ABS_GDP_DIRECTORY
The original features, plan and scores are read only. No encoder is trained.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from evaluate_stage1_frozen_gdp import load_plan, sha, write, FOLD


def row_correlation(a, b):
    a = rankdata(a, axis=1); b = rankdata(b, axis=1)
    a -= a.mean(axis=1, keepdims=True); b -= b.mean(axis=1, keepdims=True)
    return (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output
    assert out.is_absolute()
    plan = load_plan(out)
    df = pd.read_csv(out/'data_snapshot.csv')
    preds = pd.read_csv(out/'predictions.csv')
    folds = pd.read_csv(out/'fold_metrics.csv')
    summary = pd.read_csv(out/'summary.csv')
    metrics = json.loads((out/'metrics.json').read_text())
    assert metrics['predictions_sha256'] == sha(out/'predictions.csv')
    keys = ['arm', 'target', 'fraction', 'seed', 'fold']
    assert not preds.duplicated(keys+['row']).any()
    assert len(folds) == 800
    checked = 0; max_error = 0.
    for arm in plan['arms']:
        with np.load(out/(arm+'_features.npz')) as data:
            x = data['features'].astype(np.float64)
        for target in plan['targets']:
            y = df[target].to_numpy(dtype=float)
            for fraction in plan['fractions']:
                for seed in (plan['subset_seeds'] if fraction < 1 else [11]):
                    for fold in range(5):
                        p = preds[(preds.arm == arm) & (preds.target == target) &
                                  (preds.fraction == fraction) & (preds.seed == seed) & (preds.fold == fold)]
                        rows = np.flatnonzero(np.isfinite(y) & (df[FOLD].to_numpy() == fold))
                        assert np.array_equal(p.row, rows)
                        assert np.array_equal(p.id.astype(str), df.antibody_id.iloc[rows].astype(str))
                        np.testing.assert_allclose(p.truth, y[rows], rtol=1e-12)
                        train = np.flatnonzero(np.isfinite(y) & (df[FOLD].to_numpy() != fold))
                        rng = np.random.default_rng(seed+100*fold)
                        train = np.sort(rng.permutation(train)[:max(2, int(len(train)*fraction))])
                        assert not set(train) & set(rows)
                        m = folds[(folds.arm == arm) & (folds.target == target) &
                                  (folds.fraction == fraction) & (folds.seed == seed) & (folds.fold == fold)]
                        assert len(m) == 1 and int(m.iloc[0].train_n) == len(train)
                        assert int(m.iloc[0].test_n) == len(rows)
                        np.testing.assert_allclose(m.iloc[0].spearman, spearmanr(p.truth, p.prediction).statistic, atol=1e-12)
                        np.testing.assert_allclose(m.iloc[0].rmse, np.sqrt(np.mean((p.truth-p.prediction)**2)), atol=1e-12)
                        # Refit every full-data readout through explicit scaling and estimator calls.
                        if fraction == 1:
                            scaler = StandardScaler().fit(x[train])
                            with threadpool_limits(limits=4):
                                head = Ridge(alpha=1.).fit(scaler.transform(x[train]), y[train])
                                again = head.predict(scaler.transform(x[rows]))
                            max_error = max(max_error, float(np.max(np.abs(again-p.prediction))))
                            np.testing.assert_allclose(again, p.prediction, atol=1e-9, rtol=1e-9)
                        checked += 1
    for row in summary.itertuples():
        selected = folds[(folds.arm == row.arm) & (folds.target == row.target) & (folds.fraction == row.fraction)]
        np.testing.assert_allclose(row.spearman, selected.spearman.mean(), atol=1e-12)
    differences = []; bootstrap_seed = 20261025
    rng = np.random.default_rng(bootstrap_seed)
    for target in plan['targets']:
        delta = []; sampled = []
        for fold in range(5):
            p = preds[(preds.target == target) & (preds.fraction == 1) & (preds.fold == fold)]
            a = p[p.arm == 'mlm'].sort_values('row'); b = p[p.arm == 'pair_kd'].sort_values('row')
            assert np.array_equal(a.row, b.row)
            y, va, vb = a.truth.to_numpy(), a.prediction.to_numpy(), b.prediction.to_numpy()
            delta.append(float(spearmanr(y, vb).statistic-spearmanr(y, va).statistic))
            idx = rng.integers(len(y), size=(2000, len(y)))
            sampled.append(row_correlation(y[idx], vb[idx])-row_correlation(y[idx], va[idx]))
        ci = np.quantile(np.mean(sampled, axis=0), [.025, .975])
        differences.append(dict(target=target, delta=float(np.mean(delta)), fold_deltas=delta,
                                descriptive_row_bootstrap_low=float(ci[0]), descriptive_row_bootstrap_high=float(ci[1])))
    write(out/'verification.json', dict(status='passed', scorer_python=sys.executable,
        groups_verified=checked, full_data_readouts_refit=50, max_prediction_error=max_error,
        predictions_sha256=sha(out/'predictions.csv'), plan_sha256=sha(out/'plan.json'),
        verifier_sha256=sha(__file__), differences=differences, bootstrap_seed=bootstrap_seed,
        uncertainty='Paired row bootstrap within held-out folds, fixed predictions, 2000 repeats; descriptive only. '
        'No homology-cluster IDs; ignores model/head training uncertainty and residual dependence. Not a significance test.'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
        'font.size': 16, 'axes.titlesize': 20, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    labels = {'mlm': '纯 MLM', 'pair_kd': 'Pair + 链内 KD'}
    colors = {'mlm': '#2563eb', 'pair_kd': '#ea580c'}
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for ax, target in zip(axes.flat, plan['targets']):
        for arm in plan['arms']:
            sub = summary[(summary.arm == arm) & (summary.target == target)].sort_values('fraction')
            ax.plot(sub.mean_train_n, sub.spearman, 'o-', color=colors[arm], label=labels[arm], linewidth=2.5)
        ax.set(title=target, xlabel='每折实际训练标签数（均值）', ylabel='测试折 Spearman（均值）', ylim=(-.15, .5))
        ax.axhline(0, color='gray', linewidth=1); ax.grid(alpha=.2)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, .95, '两模型均为最终 9770 步\n编码器完全冻结\n只拟合同规格 Ridge 读出器\n\n官方分组五折；alpha = 1\n少样本比例：10 / 25 / 50 / 100%\n前三档各 5 次固定抽样\n\n所有任务完整展示\n不按测试分数选择超参数', va='top', fontsize=18)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=2, bbox_to_anchor=(.5, .945), frameon=False)
    fig.suptitle('现有模型的下游表现：实验性质与标签量', fontsize=27)
    fig.tight_layout(rect=(0, .02, 1, .91))
    fig.savefig(out/'gdp_fewshot_zh.png', dpi=150); fig.savefig(out/'gdp_fewshot_zh.pdf'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 7))
    for i, row in enumerate(differences):
        ax.scatter(row['fold_deltas'], np.full(5, i), color='#94a3b8', s=65, alpha=.8)
        ax.plot(row['delta'], i, 'D', color='#ea580c', markersize=10)
        ax.text(.98, i, f"均值 {row['delta']:+.3f}", transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=17)
    ax.set_yticks(range(5), plan['targets']); ax.invert_yaxis(); ax.axvline(0, color='black', linewidth=1)
    ax.set(xlabel='Pair − MLM：测试折 Spearman 差值', title='全标签评估：每个测试折的配对差值')
    ax.grid(axis='x', alpha=.2)
    fig.text(.5, .01, '灰点：五个测试折；橙色菱形：折均值。单次预训练，不代表跨种子的稳定性。', ha='center', fontsize=15)
    fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(out/'gdp_deltas_zh.png', dpi=150)
    fig.savefig(out/'gdp_deltas_zh.pdf'); plt.close(fig)
    print(json.dumps({'status':'passed','groups':checked,'max_prediction_error':max_error,'differences':differences}, ensure_ascii=False))


if __name__ == '__main__':
    main()
