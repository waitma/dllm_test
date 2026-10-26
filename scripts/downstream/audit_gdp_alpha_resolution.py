"""Diagnose Ridge alpha-grid resolution equally for teacher and final features.

Activate pllm then protenix_abtcr; from the repository root:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B \
  scripts/downstream/audit_gdp_alpha_resolution.py --out ABS_NEW_DIRECTORY
Keep original GDP folds, saved transformed targets and float64 Ridge. Add
quarter-decade alphas to the complete original grid for all five properties
and all five predefined readouts. This is same-CV exploratory probe analysis,
not a new backbone result or a replacement for the original reported protocol.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from downstream.ophiuchus_eval.developability import LAMBDA_GRID
    root = ROOT / 'output/diagnostics'
    depth = root / 'fixed50000_depth_20261018'
    sources = {}

    def pin(path):
        sources[str(path)] = sha(path)
        return path

    teacher = np.load(pin(depth/'features.npz'))['native_block2'].astype(np.float64)
    cases = {'block2': (teacher, depth/'native_block2')}
    for label, folder in [('token1700', root/'final_layer_token_direction_pilot_20261018/token_all_direction'),
                          ('token5000', root/'final_layer_convergence_pilot_20261018/token_all_direction')]:
        verified = json.loads(pin(folder/'verification.json').read_text())
        assert verified['status'] == 'passed' and verified['reloaded_features_bitwise_equal']
        for mode, filename, metrics_dir in [('native', 'final_features.npy', 'gdp_a1'),
                                            ('no_mask_keys', 'final_no_mask_features.npy', 'gdp_no_mask_keys')]:
            cases[label+'_'+mode] = (np.load(pin(folder/filename)).astype(np.float64), folder/metrics_dir)
    coarse = [float(a) for a in LAMBDA_GRID]
    # Retain original decade floats/order exactly, including zero and tie rules.
    extra = [float(10.0**(-6+i/4)) for i in range(37) if i % 4]
    alphas = coarse+extra
    assert len(set(alphas)) == len(alphas)
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), original_grid=coarse, diagnostic_grid=alphas,
                cases=list(cases), properties='all original five',
                protocol='same saved all-label transform, same five folds, Ridge float64; same-CV alpha selection/reporting',
                limitations='exploratory diagnostic; no independent test, no new backbone, no encoder training or reference-result overwrite')
    write(out/'plan.json', plan)
    status = dict(status='running', pid=os.getpid(), started_utc=plan['created_utc'])
    write(out/'status.json', status)
    metrics = {name: json.loads(pin(folder/'metrics.json').read_text()) for name, (_, folder) in cases.items()}
    result, checks = {}, []
    for prop in metrics['block2']['properties']:
        canonical = np.load(pin(depth/'native_block2'/prop/'legacy_oof.npz'))
        result[prop] = {}
        for name, (full, folder) in cases.items():
            saved = np.load(pin(folder/prop/'legacy_oof.npz'))
            for key in ('row_index', 'fold', 'y_true', 'y_transformed'):
                np.testing.assert_array_equal(saved[key], canonical[key])
            x, y, fold = full[saved['row_index']], saved['y_transformed'], saved['fold']
            predicted = np.empty((len(alphas), len(y)), dtype=np.float64)
            scores = np.empty((len(alphas), 5), dtype=np.float64)
            for j, alpha in enumerate(alphas):
                for k in range(5):
                    train, test = fold != k, fold == k
                    model = Ridge(alpha=alpha).fit(x[train], y[train])
                    predicted[j, test] = model.predict(x[test])
                    scores[j, k] = spearmanr(y[test], predicted[j, test]).statistic
            assert np.isfinite(predicted).all() and np.isfinite(scores).all()
            means = scores.mean(1)
            old = metrics[name]['properties'][prop]['legacy_reference']
            old_index = alphas.index(old['alpha'])
            np.testing.assert_allclose(predicted[old_index], saved['prediction'], atol=1e-9, rtol=1e-9)
            assert abs(means[old_index]-old['cv_mean_spearman']) < 1e-12
            assert abs(means[:len(coarse)].max()-old['cv_mean_spearman']) < 1e-12
            best = int(np.argmax(means))
            destination = out/prop/name
            destination.mkdir(parents=True)
            np.savez_compressed(destination/'diagnostic_oof.npz', row_index=saved['row_index'], fold=fold,
                 y_true=saved['y_true'], y_transformed=y, prediction=predicted[best], selected_alpha=alphas[best])
            item = dict(original_alpha=old['alpha'], original_cv_mean=old['cv_mean_spearman'],
                        diagnostic_alpha=alphas[best], diagnostic_cv_mean=float(means[best]),
                        difference=float(means[best]-old['cv_mean_spearman']),
                        alpha_curve=[dict(alpha=a, cv_mean=float(means[j]), folds=scores[j].tolist()) for j, a in enumerate(alphas)])
            result[prop][name] = item
            checks.append(dict(property=prop, readout=name, original_oof_reproduced=True,
                               original_grid_best_reproduced=True, rows_labels_folds_equal=True))
            write(out/'results.json', result)
            status.update(property=prop, readout=name)
            write(out/'status.json', status)
            print(prop, name, {k:v for k,v in item.items() if k != 'alpha_curve'}, flush=True)
    pin(Path(__file__))
    pin(ROOT/'downstream/ophiuchus_eval/developability.py')
    for path, digest in sources.items():
        assert sha(Path(path)) == digest, path
    write(out/'verification.json', dict(status='passed', checks=checks, source_sha256=sources))
    status.update(status='success', finished_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', status)


if __name__ == '__main__':
    main()
