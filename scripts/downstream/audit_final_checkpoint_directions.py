"""Diagnose property-direction errors in completed actual final checkpoints.

Activate pllm then protenix_abtcr; run from the repository root:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/audit_final_checkpoint_directions.py \
  --out ABS_NEW_DIRECTORY --arm token1700=ABS_VERIFIED_ARM --arm token5000=ABS_VERIFIED_ARM
Reproduce original teacher Ridge OOF and transfer each fixed fold probe through
the OAS-only calibration. This uses GDP labels for fold probes, not backbone
training, target fitting, checkpoint selection, or a new headline model score.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', action='append', required=True, help='NAME=ABS_VERIFIED_ARM')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge

    root = ROOT / 'output/diagnostics'
    depth = root / 'fixed50000_depth_20261018'
    cal_path = root / 'final_layer_coordinate_pilot_20261018/calibration.npz'
    teacher_path = depth / 'features.npz'
    teacher_dir = depth / 'native_block2'
    runs = {}
    identity_paths = []
    for entry in args.arm:
        label, raw = entry.split('=', 1)
        assert label.isidentifier(), label
        folder = Path(raw).resolve()
        verification = json.loads((folder/'verification.json').read_text())
        assert verification['status'] == 'passed' and verification['reloaded_features_bitwise_equal']
        assert json.loads((folder.parent/'status.json').read_text())['status'] == 'success'
        side = json.loads((folder/'checkpoint.json').read_text())
        assert sha(Path(side['parameter_file'])) == side['sha256']
        identity_paths += [folder/'verification.json', folder/'checkpoint.json', folder.parent/'status.json', Path(side['parameter_file'])]
        assert label+'_native' not in runs
        runs[label+'_native'] = (folder/'final_features.npy', folder/'gdp_a1')
        runs[label+'_no_mask_keys'] = (folder/'final_no_mask_features.npy', folder/'gdp_no_mask_keys')
    write(out/'status.json', dict(status='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat()))

    sources = {}
    def pin(path):
        sources[str(path)] = sha(path)
    def read(path):
        pin(path)
        return json.loads(path.read_text())
    def array(path):
        pin(path)
        return np.load(path, allow_pickle=False)

    for path in identity_paths:pin(path)
    cal = array(cal_path)
    teacher = array(teacher_path)['native_block2'].astype(np.float64)
    rotation = cal['rotation'].astype(np.float64)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(teacher.shape[1]), atol=1e-10, rtol=0)
    teacher_metrics = read(teacher_dir / 'metrics.json')
    data = {}
    for name, (feature_path, metrics_dir) in runs.items():
        student = array(feature_path).astype(np.float64)
        assert student.shape == teacher.shape and np.isfinite(student).all()
        estimate = ((student-cal['mu_s'])/cal['rms_s']) @ rotation.T*cal['rms_t']+cal['mu_t']
        # Verify that inverse calibration is exactly the intended affine map.
        roundtrip = ((estimate-cal['mu_t'])/cal['rms_t']) @ rotation*cal['rms_s']+cal['mu_s']
        np.testing.assert_allclose(roundtrip, student, atol=1e-10, rtol=1e-10)
        data[name] = dict(student=student, estimate=estimate, metrics_dir=metrics_dir,
                          metrics=read(metrics_dir / 'metrics.json'))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                purpose='Property-specific error diagnosis; not a trained final-layer result',
                teacher='Original fixed50000 native block2',
                calibration='Frozen OAS-only affine rotation, no GDP refit',
                probe='Original selected teacher alpha; each Ridge refit on exactly the other four folds',
                limitations='Exploratory same-CV alpha selection/all-label transform inherited from reference; this analysis uses GDP labels and must not supply training targets',
                methods={name: dict(features=str(paths[0]), metrics=str(paths[1])) for name, paths in runs.items()},
                script_sha256=sha(Path(__file__)))
    write(out / 'plan.json', plan)
    result = {}
    checks = []
    for prop in teacher_metrics['properties']:
        saved = array(teacher_dir / prop / 'legacy_oof.npz')
        idx, fold, y = saved['row_index'], saved['fold'], saved['y_transformed']
        assert len(np.unique(idx)) == len(idx) and set(fold) == set(range(5))
        x = teacher[idx]
        alpha = teacher_metrics['properties'][prop]['legacy_reference']['alpha']
        actual = {}
        for name, item in data.items():
            actual[name] = array(item['metrics_dir'] / prop / 'legacy_oof.npz')
            for key in ('row_index', 'fold', 'y_true', 'y_transformed'):
                np.testing.assert_array_equal(actual[name][key], saved[key])
        predictions = {name: np.empty(len(idx), dtype=np.float64) for name in data}
        teacher_prediction = np.empty(len(idx), dtype=np.float64)
        folds = {name: [] for name in data}
        coefficient = []
        intercept = []
        for k in range(5):
            train, test = fold != k, fold == k
            probe = Ridge(alpha=alpha).fit(x[train], y[train])
            target = probe.predict(x[test])
            teacher_prediction[test] = target
            np.testing.assert_allclose(target, saved['prediction'][test], atol=1e-9, rtol=1e-9)
            coefficient.append(probe.coef_)
            intercept.append(probe.intercept_)
            for name, item in data.items():
                estimate = item['estimate'][idx[test]]
                pred = probe.predict(estimate)
                predictions[name][test] = pred
                error = pred-target
                np.testing.assert_allclose(error, (estimate-x[test]) @ probe.coef_, atol=1e-10, rtol=1e-10)
                error_centered = error-error.mean()
                signal = target-target.mean()
                denominator = float(np.mean(signal**2))
                assert denominator > 0
                folds[name].append(dict(fold=k, n=int(test.sum()),
                    teacher_rho=float(spearmanr(y[test], target).statistic),
                    teacher_probe_on_aligned_student_rho=float(spearmanr(y[test], pred).statistic),
                    independently_refit_student_rho=float(spearmanr(y[test], actual[name]['prediction'][test]).statistic),
                    prediction_rank_agreement=float(spearmanr(target, pred).statistic),
                    centered_direction_error_over_teacher_signal=float(np.mean(error_centered**2)/denominator),
                    mean_prediction_offset=float(error.mean()),
                    error_signal_correlation=float(np.corrcoef(error_centered, signal)[0, 1])))
        result[prop] = {}
        for name in data:
            item = dict(folds=folds[name])
            for key in ('teacher_rho', 'teacher_probe_on_aligned_student_rho',
                        'independently_refit_student_rho', 'prediction_rank_agreement',
                        'centered_direction_error_over_teacher_signal'):
                item['mean_'+key] = float(np.mean([v[key] for v in folds[name]]))
            assert abs(item['mean_teacher_rho']-teacher_metrics['properties'][prop]['legacy_reference']['cv_mean_spearman']) < 1e-12
            expected = data[name]['metrics']['properties'][prop]['legacy_reference']['cv_mean_spearman']
            assert abs(item['mean_independently_refit_student_rho']-expected) < 1e-12
            result[prop][name] = item
            checks.append(dict(property=prop, method=name, teacher_predictions_reproduced=True,
                               row_label_fold_alignment=True, actual_student_score_reproduced=True,
                               affine_direction_error_identity=True))
        folder = out / prop
        folder.mkdir()
        np.savez_compressed(folder / 'diagnostic_predictions.npz', row_index=idx, fold=fold,
                            y_transformed=y, teacher_prediction=teacher_prediction,
                            teacher_coef=np.stack(coefficient), teacher_intercept=np.array(intercept),
                            **predictions)
    pin(Path(__file__))
    for path, digest in sources.items():
        assert sha(Path(path)) == digest, path
    write(out / 'results.json', result)
    write(out / 'verification.json', dict(status='passed', checks=checks, source_sha256=sources))
    write(out / 'status.json', dict(status='success', finished_utc=datetime.now(timezone.utc).isoformat()))
    for prop, methods in result.items():
        for name, item in methods.items():
            print(prop, name, {k: round(v, 6) for k, v in item.items() if k.startswith('mean_')}, flush=True)


if __name__ == '__main__':
    main()
