"""Audit completed spectral continuation without refitting or GPU inference.

Activate pllm/protenix_abtcr and run from the repository root:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/analyze_final_spectral_pilot.py
Uses the pre-existing OAS-only PCA and rotation; no GDP label fitting.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write, sha


def main():
    import numpy as np
    from scipy.stats import spearmanr

    run = ROOT / 'output/diagnostics/final_layer_spectral_pilot_20261018'
    prior = ROOT / 'output/diagnostics/final_layer_coordinate_pilot_20261018'
    depth = ROOT / 'output/diagnostics/fixed50000_depth_20261018'
    assert json.loads((run / 'status.json').read_text())['status'] == 'success'
    plan = json.loads((run / 'plan.json').read_text())
    for path, digest in plan['source_sha256'].items():
        assert sha(Path(path)) == digest, path
    cal = np.load(prior / 'calibration.npz')
    spectrum = np.load(run / 'spectrum.npz')
    axes, eigen = spectrum['axes'], spectrum['eigenvalues']
    teacher = np.load(depth / 'features.npz')['native_block2'].astype(np.float64)
    target = ((teacher - cal['mu_t']) / cal['rms_t']) @ axes.T
    result = {'interpretation': 'OAS-only fixed PCA; GDP descriptive analysis, not new readout or label fitting', 'arms': {}}
    for name, folder in [('coordinate300', prior / 'joint_coordinate')] + [(a, run / a) for a in plan['arms']]:
        verification = json.loads((folder / 'verification.json').read_text())
        assert verification['status'] == 'passed'
        student = np.load(folder / 'final_features.npy').astype(np.float64)
        predicted = ((student - cal['mu_s']) / cal['rms_s']) @ cal['rotation'].T @ axes.T
        bands = {}
        for start, stop in [(0, 16), (16, 64), (64, 256), (256, 768)]:
            a, b = predicted[:, start:stop], target[:, start:stop]
            centered_a, centered_b = a - a.mean(0), b - b.mean(0)
            ka, kb = centered_a @ centered_a.T, centered_b @ centered_b.T
            bands[f'PC{start+1}-{stop}'] = {
                'training_variance_fraction': float(eigen[start:stop].sum() / eigen.sum()),
                'mse_over_GDP_variance': float(np.mean((a-b)**2) / np.mean(centered_b**2)),
                'centered_prediction_cka': float(np.sum(ka*kb) / np.sqrt(np.sum(ka*ka)*np.sum(kb*kb))),
            }
        comparisons = {}
        metrics = json.loads((folder / 'gdp_a1/metrics.json').read_text())
        for prop, metric in metrics['properties'].items():
            saved = np.load(folder / 'gdp_a1' / prop / 'legacy_oof.npz')
            original = np.load(depth / 'native_block2' / prop / 'legacy_oof.npz')
            for key in ['row_index', 'fold', 'y_true', 'y_transformed']:
                np.testing.assert_array_equal(saved[key], original[key])
            scores, teacher_scores = [], []
            for fold in range(5):
                selected = saved['fold'] == fold
                scores.append(float(spearmanr(saved['y_transformed'][selected], saved['prediction'][selected]).statistic))
                teacher_scores.append(float(spearmanr(original['y_transformed'][selected], original['prediction'][selected]).statistic))
            assert abs(np.mean(scores)-metric['legacy_reference']['cv_mean_spearman']) < 1e-12
            comparisons[prop] = {'student_folds': scores, 'teacher_folds': teacher_scores,
                                 'student_minus_teacher': (np.array(scores)-teacher_scores).tolist()}
        result['arms'][name] = {'bands': bands, 'fold_comparisons': comparisons,
                               'features_sha256': sha(folder / 'final_features.npy')}
    result['source_sha256'] = {str(p): sha(p) for p in [Path(__file__), prior / 'calibration.npz', run / 'spectrum.npz', depth / 'features.npz']}
    write(run / 'feature_error_analysis.json', result)
    print(json.dumps({name: item['bands'] for name, item in result['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
