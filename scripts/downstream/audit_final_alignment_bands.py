"""Measure fixed OAS PCA-band errors of saved actual final-layer features.

Activate pllm then protenix_abtcr; from the repository root:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/audit_final_alignment_bands.py \
  --out ABS_NEW_DIRECTORY --arm ABS_VERIFIED_ARM [--arm ABS_VERIFIED_ARM ...]
No inference, label fitting, PCA refitting, or new downstream readout. Uses the
existing OAS calibration and original block2 GDP features as geometric targets.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', type=Path, action='append', required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np

    root = ROOT / 'output/diagnostics'
    sources = {}

    def pin(path):
        sources[str(path)] = sha(path)
        return path

    cal = np.load(pin(root / 'final_layer_coordinate_pilot_20261018/calibration.npz'))
    spectrum = np.load(pin(root / 'final_layer_spectral_pilot_20261018/spectrum.npz'))
    teacher = np.load(pin(root / 'fixed50000_depth_20261018/features.npz'))['native_block2'].astype(np.float64)
    axes, eigen = spectrum['axes'], spectrum['eigenvalues']
    target = ((teacher-cal['mu_t'])/cal['rms_t']) @ axes.T
    np.testing.assert_allclose(axes @ axes.T, np.eye(768), atol=1e-10, rtol=0)
    np.testing.assert_allclose(cal['rotation'] @ cal['rotation'].T, np.eye(768), atol=1e-10, rtol=0)
    results = {}
    checks = []
    for raw in args.arm:
        arm = raw.resolve()
        verification = json.loads(pin(arm / 'verification.json').read_text())
        assert verification['status'] == 'passed' and verification['reloaded_features_bitwise_equal']
        assert len(verification['oof_checks']) == 10
        checkpoint = json.loads(pin(arm / 'checkpoint.json').read_text())
        assert sha(pin(Path(checkpoint['parameter_file']))) == checkpoint['sha256']
        for mode, filename in [('native', 'final_features.npy'), ('no_mask_keys', 'final_no_mask_features.npy')]:
            student = np.load(pin(arm / filename)).astype(np.float64)
            assert student.shape == teacher.shape and np.isfinite(student).all()
            predicted = ((student-cal['mu_s'])/cal['rms_s']) @ cal['rotation'].T @ axes.T
            original_error = (student-cal['mu_s'])/cal['rms_s']-((teacher-cal['mu_t'])/cal['rms_t']) @ cal['rotation']
            np.testing.assert_allclose(np.sum((predicted-target)**2), np.sum(original_error**2), atol=1e-8, rtol=1e-10)
            bands = {}
            for start, stop in [(0, 16), (16, 64), (64, 256), (256, 768)]:
                a, b = predicted[:, start:stop], target[:, start:stop]
                ac, bc = a-a.mean(0), b-b.mean(0)
                variance = np.mean(bc**2)
                assert variance > 0
                ka, kb = ac @ ac.T, bc @ bc.T
                bands[f'PC{start+1}-{stop}'] = dict(
                    teacher_oas_variance_fraction=float(eigen[start:stop].sum()/eigen.sum()),
                    mse_over_gdp_teacher_variance=float(np.mean((a-b)**2)/variance),
                    centered_mse_over_gdp_teacher_variance=float(np.mean((ac-bc)**2)/variance),
                    mean_offset_over_gdp_teacher_variance=float(np.mean((a.mean(0)-b.mean(0))**2)/variance),
                    centered_cka=float(np.sum(ka*kb)/np.sqrt(np.sum(ka*ka)*np.sum(kb*kb))))
                item = bands[f'PC{start+1}-{stop}']
                assert abs(item['mse_over_gdp_teacher_variance']-item['centered_mse_over_gdp_teacher_variance']-item['mean_offset_over_gdp_teacher_variance']) < 1e-10
            key = arm.name+'/'+mode
            assert key not in results
            results[key] = dict(checkpoint=str(arm), bands=bands)
            checks.append(dict(readout=key, orthogonal_error_energy_preserved=True, bias_variance_decomposition=True))
    pin(Path(__file__))
    for path, digest in sources.items():
        assert sha(Path(path)) == digest, path
    write(out / 'results.json', results)
    write(out / 'verification.json', dict(status='passed', checks=checks, source_sha256=sources,
          interpretation='Unlabelled geometry on GDP rows; not a new model score, not causal attribution of property signals'))
    write(out / 'status.json', dict(status='success', finished_utc=datetime.now(timezone.utc).isoformat()))
    for name, value in results.items():
        print(name, {k: round(v['centered_mse_over_gdp_teacher_variance'], 6) for k, v in value['bands'].items()})


if __name__ == '__main__':
    main()
