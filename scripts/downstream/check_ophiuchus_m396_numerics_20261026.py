"""Check m396 Ridge precision without replacing the frozen evaluation results.

Activate pllm then protenix_abtcr; run this absolute script path on CPU.
All 12 saved splits and alpha=0.01 are retained, with the FP32 feature
differences cast to float64 before fitting. Output is a diagnostic JSON.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
RUN = ROOT / 'output/downstream_generation/ophiuchus_probe_audit_20261026'
sys.path.insert(0, str(ROOT))
from downstream.grammar.ab_probes import prepare_task
from downstream.ophiuchus_eval.affinity_m396 import SCORE_COLS


def main():
    folder = RUN / 'm396'
    metric_bytes = (folder / 'metrics.json').read_bytes()
    metrics = json.loads(metric_bytes)
    features = np.load(folder / 'features.npy', mmap_mode='r')
    delta = (features[:1] - features[1:]).astype(np.float64)
    df, _, _, _ = prepare_task('m396')
    y = df[SCORE_COLS].to_numpy(float)
    rows = []
    for result in metrics['splits']:
        mode, frac = result['split_protocol'], result['train_fraction']
        saved = np.load(folder / f'{mode}_{frac:g}' / 'predictions.npz')
        tr, te = saved['train_index'], saved['test_index']
        mean, std = y[tr].mean(0), np.clip(y[tr].std(0), 1e-8, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            head = Ridge(alpha=0.01).fit(delta[tr], (y[tr] - mean) / std)
            pred = head.predict(delta[te])
        if mode == 'sequence_grouped':
            pred = pred * std + mean
        scores = [float(spearmanr(saved['y_true'][:, j], pred[:, j]).statistic) for j in range(5)]
        assert np.isfinite(scores).all() and np.isfinite(pred).all()
        row = dict(protocol=mode, train_fraction=frac, float64_spearman_mean=float(np.mean(scores)),
                   float64_spearman_per_target=scores, original_spearman_mean=result['spearman_mean'],
                   mean_difference=float(np.mean(scores) - result['spearman_mean']),
                   prediction_max_abs_difference=float(np.abs(pred - saved['prediction']).max()),
                   warnings=[str(w.message) for w in caught])
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert len(rows) == 12
    report = dict(status='complete', diagnostic_only=True, alpha=0.01, splits=rows,
                  max_abs_mean_difference=max(abs(row['mean_difference']) for row in rows),
                  metrics_sha256=hashlib.sha256(metric_bytes).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checked_utc=datetime.now(timezone.utc).isoformat(),
                  note='Float64 readout control on the same FP32 feature differences; no result replacement or new hyperparameter selection.')
    assert (folder / 'metrics.json').read_bytes() == metric_bytes
    (folder / 'float64_readout_diagnostic.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
