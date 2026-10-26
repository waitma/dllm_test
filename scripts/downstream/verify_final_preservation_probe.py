"""Independently rescore completed Specificity or m396 prediction artifacts.

Activate pllm/protenix_abtcr; from repository root:
python -B scripts/downstream/verify_final_preservation_probe.py --task specificity --run ABS_TASK_DIR
No model inference or head refitting; checks original source rows and splits.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['specificity', 'm396'])
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    out = args.run.resolve()
    assert json.loads((out / 'status.json').read_text())['status'] == 'success'
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from scipy.stats import spearmanr
    from downstream.grammar.ab_probes import prepare_task, m396_splits, sequence_pairs, json_hash, SCORE_COLS

    df, _, sources, folds = prepare_task(args.task)
    metrics = json.loads((out / 'metrics.json').read_text())
    checks = []
    if args.task == 'specificity':
        all_frames = []
        for k, (train, test) in enumerate(folds):
            frame = pd.read_csv(out / f'fold_{k}/predictions.csv')
            selected = json.loads((out / f'fold_{k}/selection.json').read_text())
            assert selected['train_ids'] == df.name.iloc[train].tolist()
            assert selected['test_ids'] == df.name.iloc[test].tolist()
            assert selected['selected_epoch'] == 100 and selected['rule'] == 'fixed_last_no_test_selection'
            np.testing.assert_array_equal(frame.sample_id, df.name.iloc[test])
            np.testing.assert_array_equal(frame.y_true, df.label.iloc[test])
            np.testing.assert_array_equal(frame.fold, k)
            logits = frame[[f'logit_{c}' for c in range(3)]].to_numpy()
            np.testing.assert_array_equal(frame.y_pred, logits.argmax(1))
            probabilities = np.exp(logits-logits.max(1, keepdims=True))
            probabilities /= probabilities.sum(1, keepdims=True)
            np.testing.assert_allclose(frame[[f'prob_{c}' for c in range(3)]], probabilities, atol=2e-7, rtol=2e-6)
            score = dict(accuracy=accuracy_score(frame.y_true, frame.y_pred),
                         f1_macro=f1_score(frame.y_true, frame.y_pred, average='macro'),
                         mcc=matthews_corrcoef(frame.y_true, frame.y_pred))
            for key, value in score.items():
                assert abs(value-metrics['folds'][k][key]) < 1e-12
            checks.append(score)
            all_frames.append(frame)
        combined = pd.concat(all_frames, ignore_index=True)
        pd.testing.assert_frame_equal(combined, pd.read_csv(out / 'oof_predictions.csv'))
        assert not combined.sample_id.duplicated().any() and len(combined) == len(df)
        for key in checks[0]:
            assert abs(np.mean([s[key] for s in checks])-metrics['mean'][key]) < 1e-12
    else:
        groups = np.array([json_hash(pair) for pair in sequence_pairs(df)])
        y = df[SCORE_COLS].to_numpy(dtype=float)
        splits = list(m396_splits(df, groups, 2023))
        assert len(splits) == len(metrics['splits']) == 12
        for (mode, fraction, train, test), row in zip(splits, metrics['splits']):
            assert row['split_protocol'] == mode and row['train_fraction'] == fraction
            saved = np.load(out / f'{mode}_{fraction:g}/predictions.npz')
            np.testing.assert_array_equal(saved['train_index'], train)
            np.testing.assert_array_equal(saved['test_index'], test)
            target = y[test]
            if mode == 'row_reference':
                target = (target-target.mean(0)) / np.clip(target.std(0), 1e-8, None)
            np.testing.assert_array_equal(saved['y_true'], target)
            scores = [float(spearmanr(target[:, j], saved['prediction'][:, j]).statistic) for j in range(len(SCORE_COLS))]
            assert abs(np.mean(scores)-row['spearman_mean']) < 1e-12
            checks.append(dict(mode=mode, fraction=fraction, spearman_targets=scores, mean=float(np.mean(scores))))
    write(out / 'prediction_verification.json', dict(status='passed', task=args.task, checks=checks,
          original_rows_labels_splits_verified=True, saved_predictions_rescored=True,
          source_sha256={str(p): sha(p) for p in [Path(__file__), *sources, out / 'metrics.json']}))
    print(json.dumps({'status': 'passed', 'task': args.task, 'checks': len(checks)}))


if __name__ == '__main__':
    main()
