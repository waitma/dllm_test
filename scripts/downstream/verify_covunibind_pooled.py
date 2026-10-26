"""Independently audit pooled CoV-UniBind OOF outputs and one fit per task.

Activate pllm, then run with scipy/sklearn available:
python scripts/downstream/verify_covunibind_pooled.py --run /absolute/run
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from downstream.benchmark.ab_antigen.ridge_cv import join_features, read_rows, sha
from downstream.benchmark.ab_antigen.sequence_data import write_json


def independent_metrics(y, p, task):
    if task == 'spr':
        return {'spearman': float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])}
    positive, negative = p[y == 1], p[y == 0]
    delta = positive[:, None] - negative[None, :]
    auc = float(((delta > 0) + .5 * (delta == 0)).mean())
    order = np.argsort(-p, kind='stable')
    labels, scores = y[order], p[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
    tp = np.cumsum(labels)[ends]
    precision, recall = tp / (ends+1), tp / labels.sum()
    return dict(auroc=auc, auprc=float(np.sum(np.diff(np.r_[0., recall])*precision)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads((args.run/'plan.json').read_text())
    folds = json.loads((args.run/'folds.json').read_text())
    summary = json.loads((args.run/'summary.json').read_text())
    assert summary['status'] == 'complete_available_caches'
    for path, checksum in plan['code'].items():
        assert sha(path) == checksum
    rows = {t: read_rows(d['path']) for t, d in plan['data'].items()}
    pairs = {}
    fold_audit = {}
    for task, items in rows.items():
        assert sha(plan['data'][task]['path']) == plan['data'][task]['sha256']
        assert len(items) == plan['data'][task]['n']
        assert set(folds[task]) == {r['id'] for r in items}
        for r in items:
            key = (r['heavy'], r['light'])
            fold = folds[task][r['id']]
            assert pairs.setdefault(key, fold) == fold
        fold_audit[task] = []
        for fold in range(5):
            train = [r for r in items if folds[task][r['id']] != fold]
            test = [r for r in items if folds[task][r['id']] == fold]
            assert train and test
            assert not ({(r['heavy'], r['light']) for r in train} &
                        {(r['heavy'], r['light']) for r in test})
            if task == 'binding':
                assert {r['label'] for r in train} == {0, 1}
                assert {r['label'] for r in test} == {0, 1}
            fold_audit[task].append(dict(fold=fold, train=len(train), test=len(test), overlap=0))
    report = dict(status='passed', global_pair_fold_isolation=True,
                  folds=fold_audit, models={}, independent_solver={}, verifier_sha256=sha(__file__))
    for name, entry in summary['models'].items():
        results = json.loads(Path(entry['metrics']).read_text())
        report['models'][name] = {}
        for task, metric in results['tasks'].items():
            items = rows[task]
            output_path = args.run/name/f'{task}_oof.json'
            predictions = json.loads(output_path.read_text())
            assert [r['id'] for r in predictions] == [r['id'] for r in items]
            assert len({r['id'] for r in predictions}) == len(items)
            for original, pred in zip(items, predictions):
                for key in ('id', 'source_id', 'lineage', 'antibody_key', 'label'):
                    assert original[key] == pred[key]
                assert pred['fold'] == folds[task][pred['id']]
            y = np.array([r['label'] for r in predictions])
            p = np.array([r['prediction'] for r in predictions])
            assert np.isfinite(p).all()
            recalculated = independent_metrics(y, p, task)
            for key, value in recalculated.items():
                assert np.isclose(value, metric['pooled'][key], rtol=0, atol=1e-12)
                assert value == entry[task][key] or abs(value-entry[task][key]) < 1e-12
            assert sha(metric['cache_path']) == metric['cache_sha256']
            evidence = dict(n=len(items), raw_order_labels_and_multiplicity_preserved=True,
                            metrics_independently_recomputed=recalculated,
                            oof_sha256=sha(output_path), source_cache_sha_unchanged=True)
            if task == 'binding':
                xbb = [r for r in predictions if r['lineage'] == 'XBB']
                assert len(xbb) == 27 and all(r['label'] == 0 for r in xbb)
                evidence['xbb_negative_rows_included'] = len(xbb)
            report['models'][name][task] = evidence
            if task not in report['independent_solver']:
                x, audit = join_features(items, metric['cache_path'])
                assert x is not None and audit['missing_rows'] == 0
                test = np.array([r['fold'] == 0 for r in predictions])
                train = ~test
                mean, scale = x[train].mean(0), x[train].std(0)
                scale[scale == 0] = 1
                ztrain, ztest = (x[train]-mean)/scale, (x[test]-mean)/scale
                # Center a second time for Ridge's fitted intercept, without sklearn.
                center, intercept = ztrain.mean(0), y[train].mean()
                ztrain, ztest = ztrain-center, ztest-center
                gram = ztrain @ ztrain.T + plan['alpha']*np.eye(int(train.sum()))
                dual = np.linalg.solve(gram, y[train]-intercept)
                independent = ztest @ ztrain.T @ dual + intercept
                error = float(np.max(np.abs(independent-p[test])))
                assert error < 1e-7, error
                report['independent_solver'][task] = dict(model=name, fold=0, max_absolute_error=error)
    report['n_models'] = len(report['models'])
    report['n_completed_model_tasks'] = sum(len(v) for v in report['models'].values())
    report['summary_sha256'] = sha(args.run/'summary.json')
    write_json(args.run/'verification.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'n_models', 'n_completed_model_tasks',
                                           'independent_solver')}, indent=2))


if __name__ == '__main__':
    main()
