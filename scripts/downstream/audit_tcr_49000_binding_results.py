"""Read-only independent artifact audit of full 49000 T1 tracks.

Activate pllm then protenix_abtcr; no GPU is required. Example:
CUDA_VISIBLE_DEVICES='' python scripts/downstream/audit_tcr_49000_binding_results.py \
  --tracks others_cdr3b cdr3ab others_longab --report /abs/new_report.json
Never modifies predictions, summaries, heads, source datasets or feature caches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from downstream.benchmark.tcr_binding import run_retrained_ours as binding
from scripts.downstream.run_tcr_native_repr import EXPECTED_SHA, CKPT
from scripts.downstream.run_tcr_local_short import now, sha

ROOT = Path(__file__).resolve().parents[2]


def audit(track):
    tag = f'ours_tcr_v5_49000_20260913_{track}'
    output = ROOT / 'downstream/benchmark/outputs/tcr_binding_nm2025_retrained' / tag / track / 'AS'
    config = json.loads((output / 'run_config.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    assert config['checkpoint_sha256'] == EXPECTED_SHA and config['checkpoint'] == str(CKPT)
    assert config['folds_requested'] == summary['folds_expected'] == [1, 2, 3, 4, 5]
    assert config['eval_sets_requested'] == list(binding.EVAL_SETS)
    assert summary['track'] == config['track'] == track
    manifest = config['feature_manifest']
    assert manifest['status'] == 'complete' and manifest['next_index'] == manifest['n_pairs']
    assert manifest['checkpoint_sha256'] == EXPECTED_SHA
    assert manifest['input_protocol']['recognition'] == 'decoder_mask_only_no_gold'
    assert summary['feature_identity'] == binding._feature_identity(manifest)
    frames, provenance = binding.load_protocol('AS', track)
    pairs = binding.build_pair_universe(frames, track)
    assert len(pairs) == manifest['n_pairs']
    lookup = {pair: i for i, pair in enumerate(pairs)}
    features = np.load(manifest['features_path'], mmap_mode='r')
    assert features.shape == (len(pairs), manifest['feature_dim'])
    records, epochs, hashes = [], [], {}
    for fold in binding.FOLDS:
        headpath = output / f'fold_{fold}/head.pt'
        head = torch.load(headpath, map_location='cpu', weights_only=False)
        assert head['checkpoint_sha256'] == EXPECTED_SHA
        assert head['feature_identity'] == summary['feature_identity'] and head['fold'] == fold
        source = provenance['folds'][str(fold)]
        assert head['train_member_sha256'] == source['train']['train_member_sha256']
        train = frames[fold]['train']
        indices = binding.frame_feature_indices(train, lookup, track)
        split, _ = train_test_split(np.arange(len(train)), test_size=head['head_config']['validation_fraction'],
            random_state=head['head_config']['fold_seed'], stratify=train.Affinity.to_numpy())
        selected = np.asarray(features[indices[split]], dtype=np.float32)
        np.testing.assert_allclose(head['feature_mean'], selected.mean(axis=0, dtype=np.float64).astype(np.float32), rtol=0, atol=1e-6)
        assert np.isfinite(selected).all() and np.isfinite(head['feature_std']).all()
        assert (head['feature_std'] > 0).all()
        assert 1 <= head['epochs_run'] <= head['head_config']['max_epochs'] == 200
        epochs.append({'fold': fold, 'epochs_run': head['epochs_run']})
        for name in binding.EVAL_SETS:
            directory = output / f'fold_{fold}' / name
            record = json.loads((directory / 'metrics.json').read_text())
            pred = pd.read_csv(directory / 'predictions.csv', keep_default_na=False)
            source_rows = frames[fold][name]
            assert len(pred) == record['n_input'] == record['n_scored'] == len(source_rows)
            assert pred.row_id.tolist() == list(range(len(source_rows)))
            for saved, raw in [('peptide', 'Epitope'), ('cdr3b', 'CDR3B'), ('label', 'Affinity')]:
                np.testing.assert_array_equal(pred[saved].to_numpy(), source_rows[raw].to_numpy())
            for saved, raw in [('cdr3a','CDR3A'), ('longa','LongA'), ('longb','LongB')]:
                if saved in pred:
                    np.testing.assert_array_equal(pred[saved].astype(str), source_rows[raw].astype(str))
            expected_identity = {'feature_identity': summary['feature_identity'], 'head_sha256': sha(headpath),
                'train_member_sha256': source['train']['train_member_sha256'],
                'test_member_sha256': source['tests'][name]['test_member_sha256'], 'fold': fold, 'eval_set': name}
            assert record['prediction_identity'] == expected_identity
            assert record['model_protocol']['checkpoint_sha256'] == EXPECTED_SHA
            assert np.isfinite(pred.score).all() and pred.score.between(0,1).all()
            metrics = binding.paper_metrics(pred.label.to_numpy(), pred.score.to_numpy(dtype=np.float32))
            for key, value in metrics.items():
                np.testing.assert_allclose(value, record['metrics'][key], rtol=0, atol=2e-6, err_msg=f'{track}/{fold}/{name}/{key}')
            records.append({'fold':fold, 'eval_set':name, 'rows':len(pred), 'metrics':record['metrics']})
            hashes[str(directory / 'predictions.csv')] = sha(directory / 'predictions.csv')
    for name, aggregate in summary['eval_sets'].items():
        assert aggregate['folds_complete'] == [1,2,3,4,5]
        for saved, raw in [('auroc','auroc'), ('auprc_precrec','auprc_precrec'), ('average_precision','average_precision_sklearn')]:
            values = [r['metrics'][raw] for r in records if r['eval_set'] == name]
            assert len(values) == 5
            for suffix, value in [('mean',np.mean(values)), ('std_sample',np.std(values,ddof=1))]:
                np.testing.assert_allclose(value, aggregate['local'][saved+'_'+suffix], rtol=0, atol=1e-12)
    assert len(records) == 15
    return {'status':'passed', 'track':track, 'summary_sha256':sha(output/'summary.json'),
        'run_config_sha256':sha(output/'run_config.json'), 'output':str(output),
        'n_unique_inputs':len(pairs), 'fold_epochs':epochs, 'prediction_files_sha256':hashes,
        'checks':['all five folds and three evaluation sets', 'original row identity and labels',
                  'checkpoint, feature, head and member hashes', 'training-only normalization mean',
                  'prediction CSV metric recomputation', 'five-fold mean and sample std recomputation']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', nargs='+', required=True,
        choices=['cdr3b','cdr3ab','others_cdr3b','others_longab'])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('retain prior audit reports; choose a fresh report path')
    result = {'status':'running', 'start_utc':now(), 'global_step':49000,
        'checkpoint_sha256':EXPECTED_SHA, 'auditor_sha256':sha(__file__), 'tracks':{}}
    try:
        for track in args.tracks:
            result['tracks'][track] = audit(track)
            print('PASS',track,flush=True)
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result['end_utc'] = now()
        result['limits'] = ['artifact consistency is not a pretraining decontamination audit',
            'no new backbone forward or full parameter finetuning', 'baseline protocol parity remains separate']
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as handle:
            json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()
