#!/usr/bin/env python
"""Evaluate frozen BioSeq on unmodified UpTCR splits with official validation.

Activate pllm, then run this file with --checkpoint PATH --output PATH.
The initial input track is FV alpha/beta + peptide (no HLA). All original rows,
labels and partitions are retained. Backbone features resume by source row;
only the MLP head trains, using validation loss for checkpoint selection.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

BENCH = Path(__file__).resolve().parents[1]
ROOT = BENCH.parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT))

from common.model_api import build_embedder
from tcr_binding.query_protocol import TCRBindingEmbedder, TCRBindingQueryProtocol
from tcr_binding.retrained_protocol import sha256_file
from tcr_binding.run_retrained_ours import (
    _atomic_json, _atomic_csv, _model_assets, train_head, predict_head,
)

DEFAULT_DATA = ROOT / 'output/research/uptcr_20260930/released_tables/data/finetune/abpm'
COUNTS = {'majority': (88315, 15585, 25993), 'few-shot': (13345, 2356, 4044),
          'zero-shot': (132180, 23326, 8958)}
SPLITS = ('train', 'valid', 'test')


def load_frames(data: Path, setting: str):
    """Read all source rows in order; validation never drops or relabels rows."""
    frames, provenance = {}, {}
    for split, expected in zip(SPLITS, COUNTS[setting]):
        path = data / setting / f'{split}.csv'
        frame = pd.read_csv(path, keep_default_na=False)
        if len(frame) != expected:
            raise ValueError(f'{path}: expected {expected} rows, got {len(frame)}')
        label = 'Label' if 'Label' in frame else 'label'
        required = ['Epitope', 'FV_alpha', 'FV_beta', 'TRA.CDR3', 'TRB.CDR3', 'MHCseq']
        if not set(required + [label]).issubset(frame.columns):
            raise ValueError(f'{path}: required columns missing')
        if not frame[label].isin([0, 1]).all():
            raise ValueError(f'{path}: invalid binary labels')
        for column in required:
            if not frame[column].map(lambda x: isinstance(x, str) and bool(x)).all():
                raise ValueError(f'{path}: empty/non-string {column}')
        frames[split] = (frame, label)
        provenance[split] = {'path': str(path.resolve()), 'sha256': sha256_file(path),
                             'rows': len(frame), 'positive': int(frame[label].sum())}
    return frames, provenance


def features(frame, split, folder, embedder, identity, chunk_size):
    """Persist every source row, including duplicates, with resumable progress."""
    path = folder / f'{split}_features.npy'
    state_path = folder / f'{split}_features.json'
    manifest = {'identity': identity, 'split': split, 'rows': len(frame),
                'dimension': int(embedder.hidden), 'next_row': 0, 'status': 'extracting'}
    if state_path.exists():
        old = json.loads(state_path.read_text())
        for key in ('identity', 'split', 'rows', 'dimension'):
            if old[key] != manifest[key]:
                raise RuntimeError(f'feature provenance mismatch: {key}')
        manifest = old
        values = np.load(path, mmap_mode='r+')
        if values.shape != (len(frame), embedder.hidden) or values.dtype != np.float32:
            raise RuntimeError('invalid feature cache shape/dtype')
    else:
        if path.exists():
            raise RuntimeError('feature file without manifest; use another output directory')
        values = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                         shape=(len(frame), embedder.hidden))
        _atomic_json(state_path, manifest)
    start_time = time.time()
    initial = manifest['next_row']
    for start in range(initial, len(frame), chunk_size):
        end = min(start + chunk_size, len(frame))
        batch = frame.iloc[start:end]
        while True:
            try:
                vectors = embedder.embed_pairs(beta=batch.FV_beta.tolist(),
                    alpha=batch.FV_alpha.tolist(), peptide=batch.Epitope.tolist())
                break
            except torch.cuda.OutOfMemoryError:
                if embedder.batch_size == 1:
                    raise
                embedder._cache.clear()
                torch.cuda.empty_cache()
                embedder.batch_size = max(1, embedder.batch_size // 2)
                print(f'OOM retry with batch={embedder.batch_size}', flush=True)
        if vectors.shape != (end-start, embedder.hidden) or not np.isfinite(vectors).all():
            raise RuntimeError('invalid model features')
        values[start:end] = vectors
        values.flush()
        embedder._cache.clear()
        manifest.update(next_row=end, effective_batch=embedder.batch_size,
                        status='complete' if end == len(frame) else 'extracting')
        _atomic_json(state_path, manifest)
        print(f'{folder.name}/{split}: {end}/{len(frame)}; '
              f'{(end-initial)/max(time.time()-start_time, 1e-6):.2f} rows/s', flush=True)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if min(args.batch_size, args.chunk_size) < 1:
        parser.error('batch/chunk sizes must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    protocol = TCRBindingQueryProtocol(input_mode='longab')
    datasets = {}
    for setting in ('few-shot', 'majority', 'zero-shot'):
        frames, provenance = load_frames(args.data, setting)
        # Validate runtime compatibility without changing any source records.
        for split, (frame, _) in frames.items():
            for row in frame.itertuples(index=False):
                protocol.record(row.Epitope, row.FV_beta, row.FV_alpha)
        datasets[setting] = (frames, provenance)
        print(f'validated {setting}: {provenance}', flush=True)
    if args.validate_only:
        _atomic_json(args.output / 'input_validation.json', {
            'status': 'passed', 'source': {k: v[1] for k, v in datasets.items()},
            'protocol': protocol.provenance, 'filtered_rows': 0})
        return
    checkpoint_sha = sha256_file(args.checkpoint / 'model.safetensors')
    identity = {'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_sha,
                'model_assets': _model_assets(args.checkpoint), 'protocol': protocol.provenance,
                'runner_sha256': sha256_file(Path(__file__)),
                'head_code_sha256': sha256_file(Path(__file__).with_name('run_retrained_ours.py')),
                'seed': args.seed, 'input': 'FV_alpha+FV_beta+peptide; HLA not used',
                'method': 'frozen backbone + MLP; original validation loss selection'}
    status_path = args.output / 'status.json'
    _atomic_json(status_path, {'status': 'loading_backbone', **identity})
    backbone = build_embedder(f'grammar:decoder:global:{args.checkpoint}',
                             batch_size=args.batch_size, max_length=1024)
    embedder = TCRBindingEmbedder(backbone, protocol)
    embedder.model.eval()
    for parameter in embedder.model.parameters():
        parameter.requires_grad_(False)
    all_results = {}
    for setting, (frames, provenance) in datasets.items():
        folder = args.output / setting
        folder.mkdir(exist_ok=True)
        run_identity = {**identity, 'source': provenance}
        manifest_path = folder / 'run_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != run_identity:
            raise RuntimeError('run provenance mismatch')
        _atomic_json(manifest_path, run_identity)
        metric_path = folder / 'metrics.json'
        if metric_path.exists():
            all_results[setting] = json.loads(metric_path.read_text())
            continue
        _atomic_json(status_path, {'status': 'extracting', 'setting': setting, **identity})
        cache = {split: features(frame, split, folder, embedder, run_identity, args.chunk_size)
                 for split, (frame, _) in frames.items()}
        _atomic_json(status_path, {'status': 'training_head', 'setting': setting, **identity})
        x = np.concatenate([cache['train'], cache['valid']])
        y = np.concatenate([frames[s][0][frames[s][1]].to_numpy(dtype=np.int64)
                            for s in ('train', 'valid')])
        validation = np.arange(len(y)) >= len(frames['train'][0])
        head = train_head(x, np.arange(len(y)), y, seed=args.seed, hidden=256, dropout=0.3,
            learning_rate=1e-3, weight_decay=1e-5, batch_size=512, max_epochs=200,
            patience=15, val_fraction=0.1, validation_mask=validation)
        torch.save(head, folder / 'head.pt')
        del x
        scores = predict_head(head, cache['test'], np.arange(len(cache['test'])), batch_size=1024)
        test, label = frames['test']
        if not np.isfinite(scores).all():
            raise RuntimeError('nonfinite predictions')
        predictions = test.copy()
        predictions.insert(0, 'source_row_index', np.arange(len(test)))
        predictions['prediction'] = scores
        _atomic_csv(folder / 'predictions.csv', predictions)
        result = {'status': 'complete', 'setting': setting, 'n_test': len(test),
                  'auroc': float(roc_auc_score(test[label], scores)),
                  'auprc_average_precision': float(average_precision_score(test[label], scores)),
                  'best_val_loss': head['best_val_loss'], 'epochs_run': head['epochs_run'],
                  'n_train': head['n_train'], 'n_validation': head['n_validation'],
                  'input': identity['input'], 'checkpoint_sha256': checkpoint_sha,
                  'evidence': 'local_evaluation', 'baseline_comparison': 'paper_reported_reference'}
        _atomic_json(metric_path, result)
        all_results[setting] = result
        _atomic_json(args.output / 'summary.json', all_results)
        print(json.dumps(result), flush=True)
        del cache, head
        gc.collect()
        torch.cuda.empty_cache()
    _atomic_json(status_path, {'status': 'complete', 'results': all_results, **identity})


if __name__ == '__main__':
    main()
