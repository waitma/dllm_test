"""Independent T1–T4 GPU execution/scoring audit, never a quality result.

Run after the task-specific input gates. Original datasets/checkpoints/results
are read-only; use a new --out-dir for every local/cloud validation.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from downstream.grammar.ab_features import file_sha256, write_json
from downstream.grammar.tcr_features import TCRRuntimeEmbedder
from downstream.benchmark.tcr_binding import run_retrained_ours as binding
from downstream.benchmark.tcr_binding.query_protocol import TCRBindingEmbedder, TCRBindingQueryProtocol
from scripts.downstream import run_tcr_native_repr as representation
from scripts.downstream import run_tcr_native_generation as generation
from scripts.downstream.prepare_tcr_native_jobs import ROOT, CKPT, OUT
from scripts.downstream.tcr_eval_pin import PIN


def verify_gates(generation_gate=None):
    representation.assert_checkpoint()
    paths = {name: OUT / f'{directory}/passed.json' for name, directory in
             [('binding', 'binding_preflight'), ('repr', 'repr_preflight_v2'),
              ('generation', 'generation_preflight')]}
    if generation_gate is not None:
        paths['generation'] = Path(generation_gate)
    gates = {key: json.loads(path.read_text()) for key, path in paths.items()}
    assert all(value['status'] == 'passed' for value in gates.values())
    assert gates['binding']['checkpoint_sha256'] == representation.EXPECTED_SHA
    for track, report in gates['binding']['tracks'].items():
        assert TCRBindingQueryProtocol(input_mode=binding.input_mode_for_track(track)).sha256 == report['input_protocol_sha256']
    for task in representation.DATASETS:
        assert representation.identity(task) == gates['repr']['tasks'][task]['identity']
    assert generation.identity() == gates['generation']['identity']
    return {str(path): file_sha256(path) for path in paths.values()}


def data_audit():
    """Check local dataset identities/splits, not pretraining decontamination."""
    result = {}
    frames = {}
    for task in representation.DATASETS:
        split, paths, paired = representation.inputs(task)
        frames[task] = split
        result[task] = {'rows': [len(f) for f in split],
                        'data_sha256': {str(p): file_sha256(p) for p in paths}}
    assert len(frames['t2a'][0]) == 4779 and len(frames['t2b'][0]) == 9033
    for task in ('t3deep', 't3broad'):
        a, b = frames[task]
        keys = lambda f: set(zip(f.cdr3b, f.cdr3a))
        overlap = keys(a) & keys(b)
        result[task].update(pair_overlap=len(overlap), unique_pairs=[len(keys(a)), len(keys(b))])
        assert not overlap, f'{task}: support/background or train/test receptor overlap'
    # Broad intentionally preserves multi-epitope identities rather than picking
    # an arbitrary majority label. Deep currently has one row per paired TCR.
    assert all(len(f) == len(set(zip(f.cdr3b, f.cdr3a))) for f in frames['t3deep'])
    result['t3broad']['multilabel_pairs'] = [int(representation.receptor_groups(f).peptide.map(len).gt(1).sum())
                                           for f in frames['t3broad']]
    entries = json.loads(generation.EVAL.read_text())
    counts = pd.Series([e['category'] for e in entries.values()]).value_counts().to_dict()
    assert counts == {'held20': 20, 'benchmark14': 14}
    assert all(len(e['mhc_pseudo']) == 34 and e['ref_binders'] for e in entries.values())
    result['t4'] = {'categories': counts, 'eval_sha256': file_sha256(generation.EVAL),
        'nonstandard_reference_entries': sum(any(c not in 'ACDEFGHIKLMNPQRSTVWY' for c in s)
            for e in entries.values() for s in e['ref_binders']),
        'reference_policy': 'unchanged; count disclosed; never generator input'}
    result['scientific_limits'] = ['T1/T2/T3 pretraining membership and homology audit not complete',
        'T3 local-control datasets/episodes, not paper-exact reproduction',
        'T4 held20 is not v5-unseen; use existing full-scan overlap strata',
        'baseline native input/training differences remain; smoke does not prove fair ranking']
    return result


def binding_smoke(backbone):
    from sklearn.model_selection import train_test_split
    reports = {}
    others_identity = None
    for track in ('cdr3b', 'cdr3ab', 'others_cdr3b', 'others_longab'):
        frames, provenance = binding.load_protocol('AS', track)
        binding.protocol_overlap_audit(frames, track)
        if track != 'cdr3b':
            # Compare all official rows, not just sampled GPU inputs.
            rows = {f'{fold}/{name}': f[['Epitope', 'CDR3B', 'CDR3A', 'Affinity']].to_json(orient='split', index=False)
                    for fold, splits in frames.items() for name, f in splits.items()}
            if others_identity is None:
                others_identity = rows
            assert rows == others_identity, 'others tracks changed their row universe'
        fold = frames[min(frames)]
        subset = lambda f, n: f.groupby('Affinity', sort=True).head(n).reset_index(drop=True)
        tr, te = subset(fold['train'], 16), subset(fold['seen_test'], 8)
        assert len(tr) == 32 and len(te) == 16
        rows = pd.concat([tr, te], ignore_index=True)
        keys = list(binding._frame_keys(rows, track))
        protocol = TCRBindingQueryProtocol(input_mode=binding.input_mode_for_track(track))
        backbone.batch_size = 16
        wrapper = TCRBindingEmbedder(backbone, protocol)
        x = wrapper.embed_pairs(beta=[k[1] for k in keys],
            alpha=[k[2] for k in keys] if len(keys[0]) == 3 else None,
            peptide=[k[0] for k in keys])
        original = x.copy()
        head = binding.train_head(x, np.arange(len(tr)), tr.Affinity.to_numpy(), seed=0,
            hidden=256, dropout=.3, learning_rate=.001, weight_decay=1e-5,
            batch_size=512, max_epochs=2, patience=2, val_fraction=.25)
        train_indices, _ = train_test_split(np.arange(len(tr)), test_size=.25,
            random_state=0, stratify=tr.Affinity.to_numpy())
        np.testing.assert_array_equal(head['feature_mean'], original[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32))
        np.testing.assert_array_equal(x, original)
        scores = binding.predict_head(head, x, np.arange(len(tr), len(rows)), batch_size=512)
        metrics = binding.paper_metrics(te.Affinity.to_numpy(), scores)
        assert len(scores) == len(te) and all(np.isfinite(v) for v in metrics.values())
        reports[track] = {'train_rows': len(tr), 'test_rows': len(te), 'head_epochs': head['epochs_run'],
            'feature_shape': list(x.shape), 'train_only_scaling': True, 'metrics_finite': True,
            'raw_data_provenance': provenance}
        print('PASS suite binding', track, flush=True)
    return reports


def representation_smoke(backbone, output):
    from downstream.benchmark.tcr_clustering import run as cluster
    from downstream.benchmark.tcr_clustering import run_embed_bench as embed_bench
    from downstream.benchmark.common import fewshot, metrics
    from unittest.mock import patch
    result = {}
    for task in representation.DATASETS:
        frames, _, paired = representation.inputs(task)
        wrapper = TCRRuntimeEmbedder(backbone=backbone, batch_size=8)
        if task == 't3broad':
            a, b = [representation.receptor_groups(f) for f in frames]
            # Select by known support category for coverage, never feed category
            # into embeddings. This reduced smoke is not a reported episode set.
            a, b = a.head(96), b.head(96)
            x, y = [wrapper.embed_pairs(f.cdr3b.tolist(), f.cdr3a.tolist()) for f in (a, b)]
            report = representation.multilabel_fewshot(fewshot.cosine_distance_matrix(y, x), a, b, shots=[1], seeds=2)
            assert report['episodes'] and all(np.isfinite(e['auroc']) for e in report['episodes'])
            # Exercise the built auxiliary probe separately on original labels.
            original = frames[0].groupby('peptide', sort=False).head(4).head(96)
            x = wrapper.embed_pairs(original.cdr3b.tolist(), original.cdr3a.tolist())
            labels = pd.factorize(original.peptide)[0]
            train = original.groupby('peptide', sort=False).cumcount().to_numpy() < 2
            probe = metrics.linear_probe_metrics(x[train], labels[train], x[~train], labels[~train], multiclass=True, seed=0)
            probe.update(metrics.knn_top1_metric(x[train], labels[train], x[~train], labels[~train], metric='cosine'))
            assert probe
            result[task] = {'episodes': len(report['episodes']), 'auxiliary_probe_executed': True}
            continue
        # Basis B's official label column is epitope; A/deep use peptide.
        # Adapt only this diagnostic view; leave source CSVs/production intact.
        source = frames[0].rename(columns={'epitope': 'peptide'}) if task == 't2b' else frames[0]
        eligible = source.groupby('peptide', sort=False).filter(lambda f: len(f) >= 3)
        frame = eligible.groupby('peptide', sort=False).head(3).head(96).copy()
        if task == 't3deep':
            background = frames[1].head(32).copy()
            background['peptide'] = '__background__'
            frame = pd.concat([frame, background], ignore_index=True)
        x = wrapper.embed_pairs(frame.cdr3b.tolist(), frame.cdr3a.tolist() if paired else None)
        if task == 't2a':
            normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
            with (patch.object(cluster, '_embed_features', return_value=(normalized, 'audit-only')),
                  patch.object(cluster, 'OUT', output)):
                report = cluster.run_embed_threshold(frame, 'audit-only', 't2a', False, None)
            curve = pd.read_csv(output / 't2a/curve.csv', float_precision='round_trip')
            for row in curve.itertuples():
                assert (cluster._components_at_threshold(normalized @ normalized.T, row.tau) >= 0).sum() == row.n_clustered
            result[task] = {'n_inputs': len(frame), 'n_curve_points': len(curve),
                           'scoring_protocol': report['scoring_protocol'], 'threshold_roundtrip': True}
        elif task == 't2b':
            labels = embed_bench._cluster_all(x, 'kmeans', [10], seed=0)[10]
            report = embed_bench._score(labels, frame.peptide.to_numpy())
            assert all(np.isfinite(v) for v in report.values())
            result[task] = {'n_inputs': len(frame), 'fixed_k': 10, 'metrics_finite': True}
        else:
            report = fewshot.few_shot_paper_auroc(frame.peptide, frames[0].peptide.unique(),
                universe_vec=x, shots=[1, 2], seeds=2)
            assert all(np.isfinite(v['mean']) for v in report['auroc_by_shot'].values())
            result[task] = {'n_inputs': len(frame), 'shots': [1, 2], 'metrics_finite': True}
        print('PASS suite representation', task, flush=True)
    return result


def generation_smoke(output):
    from downstream.grammar.tcr_generation_v5 import V5BetaSampler
    from downstream.benchmark.tcr_generation_bench import scoring
    sampler = V5BetaSampler(CKPT)
    entries = json.loads(generation.EVAL.read_text())
    reports = {}
    for category in ('benchmark14', 'held20'):
        key, entry = next((k, e) for k, e in entries.items() if e['category'] == category)
        condition = {name: entry[name] for name in ('epitope', 'mhc_pseudo')}
        generated = sampler.generate(condition['epitope'], condition['mhc_pseudo'], key, k=8, batch_size=8, seed=42)
        controls, checks = sampler.sample_records([sampler.protocol.record(condition['epitope'],
            condition['mhc_pseudo'], 12, 'audit-argmax')], seed=42, strategy='argmax')
        assert len(generated) == 8 and all(g['valid'] for g in generated + controls)
        report = scoring.score_conditional({key: [g['sequence'] for g in generated]}, {key: entry},
            k=8, greedy_map={key: controls[0]['sequence']})
        assert report['n_pmhc'] == report['summary']['overall']['n_pmhc_scored'] == 1
        assert all(np.isfinite(report['per_pmhc'][key][m]) for m in ('d_edit', 'seq_recovery', 'precision', 'recall', 'f1'))
        write_json(output / f'{category}_raw_smoke.json', {'quality_evaluation': False,
            'condition': condition, 'candidates': generated, 'argmax': controls, 'checks': checks})
        reports[category] = {'pmhc': key, 'batch_size': 8, 'steps': 32, 'n_raw': 8,
                             'n_argmax': 1, 'scoring_executed': True, 'n_scored': 1}
        print('PASS suite generation', category, flush=True)
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'quality_evaluation': False, 'checkpoint': str(CKPT),
              'global_step': PIN.step, 'source_sha256': file_sha256(Path(__file__))}
    try:
        assert torch.cuda.is_available(), 'GPU required'
        report.update(gpu=torch.cuda.get_device_name(0), input_gates=verify_gates(), datasets=data_audit())
        from downstream.benchmark.common.model_api import FusionGrammarEmbedder
        backbone = FusionGrammarEmbedder(str(CKPT), device='cuda', batch_size=16)
        report['binding'] = binding_smoke(backbone)
        report['representation'] = representation_smoke(backbone, args.out_dir)
        del backbone
        gc.collect()
        torch.cuda.empty_cache()
        report['generation'] = generation_smoke(args.out_dir)
        # Detect source/data drift during the audit, not just at startup.
        assert verify_gates() == report['input_gates']
        report.update(status='passed', peak_gpu_bytes=torch.cuda.max_memory_allocated())
        write_json(args.out_dir / 'passed.json', report)
        print('TCR_SUITE_EXECUTION_SCORING_GATE_PASSED', flush=True)
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(args.out_dir / 'report.json', report)


if __name__ == '__main__':
    main()
