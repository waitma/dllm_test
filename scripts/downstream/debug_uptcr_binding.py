"""Activate pllm/protenix_abtcr; python -B SCRIPT --output ABS [--inspect-run ABS].

Read-only audit of saved UpTCR predictions and an optional fixed-chain CPU
forward. Writes a separate JSON report; never retrains or changes benchmark rows.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'output/research/uptcr_20260930/released_tables/data/finetune/abpm'
RUNS = ['v5_native_best_20261008', 'v5_native_best138000_20261009',
        'v5_native_latest269000_20261009', 'v5_native_270m_best32000_20261012',
        'fixed_chain_75m_step50000_20261013']
INPUT = ['Epitope', 'FV_beta', 'FV_alpha']


def saved_predictions():
    reports = {}
    for setting in ('majority', 'few-shot', 'zero-shot'):
        source = pd.read_csv(DATA / setting / 'test.csv', keep_default_na=False)
        label = 'Label' if 'Label' in source else 'label'
        y = source[label].to_numpy(dtype=int)
        lengths = source[INPUT].map(len)
        meta = dict(rows=len(source), positive=int(y.sum()),
                    positive_fraction=float(y.mean()),
                    mean_peptide_pool_weight=float((lengths.Epitope / lengths.sum(axis=1)).mean()),
                    input_conflicting_label_groups=int(source.groupby(INPUT)[label].nunique().gt(1).sum()))
        results = {}
        for run in RUNS:
            folder = ROOT / 'output/downstream_generation' / run / 'uptcr/protocol_outputs' / setting
            path = folder / 'predictions.csv'
            if not path.is_file():
                results[run] = {'status': 'missing_predictions'}
                continue
            pred = pd.read_csv(path, keep_default_na=False)
            assert len(pred) == len(source), path
            assert np.array_equal(pred.source_row_index, np.arange(len(source))), path
            for column in source.columns:
                assert np.array_equal(pred[column].astype(str), source[column].astype(str)), (path, column)
            scores = pred.prediction.to_numpy(dtype=float)
            assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(), path
            metrics = json.loads((folder / 'metrics.json').read_text())
            auc = float(roc_auc_score(y, scores))
            ap = float(average_precision_score(y, scores))
            assert abs(auc - metrics['auroc']) < 1e-6, path
            assert abs(ap - metrics['auprc_average_precision']) < 1e-6, path
            within = [roc_auc_score(g[label], g.prediction) for _, g in pred.groupby('Epitope')
                      if g[label].nunique() == 2]
            per_input = pred.groupby(INPUT).prediction.agg(['min', 'max'])
            results[run] = dict(
                status='verified', auroc=auc, average_precision=ap,
                auc_reversed=float(roc_auc_score(y, -scores)),
                f1_at_05=float(f1_score(y, scores >= .5, zero_division=0)),
                confusion_at_05=confusion_matrix(y, scores >= .5).tolist(),
                predicted_positive_fraction_at_05=float((scores >= .5).mean()),
                score_quantiles=np.quantile(scores, [0, .01, .1, .5, .9, .99, 1]).tolist(),
                unique_scores=int(len(np.unique(scores))),
                positive_score_mean=float(scores[y == 1].mean()),
                negative_score_mean=float(scores[y == 0].mean()),
                within_epitope_macro_auroc=float(np.mean(within)),
                eligible_epitopes=len(within),
                max_identical_input_score_range=float((per_input['max'] - per_input['min']).max()),
                n_train=metrics['n_train'], n_validation=metrics['n_validation'],
                epochs_run=metrics['epochs_run'], best_val_loss=metrics['best_val_loss'],
                head_retained=(folder / 'head.pt').exists(),
                feature_retained=(folder / 'test_features.npy').exists())
        reports[setting] = dict(data=meta, runs=results)
    return reports


def inspect_fixed_chain(run):
    # Load the pinned implementation BEFORE importing model/data classes.
    import torch
    torch.set_num_threads(4)
    from downstream.fixed_chain_eval import load, FixedChainBackbone
    model, identity = load(run / 'checkpoint', device='cpu')
    from downstream.benchmark.tcr_binding.query_protocol import (
        TCRBindingQueryProtocol, observed_residue_pool)
    from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
    backbone = FixedChainBackbone(model, identity, batch_size=2)
    protocol = TCRBindingQueryProtocol(input_mode='longab')
    source = pd.read_csv(DATA / 'zero-shot/test.csv', keep_default_na=False)
    # One real negative and one real positive; synthetic variants are diagnostics only.
    label = 'Label' if 'Label' in source else 'label'
    selected = source.groupby(label, sort=True).head(1)
    records = [protocol.record(row.Epitope, row.FV_beta, row.FV_alpha)
               for row in selected.itertuples()]
    tok = model.tokenizer

    def forward(items):
        batch = protocol.collate(items, backbone.collator, decoder_mask_token_id=tok.mask_token_id)
        hidden, _ = backbone._final_hidden(batch)
        return batch, hidden, observed_residue_pool(hidden, batch)

    batch, hidden, pooled = forward(records)
    assert all(not r.labels for r in records)
    assert 'labels' not in batch
    assert batch['relation_token_mask'].sum(1).eq(1).all()
    assert batch['input_ids'][batch['relation_token_mask']].eq(tok.mask_token_id).all()
    items = []
    for i, (idx, row) in enumerate(selected.iterrows()):
        lengths = {role: len(row[col]) for role, col in
                   [('peptide', 'Epitope'), ('tcr_beta', 'FV_beta'), ('tcr_alpha', 'FV_alpha')]}
        counts, positions = {}, {}
        for role in lengths:
            mask = (batch['role_ids'][i] == CHAIN_ROLE_TO_ID[role]) & batch['residue_mask'][i]
            counts[role] = int(mask.sum())
            assert counts[role] == lengths[role], (role, counts, lengths)
            ids = batch['input_ids'][i, mask].tolist()
            col = {'peptide': 'Epitope', 'tcr_beta': 'FV_beta', 'tcr_alpha': 'FV_alpha'}[role]
            assert ids == tok.encode_residues(row[col]), role
            positions[role] = mask.nonzero().flatten().tolist()
        items.append(dict(source_row_index=int(idx), gold_label=int(row[label]),
                          peptide=row.Epitope, FV_beta=row.FV_beta, FV_alpha=row.FV_alpha,
                          residue_counts=counts, residue_positions=positions,
                          attended_tokens=int(batch['attention_mask'][i].sum()),
                          relation_position=batch['relation_token_mask'][i].nonzero().flatten().tolist(),
                          tokens=[tok.token(v) for v in batch['input_ids'][i].tolist()],
                          feature_norm=float(pooled[i].norm())))
    _, _, reversed_pool = forward(records[::-1])
    single = torch.cat([forward([r])[2] for r in records])
    reverse_delta = float((pooled - reversed_pool.flip(0)).abs().max())
    batch_delta = float((pooled - single).abs().max())
    assert reverse_delta < 1e-4 and batch_delta < 1e-4, (reverse_delta, batch_delta)
    # Same-length peptide perturbation: checks peptide can affect receptor features.
    row = selected.iloc[0]
    peptide = row.Epitope[::-1]
    variant = protocol.record(peptide, row.FV_beta, row.FV_alpha)
    vb, vh, vp = forward([variant])
    role_deltas = {}
    for role in ('peptide', 'tcr_beta', 'tcr_alpha'):
        orig_mask = batch['role_ids'][0].eq(CHAIN_ROLE_TO_ID[role]) & batch['residue_mask'][0]
        new_mask = vb['role_ids'][0].eq(CHAIN_ROLE_TO_ID[role]) & vb['residue_mask'][0]
        orig, new = hidden[0, orig_mask].float().mean(0), vh[0, new_mask].float().mean(0)
        role_deltas[role] = dict(l2_delta=float((orig - new).norm()),
                                relative_l2_delta=float((orig - new).norm() / orig.norm()))
    return dict(identity=identity, device='cpu_float32', examples=items,
                hidden_shape=list(hidden.shape), pooled_shape=list(pooled.shape),
                frozen_parameters=all(not p.requires_grad for p in model.parameters()),
                batch_max_abs_delta=batch_delta, reverse_max_abs_delta=reverse_delta,
                peptide_perturbation=dict(peptide=peptide, role_deltas=role_deltas,
                                         pooled_l2_delta=float((pooled[0] - vp[0]).norm())),
                limitation='Two-sample CPU wiring check; no trained head replay and no quality ablation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--inspect-run', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  saved_predictions=saved_predictions())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Saved prediction audit:', args.output, flush=True)
    if args.inspect_run:
        report['cpu_input_forward'] = inspect_fixed_chain(args.inspect_run)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('CPU input/forward checks passed', flush=True)


if __name__ == '__main__':
    main()
