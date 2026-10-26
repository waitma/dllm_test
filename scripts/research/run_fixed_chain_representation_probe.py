"""Screen frozen representations with the same training-only relation-label probe.

Activate pllm/protenix_abtcr; python -m scripts.research.run_fixed_chain_representation_probe
    --plan /absolute/plan.json --recipe contrastive --device 0
This is a dataset-label diagnostic, not a downstream benchmark or affinity reward.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import traceback
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dllm.pipelines.immune_llada.fixed_chain import FixedChainModel
from dllm.pipelines.immune_llada.fixed_chain.post_training import PostTrainingModel
from dllm.pipelines.immune_llada.data.records import BioSeqRecord
from downstream.fixed_chain_eval import FixedChainBackbone
from scripts.research.run_fixed_chain_post_training import sha, write, assign_source_groups, source_of, receptor_cluster


class RelationBlindProbeBackbone(FixedChainBackbone):
    def collator(self, records):
        batch = super().collator(records)
        batch['input_ids'] = batch['input_ids'].masked_fill(
            batch['relation_token_mask'], self.tokenizer.mask_token_id)
        return batch


def select_training_rows(rows, cap=256):
    grouped = defaultdict(list)
    for row in rows:
        label = row.get('labels', {}).get('relation')
        if label in ('binding', 'nonbinding'):
            grouped[(source_of(row), label)].append(row)
    rng, result = random.Random(20261014), []
    for key in sorted(grouped):
        values = grouped[key]
        rng.shuffle(values)
        result.extend(values[:cap])
    return result


def metrics(labels, probabilities, rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[source_of(row)].append(i)
    result = {}
    for source, indices in groups.items():
        y, p = labels[indices], probabilities[indices]
        result[source] = {'n': len(y), 'positives': int(y.sum()),
                          'auroc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                          'brier': float(np.mean((p-y)**2))}
    aucs = [x['auroc'] for x in result.values() if x['auroc'] is not None]
    return {'sources': result, 'macro_auroc': float(np.mean(aucs)) if aucs else None}


def fit_probe(train_vectors, screen_vectors, train_rows, screen_rows):
    labels = np.array([r['labels']['relation'] == 'binding' for r in train_rows], dtype=int)
    truth = np.array([r['labels']['relation'] == 'binding' for r in screen_rows], dtype=int)
    inner = np.array([int(receptor_cluster(r)[:8], 16) % 5 == 0 for r in train_rows])
    if len(set(labels[inner])) != 2 or len(set(labels[~inner])) != 2:
        raise ValueError('Both labels required in training and inner-validation partitions')
    scores = {}
    for c in (.01, .1, 1.):
        head = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight='balanced', max_iter=1000, random_state=42))
        head.fit(train_vectors[~inner], labels[~inner])
        scores[str(c)] = metrics(labels[inner], head.predict_proba(train_vectors[inner])[:, 1],
                                [row for row, flag in zip(train_rows, inner) if flag])['macro_auroc']
    if any(value is None for value in scores.values()):
        raise ValueError('Inner validation lacks a source with both classes')
    best = max(scores, key=lambda c: (scores[c], -float(c)))
    head = make_pipeline(StandardScaler(), LogisticRegression(
        C=float(best), class_weight='balanced', max_iter=1000, random_state=42))
    head.fit(train_vectors, labels)
    probabilities = head.predict_proba(screen_vectors)[:, 1]
    shuffled = labels.copy()
    rng = np.random.default_rng(9173)
    for source in sorted({source_of(row) for row in train_rows}):
        indices = [i for i, row in enumerate(train_rows) if source_of(row) == source]
        shuffled[indices] = rng.permutation(shuffled[indices])
    null_head = make_pipeline(StandardScaler(), LogisticRegression(
        C=float(best), class_weight='balanced', max_iter=1000, random_state=42))
    null_head.fit(train_vectors, shuffled)
    return {'selected_C': float(best), 'inner_training_auroc': scores,
            'screen': metrics(truth, probabilities, screen_rows),
            'shuffled_training_labels': metrics(truth, null_head.predict_proba(screen_vectors)[:, 1], screen_rows),
            'predictions': [{'source': source_of(row), 'cluster': receptor_cluster(row),
                             'label': int(y), 'probability': float(p)}
                            for row, y, p in zip(screen_rows, truth, probabilities)]}


def run(path, name, device_index):
    plan = json.loads(path.read_text())
    for file, expected in plan['source_hashes'].items():
        if sha(file) != expected:
            raise ValueError('Frozen probe source changed')
    spec = plan['models'][name]
    for filename, expected in spec['files'].items():
        if sha(Path(spec['checkpoint']) / filename) != expected:
            raise ValueError('Probe model changed')
    if sha(plan['data_manifest']) != plan['data_manifest_sha256']:
        raise ValueError('Pilot manifest changed')
    manifest = json.loads(Path(plan['data_manifest']).read_text())
    rows = {}
    for split in ('train', 'screen'):
        ref = manifest['files'][split]
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Pilot rows changed')
        rows[split] = assign_source_groups([json.loads(x) for x in Path(ref['path']).read_text().splitlines()],
                                          manifest['source_counts'], split)
    train = select_training_rows(rows['train'])
    screen = [r for r in rows['screen'] if r.get('labels', {}).get('relation') in ('binding', 'nonbinding')]
    out = Path(plan['output']) / name
    out.mkdir(exist_ok=False)
    write(out / 'status.json', {'status': 'running'})
    try:
        torch.set_num_threads(2)
        torch.cuda.set_device(device_index)
        device = torch.device('cuda', device_index)
        cls = FixedChainModel if spec.get('parent') else PostTrainingModel
        model = cls.from_pretrained(spec['checkpoint']).to(device).eval().requires_grad_(False)
        backbone = RelationBlindProbeBackbone(model, {'step': spec.get('step', 1000)})
        # Metadata labels remain available for scoring; the adapter must hide them.
        records = [BioSeqRecord.from_prepared_dict(row) for row in train + screen]
        batch = backbone.collator(records[:8])
        for token in ('<binding>', '<nonbinding>'):
            if batch['input_ids'].eq(model.tokenizer.special_id(token)).any():
                raise ValueError('Relation label leaked into representation input')
        vectors = backbone.vectors(records)
        result = fit_probe(vectors[:len(train)], vectors[len(train):], train, screen)
        result.update(model=name, train_rows=len(train), screen_rows=len(screen),
                      frozen_model_sha256=spec['files']['model.safetensors'],
                      scope='source-balanced pilot label diagnostic; no confirmation/test access',
                      head_selection='C chosen on receptor-cluster partition of training rows only')
        write(out / 'metrics.json', result)
        np.savez_compressed(out / 'vectors.npz', train=vectors[:len(train)], screen=vectors[len(train):])
        write(out / 'status.json', {'status': 'success', 'model_optimizer_updates': 0})
    except Exception:
        write(out / 'status.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    run(args.plan, args.recipe, args.device)
