"""Run one frozen post-training trial, with independent seeds and evaluation RNG.

Activate pllm/protenix_abtcr, then:
python -m scripts.research.run_fixed_chain_post_training --plan /absolute/plan.json --recipe task_mix
The pilot uses existing train/valid rows; it never opens benchmark or confirmation data.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback

import torch
from dllm.pipelines.immune_llada.fixed_chain import FixedChainConfig, FixedChainCollator
from dllm.pipelines.immune_llada.fixed_chain.post_training import PostTrainingModel, RECIPES
from dllm.pipelines.immune_llada.fixed_chain.post_training_data import PostTrainingCollator, observed_end_batch, relation_blind_features
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from scripts.research.fixed_chain_validation import selected_token_metrics, summarize, source_macro, preserved_rng


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_model(checkpoint, recipe):
    from safetensors.torch import load_file
    config = FixedChainConfig(**json.loads((checkpoint / 'immune_config.json').read_text()))
    model = PostTrainingModel(config, recipe)
    expected = [model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]
    if json.loads((checkpoint / 'vocabulary.json').read_text()) != expected:
        raise ValueError('Parent vocabulary differs')
    model.load_state_dict(load_file(str(checkpoint / 'model.safetensors')), strict=True)
    model.install_adaptation()
    return model


def device_batch(collator, rows, device):
    return {k: v.to(device) for k, v in collator(rows).items()}


def assign_source_groups(rows, counts, split):
    """Use shard provenance: tcr_papers records also call themselves tcr_native."""
    cursor = 0
    for key, count in counts.items():
        name, source = key.split('/', 1)
        if name != split:
            continue
        for row in rows[cursor:cursor + count]:
            row['_exploration_source'] = source
        cursor += count
    if cursor != len(rows):
        raise ValueError('Source ranges do not cover pilot rows')
    return rows


def source_of(row):
    return row.get('_exploration_source', row['source'])


def receptor_cluster(row):
    roles = {'antibody_heavy', 'antibody_light', 'tcr_beta', 'tcr_alpha'}
    receptors = sorted((role, sequence) for role, sequence in zip(row['chain_roles'], row['chains'])
                       if role in roles)
    return hashlib.sha256(json.dumps(receptors, separators=(',', ':')).encode()).hexdigest()


class PreferenceSampler:
    """Independent source-balanced sampler; fixed shuffled label per training pair."""
    def __init__(self, pairs, seed):
        self.rng = random.Random(seed)
        self.grouped = defaultdict(list)
        direction_rng = random.Random(9173)
        for pair in pairs:
            self.grouped[pair['source']].append((pair, direction_rng.choice((-1, 1))))
        self.sources = sorted(self.grouped)

    def sample(self, collator, device, count=2, shuffle=False):
        chosen = [self.rng.choice(self.grouped[self.rng.choice(self.sources)]) for _ in range(count)]
        rows = [row for pair, _ in chosen for row in (pair['chosen'], pair['rejected'])]
        return {'preference_batch': device_batch(collator, rows, device),
                'preference_directions': torch.tensor([sign for _, sign in chosen], device=device)
                if shuffle else None}


@torch.no_grad()
def preference_probe(model, reference, rows, device):
    """Small held-out matched-context diagnostic, never a biological reward claim."""
    from scripts.research.prepare_fixed_chain_preferences import signature, binding_condition
    from dllm.pipelines.immune_llada.fixed_chain.preference import pair_features, normalized_nll
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        label = row.get('labels', {}).get('relation')
        key, identity, context = signature(row)
        if context and label in ('binding', 'nonbinding'):
            grouped[(source_of(row), key)][label][identity] = row
    was_training = model.training
    model.eval()
    scores, skipped = [], 0
    collator = PostTrainingCollator(model.config)
    for (source, key), values in sorted(grouped.items()):
        conflict = values['binding'].keys() & values['nonbinding'].keys()
        positives = [row for identity, row in values['binding'].items() if identity not in conflict]
        negatives = [row for identity, row in values['nonbinding'].items() if identity not in conflict]
        if not positives or not negatives:
            continue
        for i in range(min(32, max(len(positives), len(negatives)))):
            pair = [binding_condition(positives[i % len(positives)]),
                    binding_condition(negatives[i % len(negatives)])]
            batch = device_batch(collator, pair, device)
            try:
                features, batch = pair_features(model, batch)
            except ValueError:
                skipped += 1
                continue
            with torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext():
                current = normalized_nll(model.denoise(**features)[0], batch)
                baseline = normalized_nll(reference.denoise(**features)[0], batch)
            scores.append({'source': source, 'group_sha256': hashlib.sha256(key.encode()).hexdigest(),
                           'chosen_nll': float(current[0]), 'rejected_nll': float(current[1]),
                           'parent_chosen_nll': float(baseline[0]), 'parent_rejected_nll': float(baseline[1]),
                           'relative_margin': float((baseline-current)[0]-(baseline-current)[1])})
    model.train(was_training)
    return {'pairs': scores, 'skipped_feature_mismatch': skipped,
            'scope': 'small screening-only weak-label matched pairs; not affinity validation'}


@torch.no_grad()
def evaluate(model, rows, device, batch_size=8, semantic=False, collect_records=False):
    """Four corruption levels plus both conditional-chain directions; equal source weights."""
    was_training = model.training
    model.eval()
    collator = PostTrainingCollator(model.config)
    grouped = defaultdict(list)
    for row in rows:
        grouped[source_of(row)].append(row)
    results = {}
    with preserved_rng([device.index] if device.type == 'cuda' else []):
        torch.manual_seed(9173)
        views = ('mask25', 'mask50', 'mask75', 'mask100', 'first_chain', 'second_chain')
        if semantic:
            views += tuple('observed_' + view for view in views) + ('cdr',)
        for view in views:
            totals, records = {}, {}
            for source, items in grouped.items():
                accum = []
                for start in range(0, len(items), batch_size):
                    batch = device_batch(collator, items[start:start + batch_size], device)
                    kind = view.removeprefix('observed_')
                    ratio = float(kind[4:]) / 100 if kind.startswith('mask') else 1.0
                    if view.startswith('observed_') or view == 'cdr':
                        batch = observed_end_batch(batch, model.tokenizer.mask_token_id)
                        evaluable = torch.zeros_like(batch['target_slot_mask'])
                        for r in range(len(batch['input_ids'])):
                            for uid in batch['chain_ids'][r][batch['observed_terminal_mask'][r]].unique():
                                evaluable[r] |= batch['chain_ids'][r].eq(uid)
                        batch['endpoint_evaluable_mask'] = evaluable
                    if view == 'cdr':
                        batch['eligible_mask'] &= batch['cdr_mask']
                        batch['labels'] = batch['labels'].masked_fill(~batch['eligible_mask'], -100)
                        batch['terminal_mask'] = torch.zeros_like(batch['terminal_mask'])
                        batch['endpoint_evaluable_mask'].zero_()
                    if kind.endswith('_chain'):
                        # Reveal the other chain, and score only the requested chain.
                        keep = torch.zeros_like(batch['eligible_mask'])
                        for r in range(len(batch['input_ids'])):
                            uids = batch['chain_ids'][r][batch['eligible_mask'][r]].unique(sorted=True)
                            if len(uids) >= 2:
                                keep[r] = batch['chain_ids'][r].eq(uids[0 if kind == 'first_chain' else -1])
                        if not keep.any():
                            continue
                        batch['eligible_mask'] &= keep
                        batch['labels'] = batch['labels'].masked_fill(~keep, -100)
                        batch['target_slot_mask'] &= keep
                        batch['terminal_mask'] &= keep
                    with torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext():
                        totals_batch = selected_token_metrics(model, batch, ratio)['sample_totals']
                    accum.append(totals_batch.cpu().sum(0))
                    if collect_records:
                        records.setdefault(source, []).extend(
                            {'row': start+i, 'cluster': receptor_cluster(item), 'totals': values}
                            for i, (item, values) in enumerate(zip(
                                items[start:start + batch_size], totals_batch.cpu().tolist())))
                if accum:
                    totals[source] = summarize(torch.stack(accum).sum(0))
            results[view] = {'macro': source_macro(totals), 'sources': totals} if totals else None
            if collect_records and results[view] is not None:
                results[view]['records'] = records
    model.train(was_training)
    return results


@torch.no_grad()
def binding_probe(model, rows, device, mask_condition=False):
    """Dataset-label prediction with all binding/nonbinding tokens hidden."""
    from sklearn.metrics import roc_auc_score
    was_training = model.training
    model.eval()
    results = defaultdict(list)
    collator = PostTrainingCollator(model.config)
    for start in range(0, len(rows), 8):
        items = rows[start:start + 8]
        batch = observed_end_batch(device_batch(collator, items, device), model.tokenizer.mask_token_id)
        with torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext():
            features = relation_blind_features(model, batch)
            if mask_condition:
                ids = features['input_ids']
                condition_residues = (~features['target_slot_mask'] & model.allowed_target[ids]
                                      & ids.ne(model.terminal_id))
                features['input_ids'] = ids.masked_fill(condition_residues, model.tokenizer.mask_token_id)
            _, hidden = model.denoise(**features, output_hidden_states=True)
            mask = batch['residue_mask'] & batch['target_slot_mask']
            pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            probs = model.binding_head(pooled).float().softmax(-1)[:, 1]
        for row, label, prob in zip(items, batch['binding_labels'].tolist(), probs.tolist()):
            if label >= 0:
                results[source_of(row)].append({'label': label, 'probability': prob})
    metrics = {}
    for source, values in results.items():
        labels, scores = [v['label'] for v in values], [v['probability'] for v in values]
        metrics[source] = {'records': len(values), 'positives': sum(labels),
                           'auroc': float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
                           'accuracy': sum(int(s >= .5) == y for s, y in zip(scores, labels)) / len(values),
                           'brier': sum((s-y)**2 for s, y in zip(scores, labels)) / len(values)}
    model.train(was_training)
    return {'metrics': metrics, 'predictions': dict(results),
            'condition_residues_hidden': mask_condition,
            'scope': 'pilot dataset relation labels; not an independently validated biological reward'}


@torch.no_grad()
def generation_probe(model, rows, device, per_source_limit=8, steps=16):
    """Native EOS generation on fully observed validation receptors, never reference length."""
    chosen, per_source = [], defaultdict(int)
    receptor_roles = {'antibody_heavy', 'antibody_light', 'tcr_beta', 'tcr_alpha'}
    for row in rows:
        if per_source[source_of(row)] >= per_source_limit:
            continue
        metas = row.get('chain_metadata', {})
        if any(any(metas.get(str(i), {}).get('synthetic_residue_mask', []))
               for i, role in enumerate(row['chain_roles']) if role in receptor_roles):
            continue
        chosen.append(row)
        per_source[source_of(row)] += 1
    collator = FixedChainCollator(model.config, mode='generation')
    output = []
    for start in range(0, len(chosen), 8):
        group = chosen[start:start + 8]
        batch = device_batch(collator, group, device)
        with torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext():
            predictions = model.generate(batch, steps=steps, temperature=0, seed=9173)
        for row, pred in zip(group, predictions):
            truth = dict(zip(row['chain_roles'], row['chains']))
            chains = []
            for c in pred['chains']:
                if c['target']:
                    seq = c.get('sequence')
                    chains.append(dict(role=c['role'], status=c['status'], sequence=seq,
                                       true_length=len(truth[c['role']]),
                                       length_error=abs(len(seq)-len(truth[c['role']])) if seq else None))
            output.append({'source': source_of(row), 'cluster': receptor_cluster(row), 'chains': chains})
    return {'steps': steps, 'temperature': 0, 'per_source_counts': dict(per_source), 'records': output,
            'scope': 'small fully-observed validation probe, not a downstream benchmark'}


def run(plan_path, recipe, device_index=0):
    plan = json.loads(plan_path.read_text())
    if recipe not in plan['recipes'] or RECIPES[recipe] != plan['recipes'][recipe]:
        raise ValueError('Recipe binding differs')
    for path, expected in plan['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    for path, expected in plan['parent_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Parent snapshot changed: ' + path)
    manifest_path = Path(plan['data_manifest'])
    if sha(manifest_path) != plan['data_manifest_sha256']:
        raise ValueError('Data manifest changed')
    manifest = json.loads(manifest_path.read_text())
    data = {}
    for name in ('train', 'screen'):
        ref = manifest['files'][name]
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Pilot rows changed')
        rows = [json.loads(line) for line in Path(ref['path']).read_text().splitlines()]
        data[name] = assign_source_groups(rows, manifest['source_counts'], name)
    pairs = None
    if RECIPES[recipe].get('preference_weight'):
        ref = plan['preference_data']
        if sha(ref['manifest']) != ref['manifest_sha256']:
            raise ValueError('Preference manifest changed')
        pref_manifest = json.loads(Path(ref['manifest']).read_text())
        if (pref_manifest['status'] != 'ready'
                or pref_manifest['source_train_sha256'] != manifest['files']['train']['sha256']
                or sha(pref_manifest['file']['path']) != pref_manifest['file']['sha256']):
            raise ValueError('Preference training provenance changed')
        pairs = [json.loads(line) for line in Path(pref_manifest['file']['path']).read_text().splitlines()]
    out = Path(plan['output']) / recipe
    out.mkdir(exist_ok=False)
    started = time.time()
    write(out / 'status.json', {'status': 'running', 'started_unix': started, 'recipe': recipe})
    try:
        torch.set_num_threads(2)
        device = torch.device('cuda', device_index)
        torch.cuda.set_device(device)
        torch.manual_seed(plan['seed'])
        model = load_model(Path(plan['parent']), recipe).to(device)
        teacher = (load_model(Path(plan['parent']), 'continued').to(device).eval().requires_grad_(False)
                   if (model.post_options.get('distill_weight') or
                       plan.get('preference_data'))
                   else None)
        model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=plan['learning_rate'], weight_decay=0.01)
        grouped = defaultdict(list)
        for row in data['train']:
            grouped[source_of(row)].append(row)
        sources = sorted(grouped)
        sampler = random.Random(plan['seed'])
        collator = PostTrainingCollator(model.config)
        preference_sampler = PreferenceSampler(pairs, plan['seed'] + 10000) if pairs else None
        def preference_inputs():
            return (preference_sampler.sample(collator, device,
                    shuffle=model.post_options.get('shuffle_preference', False))
                    if preference_sampler else {})
        semantic = plan.get('semantic_validation', False)
        # Verify real full-model forward/backward at the longest sampled inputs.
        longest = sorted(data['train'], key=lambda r: sum(map(len, r['chains'])), reverse=True)[:plan['micro_batch']]
        with preserved_rng([device.index]):
            gate = device_batch(collator, longest, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                gate_loss = model(teacher=teacher, **preference_inputs(), **gate)['loss']
            gate_loss.backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not torch.isfinite(gate_loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError('Full-model CUDA gate failed')
            write(out / 'cuda_gate.json', {'passed': True, 'input_shape': list(gate['input_ids'].shape),
                                          'loss': float(gate_loss.detach()),
                                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)})
        optimizer.zero_grad(set_to_none=True)
        del gate_loss, gradients, gate
        write(out / 'initial_metrics.json', evaluate(model, data['screen'], device, semantic=semantic))
        if hasattr(model, 'binding_head'):
            write(out / 'binding_initial.json', binding_probe(model, data['screen'], device))
        if plan.get('preference_data'):
            write(out / 'preference_initial.json', preference_probe(model, teacher, data['screen'], device))
        write(out / 'parameters.json', {'total': sum(p.numel() for p in model.parameters()),
                                        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)})
        torch.manual_seed(plan['seed'])
        if preference_sampler:
            preference_sampler.rng.seed(plan['seed'] + 10000)
        model.train()
        for step in range(1, plan['steps'] + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            lr = plan['learning_rate'] * min(1, step / plan['warmup']) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / plan['steps'])))
            for group in optimizer.param_groups:
                group['lr'] = lr
            for _ in range(plan['accumulation']):
                items = [sampler.choice(grouped[sampler.choice(sources)]) for _ in range(plan['micro_batch'])]
                batch = device_batch(collator, items, device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(teacher=teacher, **preference_inputs(), **batch)['loss'] / plan['accumulation']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                total_loss += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                                 error_if_nonfinite=True)
            optimizer.step()
            if step % 10 == 0 or step == 1:
                record = dict(step=step, loss=total_loss, grad_norm=float(norm), lr=lr, time_unix=time.time())
                with (out / 'train.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                write(out / 'progress.json', record)
            if step in (plan['steps'] // 2, plan['steps']):
                write(out / f'metrics_{step}.json', evaluate(model, data['screen'], device, semantic=semantic))
                if hasattr(model, 'binding_head'):
                    write(out / f'binding_{step}.json', binding_probe(model, data['screen'], device))
        write(out / 'generation.json', generation_probe(model, data['screen'], device))
        if plan.get('preference_data'):
            write(out / 'preference_final.json', preference_probe(model, teacher, data['screen'], device))
        # A distinct checkpoint contract prevents accidental loading as the old architecture.
        torch.save({'state_dict': model.cpu().state_dict(), 'optimizer': optimizer.state_dict(),
                    'step': plan['steps'], 'recipe': recipe, 'options': RECIPES[recipe],
                    'parent': plan['parent'], 'plan_sha256': sha(plan_path),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(device),
                    'sampler_state': sampler.getstate(),
                    'preference_sampler_state': preference_sampler.rng.getstate() if preference_sampler else None},
                    out / 'checkpoint.pt')
        write(out / 'status.json', {'status': 'success', 'step': plan['steps'], 'started_unix': started,
                                   'finished_unix': time.time(), 'checkpoint_sha256': sha(out / 'checkpoint.pt')})
    except Exception:
        write(out / 'status.json', {'status': 'failed', 'started_unix': started, 'finished_unix': time.time(),
                                   'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--recipe', choices=tuple(RECIPES), required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    run(args.plan, args.recipe, args.device)
