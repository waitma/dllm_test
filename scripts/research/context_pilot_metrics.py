"""Evaluate context pilots with explicit positive-relation and context counts.

Activate conda pllm. Imported by run_v7_context_ablation.py; no standalone job.
Uses the existing fixed per-record masks and never exports sequences or logits.
"""
from contextlib import nullcontext

import torch
from torch.nn import functional as F

from scripts.run_immune_architecture import MASK_LEVELS
from scripts.run_immune_research import fixed_validation_mask

CONTEXT_ROLES = frozenset({'antigen', 'peptide', 'mhc'})
COUNT_FOR = {'loss': 'selected_count', 'aa_loss': 'aa_count',
             'aa_accuracy': 'aa_count', 'terminal_loss': 'terminal_count',
             'terminal_accuracy': 'terminal_count', 'relation_loss': 'relation_count',
             'relation_accuracy': 'relation_count'}


@torch.no_grad()
def measure(model, batch):
    device = batch['input_ids'].device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    was_training, levels = model.training, {}
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            for name, ratio in MASK_LEVELS:
                selected = fixed_validation_mask(batch, ratio).bool() & batch['eligible_mask'] & batch['attention_mask']
                if not torch.equal(batch['input_ids'][selected], batch['labels'][selected]):
                    raise ValueError('Selected targets differ from clean input')
                features = model.feature_args(batch)
                features['input_ids'] = batch['input_ids'].masked_fill(selected, model.tokenizer.mask_token_id)
                logits = model.denoise(**features)[0]
                labels = batch['labels'].masked_fill(~selected, -100)
                ce = F.cross_entropy(logits.float().transpose(1, 2), labels, reduction='none', ignore_index=-100)
                correct = logits.argmax(-1).eq(batch['labels']).float()
                masks = {'selected_count': selected, 'aa_count': selected & batch['residue_mask'],
                         'terminal_count': selected & batch['terminal_mask'],
                         'relation_count': selected & (batch['chain_ids'] < 0)}
                if not torch.equal(masks['relation_count'], selected & ~batch['residue_mask'] & ~batch['terminal_mask']):
                    raise ValueError('Unexpected supervised non-residue/terminal field')
                level = {'mask_ratio': ratio, **{key: int(mask.sum()) for key, mask in masks.items()}}
                for metric, count_name in COUNT_FOR.items():
                    values = correct if metric.endswith('accuracy') else ce
                    count = level[count_name]
                    total = float(values[masks[count_name]].sum()) if count else 0.
                    level[metric + '_sum'] = total
                    level[metric] = total / count if count else None
                levels[name] = level
    finally:
        model.train(was_training)
    return {'mask_levels': levels}


def combine(items):
    if not items:
        raise ValueError('Cannot pool empty validation portions')
    levels = {}
    for name, ratio in MASK_LEVELS:
        parts = [item['mask_levels'][name] for item in items]
        level = {'mask_ratio': ratio}
        for count_name in set(COUNT_FOR.values()):
            level[count_name] = sum(part[count_name] for part in parts)
        for metric, count_name in COUNT_FOR.items():
            total = sum(part[metric + '_sum'] for part in parts)
            level[metric + '_sum'] = total
            level[metric] = total / level[count_name] if level[count_name] else None
        levels[name] = level
    means = {}
    for metric in COUNT_FOR:
        values = [level[metric] for level in levels.values() if level[metric] is not None]
        means[metric] = sum(values) / len(values) if values else None
    return {'mask_levels': levels, 'mask_mean': means,
            **{metric: levels['mask100'][metric] for metric in COUNT_FOR},
            'mask_protocol': 'fixed_per_record_selected_tokens_v1'}


def evaluate(model, collate, rows, sources, device, micro, cpu):
    grouped, source_parts, context_parts = {}, {}, {}
    for row, source in zip(rows, sources):
        has_context = bool(CONTEXT_ROLES.intersection(row.chain_roles))
        grouped.setdefault((source, has_context), []).append(row)
    all_parts = []
    with nullcontext() if cpu else torch.autocast('cuda', dtype=torch.bfloat16):
        for (source, has_context), records in sorted(grouped.items()):
            for start in range(0, len(records), micro):
                selected_rows = records[start:start + micro]
                batch = {key: value.to(device) for key, value in collate(selected_rows).items()}
                part = measure(model, batch)
                part['records'] = len(selected_rows)
                all_parts.append(part)
                source_parts.setdefault(source, []).append(part)
                context_parts.setdefault('with_context' if has_context else 'without_context', []).append(part)
    def aggregate(parts):
        return {**combine(parts), 'records': sum(part['records'] for part in parts)}
    strata = {source: aggregate(parts) for source, parts in source_parts.items()}
    context = {name: aggregate(parts) for name, parts in context_parts.items()}
    macro = {}
    for name, _ in MASK_LEVELS:
        macro[name] = {}
        for metric in COUNT_FOR:
            values = [source['mask_levels'][name][metric] for source in strata.values()
                      if source['mask_levels'][name][metric] is not None]
            macro[name][metric] = sum(values) / len(values) if values else None
    return {'records': len(rows), 'token_pooled': combine(all_parts), 'source_macro': macro,
            'sources': strata, 'context_strata': context,
            'relation_scope': 'existing supervised relation tokens; positive-only data; no binary-discrimination claim'}
