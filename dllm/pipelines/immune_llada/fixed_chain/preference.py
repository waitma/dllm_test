"""Weak-label, full-mask denoising preference surrogate (not sequence DPO).

Run: python -m pytest scripts/tests/immune_llada/test_fixed_chain_preference.py -q
Chosen and rejected records share all model inputs; only observed targets differ.
"""
import torch
from torch.nn import functional as F
from .post_training_data import observed_end_batch


def pair_features(model, batch):
    """Hide the entire target canvas, including uncertain synthetic endpoints."""
    batch = observed_end_batch(batch, model.tokenizer.mask_token_id)
    features = model.feature_args(batch)
    features['input_ids'] = features['input_ids'].masked_fill(
        features['target_slot_mask'], model.tokenizer.mask_token_id)
    if len(batch['input_ids']) % 2:
        raise ValueError('Expected interleaved chosen/rejected records')
    for name, value in features.items():
        if not torch.equal(value[0::2], value[1::2]):
            raise ValueError('Preference condition differs: ' + name)
    if not batch['eligible_mask'].any(1).all():
        raise ValueError('Each preference candidate needs observed targets')
    return features, batch


def normalized_nll(logits, batch):
    """Per-record mean over observed target residues and observed EOS only."""
    labels = batch['labels'].masked_fill(~batch['eligible_mask'], -100)
    losses = F.cross_entropy(logits.float().transpose(1, 2), labels,
                             ignore_index=-100, reduction='none')
    return losses.sum(1) / batch['eligible_mask'].sum(1)


def preference_objective(model, reference, batch, *, beta=1.0, positive_only=False,
                         directions=None):
    """Pairwise relative improvement in denoising NLL against a frozen parent.

    Full-mask marginal token losses are a surrogate, not log sequence likelihood.
    A shuffled-direction control uses fixed random signs supplied by its sampler.
    """
    features, batch = pair_features(model, batch)
    current = normalized_nll(model.denoise(**features)[0], batch)
    if positive_only:
        return current[0::2].mean()
    if reference is None or reference.training or any(p.requires_grad for p in reference.parameters()):
        raise ValueError('Preference reference must be frozen and in eval mode')
    ids = features['input_ids']
    with torch.no_grad(), torch.autocast(ids.device.type,
            enabled=torch.is_autocast_enabled(ids.device.type),
            dtype=torch.get_autocast_dtype(ids.device.type), cache_enabled=False):
        baseline = normalized_nll(reference.denoise(**features)[0], batch)
    improvement = baseline - current
    margin = beta * (improvement[0::2] - improvement[1::2])
    if directions is not None:
        if directions.shape != margin.shape or not ((directions == 1) | (directions == -1)).all():
            raise ValueError('One +/-1 direction is required per pair')
        margin = margin * directions
    return F.softplus(-margin).mean()
