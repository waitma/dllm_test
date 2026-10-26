"""Activate pllm; python -B -m pytest -q scripts/tests/test_cdr_block_ablation.py.

Training-free grouped-context ablation, copied independently from cdr_block_soft
so a frozen file can be imported without the working tree. Every scored group
keeps its original MASK embeddings; only other target positions receive context.
Sequential feedback updates only previously scored positions with raw query
probabilities. Output anchoring is deliberately left to the calling runner.
"""
import math

import torch


def partitions(target, groups, rule):
    """Use the same ordinal groups as the frozen grouped-soft implementation."""
    if groups < 1 or rule not in ('interleaved', 'contiguous'):
        raise ValueError((groups, rule))
    ordinal = target.long().cumsum(-1) - 1
    labels = (ordinal.remainder(groups) if rule == 'interleaved'
              else ordinal * groups // target.sum(-1, keepdim=True).clamp_min(1))
    return [target & labels.eq(i) for i in range(groups)]


def check_probabilities(probs):
    assert torch.isfinite(probs).all() and (probs >= 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0]), atol=1e-5)


def posterior(model, predict, batch, target, features, maps, prior, groups, rule,
              temperature, strength, context='soft', feedback='frozen', observer=None):
    """Return unanchored target posteriors and logical NFE including the prior.

    The observer receives (index, scored, feedback_mask, context_weights,
    query_probs). Context weights are the actual coefficients multiplied by
    embedding.weight; non-feedback positions have zero weight.
    Its arguments are independent snapshots, so logging cannot mutate sampling.
    Hard context selects an argmax embedding but still uses the same MASK blend
    as soft context. All groups, including empty padded groups, count as queries.
    """
    if (not math.isfinite(temperature) or temperature <= 0
            or not 0 <= strength <= 1):
        raise ValueError((temperature, strength))
    if context not in ('soft', 'hard') or feedback not in ('frozen', 'sequential'):
        raise ValueError((context, feedback))
    tokens, native_ids = batch['input_ids'], features['input_ids']
    mask_id = model.tokenizer.mask_token_id
    assert target.dtype == torch.bool and target.shape == tokens.shape
    assert tokens[target].eq(mask_id).all(), 'Every scored target must begin masked'
    assert prior.shape[:2] == tokens.shape
    assert len(maps) == len(tokens) == len(native_ids)
    check_probabilities(prior)
    native = []
    for i, mapping in enumerate(maps):
        coordinates = dict(mapping)
        assert len(coordinates) == len(mapping) == len(set(coordinates.values())), 'Coordinate map must be injective'
        assert all(0 <= v < tokens.shape[1] and 0 <= n < native_ids.shape[1] for v, n in mapping)
        positions = target[i].nonzero().flatten().tolist()
        assert all(v in coordinates for v in positions)
        assert all(native_ids[i, coordinates[v]] == mask_id for v in positions)
        native.append(coordinates)

    embed = model.backbone.transformer.wte
    original = embed(native_ids)
    bank = prior.clone()

    def bank_weights():
        if context == 'hard':
            # Positive-temperature sharpening preserves this argmax. Explicit
            # one-hot weights let the independent auditor verify the actual
            # coefficients used for both context representations.
            return torch.nn.functional.one_hot(bank.argmax(-1), bank.shape[-1]).to(bank.dtype)
        weights = bank.pow(1 / temperature)
        normalizer = weights.sum(-1, keepdim=True)
        assert (normalizer > 0).all(), 'Temperature caused probability underflow'
        weights /= normalizer
        return weights

    fixed_weights = bank_weights() if feedback == 'frozen' else None
    masks = partitions(target, groups, rule)
    active = torch.zeros_like(native_ids, dtype=torch.bool)
    replacement = original.clone()
    calls = [0]
    counts = torch.zeros_like(target, dtype=torch.int)
    result = prior.clone()

    def hook(layer, inputs, output):
        calls[0] += 1
        assert torch.equal(inputs[0], native_ids)
        revised = torch.where(active[..., None], replacement, output)
        assert torch.equal(revised[~active], output[~active])
        return revised

    handle = embed.register_forward_hook(hook)
    try:
        for index, scored in enumerate(masks):
            active.zero_()
            replacement.copy_(original)
            feedback_mask = target & ~scored
            weights = fixed_weights if feedback == 'frozen' else bank_weights()
            context_weights = weights * feedback_mask[..., None]
            expected = context_weights @ embed.weight
            for i in range(len(target)):
                for virtual in feedback_mask[i].nonzero().flatten().tolist():
                    position = native[i][virtual]
                    active[i, position] = True
                    replacement[i, position] = ((1 - strength) * original[i, position]
                                                + strength * expected[i, virtual])
                for virtual in scored[i].nonzero().flatten().tolist():
                    assert not active[i, native[i][virtual]]
            before = calls[0]
            probs = predict(tokens)
            assert calls[0] == before + 1
            assert probs.shape == prior.shape
            check_probabilities(probs)
            result[scored] = probs[scored]
            counts += scored
            if feedback == 'sequential':
                # No output anchor here: this factor changes only the feedback
                # source, and preserves the original bank for all unscored sites.
                bank[scored] = probs[scored]
            if observer is not None:
                observer(index, scored.clone(), feedback_mask.clone(),
                         context_weights.detach().clone(), probs.detach().clone())
    finally:
        handle.remove()
    assert counts[target].eq(1).all() and counts[~target].eq(0).all()
    assert torch.equal(result[~target], prior[~target])
    return result, groups + 1
