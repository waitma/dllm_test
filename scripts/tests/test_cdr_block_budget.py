"""Activate pllm; python -B -m pytest -q scripts/tests/test_cdr_block_budget.py.

Focused CPU checks for the frozen sampler's seven-group/eight-forward extension.
Import the frozen source directly; do not change the existing sampler or tests.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def frozen_module(relative, expected_sha, name):
    path = ROOT / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SAMPLER = frozen_module(
    'output/downstream_generation/cdr_block_ablation_20261016/source/cdr_block_ablation.py',
    '8791e7867a278fcce0c0764424166092050895f0265dc75449c25bcad6835593', 'budget_frozen_sampler')
LEGACY = frozen_module(
    'output/downstream_generation/cdr_block_context_20261015/source/legacy_soft.py',
    'afe908fb028b978aac34796018abb5e89a9f0b3f26367726509e1c6ec9218cc9', 'budget_legacy_loo')


def fixture(lengths):
    """Short and long targets, gapped virtual coordinates, permuted native maps."""
    torch.manual_seed(18)
    embed = torch.nn.Embedding(7, 5)
    with torch.no_grad():
        embed.weight.mul_(.2)
    model = SimpleNamespace(backbone=SimpleNamespace(transformer=SimpleNamespace(wte=embed)),
                            tokenizer=SimpleNamespace(mask_token_id=0))
    tokens = torch.full((len(lengths), 13), 2, dtype=torch.long)
    target_positions = [1, 3, 4, 5, 7, 8, 9, 10, 11]
    for i, length in enumerate(lengths):
        assert 0 < length <= len(target_positions)
        tokens[i, target_positions[:length]] = 0
    target = tokens.eq(0)
    maps = [[(v, (v * 2 + i) % 15) for v in range(13)] for i in range(len(lengths))]
    native_ids = torch.full((len(lengths), 15), 6, dtype=torch.long)
    for i, mapping in enumerate(maps):
        for virtual, native in mapping:
            native_ids[i, native] = tokens[i, virtual]
    seen = []

    def predict(inputs):
        assert torch.equal(inputs, tokens)
        x = embed(native_ids)
        seen.append(x.detach().clone())
        logits = (x + x.mean(1, keepdim=True)) @ embed.weight.T
        logits[..., 0] = -torch.inf
        probs = logits.softmax(-1)
        return torch.stack([probs[i, [n for _, n in mapping]] for i, mapping in enumerate(maps)])

    prior = predict(tokens)
    return (model, predict, {'input_ids': tokens}, target,
            {'input_ids': native_ids}, maps, prior), seen


def run(args, observer=None, strength=.5):
    return SAMPLER.posterior(*args, groups=7, rule='interleaved', temperature=.5,
        strength=strength, context='soft', feedback='frozen', observer=observer)


def test_seven_group_coverage_for_short_exact_and_long_targets():
    lengths = [1, 4, 7, 9]
    args, _ = fixture(lengths)
    masks = torch.stack(SAMPLER.partitions(args[3], 7, 'interleaved'))
    assert torch.equal(masks.sum(0), args[3].long())
    assert masks.any(-1).sum(0).tolist() == [min(7, length) for length in lengths]
    assert (~masks.any(-1)).sum(0).tolist() == [6, 3, 0, 0]
    # Seven groups are not leave-one-out when the target is longer than seven.
    assert masks[:, 3].sum(-1).tolist() == [2, 2, 1, 1, 1, 1, 1]


def test_seven_queries_preserve_scored_masks_and_visible_native_context():
    args, seen = fixture([1, 4, 7, 9])
    events = []
    result, nfe = run(args, observer=lambda *event: events.append(event))
    assert nfe == 8 and len(seen) == 8 and len(events) == 7
    counted = torch.zeros_like(args[3], dtype=torch.int)
    original = seen[0]
    prior_weights = args[6].pow(2)
    prior_weights /= prior_weights.sum(-1, keepdim=True)
    for (index, scored, feedback, weights, probs), actual in zip(events, seen[1:]):
        assert index < 7 and not (scored & feedback).any()
        assert torch.equal(scored | feedback, args[3])
        assert torch.equal(weights, prior_weights * feedback[..., None])
        assert weights[scored].eq(0).all()
        expected = original.clone()
        native_feedback = torch.zeros_like(args[4]['input_ids'], dtype=torch.bool)
        soft = weights @ args[0].backbone.transformer.wte.weight
        for i, mapping in enumerate(args[5]):
            for virtual, native in mapping:
                if feedback[i, virtual]:
                    native_feedback[i, native] = True
                    expected[i, native] = .5 * original[i, native] + .5 * soft[i, virtual]
                if scored[i, virtual]:
                    assert torch.equal(actual[i, native], original[i, native])
        assert torch.equal(actual[~native_feedback], original[~native_feedback])
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
        assert torch.equal(result[scored], probs[scored])
        counted += scored
    assert torch.equal(counted, args[3].int())
    assert torch.equal(result[~args[3]], args[6][~args[3]])
    assert not args[0].backbone.transformer.wte._forward_hooks


def test_empty_whole_batch_queries_are_counted_but_do_not_enter_result():
    args, seen = fixture([1, 4])
    events = []
    result, nfe = run(args, observer=lambda *event: events.append(event))
    assert nfe == len(seen) == 8
    assert len(events) == 7 and all(not event[1].any() for event in events[4:])
    rebuilt = args[6].clone()
    for index, scored, feedback, weights, probs in events[:4]:
        rebuilt[scored] = probs[scored]
    assert torch.equal(result, rebuilt)
    for index, scored, feedback, weights, probs in events[4:]:
        assert torch.equal(feedback, args[3])
        assert torch.allclose(weights[feedback].sum(-1), torch.ones_like(probs[..., 0][feedback]))


def test_zero_strength_is_exact_baseline_even_with_empty_groups():
    args, seen = fixture([1, 4, 7, 9])
    result, nfe = run(args, strength=0.)
    assert nfe == len(seen) == 8
    assert torch.equal(result, args[6])


@pytest.mark.parametrize('length', [1, 4, 6])
def test_seven_groups_equal_independent_legacy_leave_one_out_for_short_targets(length):
    args, seen = fixture([length])
    result, nfe = run(args)
    before = len(seen)
    legacy = LEGACY.selected_posteriors(args[0], None, args[1], args[2], args[3],
        args[4], args[5], args[6], dict(kind='soft_context', temperature=.5, strength=.5))
    expected, legacy_nfe = legacy['candidate']
    assert torch.equal(result, expected)
    assert nfe == 8 and legacy_nfe == length + 1
    assert len(seen) - before == length
