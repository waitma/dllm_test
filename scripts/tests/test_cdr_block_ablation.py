"""Activate pllm; python -B -m pytest -q scripts/tests/test_cdr_block_ablation.py."""
from types import SimpleNamespace

import pytest
import torch

from scripts.downstream.cdr_block_ablation import partitions, posterior
from scripts.downstream.cdr_block_soft import posterior as frozen_soft_posterior


def fixture():
    """A contextual predictor with permuted native positions and legal outputs."""
    torch.manual_seed(18)
    embed = torch.nn.Embedding(7, 5)
    with torch.no_grad():
        embed.weight.mul_(.2)
    model = SimpleNamespace(backbone=SimpleNamespace(transformer=SimpleNamespace(wte=embed)),
                            tokenizer=SimpleNamespace(mask_token_id=0))
    tokens = torch.tensor([[2, 0, 0, 0, 4, 0], [3, 0, 5, 0, 0, 0]])
    target = tokens.eq(0)
    maps = [list(enumerate([4, 1, 6, 2, 7, 0])), list(enumerate([6, 4, 2, 0, 5, 7]))]
    native_ids = torch.full((2, 8), 6, dtype=torch.long)
    for i, mapping in enumerate(maps):
        for virtual, native in mapping:
            native_ids[i, native] = tokens[i, virtual]
    seen, distributions = [], []

    def predict(inputs):
        assert torch.equal(inputs, tokens)
        value = embed(native_ids)
        seen.append(value.detach().clone())
        logits = (value + value.mean(1, keepdim=True)) @ embed.weight.T
        logits[..., 0] = -torch.inf
        probs = logits.softmax(-1)
        virtual_probs = torch.stack([probs[i, [n for _, n in mapping]] for i, mapping in enumerate(maps)])
        distributions.append(virtual_probs.detach().clone())
        return virtual_probs

    prior = predict(tokens)
    args = (model, predict, {'input_ids': tokens}, target, {'input_ids': native_ids}, maps, prior)
    return args, seen, distributions


@pytest.mark.parametrize('rule', ['interleaved', 'contiguous'])
@pytest.mark.parametrize('temperature,strength', [(.5, .5), (.5, 1.), (1., .5)])
def test_frozen_soft_exactly_matches_original(rule, temperature, strength):
    args, seen, _ = fixture()
    original, old_nfe = frozen_soft_posterior(*args, groups=3, rule=rule,
        temperature=temperature, strength=strength)
    original_queries = [x.clone() for x in seen[1:]]
    actual, nfe = posterior(*args, groups=3, rule=rule, temperature=temperature,
        strength=strength, context='soft', feedback='frozen')
    assert nfe == old_nfe == 4 and torch.equal(actual, original)
    assert all(torch.equal(x, y) for x, y in zip(original_queries, seen[4:]))
    assert not args[0].backbone.transformer.wte._forward_hooks


@pytest.mark.parametrize('context', ['soft', 'hard'])
@pytest.mark.parametrize('feedback', ['frozen', 'sequential'])
def test_native_mapping_and_feedback_bank(context, feedback):
    args, seen, distributions = fixture()
    prior = args[6].clone()
    events = []
    result, nfe = posterior(*args, groups=3, rule='interleaved', temperature=.5,
        strength=.5, context=context, feedback=feedback,
        observer=lambda index, scored, used, weights, probs: events.append((index, scored, used, weights, probs)))
    assert nfe == 4 and len(events) == 3 and len(seen) == 4
    bank = prior.clone()
    counted = torch.zeros_like(args[3], dtype=torch.int)
    for event, actual, probs in zip(events, seen[1:], distributions[1:]):
        index, scored, used, observed_weights, observed_probs = event
        assert torch.equal(observed_probs, probs)
        assert torch.equal(scored, partitions(args[3], 3, 'interleaved')[index])
        assert not (scored & used).any() and torch.equal(scored | used, args[3])
        expected = seen[0].clone()
        native_used = torch.zeros(expected.shape[:2], dtype=torch.bool)
        if context == 'soft':
            weights = bank.pow(2); weights /= weights.sum(-1, keepdim=True)
        else:
            weights = torch.nn.functional.one_hot(bank.argmax(-1), bank.shape[-1]).float()
        weights *= used[..., None]
        assert torch.equal(observed_weights, weights)
        assert observed_weights[~used].eq(0).all()
        assert torch.allclose(observed_weights[used].sum(-1), torch.ones_like(observed_weights[..., 0][used]))
        embed = args[0].backbone.transformer.wte
        feedback_vectors = weights @ embed.weight
        for i, mapping in enumerate(args[5]):
            for virtual, native in mapping:
                if used[i, virtual]:
                    native_used[i, native] = True
                    expected[i, native] = .5 * seen[0][i, native] + .5 * feedback_vectors[i, virtual]
                if scored[i, virtual]:
                    assert torch.equal(actual[i, native], seen[0][i, native])
        assert torch.equal(actual[~native_used], seen[0][~native_used])
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
        assert not torch.equal(actual[native_used], seen[0][native_used])
        assert torch.equal(result[scored], probs[scored])
        counted += scored
        if feedback == 'sequential':
            bank[scored] = probs[scored]
            assert torch.equal(bank[counted.eq(0)], prior[counted.eq(0)])
    if feedback == 'sequential':
        assert not torch.equal(bank[args[3]], prior[args[3]]), 'Fixture must exercise a changed feedback bank'
    assert torch.equal(args[6], prior), 'Caller prior must not be mutated'
    assert torch.equal(result[~args[3]], prior[~args[3]])
    assert not args[0].backbone.transformer.wte._forward_hooks


def test_hard_context_matches_one_hot_soft_context_and_ignores_temperature():
    args, _, _ = fixture()
    hot = torch.nn.functional.one_hot(args[6].argmax(-1), args[6].shape[-1]).float()
    soft_args = (*args[:6], hot)
    onehot_soft, _ = posterior(*soft_args, groups=3, rule='interleaved', temperature=.5,
        strength=.5, context='soft', feedback='frozen')
    for temperature in (.5, 1.):
        hard, _ = posterior(*args, groups=3, rule='interleaved', temperature=temperature,
            strength=.5, context='hard', feedback='frozen')
        assert torch.equal(hard[args[3]], onehot_soft[args[3]])


@pytest.mark.parametrize('context', ['soft', 'hard'])
def test_sequential_bank_uses_raw_queries_only_at_previously_scored_positions(context):
    args, _, _ = fixture()
    initial = torch.zeros_like(args[6]); initial[..., 1] = 1.
    query_count = [0]
    def changing_predict(tokens):
        args[1](tokens)  # Exercise the actual native input hook once per query.
        query_count[0] += 1
        query = torch.zeros_like(initial)
        query[..., query_count[0] + 1] = 1.
        return query
    modified = (args[0], changing_predict, *args[2:6], initial)
    events = []
    actual, _ = posterior(*modified, groups=3, rule='interleaved', temperature=.5,
        strength=.5, context=context, feedback='sequential',
        observer=lambda index, scored, used, weights, probs: events.append((scored, used, weights, probs)))
    bank = initial.clone()
    for index, (scored, used, weights, probs) in enumerate(events):
        assert torch.equal(weights, bank * used[..., None])
        assert probs[..., index + 2].eq(1).all()
        bank[scored] = probs[scored]
        assert torch.equal(actual[scored], probs[scored])
    assert torch.equal(actual, bank)
    assert torch.equal(actual[~args[3]], initial[~args[3]])


@pytest.mark.parametrize('context', ['soft', 'hard'])
@pytest.mark.parametrize('feedback', ['frozen', 'sequential'])
@pytest.mark.parametrize('groups,strength', [(1, .5), (3, 0.)])
def test_no_feedback_or_zero_strength_is_baseline(context, feedback, groups, strength):
    args, _, _ = fixture()
    result, nfe = posterior(*args, groups=groups, rule='interleaved', temperature=.5,
        strength=strength, context=context, feedback=feedback)
    assert nfe == groups + 1 and torch.equal(result, args[6])


@pytest.mark.parametrize('failure', ['predict', 'observer'])
@pytest.mark.parametrize('feedback', ['frozen', 'sequential'])
def test_hook_cleanup_after_failure(failure, feedback):
    args, _, _ = fixture()
    def fail(*unused):
        raise RuntimeError('intentional failure')
    altered = list(args)
    if failure == 'predict':
        altered[1] = fail
    with pytest.raises(RuntimeError, match='intentional failure'):
        posterior(*altered, groups=3, rule='interleaved', temperature=.5, strength=.5,
            context='soft', feedback=feedback, observer=fail if failure == 'observer' else None)
    assert not args[0].backbone.transformer.wte._forward_hooks
    assert torch.equal(args[1](args[2]['input_ids']), args[6])


def test_observer_snapshots_cannot_change_sampling():
    args, _, _ = fixture()
    expected, _ = posterior(*args, groups=3, rule='interleaved', temperature=.5,
        strength=.5, context='soft', feedback='sequential')
    def mutate(index, scored, used, weights, probs):
        scored.zero_(); used.zero_(); weights.zero_(); probs.zero_()
    actual, _ = posterior(*args, groups=3, rule='interleaved', temperature=.5,
        strength=.5, context='soft', feedback='sequential', observer=mutate)
    assert torch.equal(actual, expected)
