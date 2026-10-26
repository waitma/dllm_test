"""Activate pllm; python -B -m pytest -q scripts/tests/test_cdr_left_to_right.py."""
import pytest
import torch

from scripts.downstream.cdr_left_to_right import generate


def fixture():
    tokens = torch.tensor([[9, 0, 8, 0, 0, 7], [9, 0, 8, 7, 6, 5], [9, 8, 7, 6, 5, 4]])
    target = tokens.eq(0)
    calls = []
    def predict(inputs):
        assert torch.equal(inputs[~target], tokens[~target])
        calls.append(inputs.clone())
        # Later guesses depend on the accepted prefix. The position-dependent
        # confidences deliberately favor later sites, which must not change order.
        prefix = (inputs * target).sum(-1, keepdim=True)
        preferred = (prefix + torch.arange(tokens.shape[1])) % 3 + 1
        probs = torch.zeros((*tokens.shape, 10))
        confidence = .55 + .04 * torch.arange(tokens.shape[1])
        probs[..., 1:4] = (1 - confidence)[None, :, None] / 2
        probs.scatter_(-1, preferred[..., None], confidence[None, :, None].expand(len(tokens), -1, 1))
        return probs
    prior = predict(tokens)
    return predict, {'input_ids': tokens}, target, 0, prior, calls


def test_causal_inputs_ascending_positions_and_varying_length_costs():
    predict, batch, target, mask, prior, calls = fixture()
    events = []
    prediction, logical, physical = generate(predict, batch, target, mask, prior,
        observer=lambda *event: events.append(event))
    assert logical.tolist() == [3, 1, 0] and physical == 3
    assert len(calls) == physical and len(events) == 3
    assert torch.equal(batch['input_ids'], calls[0])
    assert torch.equal(events[0][-1], prior)
    positions = [row.nonzero().flatten().tolist() for row in target]
    expected = batch['input_ids'].clone()
    counted = torch.zeros_like(target, dtype=torch.int)
    for step, inputs, selected, accepted, probs in events:
        assert torch.equal(inputs, calls[step]) and torch.equal(inputs, expected)
        assert torch.equal(inputs[~target], batch['input_ids'][~target])
        for row, row_positions in enumerate(positions):
            assert inputs[row, row_positions[step:]].eq(mask).all()
            if step < len(row_positions):
                position = row_positions[step]
                assert selected[row].nonzero().flatten().tolist() == [position]
                assert accepted[row] == probs[row, position].argmax()
                expected[row, position] = accepted[row]
            else:
                assert not selected[row].any() and accepted[row] == -1
        counted += selected
    assert torch.equal(counted, target.int()) and torch.equal(prediction, expected)
    assert torch.equal(prediction[~target], batch['input_ids'][~target])
    # Prefix-sensitive decoding differs from independently accepting every p0 argmax.
    assert not torch.equal(prediction[0, target[0]], prior.argmax(-1)[0, target[0]])


def test_first_target_uses_provided_prior_without_recomputing():
    _, batch, target, mask, prior, _ = fixture()
    batch = {'input_ids': batch['input_ids'][1:]}
    target, prior = target[1:], prior[1:]
    def unexpected_call(inputs):
        raise AssertionError('A one-target batch must not request another forward')
    prediction, logical, physical = generate(unexpected_call, batch, target, mask, prior)
    assert logical.tolist() == [1, 0] and physical == 1
    assert torch.equal(prediction[target], prior.argmax(-1)[target])
    assert torch.equal(prediction[~target], batch['input_ids'][~target])


def test_eos_or_separator_like_token_does_not_terminate_fixed_length_infilling():
    _, batch, target, mask, prior, _ = fixture()
    calls = []
    special_probs = torch.zeros_like(prior); special_probs[..., 4] = 1.
    def predict(inputs):
        calls.append(inputs.clone())
        return special_probs
    result, logical, physical = generate(predict, batch, target, mask, special_probs)
    assert logical.tolist() == [3, 1, 0] and physical == 3 and len(calls) == 2
    assert result[target].eq(4).all()


def test_argmax_ties_use_predictor_token_order_and_observer_cannot_change_decoding():
    _, batch, target, mask, prior, _ = fixture()
    tied = torch.zeros_like(prior); tied[..., 2] = .5; tied[..., 3] = .5
    def predict(inputs):
        return tied
    def mutate(step, inputs, selected, accepted, probs):
        inputs.zero_(); selected.zero_(); accepted.zero_(); probs.zero_()
    prediction, _, _ = generate(predict, batch, target, mask, tied, observer=mutate)
    assert prediction[target].eq(2).all()
    assert tied[..., 2].eq(.5).all() and tied[..., 3].eq(.5).all()
    assert torch.equal(prediction[~target], batch['input_ids'][~target])


def test_extra_reference_metadata_is_never_used():
    predict, batch, target, mask, prior, _ = fixture()
    first, _, _ = generate(predict, {**batch, 'reference_ids': torch.ones_like(batch['input_ids'])}, target, mask, prior)
    second, _, _ = generate(predict, {**batch, 'reference_ids': torch.full_like(batch['input_ids'], 6)}, target, mask, prior)
    assert torch.equal(first, second)


def test_rejects_visible_target_and_wholly_inactive_batch():
    predict, batch, target, mask, prior, _ = fixture()
    invalid = batch['input_ids'].clone(); invalid[0, 1] = 2
    with pytest.raises(AssertionError, match='begin masked'):
        generate(predict, {'input_ids': invalid}, target, mask, prior)
    with pytest.raises(ValueError, match='at least one target'):
        generate(predict, batch, torch.zeros_like(target), mask, prior)


def test_rejects_predictor_mutation_and_illegal_mask_proposal():
    _, batch, target, mask, prior, _ = fixture()
    def mutating_predict(inputs):
        inputs[0, 0] = 4
        return prior
    with pytest.raises(AssertionError, match='must not mutate'):
        generate(mutating_predict, batch, target, mask, prior)
    invalid = torch.zeros_like(prior); invalid[..., mask] = 1.
    with pytest.raises(AssertionError, match='exclude MASK'):
        generate(lambda inputs: invalid, batch, target, mask, invalid)


def test_extra_vocabulary_restriction_is_rejected():
    predict, batch, target, mask, prior, _ = fixture()
    with pytest.raises(ValueError, match='without extra filtering'):
        generate(predict, batch, target, mask, prior, allowed_token_ids=[1, 2, 3])
