"""Activate pllm; python -B -m pytest -q scripts/tests/test_cdr_left_to_right.py.

Strict left-to-right greedy decoding of fixed-length masked targets. Reuse the
supplied all-target-MASK prior for the first accepted token, then expose only
earlier accepted predictions. Keep the predictor's legal vocabulary unchanged:
EOS/separator predictions do not terminate fixed-length infilling. No reference
residues, embedding feedback, output anchoring, or random sampling are used.
"""
import torch


def check_probabilities(probs, shape):
    assert probs.shape == shape
    assert torch.isfinite(probs).all() and (probs >= 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0]), atol=1e-5)


def generate(predict, batch, target, mask_id, prior, observer=None, allowed_token_ids=None):
    """Return (full_prediction, per_row_logical_nfe, physical_batch_forwards).

    NFE includes the supplied prior. For a nonempty batch, this function makes
    max(target lengths)-1 additional predictor calls, even when some rows have
    already finished. A row with no targets remains inactive and has logical
    NFE zero; a wholly target-free batch is rejected because a supplied prior
    would otherwise make the physical-cost convention ambiguous.

    observer(step, actual_input_ids, selected_mask, accepted_token_ids, probs)
    receives independent snapshots. accepted_token_ids has shape [batch], with
    -1 for inactive rows. probs is the actual full predictor distribution; its
    nonselected outputs are never accepted or used as feedback.
    """
    if allowed_token_ids is not None:
        raise ValueError('Use the frozen predictor vocabulary without extra filtering')
    original = batch['input_ids']
    assert original.ndim == 2 and original.dtype == torch.long and len(original) > 0
    assert target.dtype == torch.bool and target.shape == original.shape
    assert target.device == original.device == prior.device
    assert prior.ndim == 3 and prior.shape[:2] == original.shape
    assert 0 <= mask_id < prior.shape[-1]
    assert original[target].eq(mask_id).all(), 'All target positions must begin masked'
    check_probabilities(prior, prior.shape)
    lengths = target.long().sum(-1)
    max_length = int(lengths.max())
    if max_length == 0:
        raise ValueError('A prior-bearing decoding batch must contain at least one target')
    positions = [row.nonzero().flatten() for row in target]
    tokens = original.clone()
    committed = torch.zeros_like(target)
    additional_calls = 0
    for step in range(max_length):
        selected = torch.zeros_like(target)
        for row, row_positions in enumerate(positions):
            if step < len(row_positions):
                selected[row, row_positions[step]] = True
        assert torch.equal(selected.sum(-1), lengths.gt(step).long())
        assert not (selected & committed).any()
        query_inputs = tokens.clone()
        assert torch.equal(query_inputs[~target], original[~target])
        assert query_inputs[target & ~committed].eq(mask_id).all()
        assert query_inputs[committed].ne(mask_id).all()
        if step == 0:
            probs = prior
        else:
            probs = predict(query_inputs)
            additional_calls += 1
        assert torch.equal(query_inputs, tokens), 'Predictor must not mutate input tokens'
        check_probabilities(probs, prior.shape)
        proposal = probs.argmax(-1)
        assert proposal[selected].ne(mask_id).all(), 'Frozen predictor must exclude MASK from accepted outputs'
        accepted = torch.full_like(lengths, -1)
        active = selected.any(-1)
        accepted[active] = proposal[selected]
        tokens[selected] = proposal[selected]
        committed |= selected
        if observer is not None:
            observer(step, query_inputs.detach().clone(), selected.clone(), accepted.clone(),
                     probs.detach().clone())
    assert additional_calls == max_length - 1
    assert torch.equal(committed, target)
    assert torch.equal(tokens[~target], original[~target])
    assert tokens[target].ne(mask_id).all()
    return tokens, lengths, max_length
