"""Artificial integer labels and tiny CPU linear-model normalization tests.

Activate pllm, then:
  python -m pytest scripts/tests/immune_llada/test_record_loss.py -q
No corpus/sequence/benchmark input, model runner, Trainer, GPU, or checkpoint IO.
"""
from copy import deepcopy

import pytest
import torch

from scripts.architecture_prototypes.record_loss import actual_global_record_count, record_normalized_ce


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def artificial(dtype=torch.float64):
    generator = torch.Generator().manual_seed(182)
    features = torch.randn(7, 11, 5, generator=generator, dtype=dtype)
    labels = torch.randint(0, 7, (7, 11), generator=generator)
    lengths, counts = [1, 3, 11, 8, 4, 10, 6], [1, 1, 8, 2, 4, 3, 6]
    valid = torch.arange(11)[None, :] < torch.tensor(lengths)[:, None]
    selected = torch.zeros_like(valid)
    for row, (length, count) in enumerate(zip(lengths, counts)):
        selected[row, torch.randperm(length, generator=generator)[:count]] = True
    labels[~selected] = -100
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(921)
        model = torch.nn.Linear(5, 7).to(dtype=dtype)
    return model, features, labels, selected, valid


def oracle(logits, labels, selected):
    # Independently loop over records and use log_softmax, not the tested reducer.
    return torch.stack([-logits[row, mask].log_softmax(-1)
        .gather(1, labels[row, mask, None]).mean() for row, mask in enumerate(selected)]).mean()


GROUPS = [
    [[0, 1, 2, 3, 4, 5, 6]],
    [[i] for i in range(7)],
    [[0, 1], [2, 3], [4, 5], [6]],
    [[0], [1, 2, 3, 4, 5, 6]],
    [[6, 0], [2], [3, 1, 5, 4]],
]


@pytest.mark.parametrize('groups', GROUPS)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_all_parameter_gradients_match_full_batch_for_every_group_and_tail(groups, dtype):
    model, features, labels, selected, valid = artificial(dtype)
    reference = deepcopy(model)
    expected = oracle(reference(features), labels, selected)
    expected.backward()
    global_count = actual_global_record_count([len(group) for group in groups])
    assert global_count == len(features) == 7
    contributions = []
    for indices in groups:
        # Each microbatch pads only to its own longest record.
        width = int(valid[indices].sum(1).max())
        result = record_normalized_ce(model(features[indices, :width]), labels[indices, :width],
            selected[indices, :width], valid[indices, :width], global_record_count=global_count)
        result.contribution.backward()
        contributions.append(result.contribution.detach())
        assert result.local_record_count == len(indices)
        assert result.selected_counts.tolist() == selected[indices].sum(1).tolist()
    tolerance = 3e-7 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(sum(contributions), expected.detach(), rtol=tolerance, atol=tolerance)
    assert set(dict(model.named_parameters())) == {'weight', 'bias'}
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name and parameter.grad is not None and other.grad is not None
        torch.testing.assert_close(parameter.grad, other.grad, rtol=tolerance, atol=tolerance)


def test_unselected_labels_and_nonfinite_logits_do_not_enter_loss_or_gradient():
    model, features, labels, selected, valid = artificial()
    logits = model(features).detach().requires_grad_()
    contaminated = logits.detach().clone()
    contaminated[~selected] = float('nan')
    contaminated.requires_grad_()
    altered_labels = labels.clone(); altered_labels[~selected] = 900_000
    left = record_normalized_ce(logits, labels, selected, valid, global_record_count=7)
    right = record_normalized_ce(contaminated, altered_labels, selected, valid, global_record_count=7)
    left.contribution.backward(); right.contribution.backward()
    assert torch.equal(left.contribution, right.contribution)
    assert torch.equal(logits.grad, contaminated.grad)
    assert torch.equal(logits.grad[~selected], torch.zeros_like(logits.grad[~selected]))


def test_tail_uses_actual_records_and_never_averages_contributions_again():
    model, features, labels, selected, valid = artificial()
    direct = oracle(model(features), labels, selected)
    groups = [[0, 1, 2, 3], [4, 5, 6]]
    count = actual_global_record_count(map(len, groups))
    parts = [record_normalized_ce(model(features[index]), labels[index], selected[index], valid[index],
        global_record_count=count).contribution for index in groups]
    torch.testing.assert_close(sum(parts), direct)
    assert not torch.isclose(sum(parts)/len(parts), direct)
    assert not torch.isclose(sum(parts)*count/8, direct)  # nominal 8 is wrong for actual 7


def test_target_is_record_mean_and_different_from_global_token_mean():
    logits = torch.tensor([[[4., 0.]]*4, [[0., 4.]]*4], dtype=torch.float64)
    labels = torch.zeros(2, 4, dtype=torch.long)
    selected = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    result = record_normalized_ce(logits, labels, selected, torch.ones_like(selected), global_record_count=2)
    token_mean = -logits[selected].log_softmax(-1)[:, 0].mean()
    torch.testing.assert_close(result.contribution, oracle(logits, labels, selected))
    assert not torch.isclose(result.contribution, token_mean)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_logits_use_float32_ce_and_preserve_gradients(dtype):
    logits = torch.tensor([[[.5, -.5], [1., -.5]]], dtype=dtype, requires_grad=True)
    labels = torch.tensor([[0, 1]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    result = record_normalized_ce(logits, labels, mask, mask, global_record_count=1)
    assert result.contribution.dtype == torch.float32
    torch.testing.assert_close(result.contribution, oracle(logits.float(), labels, mask))
    result.contribution.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_negative_infinity_may_exclude_non_target_classes():
    logits = torch.tensor([[[0., float('-inf'), 1.]]], requires_grad=True)
    labels = torch.tensor([[2]]); mask = torch.ones(1, 1, dtype=torch.bool)
    result = record_normalized_ce(logits, labels, mask, mask, global_record_count=1)
    result.contribution.backward()
    assert torch.isfinite(result.contribution) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 1] == 0


def test_finite_large_means_do_not_overflow_intermediate_sums():
    logits = torch.tensor([[[0., -2e38]]*3]*2, requires_grad=True)
    labels = torch.ones(2, 3, dtype=torch.long); mask = torch.ones(2, 3, dtype=torch.bool)
    result = record_normalized_ce(logits, labels, mask, mask, global_record_count=2)
    torch.testing.assert_close(result.contribution, torch.tensor(2e38))
    result.contribution.backward()
    assert torch.isfinite(result.contribution) and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('counts', [[], [0], [1, 0], [-1], [True], [2.0], ['2']])
def test_invalid_window_counts_are_rejected(counts):
    with pytest.raises(ValueError, match='positive integer'):
        actual_global_record_count(counts)


@pytest.mark.parametrize('denominator', [0, -1, 6, 7.0, True, '7'])
def test_invalid_denominator_cannot_silently_change_population(denominator):
    model, features, labels, selected, valid = artificial()
    with pytest.raises(ValueError, match='global_record_count'):
        record_normalized_ce(model(features), labels, selected, valid, global_record_count=denominator)


@pytest.mark.parametrize('mutation', ['empty_record', 'zero_selected', 'padding_selected', 'negative_label',
    'large_label', 'labels_dtype', 'mask_dtype', 'shape', 'integer_logits', 'zero_records', 'zero_positions', 'zero_classes'])
def test_bad_inputs_fail_closed_without_dropping_records(mutation):
    model, features, labels, selected, valid = artificial()
    logits = model(features)
    if mutation == 'empty_record': valid[0] = False; selected[0] = False
    elif mutation == 'zero_selected': selected[0] = False
    elif mutation == 'padding_selected': selected[0, -1] = True
    elif mutation == 'negative_label': labels[selected] = -100
    elif mutation == 'large_label': labels[selected] = logits.shape[-1]
    elif mutation == 'labels_dtype': labels = labels.float()
    elif mutation == 'mask_dtype': selected = selected.long()
    elif mutation == 'shape': valid = valid[:, :-1]
    elif mutation == 'integer_logits': logits = logits.long()
    elif mutation == 'zero_records': logits = logits[:0]
    elif mutation == 'zero_positions': logits = logits[:, :0]
    else: logits = logits[:, :, :0]
    with pytest.raises(ValueError):
        record_normalized_ce(logits, labels, selected, valid, global_record_count=7)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_selected_ce_is_rejected(value):
    logits = torch.tensor([[[value, 0.]]], requires_grad=True)
    labels = torch.tensor([[0]]); mask = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(FloatingPointError):
        record_normalized_ce(logits, labels, mask, mask, global_record_count=1)


def test_all_negative_infinity_classes_are_rejected():
    logits = torch.full((1, 1, 3), float('-inf'))
    labels = torch.tensor([[0]]); mask = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(FloatingPointError, match='finite'):
        record_normalized_ce(logits, labels, mask, mask, global_record_count=1)
