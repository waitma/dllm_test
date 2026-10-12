"""Activate pllm: USE_TF=0 python -m pytest -q scripts/tests/immune_llada/test_token_endpoints.py.

Scripted integer predictions expose END-recall/first-boundary counterexamples.
No actual sequence corpus, checkpoint, or GPU is read.
"""
from types import SimpleNamespace
import pytest
import torch

from scripts.architecture_prototypes.validation_metrics import (
    chain_endpoint_totals, mask_mean, selected_token_metrics, summarize,
)


class ScriptedModel:
    tokenizer = SimpleNamespace(mask_token_id=9)
    terminal_id = 1

    def __init__(self, predictions):
        self.predictions = torch.tensor(predictions)

    @staticmethod
    def feature_args(batch):
        return {key: batch[key] for key in ('input_ids', 'attention_mask', 'position_ids')}

    def denoise(self, **features):
        shape = (*self.predictions.shape, 3)
        logits = torch.zeros(shape).scatter(-1, self.predictions[..., None], 8.)
        return logits, None


def batch():
    ids = torch.tensor([[0, 0, 1, 9, 9, 9]])
    return dict(input_ids=ids, labels=torch.tensor([[0, 0, 1, -100, -100, -100]]),
        attention_mask=torch.ones_like(ids), eligible_mask=torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool),
        residue_mask=torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool),
        terminal_mask=torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.bool),
        target_slot_mask=torch.ones_like(ids, dtype=torch.bool), chain_ids=torch.zeros_like(ids),
        position_ids=torch.arange(6)[None])


@pytest.mark.parametrize('predictions,recall,precision,fpr,first,missing,mae', [
    ([1, 0, 1, 0, 0, 0], 1., .5, .5, 0., 0., 2.),  # early END plus true END
    ([0, 0, 0, 0, 1, 0], 0., None, 0., 0., 0., 2.),  # END only in unsupervised tail
    ([0, 0, 0, 0, 0, 0], 0., None, 0., 0., 1., None),  # no END
    ([0, 0, 1, 0, 0, 0], 1., 1., 0., 1., 0., 0.),
    ([0, 0, 1, 1, 1, 0], 1., 1., 0., 1., 0., 0.),  # later ENDs do not move first
])
def test_recall_is_not_first_end_accuracy(predictions, recall, precision, fpr, first, missing, mae):
    result = summarize(selected_token_metrics(ScriptedModel([predictions]), batch(), 1.)['sample_totals'])
    for key, value in dict(terminal_accuracy=recall, terminal_precision=precision,
        terminal_false_positive_rate=fpr, first_terminal_accuracy=first,
        terminal_missing_rate=missing, length_mae=mae).items():
        assert result[key] == value
    assert result['chain_count'] == 1
    assert result['length_compared_count'] == 1-int(missing)


def test_partial_masks_have_no_endpoint_metrics_or_endpoint_mask_mean():
    model = ScriptedModel([[1, 0, 1, 0, 0, 0]])
    levels = {}
    for ratio in (.25, .5, .75, 1.):
        result = summarize(selected_token_metrics(model, batch(), ratio)['sample_totals'])
        levels[f'mask{int(ratio*100)}'] = result
        if ratio != 1:
            assert result['chain_count'] == 0
            assert result['first_terminal_accuracy'] is None and result['length_mae'] is None
    assert not {'first_terminal_accuracy', 'terminal_missing_rate', 'length_mae'} & mask_mean(levels).keys()


def test_later_chain_end_does_not_fill_earlier_missing_chain():
    row = batch()
    row = {key: torch.cat((value, value), 1) for key, value in row.items()}
    row['chain_ids'][:, 6:] = 1
    prediction = torch.zeros((1, 12), dtype=torch.bool)
    prediction[0, 8] = True
    # Chain 0 missing, chain 1 exact. Only chain 1 contributes length MAE.
    assert chain_endpoint_totals(row, prediction, full_mask=True).tolist() == [[2., 1., 1., 0.]]


def test_padding_and_non_target_end_predictions_are_ignored():
    row = batch()
    # Add a context prefix and padding suffix, both predicting END.
    for key, value in list(row.items()):
        row[key] = torch.cat((value.new_zeros(1, 1), value, value.new_zeros(1, 1)), 1)
    row['attention_mask'][0, 0] = 1
    row['chain_ids'][0, (0, -1)] = -1
    predicted = torch.tensor([[1, 0, 0, 1, 0, 0, 0, 1]], dtype=torch.bool)
    assert chain_endpoint_totals(row, predicted, full_mask=True).tolist() == [[1., 1., 0., 0.]]


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'outside_slot', 'negative_chain'])
def test_invalid_boundary_metadata_is_not_silently_skipped(mutation):
    row = batch()
    if mutation == 'missing':
        row['terminal_mask'].zero_()
    elif mutation == 'duplicate':
        row['terminal_mask'][0, 1] = True
    elif mutation == 'outside_slot':
        row['target_slot_mask'][0, 2] = False
    else:
        row['chain_ids'].fill_(-1)
    with pytest.raises(ValueError, match='one true END'):
        chain_endpoint_totals(row, torch.zeros((1, 6), dtype=torch.bool), full_mask=True)


def test_three_lengths_same_prediction_can_have_perfect_recall_but_one_third_first():
    # Same predicted pattern END at positions 1, 3, 5; three different true lengths.
    predicted = torch.tensor([[0, 1, 0, 1, 0, 1]], dtype=torch.bool).expand(3, -1)
    rows = {key: value.repeat(3, 1) for key, value in batch().items()}
    rows['eligible_mask'].zero_()
    rows['terminal_mask'].zero_()
    for i, true_position in enumerate((1, 3, 5)):
        rows['eligible_mask'][i, :true_position+1] = True
        rows['terminal_mask'][i, true_position] = True
    ends = chain_endpoint_totals(rows, predicted, full_mask=True).sum(0)
    recall = (predicted & rows['terminal_mask']).sum()/rows['terminal_mask'].sum()
    assert recall == 1
    assert ends.tolist() == [3., 1., 0., 6.]
    assert ends[1]/ends[0] == pytest.approx(1/3)


def test_end_prediction_at_relation_prefix_is_not_a_target_false_positive():
    row = batch()
    for key, value in list(row.items()):
        row[key] = torch.cat((value.new_zeros(1, 1), value), 1)
    row['input_ids'][0, 0] = row['labels'][0, 0] = 2
    row['attention_mask'][0, 0] = row['eligible_mask'][0, 0] = 1
    row['chain_ids'][0, 0] = -1
    model = ScriptedModel([[1, 0, 0, 1, 0, 0, 0]])
    model.tokenizer = SimpleNamespace(mask_token_id=9, token_to_id={'<binding>': 2})
    measured = summarize(selected_token_metrics(model, row, 1.)['sample_totals'])
    assert measured['relation_count'] == 1 and measured['relation_accuracy'] == 0
    assert measured['terminal_false_positive_count'] == 0
    assert measured['terminal_precision'] == 1 and measured['first_terminal_accuracy'] == 1
