"""Check positive relation accounting on integer tensors, without a model job.

Activate pllm, then run python -B -m pytest -q with this absolute file path.
No sequence records, model training, checkpoints, or GPU execution are involved.
"""
import math

import pytest
import torch

from scripts.research.context_pilot_metrics import combine, measure


class UniformModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = type('Tokenizer', (), {'mask_token_id': 7})()
        self.inputs = []

    def feature_args(self, batch):
        return {'input_ids': batch['input_ids']}

    def denoise(self, input_ids):
        self.inputs.append(input_ids.clone())
        return torch.zeros(*input_ids.shape, 8), None


def batch(relation=True):
    eligible = torch.tensor([[False, relation, False, False, True, True, True]])
    ids = torch.arange(7)[None]
    return dict(input_ids=ids, labels=ids.masked_fill(~eligible, -100),
                attention_mask=torch.ones_like(eligible), eligible_mask=eligible,
                chain_ids=torch.tensor([[-1, -1, 0, 0, 1, 1, 1]]),
                residue_mask=torch.tensor([[False, False, True, False, True, True, False]]),
                terminal_mask=torch.tensor([[False, False, False, False, False, False, True]]))


def test_positive_relation_is_separate_from_context_and_target_counts():
    model = UniformModel().train()
    before = torch.get_rng_state().clone()
    result = measure(model, batch())
    assert model.training and torch.equal(before, torch.get_rng_state())
    full = result['mask_levels']['mask100']
    assert (full['selected_count'], full['aa_count'], full['terminal_count'], full['relation_count']) == (4, 2, 1, 1)
    assert full['loss'] == pytest.approx(math.log(8))
    assert full['relation_loss_sum'] == pytest.approx(math.log(8))
    assert full['relation_accuracy'] == 0
    for seen in model.inputs:
        assert seen[0, 2:4].tolist() == [2, 3]
    assert model.inputs[-1][0, 1].item() == 7
    assert model.inputs[-1][0, 4:].tolist() == [7, 7, 7]


def test_pooling_uses_observed_relation_denominators_and_absence_is_none():
    model = UniformModel()
    positive = measure(model, batch())
    absent = measure(model, batch(relation=False))
    assert absent['mask_levels']['mask100']['relation_loss'] is None
    pooled = combine([positive, absent])['mask_levels']['mask100']
    assert pooled['selected_count'] == 7 and pooled['relation_count'] == 1
    assert pooled['relation_loss'] == pytest.approx(math.log(8))
    assert pooled['aa_count'] == 4 and pooled['terminal_count'] == 2
