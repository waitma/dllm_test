"""Run with pllm: pytest -q scripts/tests/immune_llada/test_uptcr_original_validation.py."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'downstream/benchmark'))
from tcr_binding.run_retrained_ours import train_head


def test_original_validation_controls_normalization_and_membership(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    # Validation distribution deliberately differs greatly from training.
    features = np.array([[0., 0.], [2., 4.], [100., 200.], [200., 300.]], dtype=np.float32)
    mask = np.array([False, False, True, True])
    head = train_head(features, np.arange(4), np.array([0, 1, 0, 1]), seed=0,
                      hidden=4, dropout=0., learning_rate=.001, weight_decay=0.,
                      batch_size=2, max_epochs=2, patience=2, val_fraction=.1,
                      validation_mask=mask)
    assert head['n_train'] == head['n_validation'] == 2
    np.testing.assert_array_equal(head['feature_mean'], [1., 2.])
    np.testing.assert_array_equal(head['feature_std'], [1., 2.])
    np.testing.assert_array_equal(features[2:], [[100., 200.], [200., 300.]])


def test_empty_official_validation_rejected():
    with pytest.raises(ValueError, match='nonempty'):
        train_head(np.ones((4, 2)), np.arange(4), np.array([0, 1, 0, 1]), seed=0,
                   hidden=4, dropout=0., learning_rate=.001, weight_decay=0.,
                   batch_size=2, max_epochs=1, patience=1, val_fraction=.1,
                   validation_mask=np.zeros(4, dtype=bool))
