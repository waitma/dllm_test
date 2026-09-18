"""CPU tests for source-macro evaluation and checkpoint metric publication.

Run from the project root with the training environment:
python -m pytest scripts/tests/immune_llada/test_eval_source_mean.py -q
No model weights or datasets are loaded.
"""

from types import SimpleNamespace

import pytest
import transformers

from examples.llada.protein_pretrain_esmc import FusionTrainer


def make_trainer(monkeypatch, datasets, metrics, counts=None):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        return dict(metrics)

    monkeypatch.setattr(transformers.Trainer, "evaluate", evaluate)
    trainer = object.__new__(FusionTrainer)
    trainer.eval_dataset = datasets
    trainer.eval_source_rows = counts
    trainer.state = SimpleNamespace(log_history=[])
    trainer.log = lambda values: trainer.state.log_history.append(dict(values))
    return trainer


def test_unequal_source_sizes_use_equal_weights_and_publish_selection_loss(monkeypatch):
    metrics = {"eval_oas_loss": 1.0, "eval_trait_loss": 3.0}
    trainer = make_trainer(monkeypatch, {"oas": object(), "trait": object()}, metrics,
                           {"oas": 900, "trait": 100})
    result = trainer.evaluate()
    assert result["eval_loss"] == pytest.approx(2.0)  # Former row mean: 1.2.
    assert result["eval_oas_loss"] == 1.0
    assert result["eval_trait_loss"] == 3.0
    assert trainer.state.log_history == [{"eval_loss": 2.0}]


def test_missing_source_does_not_publish_partial_mean(monkeypatch):
    trainer = make_trainer(monkeypatch, {"oas": object(), "trait": object()},
                           {"eval_oas_loss": 1.0})
    assert "eval_loss" not in trainer.evaluate()
    assert trainer.state.log_history == []


def test_explicit_dataset_and_prefix_define_sources(monkeypatch):
    trainer = make_trainer(monkeypatch, {"old": object()},
                           {"check_a_loss": 2.0, "check_b_loss": 6.0}, {"old": 100})
    result = trainer.evaluate(eval_dataset={"a": object(), "b": object()},
                              metric_key_prefix="check")
    assert result["check_loss"] == 4.0
    assert trainer.state.log_history == [{"check_loss": 4.0}]


def test_single_source_dict_works_without_row_counts(monkeypatch):
    trainer = make_trainer(monkeypatch, {"oas": object()}, {"eval_oas_loss": 2.5})
    assert trainer.evaluate()["eval_loss"] == 2.5


@pytest.mark.parametrize("datasets,metrics", [
    (object(), {"eval_loss": 2.5}),
    ({}, {}),
    ({"oas": object()}, {"eval_oas_loss": 2.0, "eval_loss": 2.5}),
])
def test_existing_loss_leaf_and_empty_evaluation_are_unchanged(monkeypatch, datasets, metrics):
    trainer = make_trainer(monkeypatch, datasets, metrics)
    assert trainer.evaluate() == metrics
    assert trainer.state.log_history == []
