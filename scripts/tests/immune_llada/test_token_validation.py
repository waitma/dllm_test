"""Run after activating pllm: USE_TF=0 python -m pytest -q scripts/tests/immune_llada/test_token_validation.py.

Artificial arrays/fixtures only; no corpus access, GPU jobs, or retained checkpoints.
"""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import math
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import TrainerCallback, TrainingArguments

from scripts.architecture_prototypes.validation_metrics import (
    METRICS, TOKEN_METRICS, FIELDS, TokenValidationTrainer, mask_mean, selected_token_metrics, source_macro, summarize,
)
from scripts.architecture_prototypes.expanded_fixtures import build_fixture_split
from scripts.run_immune_research import ResearchTrainer, collator_for
from dllm.pipelines.immune_llada.research_refined.model import RefinedModel
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ArrayModel:
    tokenizer = SimpleNamespace(mask_token_id=9)
    terminal_id = 1

    @staticmethod
    def feature_args(batch):
        return {key: batch[key] for key in ('input_ids', 'attention_mask', 'position_ids')}

    def denoise(self, **features):
        self.features = features
        # Class 0 has probability 1/2; classes 1 and 2 each have probability 1/4.
        logits = torch.tensor([math.log(2), 0., 0.]).expand(*features['input_ids'].shape, 3)
        return logits, None


def array_batch():
    # Last position in row 0 is padding even though eligible_mask says True.
    ids = torch.tensor([[0, 1, 2, 2], [2, 0, 1, 1]])
    return dict(input_ids=ids, labels=ids.clone(),
                attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool),
                eligible_mask=torch.tensor([[1, 1, 0, 1], [1, 1, 1, 0]], dtype=torch.bool),
                residue_mask=torch.tensor([[1, 0, 0, 1], [1, 1, 0, 0]], dtype=torch.bool),
                terminal_mask=torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.bool),
                position_ids=torch.arange(4).expand(2, -1), chain_ids=torch.zeros_like(ids),
                target_slot_mask=torch.ones_like(ids, dtype=torch.bool))


def test_literal_counts_padding_visible_tokens_and_feature_whitelist():
    model, batch = ArrayModel(), array_batch()
    original = batch['input_ids'].clone()
    result = selected_token_metrics(model, batch, 1.)
    expected = summarize(result['sample_totals'])
    oracle = dict(selected_count=5, aa_count=3, terminal_count=2,
        loss=8*math.log(2)/5, aa_loss=4*math.log(2)/3, terminal_loss=2*math.log(2),
        aa_accuracy=2/3, terminal_accuracy=0.)
    assert {key: expected[key] for key in oracle} == pytest.approx(oracle, abs=2e-7)
    assert expected['chain_count'] == 2 and expected['terminal_missing_rate'] == 1
    assert expected['terminal_precision'] is None and expected['length_mae'] is None
    assert torch.equal(batch['input_ids'], original)
    assert model.features['input_ids'].tolist() == [[9, 9, 2, 2], [9, 9, 9, 1]]
    assert set(model.features) == {'input_ids', 'attention_mask', 'position_ids'}


@pytest.mark.parametrize('ratio', [.25, .5, .75, 1.])
def test_per_record_mask_and_totals_are_batch_invariant(ratio):
    batch = array_batch()
    joint = selected_token_metrics(ArrayModel(), batch, ratio)['sample_totals']
    individual = [selected_token_metrics(ArrayModel(), {k: v[i:i+1] for k, v in batch.items()}, ratio)
                  ['sample_totals'] for i in range(2)]
    assert torch.equal(joint, torch.cat(individual))


def test_missing_terminal_subset_is_null_not_zero():
    result = summarize([6, 3, 6, 3, 0, 0, 2, 0] + [0]*(len(FIELDS)-8))
    assert result['terminal_loss'] is result['terminal_accuracy'] is None
    assert result['aa_loss'] == 2
    assert all(value is None for key, value in summarize([0]*len(FIELDS)).items() if key in METRICS)


@pytest.mark.parametrize('totals', [
    [1, 2, 1, 1, 0, 0, 0, 0],  # population mismatch
    [1, 1, 1, 1, 0, 0, 2, 0],  # correct > selected
    [2, 1, 1, 1, 0, 0, 1, 0],  # additive loss mismatch
    [1, 1.5, 1, 1.5, 0, 0, 1, 0],
    [float('nan')]*8,
    [-1]*8,
    [1, 0, 1, 0, 0, 0, 0, 0],
])
def test_bad_totals_rejected(totals):
    with pytest.raises(ValueError):
        summarize(totals+[0]*(len(FIELDS)-8))


@pytest.mark.parametrize('mutation', ['labels', 'partition', 'overlap'])
def test_corrupt_evaluation_contract_rejected(mutation):
    batch = array_batch()
    if mutation == 'labels':
        batch['labels'][0, 0] = 2
    elif mutation == 'partition':
        batch['residue_mask'][0, 0] = False
    else:
        batch['terminal_mask'][0, 0] = True
    with pytest.raises(ValueError):
        selected_token_metrics(ArrayModel(), batch, 1.)


def test_macro_micro_and_mask_mean_are_distinct_and_exclude_undefined():
    large = torch.tensor([18, 9, 18, 9, 0, 0, 3, 0] + [0]*(len(FIELDS)-8))
    small = torch.tensor([1, 1, 0, 0, 1, 1, 0, 1] + [0]*(len(FIELDS)-8))
    macro = source_macro({'large': summarize(large), 'small': summarize(small)})
    micro = summarize(torch.stack([large, small]))
    assert macro['loss'] == 1.5 and micro['loss'] == 1.9
    assert macro['terminal_loss'] == 1 and macro['terminal_loss_sources'] == 1
    assert macro['aa_accuracy_sources'] == 1
    levels = dict(zip(('mask25', 'mask50', 'mask75', 'mask100'),
                      [summarize(large), summarize(small), macro, micro]))
    mean = mask_mean(levels)
    assert mean['loss'] == pytest.approx((2+1+1.5+1.9)/4)
    assert mean['terminal_loss_levels'] == 3
    with pytest.raises(ValueError):
        mask_mean({'mask100': micro})
    with pytest.raises(ValueError):
        source_macro({})


def model_data():
    config = ImmuneConfig(ablation='A4', d_model=16, n_layers=2, n_heads=2,
        mlp_hidden_size=32, full_capacity=24, fragment_capacity=16, max_length=256)
    rows = build_fixture_split(train_size=36, valid_size=18, full_capacity=24,
        fragment_capacity=16, max_length=256).valid_records
    return RefinedModel(config), collator_for(config), rows


def settings(directory, batch_size, bf16=False):
    return TrainingArguments(output_dir=directory, use_cpu=True, per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=2, bf16=bf16, remove_unused_columns=False,
        label_names=['labels'], prediction_loss_only=True, report_to='none', disable_tqdm=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_real_model_metrics_are_batch_invariant(dtype):
    model, collate, rows = model_data()
    model.eval()
    chosen = [rows[i] for i in (0, 6, 17)]
    context = nullcontext() if dtype == torch.float32 else torch.autocast('cpu', dtype=dtype)
    with sdpa_kernel(SDPBackend.MATH), context:
        for ratio in (.25, .5, .75, 1.):
            joint = summarize(selected_token_metrics(model, collate(chosen), ratio)['sample_totals'])
            parts = torch.cat([selected_token_metrics(model, collate([row]), ratio)['sample_totals']
                               for row in chosen])
            assert joint == pytest.approx(summarize(parts), abs=2e-6)


def test_trainer_integrates_four_masks_weighting_and_preserves_rng():
    model, collate, rows = model_data()
    dataset = {'short': [rows[0]], 'varied': [rows[i] for i in (6, 10, 17)]}
    with TemporaryDirectory(prefix='token-validation-') as directory:
        first = TokenValidationTrainer(model=deepcopy(model), args=settings(directory, 1),
                                       eval_dataset=dataset, data_collator=collate)
        second = TokenValidationTrainer(model=deepcopy(model), args=settings(directory, 3),
                                        eval_dataset=dataset, data_collator=collate)
        state = torch.random.get_rng_state().clone()
        with sdpa_kernel(SDPBackend.MATH):
            one, three = first.evaluate(), second.evaluate()
        assert torch.equal(state, torch.random.get_rng_state())
        stable = [key for key in one if not key.endswith(('runtime', '_time', 'samples_per_second', 'steps_per_second'))]
        assert {key: one[key] for key in stable} == pytest.approx({key: three[key] for key in stable}, abs=2e-6)
        assert one['eval_loss'] == one['eval_mask100_source_macro_loss']
        for weighting in ('source_macro', 'token_micro'):
            expected = sum(one[f'eval_mask{ratio}_{weighting}_loss'] for ratio in (25, 50, 75, 100))/4
            assert one[f'eval_mask_mean_{weighting}_loss'] == pytest.approx(expected)
        assert first._validation_ratio == 1.
        assert first._token_validation_totals is None
        with pytest.raises(ValueError, match='must have records'):
            first.evaluate({'empty': []})


def test_training_loss_and_gradients_are_unchanged():
    model, collate, rows = model_data()
    with TemporaryDirectory(prefix='token-training-parity-') as directory:
        legacy = ResearchTrainer(model=deepcopy(model), args=settings(directory, 2), data_collator=collate)
        prototype = TokenValidationTrainer(model=deepcopy(model), args=settings(directory, 2), data_collator=collate)
        batch = collate([rows[0], rows[17]])
        losses = []
        with sdpa_kernel(SDPBackend.MATH):
            for trainer in (legacy, prototype):
                torch.manual_seed(81)
                trainer.model.train()
                loss = trainer.compute_loss(trainer.model, batch)
                loss.backward()
                losses.append(loss.detach())
        assert torch.equal(*losses)
        for (name, a), (other, b) in zip(legacy.model.named_parameters(), prototype.model.named_parameters()):
            assert name == other
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                assert torch.equal(a.grad, b.grad), name


class RandomCallback(TrainerCallback):
    def __init__(self, fail=False):
        self.logs, self.evaluations, self.fail = [], [], fail

    @staticmethod
    def draw():
        random.random()
        np.random.random()
        torch.rand(1)

    def on_prediction_step(self, args, state, control, **kwargs):
        self.draw()

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.draw()
        self.logs.append(dict(logs))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.draw()
        self.evaluations.append(dict(metrics))
        if self.fail:
            raise RuntimeError('injected evaluation callback error')


@pytest.mark.parametrize('fail', [False, True])
def test_callback_once_with_complete_metrics_and_all_rng_restored(fail):
    model, collate, rows = model_data()
    callback = RandomCallback(fail=fail)
    with TemporaryDirectory(prefix='token-callback-') as directory:
        trainer = TokenValidationTrainer(model=model, args=settings(directory, 2),
            eval_dataset={'one': [rows[0]], 'three': [rows[i] for i in (6, 10, 17)]},
            data_collator=collate, callbacks=[callback])
        before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
        with sdpa_kernel(SDPBackend.MATH):
            if fail:
                with pytest.raises(RuntimeError, match='injected'):
                    trainer.evaluate()
            else:
                trainer.evaluate()
        assert random.getstate() == before[0]
        after_numpy = np.random.get_state()
        assert after_numpy[0] == before[1][0]
        assert np.array_equal(after_numpy[1], before[1][1])
        assert after_numpy[2:] == before[1][2:]
        assert torch.equal(torch.random.get_rng_state(), before[2])
        assert len(callback.logs) == len(callback.evaluations) == 1
        metrics = callback.evaluations[0]
        assert metrics['eval_records_per_mask'] == 4
        assert metrics['eval_record_evaluations'] == 16
        for prefix in ('eval_mask_mean_token_micro', 'eval_mask_mean_source_macro',
                       'eval_mask100_token_micro', 'eval_mask100_source_macro'):
            for key in TOKEN_METRICS if '_mask_mean_' in prefix else METRICS:
                assert prefix+'_'+key in callback.logs[0]
                assert prefix+'_'+key in metrics
        assert trainer._validation_ratio == 1.
        assert trainer._token_validation_totals is None


def test_persistent_worker_source_cache_rejected():
    model, collate, rows = model_data()
    with TemporaryDirectory(prefix='token-persistent-') as directory:
        trainer = TokenValidationTrainer(model=model, args=settings(directory, 2),
                                         eval_dataset=[rows[0]], data_collator=collate)
        trainer.args.dataloader_persistent_workers = True
        with pytest.raises(NotImplementedError, match='persistent workers'):
            trainer.evaluate()


def test_trainer_cpu_bf16_context_reaches_denoise(monkeypatch):
    model, collate, rows = model_data()
    actual, observations = model.denoise, []

    def observed(**features):
        result = actual(**features)
        observations.append((torch.is_autocast_enabled('cpu'), result[0].dtype))
        return result

    monkeypatch.setattr(model, 'denoise', observed)
    with TemporaryDirectory(prefix='token-autocast-') as directory:
        trainer = TokenValidationTrainer(model=model, args=settings(directory, 2, bf16=True),
                                         eval_dataset=[rows[0]], data_collator=collate)
        with sdpa_kernel(SDPBackend.MATH):
            metrics = trainer.evaluate()
        assert observations == [(True, torch.bfloat16)]*4
        assert math.isfinite(metrics['eval_loss'])


def test_failed_loop_restores_rng_and_memory_tracker_then_can_retry():
    model, collate, rows = model_data()

    def failing_collate(records):
        RandomCallback.draw()
        raise RuntimeError('injected collator error')

    with TemporaryDirectory(prefix='token-loop-error-') as directory:
        arguments = settings(directory, 2)
        arguments.skip_memory_metrics = False
        trainer = TokenValidationTrainer(model=model, args=arguments, eval_dataset=[rows[0]],
                                         data_collator=failing_collate)
        before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
        with sdpa_kernel(SDPBackend.MATH):
            with pytest.raises(RuntimeError, match='collator error'):
                trainer.evaluate()
            assert random.getstate() == before[0]
            after = np.random.get_state()
            assert after[0] == before[1][0] and np.array_equal(after[1], before[1][1])
            assert after[2:] == before[1][2:]
            assert torch.equal(torch.random.get_rng_state(), before[2])
            assert trainer._validation_ratio == 1.
            assert trainer._token_validation_totals is None
            trainer.data_collator = collate
            metrics = trainer.evaluate()
        assert math.isfinite(metrics['eval_loss'])


@pytest.mark.parametrize('relation', ['binding', 'nonbinding'])
def test_actual_collator_relation_supervision_stays_in_total_ce(relation):
    model, collate, rows = model_data()
    row = replace(rows[0], labels={'relation': relation},
                  metadata={**rows[0].metadata, 'relation_supervised': True})
    model.eval()
    with sdpa_kernel(SDPBackend.MATH):
        measured = summarize(selected_token_metrics(model, collate([row]), 1.)['sample_totals'])
    assert measured['relation_count'] == 1
    assert measured['relation_accuracy'] in (0., 1.)
    assert measured['selected_count'] == measured['aa_count']+measured['terminal_count']+1
    components = (measured['aa_loss']*measured['aa_count']+
                  measured['terminal_loss']*measured['terminal_count']+measured['relation_loss'])
    assert measured['loss']*measured['selected_count'] == pytest.approx(components)
    assert measured['chain_count'] == 2
