"""Real CPU Trainer/Accelerate accumulation over artificial integer fixtures.

Activate pllm, then USE_TF=0 python -m pytest scripts/tests/immune_llada/test_record_loss_framework.py -q
Optional RECORD_LOSS_FRAMEWORK_EVIDENCE=/absolute/directory writes small JSON
traces. No training_step/backward mocks, corpus input, GPU, or checkpoint save.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainerCallback, TrainingArguments

from scripts.architecture_prototypes.record_loss_framework import RecordWindowTrainer


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyTokenModel(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(5, 7).to(dtype=dtype)

    def forward(self, features, selected_mask, valid_mask, record_ids, _record_window):
        return dict(logits=self.linear(features), selected_mask=selected_mask,
            valid_mask=valid_mask, record_ids=record_ids, _record_window=_record_window)


class SequentialRecordTrainer(RecordWindowTrainer):
    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)


def invented_records(count, dtype):
    generator = torch.Generator().manual_seed(317)
    records = []
    for index in range(count):
        length = 2+index % 8
        selected = torch.arange(length) % (1+index % 3) == 0
        labels = torch.randint(0, 7, (length,), generator=generator)
        labels[~selected] = -100
        records.append(dict(features=torch.randn(length, 5, generator=generator, dtype=dtype),
            labels=labels, selected_mask=selected, valid_mask=torch.ones(length, dtype=torch.bool),
            record_ids=torch.tensor(index)))
    return records


def collate(records):
    result = {}
    for name in ('features', 'labels', 'selected_mask', 'valid_mask'):
        result[name] = torch.nn.utils.rnn.pad_sequence([row[name] for row in records],
            batch_first=True, padding_value=-100 if name == 'labels' else 0)
    result['record_ids'] = torch.stack([row['record_ids'] for row in records])
    return result


def independent_oracle(model, records):
    losses = []
    for row in records:
        chosen = row['selected_mask']
        log_prob = model.linear(row['features'][chosen]).log_softmax(-1)
        losses.append(-log_prob.gather(1, row['labels'][chosen, None]).mean())
    return torch.stack(losses).mean()


class CheckEveryUpdate(TrainerCallback):
    def __init__(self, reference, records, dtype):
        self.reference, self.records = reference, records
        self.optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
        self.tolerance = 3e-7 if dtype == torch.float32 else 2e-14
        self.events = []
        self.trainer = None

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        trainer = self.trainer
        window = trainer.window_trace[-1]
        assert trainer._consumed == len(window['micro_sizes'])
        ids = [index for group in window['record_ids'] for index in group]
        assert len(ids) == window['actual_records']
        self.optimizer.zero_grad()
        expected = independent_oracle(self.reference, [self.records[index] for index in ids])
        expected.backward()
        maximum = 0.
        for (name, actual), (other, reference) in zip(model.named_parameters(), self.reference.named_parameters()):
            assert name == other and actual.grad is not None and reference.grad is not None
            torch.testing.assert_close(actual.grad, reference.grad, rtol=self.tolerance, atol=self.tolerance)
            maximum = max(maximum, float((actual.grad-reference.grad).abs().max()))
        contributions = sum(item['contribution'] for item in trainer.micro_trace if item['window'] == window['window'])
        assert contributions == pytest.approx(float(expected.detach()), rel=self.tolerance, abs=self.tolerance)
        self.events.append(dict(global_step_before=state.global_step, window=window['window'],
            records=len(ids), oracle_loss=float(expected.detach()), max_parameter_gradient_error=maximum,
            parameter_names=[name for name, _ in model.named_parameters()]))
        self.optimizer.step()

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        maximum = 0.
        for actual, reference in zip(model.parameters(), self.reference.parameters()):
            torch.testing.assert_close(actual, reference, rtol=self.tolerance, atol=self.tolerance)
            maximum = max(maximum, float((actual-reference).detach().abs().max()))
        self.events[-1]['max_updated_parameter_error'] = maximum


CASES = [(4,4,1), (5,3,1), (6,2,3), (9,2,3), (7,1,3), (7,2,5), (7,7,1)]


@pytest.mark.parametrize('records,micro,ga', CASES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_real_trainer_windows_all_gradients_and_sgd_updates(tmp_path, records, micro, ga, dtype):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(941)
        model = TinyTokenModel(dtype)
    dataset = invented_records(records, dtype)
    reference = deepcopy(model)
    observer = CheckEveryUpdate(reference, dataset, dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    steps = math.ceil(math.ceil(records/micro)/ga)
    args = TrainingArguments(output_dir=str(tmp_path/'run'), use_cpu=True,
        per_device_train_batch_size=micro, gradient_accumulation_steps=ga, max_steps=steps,
        max_grad_norm=0., report_to='none', disable_tqdm=True, remove_unused_columns=False,
        save_strategy='no', eval_strategy='no', logging_strategy='no',
        dataloader_num_workers=0, dataloader_pin_memory=False, dataloader_drop_last=False,
        average_tokens_across_devices=False, seed=42, data_seed=42)
    trainer = SequentialRecordTrainer(model=model, args=args, train_dataset=dataset,
        data_collator=collate, optimizers=(optimizer, scheduler), callbacks=[observer])
    observer.trainer = trainer
    assert trainer.training_step.__func__ is Trainer.training_step
    assert trainer.compute_loss.__func__ is Trainer.compute_loss
    assert trainer.train.__func__ is Trainer.train
    assert trainer.accelerator.backward.__func__ is Accelerator.backward
    trainer.train()
    assert trainer.state.global_step == steps == len(observer.events) == len(trainer.window_trace)
    observed = [value for window in trainer.window_trace for group in window['record_ids'] for value in group]
    assert observed == list(range(records))  # no dropped, repeated, or shuffled inputs
    assert trainer._consumed == len(trainer.window_trace[-1]['micro_sizes'])
    assert all(window['native_token_count'] > window['actual_records'] for window in trainer.window_trace)
    assert not list((tmp_path/'run').rglob('checkpoint-*'))
    # Actual framework windows and parameter errors, not inferred GA behavior.
    destination = os.environ.get('RECORD_LOSS_FRAMEWORK_EVIDENCE')
    if destination:
        path = Path(destination); path.mkdir(parents=True, exist_ok=True)
        report = dict(records=records, micro=micro, ga=ga, dtype=str(dtype), optimizer='SGD', learning_rate=.03,
            optimizer_steps=steps, training_step_unmodified=True, compute_loss_unmodified=True,
            accelerator_backward_unmodified=True, token_count_not_used_as_record_count=True,
            windows=trainer.window_trace, microbatches=trainer.micro_trace,
            prefetch=trainer.fetch_trace, updates=observer.events, tolerance=observer.tolerance)
        (path/f'case_n{records}_micro{micro}_ga{ga}_{str(dtype).split(".")[-1]}.json').write_text(
            json.dumps(report,indent=2,allow_nan=False)+'\n')
