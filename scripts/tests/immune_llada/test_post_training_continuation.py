"""Activate pllm/protenix_abtcr; python -m pytest this_file.py -q.

Exercise completed-run extension, native recovery, data skipping and a second resume.
"""
import json

import pytest
import torch
from torch.utils.data import ConcatDataset
from transformers import TrainingArguments, TrainerCallback

from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
from dllm.pipelines.immune_llada.fixed_chain.post_training_data import PostTrainingCollator
from scripts.research.fixed_chain_validation import TokenValidationTrainer
from scripts.research.formal_resume_observation import state_digest
from scripts.research.post_training_trainer import PostTrainingTrainerMixin
from scripts.research.post_training_continuation import (
    ContinuationScheduleMixin, continuation_factor, coverage_steps, validate_binding)
from scripts.tests.immune_llada.test_fixed_chain_post_training import fixture


class Trainer(ContinuationScheduleMixin, PostTrainingTrainerMixin, TokenValidationTrainer):
    def _load_optimizer_and_scheduler(self, checkpoint):
        result = super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint:
            for name, value in [('optimizer.pt', self.optimizer.state_dict()),
                                ('scheduler.pt', self.lr_scheduler.state_dict())]:
                saved = torch.load(checkpoint + '/' + name, map_location='cpu', weights_only=False)
                assert state_digest(saved) == state_digest(value)
            self.loaded_state_verified = True
        return result


class StopAt(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_training_stop = True
            control.should_save = True
        return control


def test_completed_lora_run_extends_and_resumes_without_replay(tmp_path):
    plan = {'start_step': 4, 'end_step': 16, 'warmup_steps': 2}

    def build(directory, continuation=False, stop=None):
        _, model, _ = fixture('lora')
        rows = [BioSeqRecord([BioSeqChain('ACD', 'peptide'), BioSeqChain(seq, 'tcr_beta')],
                             'tcr_epitope', 'test') for seq in ('CASSF', 'CATF', 'CASRF', 'CAVF')]
        data = ConcatDataset([rows[:2] * 4, rows[2:] * 8])
        args = TrainingArguments(output_dir=str(directory), use_cpu=True, report_to='none',
            disable_tqdm=True, per_device_train_batch_size=2, gradient_accumulation_steps=1,
            max_steps=16 if continuation else 4, learning_rate=.001, weight_decay=.01,
            lr_scheduler_type='cosine', warmup_steps=2, save_strategy='steps', save_steps=4,
            eval_strategy='no', logging_strategy='no', remove_unused_columns=False,
            seed=123, data_seed=123, dataloader_num_workers=0, dataloader_drop_last=True,
            optim='adamw_torch')
        instance = Trainer(model=model, args=args, train_dataset=data,
            data_collator=PostTrainingCollator(model.config), callbacks=[StopAt(stop)] if stop else [])
        instance.continuation_plan = plan if continuation else None
        return instance

    parent = build(tmp_path / 'parent')
    parent.train()
    checkpoint = tmp_path / 'parent/checkpoint-4'
    assert parent.lr_scheduler.get_last_lr() == [0., 0.]
    parent_files = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    # Expected data-stream suffix: continuing skips precisely the four old batches.
    all_indices = list(parent._get_train_sampler())
    continuous = build(tmp_path / 'continuous', continuation=True)
    loader = continuous.get_train_dataloader()
    from accelerate import skip_first_batches
    suffix = list(skip_first_batches(loader, 4))
    expected = [continuous.data_collator([continuous.train_dataset[i] for i in all_indices[j:j+2]])
                for j in range(8, len(all_indices), 2)]
    assert len(suffix) == len(expected) == 12
    for actual, wanted in zip(suffix, expected):
        for key in wanted:
            torch.testing.assert_close(actual[key], wanted[key], rtol=0, atol=0)
    continuous.train(resume_from_checkpoint=str(checkpoint))
    stopped = build(tmp_path / 'resumed', continuation=True, stop=6)
    stopped.train(resume_from_checkpoint=str(checkpoint))
    resumed = build(tmp_path / 'resumed', continuation=True)
    resumed.train(resume_from_checkpoint=str(tmp_path / 'resumed/checkpoint-6'))
    assert continuous.loaded_state_verified and stopped.loaded_state_verified and resumed.loaded_state_verified
    assert resumed.state.global_step == 16 and resumed.state.epoch == 1.
    assert resumed.lr_scheduler.last_epoch == 16 and resumed.lr_scheduler.get_last_lr() == [0., 0.]
    for name, value in continuous.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
    assert state_digest(continuous.optimizer.state_dict()) == state_digest(resumed.optimizer.state_dict())
    changed = []
    for name, value in parent.model.state_dict().items():
        if not torch.equal(value, continuous.model.state_dict()[name]):
            changed.append(name)
    assert changed and all(name.endswith(('.a', '.b')) for name in changed)
    assert parent_files == {p.name: p.read_bytes() for p in checkpoint.iterdir()}


def test_schedule_boundary_and_balanced_coverage():
    lengths = [276412, 2485433, 2094186, 96521, 642681, 2000000, 31515]
    assert coverage_steps(lengths, 512) == 33981
    rate = lambda step: 2e-5 * continuation_factor(step, 10000, 33981, 200)
    assert rate(10000) == 0 and rate(10001) == pytest.approx(1e-7)
    assert rate(10200) == 2e-5 and 0 < rate(20000) < rate(10200)
    assert rate(33981) == rate(40000) == 0


def test_binding_rejects_changed_data_before_resume():
    with pytest.raises(ValueError, match='model/data/optimizer'):
        validate_binding({}, {'seed': 43}, {'seed': 42})
