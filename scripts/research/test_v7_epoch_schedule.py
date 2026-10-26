"""Check native CPU Trainer recovery across an explicit epoch/LR extension.

Activate conda pllm, then use the pinned ML Python:
/project_c20250601/251105016/conda/envs/protenix_abtcr/bin/python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_v7_epoch_schedule.py
Generic regression tensors only; no corpus or GPU. All temporary checkpoints
are removed when this process exits normally or raises an exception.
"""
import ast
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import tempfile

os.environ.update(CUDA_VISIBLE_DEVICES='', USE_TF='0', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', WANDB_DISABLED='true')
import torch
from torch.utils.data import Dataset, Sampler
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.optimization import get_cosine_schedule_with_warmup
from v7_epoch_schedule import epoch_restart_factor, preserve_resume_epoch

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
SOURCE = ROOT/'output/v7_formal_eight_gpu_20261002/where_continuation_20261006/source_snapshot_v1'

# Execute the exact frozen sampler class, without importing its model runner.
tree = ast.parse((SOURCE/'scripts/run_immune_research.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EpochRandomSampler')
namespace = {'torch': torch, 'Sampler': Sampler}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen EpochRandomSampler>', 'exec'), namespace)
EpochRandomSampler = namespace['EpochRandomSampler']


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


class Rows(Dataset):
    def __len__(self):
        return 14

    def __getitem__(self, index):
        x = torch.tensor([index/14, (index % 3)/3, 1.0])
        return dict(input_ids=x, labels=x[:2]*0.3, index=torch.tensor(index))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(3, 2)
        self.seen = []

    def forward(self, input_ids, labels, index):
        if self.training:
            self.seen.extend(index.tolist())
        logits = self.linear(self.dropout(input_ids))
        return dict(loss=torch.nn.functional.mse_loss(logits, labels), logits=logits)


class ExtensionTrainer(Trainer):
    def _get_train_sampler(self, train_dataset=None):
        return EpochRandomSampler(self.train_dataset if train_dataset is None else train_dataset, self.args.data_seed)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer if optimizer is None else optimizer,
                partial(epoch_restart_factor, first_phase_steps=8, total_steps=14, warmup_steps=2))
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _load_optimizer_and_scheduler(self, checkpoint):
        result = super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint:
            root = Path(checkpoint)
            assert state_equal(self.optimizer.state_dict(), torch.load(root/'optimizer.pt', weights_only=False))
            assert state_equal(self.lr_scheduler.state_dict(), torch.load(root/'scheduler.pt', weights_only=False))
            self.native_optimizer_scheduler_exact = True
        return result

    def _load_rng_state(self, checkpoint):
        result = super()._load_rng_state(checkpoint)
        if checkpoint:
            saved = torch.load(Path(checkpoint)/'rng_state.pth', weights_only=False)
            assert torch.equal(saved['cpu'], torch.random.get_rng_state())
            self.native_cpu_rng_exact = True
        return result


class OldTrainer(ExtensionTrainer):
    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer if optimizer is None else optimizer, num_warmup_steps=2, num_training_steps=8)
            self._created_lr_scheduler = True
        return self.lr_scheduler


def run(path, total, cls, resume=None):
    set_seed(42)
    model = Model()
    args = TrainingArguments(output_dir=str(path), max_steps=total, warmup_steps=2,
        per_device_train_batch_size=2, dataloader_drop_last=True, dataloader_num_workers=0,
        learning_rate=1e-3, lr_scheduler_type='cosine', seed=42, data_seed=42,
        save_strategy='steps', save_steps=8, save_total_limit=None, save_safetensors=True,
        logging_strategy='no', report_to='none', disable_tqdm=True, use_cpu=True,
        remove_unused_columns=False, optim='adamw_torch')
    trainer = cls(model=model, args=args, train_dataset=Rows())
    with preserve_resume_epoch():
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    return trainer


def main():
    torch.set_num_threads(1)
    for step in range(25043):
        expected = step/1000 if step < 1000 else 0.5*(1+torch.cos(torch.tensor(
            (step-1000)/(25042-1000)*torch.pi, dtype=torch.float64)).item())
        assert abs(epoch_restart_factor(step, first_phase_steps=25042, total_steps=50082, warmup_steps=1000)-expected)<1e-14
    assert epoch_restart_factor(26042, first_phase_steps=25042, total_steps=50082, warmup_steps=1000)==1
    assert epoch_restart_factor(50082, first_phase_steps=25042, total_steps=50082, warmup_steps=1000)==0
    with tempfile.TemporaryDirectory(prefix='v7_epoch_extension_cpu_') as temp:
        base = Path(temp)
        continuous = run(base/'continuous', 14, ExtensionTrainer)
        old = run(base/'old', 8, OldTrainer)
        resume = run(base/'resumed', 14, ExtensionTrainer, base/'old/checkpoint-8')
        assert old.state.global_step == 8 and resume.state.global_step == 14
        assert resume.native_optimizer_scheduler_exact and resume.native_cpu_rng_exact
        print(json.dumps(dict(diagnostic='native_resume_offset1',
            continuous_order=continuous.model.seen, old_order=old.model.seen,
            resumed_order=resume.model.seen,
            max_parameter_difference=max((value-resume.model.state_dict()[name]).abs().max().item()
                for name, value in continuous.model.state_dict().items()),
            continuous_scheduler=continuous.lr_scheduler.state_dict(),
            resumed_scheduler=resume.lr_scheduler.state_dict())), flush=True)
        assert state_equal(continuous.model.state_dict(), resume.model.state_dict())
        assert state_equal(continuous.optimizer.state_dict(), resume.optimizer.state_dict())
        assert state_equal(continuous.lr_scheduler.state_dict(), resume.lr_scheduler.state_dict())
        assert old.model.seen + resume.model.seen == continuous.model.seen
        assert sorted(old.model.seen[14:]+resume.model.seen) == list(range(14))
        assert continuous.model.seen[:14] != continuous.model.seen[14:]
        assert len(resume.model.seen) == 12
    print(json.dumps(dict(status='passed', actual_cpu_native_resume=True,
        optimizer_scheduler_restore_exact=True, cpu_rng_restore_exact=True,
        resumed_weights_optimizer_scheduler_equal_continuous=True,
        second_epoch_order_differs_and_matches_continuous=True,
        one_already_consumed_second_epoch_batch_skipped=True,
        original_25042_step_lr_curve_preserved=True,
        gpu_used=False, corpus_read=False, temporary_checkpoints_removed=True)))


if __name__ == '__main__':
    main()
