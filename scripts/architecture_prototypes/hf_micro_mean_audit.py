"""Audit native Trainer accumulation with artificial Linear/MSE data on CPU.

Activate pllm, then USE_TF=0 CUDA_VISIBLE_DEVICES='' python
scripts/architecture_prototypes/hf_micro_mean_audit.py --output-json /absolute/new/report.json
Runs five one-step CPU experiments, comparing every gradient/parameter with an
independent NumPy oracle. No biological data, model weights, GPU or checkpoint.
The explicit false-flag cases are diagnostic controls, not a production fix.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['USE_TF'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
from torch import nn
from accelerate import Accelerator
from transformers import Trainer, TrainerCallback, TrainingArguments

from dllm.pipelines.immune_llada.research.model import ResearchModel
from scripts.architecture_prototypes.validation_metrics import TokenValidationTrainer


class ArtificialMeanModel(nn.Module):
    """Same forward parameter kinds as ResearchModel; only synthetic MSE."""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2)
        self.variant = 'r20_refined_control'
        self.calls = []
        with torch.no_grad():
            self.layer.weight.copy_(torch.tensor([[.2, -.1, .3], [-.4, .5, .1]]))
            self.layer.bias.copy_(torch.tensor([.1, -.2]))

    def forward(self, token_t=None, **batch):
        self.calls.append(batch['record_id'].tolist())
        prediction = self.layer(batch['features'])
        loss = (prediction-batch['labels']).square().mean()
        return {'loss': loss, 'artificial_mse': loss.detach()}


def signature_kinds(method):
    return [(p.name, str(p.kind)) for p in inspect.signature(method).parameters.values()]


def collate(records):
    return {key: torch.stack([row[key] for row in records]) for key in records[0]}


def experiment(accumulation, force_false):
    torch.manual_seed(42)
    model = ArtificialMeanModel()
    x = np.array([[i/9., (i%3-1)/4., (7-i)/11.] for i in range(8)], dtype=np.float32)
    y = np.array([[i/13.-.2, (i%2)/3.-.1] for i in range(8)], dtype=np.float32)
    weight, bias = model.layer.weight.detach().numpy().copy(), model.layer.bias.detach().numpy().copy()
    error = x.astype(np.float64)@weight.astype(np.float64).T+bias-y
    expected = {'layer.weight': (2*error.T@x/error.size).astype(np.float32),
                'layer.bias': (2*error.sum(0)/error.size).astype(np.float32)}
    start = {'layer.weight': weight, 'layer.bias': bias}
    rows = [dict(features=torch.from_numpy(x[i]), labels=torch.from_numpy(y[i]),
                 record_id=torch.tensor(i)) for i in range(8)]
    observed = {}

    class Observe(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            assert not observed, 'More than one update'
            observed.update({name: parameter.grad.detach().cpu().numpy().copy()
                             for name, parameter in model.named_parameters()})

    with tempfile.TemporaryDirectory(prefix='immune-hf-micro-mean-') as output:
        args = TrainingArguments(output_dir=output, use_cpu=True,
            per_device_train_batch_size=8//accumulation, gradient_accumulation_steps=accumulation,
            max_steps=1, learning_rate=.05, lr_scheduler_type='constant', optim='sgd',
            weight_decay=0., max_grad_norm=0., save_strategy='no', eval_strategy='no',
            logging_strategy='no', report_to='none', disable_tqdm=True,
            dataloader_num_workers=0, remove_unused_columns=False, label_names=['labels'],
            seed=42, data_seed=42)
        trainer = TokenValidationTrainer(model=model, args=args, train_dataset=rows,
            data_collator=collate, callbacks=[Observe()])
        initial_flag = trainer.model_accepts_loss_kwargs
        if force_false:
            trainer.model_accepts_loss_kwargs = False
        assert type(initial_flag) is bool and trainer.accelerator.gradient_accumulation_steps == 1
        assert type(trainer).train is Trainer.train and type(trainer).training_step is Trainer.training_step
        assert trainer.compute_loss_func is None
        result = trainer.train()
        assert trainer.state.global_step == 1 and not torch.cuda.is_initialized()
        assert sorted(i for call in model.calls for i in call) == list(range(8))
        assert len(model.calls) == accumulation and all(len(call) == 8//accumulation for call in model.calls)
        assert not list(Path(output).glob('checkpoint-*'))
        flag = trainer.model_accepts_loss_kwargs
        actual_scale = sum(float((observed[k].astype(np.float64)*expected[k]).sum()) for k in expected)
        actual_scale /= sum(float((expected[k].astype(np.float64)**2).sum()) for k in expected)
        expected_scale = accumulation if flag else 1
        assert abs(actual_scale-expected_scale) < 2e-5
        per_parameter = {}
        for name, parameter in model.named_parameters():
            gradient, oracle = observed[name], expected[name]
            np.testing.assert_allclose(gradient, expected_scale*oracle, rtol=2e-5, atol=2e-7)
            np.testing.assert_allclose(parameter.detach().numpy(), start[name]-.05*expected_scale*oracle,
                                       rtol=2e-5, atol=2e-7)
            per_parameter[name] = dict(gradient_max_abs_error_to_intended_mean=float(np.max(np.abs(gradient-oracle))),
                gradient_max_abs_error_to_observed_scale=float(np.max(np.abs(gradient-expected_scale*oracle))),
                parameter_max_abs_error_to_intended_update=float(np.max(np.abs(parameter.detach().numpy()-(start[name]-.05*oracle)))))
        return dict(accumulation=accumulation, micro_batch=8//accumulation, records=8,
            trainer_initial_model_accepts_loss_kwargs=initial_flag, actual_model_accepts_loss_kwargs=flag,
            explicit_false_control=force_false, accelerator_accumulation=1, actual_micro_ids=model.calls,
            optimizer_steps=1, observed_gradient_scale_vs_intended_mean=actual_scale,
            matches_intended_micro_mean=abs(actual_scale-1)<2e-5, train_loss=result.training_loss,
            parameter_checks=per_parameter)


def run(output):
    output = Path(output)
    if not output.is_absolute() or output.exists() or output != output.resolve():
        raise ValueError('Use a new canonical absolute report path')
    assert signature_kinds(ArtificialMeanModel.forward) == signature_kinds(ResearchModel.forward)
    torch.set_num_threads(2)
    cases = [experiment(ga, False) for ga in (1, 2, 4)]
    cases += [experiment(ga, True) for ga in (2, 4)]
    sources = {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for target in (ResearchModel, *TokenValidationTrainer.__mro__[:-1], Accelerator):
        path = Path(inspect.getfile(target)).resolve()
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(schema='immune.artificial_hf_micro_mean_audit.v1',
        status=('completed_diagnostic_intended_micro_mean_matched' if all(c['matches_intended_micro_mean'] for c in cases)
                else 'completed_diagnostic_native_accumulation_mismatch_found'),
        versions={name: importlib.metadata.version(name) for name in ('torch','transformers','accelerate')},
        forward_signature=signature_kinds(ResearchModel.forward), source_sha256=sources, cases=cases,
        production_modified=False, gpu_executed=False, biological_data_read=False,
        checkpoint_written=False,
        limits='Linear/MSE isolates framework normalization and inherited Trainer behavior. '
               'It does not certify real backbone gradients, BF16/AdamW, incomplete windows or resume. '
               'False-flag controls are not a deployed correction; current manual synthetic500 divides by accumulation separately.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(output), cases=len(cases), native_scales=[c['observed_gradient_scale_vs_intended_mean'] for c in cases[:3]])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-json', type=Path, required=True)
    run(parser.parse_args().output_json)
