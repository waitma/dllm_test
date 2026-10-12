"""FSDP-compatible training adapter; imported by run_v7_fsdp.py after source binding.

Run checks after conda activate pllm with scripts/research/test_v7_fsdp.py.
The frozen f011 model and token metric functions remain the source of truth.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import TrainerCallback

from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
from scripts.architecture_prototypes import validation_metrics as metrics
from scripts.run_immune_research import EpochRandomSampler


class ForwardMetricModel(FactorialModel):
    """Keep evaluation inside the root FSDP forward, including root-owned weights."""

    def forward(self, validation_ratio=None, **batch):
        if validation_ratio is None:
            return super().forward(**batch)
        if self.training:
            raise ValueError('Validation ratio supplied during training')
        return metrics.selected_token_metrics(self, batch, validation_ratio)


class FSDPTrainer(metrics.TokenValidationTrainer):
    """Native HF training/save/load, existing token-counted validation, timed phases."""

    def _get_train_sampler(self, train_dataset=None):
        return EpochRandomSampler(self.train_dataset if train_dataset is None else train_dataset,
                                  self.args.data_seed)

    @contextmanager
    def phase(self, name):
        started = time.perf_counter()
        with torch.profiler.record_function('v7/'+name):
            yield
        if self.is_world_process_zero():
            self.phase_totals[name] = self.phase_totals.get(name, 0.) + time.perf_counter()-started

    def __init__(self, *args, **kwargs):
        self.phase_totals = {}
        super().__init__(*args, **kwargs)

    def get_batch_samples(self, *args, **kwargs):
        with self.phase('fetch_and_collate_host'):
            return super().get_batch_samples(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        with self.phase('training_step_host_including_forward_backward'):
            return super().training_step(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        with self.phase('forward_and_loss_host'):
            if model.training:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            # Calling unwrap_model(...).denoise would bypass root FSDP all-gather.
            result = model(validation_ratio=getattr(self, '_validation_ratio', 1.), **inputs)
            if self._token_validation_totals is not None:
                gathered = self.accelerator.gather_for_metrics(result['sample_totals'])
                self._token_validation_totals += gathered.sum(0)
            return (result['loss'], {'loss': result['loss']}) if return_outputs else result['loss']

    def get_eval_dataloader(self, eval_dataset=None):
        drop = self.args.dataloader_drop_last
        try:
            self.args.dataloader_drop_last = False
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.args.dataloader_drop_last = drop

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        if (self.args.use_legacy_prediction_loop or self.args.dataloader_persistent_workers
                or self.compute_metrics is not None or self.is_deepspeed_enabled
                or getattr(self, 'is_fsdp_xla_enabled', False) or not self.args.disable_tqdm):
            raise ValueError('Unsupported evaluation configuration')
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        dataset = dataset if isinstance(dataset, dict) else {'all': dataset}
        names = sorted(dataset)
        if not names or any(len(dataset[name]) == 0 for name in names):
            raise ValueError('Every validation source must be nonempty')
        declaration = [(name, len(dataset[name])) for name in names]
        if torch.distributed.is_initialized():
            declarations = [None]*torch.distributed.get_world_size()
            torch.distributed.all_gather_object(declarations, declaration)
            if any(item != declaration for item in declarations):
                raise ValueError('Ranks disagree on source order or length')
        devices = [self.args.device.index or 0] if self.args.device.type == 'cuda' else []
        result, by_level = {}, {'source_macro': {}, 'token_micro': {}}
        with metrics.preserved_rng(devices), self.phase('evaluation_host'):
            self._memory_tracker.start()
            started = time.perf_counter()
            try:
                for ratio in (.25, .5, .75, 1.):
                    self._validation_ratio = ratio
                    prefix = f'{metric_key_prefix}_mask{int(100*ratio)}'
                    sources, totals = {}, []
                    for name in names:
                        source_prefix = prefix+'_'+name
                        output = self.evaluation_loop(self.get_eval_dataloader(dataset[name]),
                            description='Evaluation', prediction_loss_only=True,
                            ignore_keys=ignore_keys, metric_key_prefix=source_prefix)
                        if output.num_samples != len(dataset[name]):
                            raise ValueError('Validation tail sample count differs')
                        result.update(output.metrics)
                        sources[name] = {key: result[source_prefix+'_'+key]
                                        for key in (*metrics.METRICS, *metrics.COUNTS)}
                        totals.append(self._last_token_validation_totals)
                    aggregate = {'source_macro': metrics.source_macro(sources),
                                 'token_micro': metrics.summarize(torch.stack(totals))}
                    for weighting, values in aggregate.items():
                        by_level[weighting][f'mask{int(100*ratio)}'] = values
                        result.update({prefix+'_'+weighting+'_'+key: value for key, value in values.items()})
                    result.update({prefix+'_'+key: value for key, value in aggregate['source_macro'].items()})
                for weighting, values in by_level.items():
                    result.update({metric_key_prefix+'_mask_mean_'+weighting+'_'+key: value
                                   for key, value in metrics.mask_mean(values).items()})
                for key in metrics.METRICS:
                    result[metric_key_prefix+'_'+key] = result[metric_key_prefix+'_mask100_'+key]
                result[metric_key_prefix+'_runtime'] = time.perf_counter()-started
                result[metric_key_prefix+'_records_per_mask'] = sum(len(dataset[n]) for n in names)
                result[metric_key_prefix+'_record_evaluations'] = 4*result[metric_key_prefix+'_records_per_mask']
            finally:
                self._validation_ratio = 1.
                self._memory_tracker.stop_and_update_metrics(result)
            # HF log/evaluate callbacks run on every rank, shared persistence on rank0.
            self.log(result)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, result)
        return result

    def _save_checkpoint(self, *args, **kwargs):
        with self.phase('native_checkpoint_save_host'):
            return super()._save_checkpoint(*args, **kwargs)

    def _load_optimizer_and_scheduler(self, checkpoint):
        result = super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is not None:
            expected = json.loads((Path(checkpoint)/'trainer_state.json').read_text())['global_step']
            states = list(self.optimizer.state.values())
            if (not states or self.lr_scheduler.state_dict()['last_epoch'] != expected
                    or any(int(item['step']) != expected for item in states if 'step' in item)):
                raise ValueError('Native FSDP optimizer/scheduler restoration did not match checkpoint step')
            self.native_optimizer_restore = {'step': expected, 'local_state_entries': len(states)}
        return result

    def _load_rng_state(self, checkpoint):
        result = super()._load_rng_state(checkpoint)
        if checkpoint is not None:
            name = f'rng_state_{self.args.process_index}.pth' if self.args.world_size > 1 else 'rng_state.pth'
            saved = torch.load(Path(checkpoint)/name, map_location='cpu', weights_only=False)
            current_numpy = np.random.get_state()
            if (random.getstate() != saved['python'] or not torch.equal(torch.get_rng_state(), saved['cpu'])
                    or current_numpy[0] != saved['numpy'][0]
                    or not np.array_equal(current_numpy[1], saved['numpy'][1])
                    or current_numpy[2:] != saved['numpy'][2:]):
                raise ValueError('Native CPU RNG restore differs')
            if torch.cuda.is_available():
                expected_cuda = saved['cuda'] if isinstance(saved['cuda'], list) else [saved['cuda']]
                actual_cuda = torch.cuda.get_rng_state_all()
                if len(actual_cuda) != len(expected_cuda) or any(not torch.equal(a, b)
                        for a, b in zip(actual_cuda, expected_cuda)):
                    raise ValueError('Native CUDA RNG restore differs')
            self.native_rng_restore = True
        return result


class BoundedTrace(TrainerCallback):
    """Trace whole-model execution for a bounded window; never run isolated kernels."""

    def __init__(self, trainer, active_steps=4):
        self.trainer, self.active_steps, self.profiler = trainer, active_steps, None

    def on_train_begin(self, args, state, control, **kwargs):
        self.begin_step = state.global_step
        self.started = time.perf_counter()
        if args.process_index == 0 and self.active_steps:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if args.device.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities,
                schedule=torch.profiler.schedule(wait=2, warmup=1, active=self.active_steps, repeat=1),
                record_shapes=True, profile_memory=True,
                on_trace_ready=lambda p: p.export_chrome_trace(str(Path(args.output_dir)/
                    f'whole_model_trace_from_{self.begin_step}.json')))
            self.profiler.start()

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler:
            self.profiler.step()
            if state.global_step-self.begin_step >= self.active_steps+3:
                self.profiler.stop()
                self.profiler = None
        if args.process_index == 0 and state.global_step % 25 == 0:
            self.write(args, state)

    def write(self, args, state):
        data = dict(step=state.global_step, elapsed_seconds=time.perf_counter()-self.started,
            phase_host_seconds=self.trainer.phase_totals,
            timing_scope='Host wall durations; nested regions overlap; CUDA trace contains device events.',
            max_memory_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.device.type == 'cuda' else None)
        path = Path(args.output_dir)/'timing_progress.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2)+'\n')
        tmp.replace(path)

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler:
            self.profiler.stop()
            self.profiler = None
        if args.process_index == 0:
            self.write(args, state)
