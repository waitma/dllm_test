"""Artificial-only CPU Trainer adapter for the independent record-mean objective.

Activate pllm, then:
  USE_TF=0 python -m pytest scripts/tests/immune_llada/test_record_loss_framework.py -q
This file imports no repository model, corpus loader, or existing trainer. The
inherited Trainer.train/compute_loss/training_step and Accelerator.backward run
unchanged. Only prefetched-window metadata and the public loss callback adapt
the objective. No production training entry point is provided.
"""
import importlib.metadata

import torch
from transformers import Trainer

from scripts.architecture_prototypes.record_loss import actual_global_record_count, record_normalized_ce


PROTOCOL = 'artificial_cpu.record_mean.real_trainer_window.v1'
SUPPORTED_TRANSFORMERS = ('4.48.1', '4.57.6')


def token_count(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError('The framework token count must be scalar')
        value = value.item()
    if type(value) is not int or value < 0:
        raise ValueError('The framework token count must be a nonnegative integer')
    return value


class RecordWindowTrainer(Trainer):
    """Fixed artificial inputs only; public compute_loss_func prevents GA division.

    The artificial model forwards features to a tiny classifier and returns
    logits plus selected_mask, valid_mask, record_ids and _record_window metadata
    unchanged. Labels follow the ordinary Trainer loss-callback contract. The
    framework num_items_in_batch remains a token count and is never a divisor.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get('compute_loss_func') is not None:
            raise ValueError('This isolated adapter owns its public loss callback')
        if (importlib.metadata.version('transformers') not in SUPPORTED_TRANSFORMERS
                or importlib.metadata.version('accelerate') != '1.14.0'):
            raise ValueError('Untested framework version; revalidate before use')
        self.window_trace, self.micro_trace, self.fetch_trace = [], [], []
        self._consumed = 0
        kwargs['compute_loss_func'] = self.record_loss_callback
        super().__init__(*args, **kwargs)
        self.check_runtime()

    def check_runtime(self):
        if (self.args.device.type != 'cpu' or self.args.world_size != 1
                or self.accelerator.num_processes != 1 or self.accelerator.gradient_accumulation_steps != 1
                or self.is_deepspeed_enabled or self.is_fsdp_enabled or self.is_fsdp_xla_enabled
                or self.args.fp16 or self.args.bf16 or self.use_apex
                or self.args.average_tokens_across_devices or self.label_smoother is not None
                or self.args.dataloader_drop_last or self.args.dataloader_num_workers != 0
                or self.args.remove_unused_columns or self.args.max_grad_norm != 0
                or self.args.torch_compile or self.model_accepts_loss_kwargs):
            raise ValueError('Requires the tested single-process CPU/no-AMP/no-clipping artificial contract')
        if self.compute_loss_func != self.record_loss_callback:
            raise ValueError('The public loss callback changed')

    def get_batch_samples(self, epoch_iterator, num_batches, *args, **kwargs):
        self.check_runtime()
        if self.window_trace and self._consumed != len(self.window_trace[-1]['micro_sizes']):
            raise ValueError('The previous prefetched window was not completely consumed')
        # HF 4.48.1 derives its last prefetch limit from examples % GA, not
        # dataloader batches % GA. Fetch up to configured GA and let real EOF
        # supply the actual tail in both versions, including a short epoch.
        fetch_limit = self.args.gradient_accumulation_steps
        batches, num_items = super().get_batch_samples(epoch_iterator, fetch_limit, *args, **kwargs)
        sizes = [int(batch['labels'].shape[0]) for batch in batches]
        self.fetch_trace.append(dict(framework_requested_batches=num_batches,
                                     actual_fetch_limit=fetch_limit, actual_micro_sizes=sizes))
        if not batches:
            return batches, num_items
        records = actual_global_record_count(sizes)
        observed_tokens = sum(int(batch['labels'].ne(-100).sum()) for batch in batches)
        if token_count(num_items) != observed_tokens:
            raise ValueError('Native framework token count differs from observed labels')
        row_ids = [batch['record_ids'].tolist() for batch in batches]
        if any(len(ids) != size for ids, size in zip(row_ids, sizes)):
            raise ValueError('Record IDs and actual batch sizes differ')
        window = len(self.window_trace)
        self.window_trace.append(dict(window=window, micro_sizes=sizes, actual_records=records,
            native_token_count=observed_tokens, record_ids=row_ids,
            configured_ga=self.args.gradient_accumulation_steps,
            accelerator_ga=self.accelerator.gradient_accumulation_steps))
        self._consumed = 0
        # Keep native num_items unchanged; add an explicitly different channel.
        return [{**batch, '_record_window': (window, index, records)}
                for index, batch in enumerate(batches)], num_items

    def record_loss_callback(self, outputs, labels, num_items_in_batch=None):
        self.check_runtime()
        window, index, records = outputs['_record_window']
        if (window != len(self.window_trace)-1 or index != self._consumed
                or records != self.window_trace[window]['actual_records']):
            raise ValueError('Microbatch does not belong to the active actual-record window')
        trace = self.window_trace[window]
        if (labels.shape[0] != trace['micro_sizes'][index]
                or outputs['record_ids'].tolist() != trace['record_ids'][index]
                or token_count(num_items_in_batch) != trace['native_token_count']):
            raise ValueError('Records or native token evidence changed during the window')
        result = record_normalized_ce(outputs['logits'], labels, outputs['selected_mask'],
            outputs['valid_mask'], global_record_count=records)
        self.micro_trace.append(dict(window=window, index=index, actual_micro_records=labels.shape[0],
            actual_window_records=records, native_window_tokens=token_count(num_items_in_batch),
            selected_token_count=int(result.selected_counts.sum()),
            configured_ga=self.args.gradient_accumulation_steps,
            trainer_current_ga=getattr(self, 'current_gradient_accumulation_steps', None),
            accelerator_ga=self.accelerator.gradient_accumulation_steps,
            contribution=float(result.contribution.detach())))
        self._consumed += 1
        return result.contribution
