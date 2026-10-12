"""Undeployed, token-counted validation for future certified corpus pilots.

After activating pllm:
USE_TF=0 python -m pytest scripts/tests/immune_llada/test_token_validation.py -q
This prototype uses only a model's explicit denoise feature interface. It does
not alter any live runner/model and does not read prepared or benchmark data.
"""
from contextlib import contextmanager
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from transformers import Trainer

from scripts.run_immune_research import ResearchTrainer, fixed_validation_mask


FIELDS = ('loss_sum', 'selected_count', 'aa_loss_sum', 'aa_count',
          'terminal_loss_sum', 'terminal_count', 'aa_correct', 'terminal_correct',
          'terminal_false_positive_count', 'chain_count', 'first_terminal_correct_count',
          'terminal_missing_count', 'length_absolute_error_sum',
          'relation_loss_sum', 'relation_count', 'relation_correct',
          'split_loss_sum', 'split_count', 'split_correct')
TOKEN_METRICS = ('loss', 'aa_loss', 'terminal_loss', 'aa_accuracy', 'terminal_accuracy',
                 'terminal_precision', 'terminal_false_positive_rate', 'relation_loss', 'relation_accuracy',
                 'split_loss', 'split_accuracy')
ENDPOINT_METRICS = ('first_terminal_accuracy', 'terminal_missing_rate', 'length_mae')
METRICS = TOKEN_METRICS+ENDPOINT_METRICS
COUNTS = ('selected_count', 'aa_count', 'terminal_count', 'terminal_true_positive_count',
          'terminal_prediction_count', 'terminal_false_positive_count', 'chain_count',
          'first_terminal_correct_count', 'terminal_missing_count', 'length_compared_count', 'relation_count', 'split_count')
METRIC_PROTOCOL = 'immune.token_validation.v3.aa_end_relation_split_and_full_mask_first_terminal'


@contextmanager
def preserved_rng(devices):
    """Evaluation, including callbacks, cannot advance subsequent training RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@torch.no_grad()
def chain_endpoint_totals(batch, predicted_end, *, full_mask):
    """Score first END over every attended target slot, including capacity tails.

    Truth labels determine only the score, never the predicted search interval.
    Partial masks leave truth visible, so first-END metrics are undefined there.
    """
    result = torch.zeros((predicted_end.shape[0], 4), dtype=torch.float64)
    if not full_mask:
        return result.to(predicted_end.device)
    slots = batch['target_slot_mask'].bool() & batch['attention_mask'].bool()
    truth = batch['terminal_mask'].bool() & batch['eligible_mask'].bool() & batch['attention_mask'].bool()
    metadata = torch.stack((batch['chain_ids'].long(), slots.long(), truth.long(), predicted_end.long()), 1).cpu()
    for row, (ids, valid_slots, true_end, prediction) in enumerate(metadata):
        for uid in torch.cat((ids[valid_slots.bool()], ids[true_end.bool()])).unique().tolist():
            true_positions = ((ids == uid) & true_end.bool()).nonzero().flatten()
            if uid < 0 or len(true_positions) != 1 or not valid_slots[true_positions[0]]:
                raise ValueError('A scored chain requires one true END in its target slots')
            positions = ((ids == uid) & valid_slots.bool()).nonzero().flatten()
            true_offset = (positions == true_positions[0]).nonzero().item()
            predicted_offsets = prediction[positions].nonzero().flatten()
            result[row, 0] += 1
            if len(predicted_offsets) == 0:
                result[row, 2] += 1
            else:
                predicted_offset = predicted_offsets[0].item()
                result[row, 1] += int(predicted_offset == true_offset)
                result[row, 3] += abs(predicted_offset-true_offset)
    return result.to(predicted_end.device)


@torch.no_grad()
def selected_token_metrics(model, batch, ratio):
    """Return per-record additive totals; padding and visible tokens never score."""
    selected = fixed_validation_mask(batch, ratio).bool()
    selected &= batch['eligible_mask'].bool() & batch['attention_mask'].bool()
    if not torch.equal(batch['input_ids'][selected], batch['labels'][selected]):
        raise ValueError('Validation labels must match clean selected inputs')
    aa, terminal = selected & batch['residue_mask'].bool(), selected & batch['terminal_mask'].bool()
    if (aa & terminal).any():
        raise ValueError('AA and terminal targets must not overlap')
    split = torch.zeros_like(selected)
    if getattr(getattr(model, 'config', None), 'ablation', None) == 'V5':
        split = (selected & batch['target_slot_mask'].bool()
                 & batch['labels'].eq(model.tokenizer.chain_separator_id()))
    relation = selected & ~(aa | terminal | split)
    if relation.any():
        tokens = getattr(model.tokenizer, 'token_to_id', {})
        allowed = [tokens[name] for name in ('<binding>', '<nonbinding>') if name in tokens]
        valid_relation = torch.zeros_like(selected)
        for token in allowed:
            valid_relation |= batch['labels'].eq(token)
        valid_relation &= ~batch['target_slot_mask'].bool() & (batch['chain_ids'] < 0)
        if (relation & ~valid_relation).any():
            raise ValueError('Unclassified supervised token outside AA/end/relation contract')
    if (batch['labels'][aa] == model.terminal_id).any() or (batch['labels'][terminal] != model.terminal_id).any():
        raise ValueError('AA/terminal scoring masks disagree with the END token')
    features = model.feature_args(batch)
    features['input_ids'] = batch['input_ids'].masked_fill(selected, model.tokenizer.mask_token_id)
    logits = model.denoise(**features)[0]
    labels = batch['labels'].masked_fill(~selected, -100)
    losses = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction='none')
    predictions = logits.argmax(-1)
    correct = predictions.eq(batch['labels'])
    predicted_end = predictions.eq(model.terminal_id)
    # Sum in float64, before batch aggregation, to limit sensitivity to grouping.
    def total(values, mask):
        return values.double().masked_fill(~mask, 0).sum(1)
    totals = torch.stack((total(losses, selected), selected.sum(1),
        total(losses, aa), aa.sum(1), total(losses, terminal), terminal.sum(1),
        (correct & aa).sum(1), (correct & terminal).sum(1), (predicted_end & aa).sum(1)), dim=1).double()
    totals = torch.cat((totals, chain_endpoint_totals(batch, predicted_end, full_mask=ratio == 1.)), 1)
    totals = torch.cat((totals, torch.stack((total(losses, relation), relation.sum(1),
                        (correct & relation).sum(1)), 1)), 1)
    totals = torch.cat((totals, torch.stack((total(losses, split), split.sum(1),
                        (correct & split).sum(1)), 1)), 1)
    if not torch.isfinite(totals).all():
        raise FloatingPointError('Nonfinite validation totals')
    loss = totals[:, 0].sum()/totals[:, 1].sum().clamp_min(1)
    return dict(loss=loss.float(), sample_totals=totals)


def summarize(totals):
    """Null means a target subset was absent; never turn it into a perfect score."""
    values = torch.as_tensor(totals, dtype=torch.float64)
    if values.ndim == 2:
        values = values.sum(0)
    if values.shape != (len(FIELDS),) or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid validation totals')
    value = dict(zip(FIELDS, values.tolist()))
    for name in FIELDS:
        if name in ('loss_sum', 'aa_loss_sum', 'terminal_loss_sum', 'relation_loss_sum', 'split_loss_sum'):
            continue
        if not value[name].is_integer():
            raise ValueError('Counts must be integers')
        value[name] = int(value[name])
    if (value['selected_count'] != value['aa_count']+value['terminal_count']+value['relation_count']+value['split_count']
            or value['aa_correct'] > value['aa_count'] or value['terminal_correct'] > value['terminal_count']
            or value['terminal_false_positive_count'] > value['aa_count']-value['aa_correct']
            or value['relation_correct'] > value['relation_count']
            or value['split_correct'] > value['split_count']):
        raise ValueError('Inconsistent validation populations')
    if not torch.isclose(values[0], values[2]+values[4]+values[FIELDS.index('relation_loss_sum')]
                         +values[FIELDS.index('split_loss_sum')], atol=1e-7, rtol=1e-7):
        raise ValueError('Loss components do not cover all selected targets')
    for count, numerator in (('selected_count', 'loss_sum'), ('aa_count', 'aa_loss_sum'),
                             ('terminal_count', 'terminal_loss_sum'), ('relation_count', 'relation_loss_sum'),
                             ('split_count', 'split_loss_sum')):
        if value[count] == 0 and value[numerator] != 0:
            raise ValueError('An empty population cannot have positive loss')
    compared = value['chain_count']-value['terminal_missing_count']
    if (compared < 0 or value['first_terminal_correct_count'] > compared
            or compared == 0 and value['length_absolute_error_sum'] != 0):
        raise ValueError('Inconsistent first-END populations')
    value.update(terminal_true_positive_count=value['terminal_correct'],
                 terminal_prediction_count=value['terminal_correct']+value['terminal_false_positive_count'],
                 length_compared_count=compared)
    result = {name: value[name] for name in COUNTS}
    for metric, numerator, count in (
        ('loss', 'loss_sum', 'selected_count'), ('aa_loss', 'aa_loss_sum', 'aa_count'),
        ('terminal_loss', 'terminal_loss_sum', 'terminal_count'),
        ('aa_accuracy', 'aa_correct', 'aa_count'),
        ('terminal_accuracy', 'terminal_correct', 'terminal_count'),
        ('terminal_precision', 'terminal_correct', 'terminal_prediction_count'),
        ('terminal_false_positive_rate', 'terminal_false_positive_count', 'aa_count'),
        ('relation_loss', 'relation_loss_sum', 'relation_count'),
        ('relation_accuracy', 'relation_correct', 'relation_count'),
        ('split_loss', 'split_loss_sum', 'split_count'),
        ('split_accuracy', 'split_correct', 'split_count'),
        ('first_terminal_accuracy', 'first_terminal_correct_count', 'chain_count'),
        ('terminal_missing_rate', 'terminal_missing_count', 'chain_count'),
        ('length_mae', 'length_absolute_error_sum', 'length_compared_count')):
        result[metric] = value[numerator]/value[count] if value[count] else None
    return result


def source_macro(metrics_by_source):
    """Equal source weight; explicitly count sources with a defined metric."""
    if not metrics_by_source:
        raise ValueError('No validation sources')
    result = {}
    for metric in METRICS:
        values = [values[metric] for values in metrics_by_source.values() if values[metric] is not None]
        result[metric] = sum(values)/len(values) if values else None
        result[metric+'_sources'] = len(values)
    for count in COUNTS:
        result[count] = sum(values[count] for values in metrics_by_source.values())
    return result


def mask_mean(metrics_by_level):
    """Equal weight for four fixed noise levels; their targets overlap."""
    if set(metrics_by_level) != {'mask25', 'mask50', 'mask75', 'mask100'}:
        raise ValueError('Four fixed mask levels are required')
    result = {}
    for metric in TOKEN_METRICS:
        values = [item[metric] for item in metrics_by_level.values() if item[metric] is not None]
        result[metric] = sum(values)/len(values) if values else None
        result[metric+'_levels'] = len(values)
    return result


class TokenValidationTrainer(ResearchTrainer):
    """Future adapter: inherited micro-mean training and richer evaluation.

    Gather per-record totals with Accelerate before summing, so padded distributed
    final batches can be removed by gather_for_metrics. The production work pool
    currently runs each method on one GPU. Unsupported distributed and persistent
    worker configurations fail explicitly; this prototype is not a GPU release.
    HF accumulation must treat compute_loss as a microbatch mean. Full-window
    CPU normalization is audited separately; short windows remain version-specific.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inherited compute_loss returns a microbatch mean and does not use
        # num_items_in_batch. ResearchModel's **batch signature otherwise makes
        # HF skip its accumulation divisor, summing microbatch means instead.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if model.training:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        unwrapped = self.accelerator.unwrap_model(model)
        with self.accelerator.autocast():
            result = selected_token_metrics(unwrapped, inputs, getattr(self, '_validation_ratio', 1.))
        if getattr(self, '_token_validation_totals', None) is not None:
            gathered = self.accelerator.gather_for_metrics(result['sample_totals'])
            self._token_validation_totals += gathered.sum(0)
        return (result['loss'], {'loss': result['loss']}) if return_outputs else result['loss']

    def evaluation_loop(self, *args, metric_key_prefix='eval', **kwargs):
        self._last_token_validation_totals = None
        self._token_validation_totals = torch.zeros(len(FIELDS), dtype=torch.float64, device=self.args.device)
        try:
            # Skip the legacy four-component accumulator, keeping Trainer's loop.
            output = Trainer.evaluation_loop(self, *args, metric_key_prefix=metric_key_prefix, **kwargs)
            self._last_token_validation_totals = self._token_validation_totals.detach().cpu().clone()
            computed = summarize(self._last_token_validation_totals)
            output.metrics.update({metric_key_prefix+'_'+key: value for key, value in computed.items()})
            return output
        finally:
            self._token_validation_totals = None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        unsupported = (self.args.world_size != 1 or self.args.use_legacy_prediction_loop
            or self.args.dataloader_persistent_workers or self.compute_metrics is not None
            or self.is_fsdp_enabled or self.is_deepspeed_enabled or self.is_fsdp_xla_enabled)
        if unsupported:
            raise NotImplementedError('Prototype requires single-process evaluation, no persistent workers, '
                                      'custom compute_metrics, legacy loop, FSDP, or DeepSpeed')
        if not self.args.disable_tqdm:
            raise NotImplementedError('Prototype requires disable_tqdm=True for multi-source progress callbacks')
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if not isinstance(dataset, dict):
            dataset = {'all': dataset}
        if not dataset:
            raise ValueError('No validation sources')
        if any(rows is None or len(rows) == 0 for rows in dataset.values()):
            raise ValueError('Every validation source must have records')
        devices = [self.args.device.index or 0] if self.args.device.type == 'cuda' else []
        metrics = {}
        by_level = {'source_macro': {}, 'token_micro': {}}
        with preserved_rng(devices):
            # These calls must stay in evaluate: HF derives its memory stage
            # from the calling method name. Always clean up on exceptions too.
            self._memory_tracker.start()
            started = time.perf_counter()
            try:
                for ratio in (.25, .5, .75, 1.):
                    self._validation_ratio = ratio
                    prefix = f'{metric_key_prefix}_mask{int(ratio*100)}'
                    sources, totals = {}, []
                    for source, rows in dataset.items():
                        source_prefix = prefix+'_'+source
                        loader = self.get_eval_dataloader(rows)
                        source_started = time.perf_counter()
                        output = self.evaluation_loop(loader, description='Evaluation', prediction_loss_only=True,
                            ignore_keys=ignore_keys, metric_key_prefix=source_prefix)
                        result = output.metrics
                        if output.num_samples != len(rows):
                            raise ValueError('Evaluation sample count does not match the selected source')
                        result[source_prefix+'_runtime'] = time.perf_counter()-source_started
                        metrics.update(result)
                        sources[source] = {key: result[source_prefix+'_'+key] for key in (*METRICS, *COUNTS)}
                        totals.append(self._last_token_validation_totals)
                    aggregate = {'source_macro': source_macro(sources),
                                 'token_micro': summarize(torch.stack(totals))}
                    for weighting, values in aggregate.items():
                        by_level[weighting][f'mask{int(ratio*100)}'] = values
                        metrics.update({prefix+'_'+weighting+'_'+key: value for key, value in values.items()})
                    # Preserve the full-mask/source-macro alias, while fixing
                    # within-source token weighting (metric schema changes).
                    metrics.update({prefix+'_'+key: value for key, value in aggregate['source_macro'].items()})
                for weighting, levels in by_level.items():
                    metrics.update({metric_key_prefix+'_mask_mean_'+weighting+'_'+key: value
                                    for key, value in mask_mean(levels).items()})
                for key in METRICS:
                    metrics[metric_key_prefix+'_'+key] = metrics[metric_key_prefix+'_mask100_'+key]
                metrics[metric_key_prefix+'_runtime'] = time.perf_counter()-started
                metrics[metric_key_prefix+'_records_per_mask'] = sum(len(rows) for rows in dataset.values())
                metrics[metric_key_prefix+'_record_evaluations'] = 4*metrics[metric_key_prefix+'_records_per_mask']
            finally:
                self._validation_ratio = 1.
                self._memory_tracker.stop_and_update_metrics(metrics)
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
