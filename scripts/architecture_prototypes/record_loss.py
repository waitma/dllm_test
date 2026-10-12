"""Independent record-normalized token CE; no training integration or data access.

Activate pllm, then run artificial CPU tests:
  python -m pytest scripts/tests/immune_llada/test_record_loss.py -q
For one complete accumulation window, determine its actual record count first.
Sum backward contributions returned by record_normalized_ce across microbatches;
do not divide them by the number of microbatches again. Framework scaling still
requires a separate integration test. This is not an official LLaDA/ELBO loss.
"""
from dataclasses import dataclass
from collections.abc import Iterable

import torch
from torch.nn import functional as F


PROTOCOL = 'token_ce.per_record_selected_mean.actual_window_records.v1'


@dataclass(frozen=True)
class RecordCELoss:
    contribution: torch.Tensor
    per_record_mean: torch.Tensor
    selected_counts: torch.Tensor
    local_record_count: int
    global_record_count: int


def actual_global_record_count(microbatch_record_counts: Iterable[int]) -> int:
    """Sum actual, nonempty microbatch sizes; nominal capacity is not a count.

    The caller must supply the complete window and check that each size matches
    its actual microbatch. This helper cannot observe any external dataloader.
    """
    counts = tuple(microbatch_record_counts)
    if not counts or any(type(count) is not int or count <= 0 for count in counts):
        raise ValueError('A window needs positive integer microbatch record counts')
    return sum(counts)


def record_normalized_ce(logits, labels, selected_mask, valid_mask, *, global_record_count):
    """Return sum_i(mean_selected_token_CE_i) / actual_window_record_count.

    logits: floating [records, positions, classes]; labels: int64 [records,
    positions]; masks: boolean [records, positions]. Every record must contain
    valid positions and at least one selected valid token. Empty populations
    raise, never silently vanish from the record denominator. Only selected
    labels/logits are scored; unselected labels may be -100 or arbitrary values.

    FP16/BF16 logits are promoted to FP32 for CE; FP32/FP64 keep their precision.
    Negative-infinite logits may exclude classes, provided selected CE is finite.
    This stateless function cannot verify that all external microbatches were
    included exactly once, nor compensate for Trainer/DDP/Accelerate scaling.
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError('logits must be a floating [records, positions, classes] tensor')
    records, positions, classes = logits.shape
    if min(records, positions, classes) <= 0:
        raise ValueError('Empty record, position, or class dimensions are invalid')
    if type(global_record_count) is not int or global_record_count < records:
        raise ValueError('global_record_count must be an integer at least as large as this microbatch')
    for name, tensor, dtype in (('labels', labels, torch.long),
                                ('selected_mask', selected_mask, torch.bool),
                                ('valid_mask', valid_mask, torch.bool)):
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != (records, positions)
                or tensor.dtype != dtype or tensor.device != logits.device):
            raise ValueError(f'{name} has an incompatible shape, dtype, or device')
    if not valid_mask.any(1).all():
        raise ValueError('An empty record is invalid; no record is dropped from the denominator')
    if (selected_mask & ~valid_mask).any():
        raise ValueError('Selected tokens must be valid, never padding')
    selected_counts = selected_mask.sum(1)
    if (selected_counts == 0).any():
        raise ValueError('Every record requires a selected token; zero-selected records are not dropped')
    chosen_labels = labels[selected_mask]
    if ((chosen_labels < 0) | (chosen_labels >= classes)).any():
        raise ValueError('Selected labels must be valid class indices')
    chosen_logits = logits[selected_mask]
    if chosen_logits.dtype in (torch.float16, torch.bfloat16):
        chosen_logits = chosen_logits.float()
    if torch.isnan(chosen_logits).any() or torch.isposinf(chosen_logits).any():
        raise FloatingPointError('Selected logits contain NaN or positive infinity')
    token_losses = F.cross_entropy(chosen_logits, chosen_labels, reduction='none')
    if not torch.isfinite(token_losses).all():
        raise FloatingPointError('Selected token CE must be finite')
    row_indices = torch.arange(records, device=logits.device)[:, None].expand(records, positions)[selected_mask]
    # Divide before summing, so finite means need not overflow an intermediate sum.
    weighted_tokens = token_losses / selected_counts[row_indices].to(token_losses.dtype)
    per_record_mean = token_losses.new_zeros(records).index_add(0, row_indices, weighted_tokens)
    contribution = (per_record_mean / global_record_count).sum()
    if not torch.isfinite(per_record_mean).all() or not torch.isfinite(contribution):
        raise FloatingPointError('Record CE reduction must be finite')
    return RecordCELoss(contribution, per_record_mean,
                        selected_counts, records, global_record_count)
