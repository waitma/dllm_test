"""Extend a fixed training horizon with an explicit warmup/cosine restart.

Activate conda pllm; run the CPU native-resume check with:
python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_epoch_schedule.py
This module does not load a model, dataset, checkpoint, or GPU.
"""
import math
from contextlib import contextmanager
from functools import partial


def epoch_restart_factor(step, *, first_phase_steps, total_steps, warmup_steps):
    """Preserve the first LR horizon, then warm up from its terminal zero LR.

    The LR boundary and data epoch boundary need not coincide: the existing
    horizon is 25042 while the observed loader has 25041 updates per epoch.
    Two complete data epochs therefore end at global step 50082.
    """
    if not (type(step) is int and step >= 0 and type(first_phase_steps) is int
            and type(total_steps) is int and total_steps > first_phase_steps
            and type(warmup_steps) is int
            and 0 < warmup_steps < min(first_phase_steps, total_steps-first_phase_steps)):
        raise ValueError('Expected two positive LR phases, valid warmup, and nonnegative step')
    if step >= total_steps:
        return 0.0
    within_phase = step if step < first_phase_steps else step-first_phase_steps
    phase_steps = first_phase_steps if step < first_phase_steps else total_steps-first_phase_steps
    if within_phase < warmup_steps:
        return float(within_phase)/warmup_steps
    progress = float(within_phase-warmup_steps)/(phase_steps-warmup_steps)
    return 0.5*(1.0+math.cos(math.pi*progress))


def skip_first_batches_preserving_epoch(dataloader, num_batches, *, native):
    """Keep the epoch that Trainer set before Accelerate constructs a skip loader."""
    epoch = getattr(dataloader, 'iteration', None)
    if type(epoch) is not int or epoch < 0:
        raise ValueError('An explicit nonnegative Accelerate loader iteration is required')
    result = native(dataloader, num_batches)
    setter = getattr(result, 'set_epoch', None)
    if not callable(setter):
        raise ValueError('The resumed skip loader must support set_epoch')
    setter(epoch)
    if getattr(result, 'iteration', None) != epoch:
        raise ValueError('The skip loader lost its original epoch')
    return result


@contextmanager
def preserve_resume_epoch():
    """Temporarily scope the skip-loader correction to one Trainer invocation."""
    import transformers.trainer as trainer_module
    original = trainer_module.skip_first_batches
    replacement = partial(skip_first_batches_preserving_epoch, native=original)
    trainer_module.skip_first_batches = replacement
    try:
        yield
    finally:
        if trainer_module.skip_first_batches is not replacement:
            raise RuntimeError('Trainer skip-loader integration changed during this run')
        trainer_module.skip_first_batches = original
