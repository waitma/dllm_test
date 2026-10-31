"""Unchanged model adapters; run python -B -m architecture_confirm.tests."""
from experiments.arch70m.models import build as baseline_build
from architecture_round2.models import build as global_build


def build(settings,variant,seed):
    if variant == 'A0':return baseline_build(settings,'A0',seed)
    if variant in ('B0','B1'):return global_build(settings,variant,seed)
    raise ValueError('Only the preselected A0 / A5(B0) / B1 confirmation models are allowed.')
