"""Check real optimizer updates, validation counting, schedule caps and resume.

Activate pllm; USE_TF=0 python -B -m pytest THIS_FILE -q.
"""
import copy
import math

import pytest
import torch

from scripts.research.diffusion_training_guard import ValidationLRCeiling
from scripts.research.diffusion_guard_repair import validate_repair

CONFIG = dict(factor=.5, patience=2, min_lr=.01, relative_threshold=.001)


def make(initial_scale=1., best=1.):
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([parameter], lr=.1)
    base = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1. + step / 100.)
    scheduler = ValidationLRCeiling(base, CONFIG, best=best, initial_scale=initial_scale)
    return parameter, optimizer, scheduler


def update(parameter, optimizer, scheduler):
    optimizer.zero_grad()
    parameter.square().backward()
    optimizer.step()
    scheduler.step()


def test_reduction_survives_next_base_scheduler_step_and_never_increases():
    p, o, s = make()
    update(p, o, s)
    assert not s.observe(1.1, 1)
    # Repeating the same validation cannot consume patience twice.
    assert not s.observe(1.1, 1)
    assert s.bad_validations == 1
    update(p, o, s)
    before = o.param_groups[0]["lr"]
    assert s.observe(1.2, 2)
    assert o.param_groups[0]["lr"] == before * .5
    update(p, o, s)
    assert o.param_groups[0]["lr"] == before * .5
    assert not s.observe(.8, 3)
    assert s.bad_validations == 0
    assert o.param_groups[0]["lr"] == before * .5


def test_optimizer_and_guard_resume_matches_uninterrupted_updates():
    p, o, s = make(initial_scale=.5)
    for loss in [1.1, 1.2, 1.3]:
        update(p, o, s)
        s.observe(loss, s.last_epoch)
    state = copy.deepcopy(s.state_dict())
    q, other_o, other_s = make()
    q.data.copy_(p.data)
    other_o.load_state_dict(copy.deepcopy(o.state_dict()))
    other_s.load_state_dict(state)
    assert other_s.state_dict() == state
    for loss in [1.4, .9, 1.5, 1.6]:
        for parameter, optimizer, scheduler in [(p, o, s), (q, other_o, other_s)]:
            update(parameter, optimizer, scheduler)
            scheduler.observe(loss, scheduler.last_epoch)
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        assert o.param_groups[0]["lr"] == other_o.param_groups[0]["lr"]
        assert s.state_dict() == other_s.state_dict()


def test_floor_and_invalid_validation_fail_closed():
    p, o, s = make()
    for _ in range(30):
        update(p, o, s)
        s.observe(2., s.last_epoch)
    assert o.param_groups[0]["lr"] == CONFIG["min_lr"]
    with pytest.raises(ValueError):
        s.observe(math.nan, s.last_epoch)
    with pytest.raises(ValueError):
        s.observe(2., s.last_epoch - 1)


def test_base_decay_below_floor_is_not_raised_and_warmup_is_preserved():
    p = torch.nn.Parameter(torch.tensor(1.))
    o = torch.optim.SGD([p], lr=.001)
    base = torch.optim.lr_scheduler.LambdaLR(o, lambda step: .5**step)
    s = ValidationLRCeiling(base, CONFIG, best=1.)
    assert o.param_groups[0]["lr"] == .001
    for _ in range(3):
        update(p, o, s)
        s.observe(2., s.last_epoch)
    assert o.param_groups[0]["lr"] == .001 * .5**3


def test_repair_preserves_original_recipe_and_source_identity():
    previous = dict(model={"d_model": 768}, recipe={"edit": .25}, world_size=8,
        training={"gradient_clip": 1.}, source_root="/old", sources={"/old/model.py": "fixed"})
    current = copy.deepcopy(previous)
    current.update(source_root="/new", sources={"/new/model.py": "fixed",
        "/new/scripts/research/diffusion_training_guard.py": "new"})
    current["training"]["validation_lr_guard"] = CONFIG
    repair = {"initial_lr_scale": .5}
    validate_repair(current, previous, repair)
    for key, value in [("recipe", {}), ("world_size", 1), ("sources", {"/new/model.py": "changed"})]:
        changed = {**current, key: value}
        with pytest.raises(ValueError):
            validate_repair(changed, previous, repair)
