"""Check continuous schedule extension, native Adam restore and provenance guards.

Activate pllm; USE_TF=0 python -B -m pytest
scripts/tests/immune_llada/test_diffusion_extension.py -q.
"""
import copy

import pytest
import torch

from scripts.research.diffusion_continuation_data import warmup_cosine_scheduler
from scripts.research.diffusion_extension import (
    extension_factor, extension_scheduler, validate_extension)


def extension():
    return dict(start_step=10000, end_step=200000, original_warmup_steps=1000,
                rewarm_steps=1000, start_fraction=.1, final_fraction=.1)


def test_schedule_is_continuous_and_has_requested_horizon():
    spec = extension()
    assert extension_factor(0, spec) == pytest.approx(.01)
    assert extension_factor(1000, spec) == pytest.approx(1.)
    assert extension_factor(10000, spec) == pytest.approx(.1)
    assert extension_factor(10001, spec) == pytest.approx(.1009)
    assert extension_factor(11000, spec) == pytest.approx(1.)
    assert extension_factor(200000, spec) == pytest.approx(.1)
    assert extension_factor(200001, spec) == pytest.approx(.1)
    assert extension_factor(10001, spec) < .11
    with pytest.raises(ValueError):
        extension_factor(10000, {**spec, "end_step": 10500})


def test_native_optimizer_and_scheduler_restore_preserve_first_update():
    spec = extension()
    p = torch.nn.Parameter(torch.tensor([.7]))
    optimizer = torch.optim.AdamW([p], lr=2e-5)
    old_scheduler = warmup_cosine_scheduler(optimizer, 10000, 1000)
    for _ in range(3):
        p.grad = torch.tensor([.25])
        optimizer.step()
        old_scheduler.step()
    # Place the scheduler at the real boundary without a costly 10000-update test.
    old_scheduler.last_epoch = 9999
    optimizer.step()
    old_scheduler.step()
    model_state = p.detach().clone()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(old_scheduler.state_dict())
    restored = torch.nn.Parameter(model_state.clone())
    new_optimizer = torch.optim.AdamW([restored], lr=2e-5)
    new_scheduler = extension_scheduler(new_optimizer, spec)
    new_optimizer.load_state_dict(saved_optimizer)
    new_scheduler.load_state_dict(saved_scheduler)
    assert new_scheduler.state_dict() == saved_scheduler
    assert new_optimizer.param_groups == [{**optimizer.param_groups[0], "params": [restored]}]
    for key, value in optimizer.state[p].items():
        torch.testing.assert_close(value, new_optimizer.state[restored][key], rtol=0, atol=0)
    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(2e-6)
    p.grad = restored.grad = torch.tensor([.35])
    optimizer.step()
    new_optimizer.step()
    new_scheduler.step()
    torch.testing.assert_close(p, restored, rtol=0, atol=0)
    assert new_scheduler.last_epoch == 10001
    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(2.018e-6)
    later = copy.deepcopy((new_optimizer.state_dict(), new_scheduler.state_dict()))
    second = torch.optim.AdamW([torch.nn.Parameter(restored.detach().clone())], lr=2e-5)
    second_scheduler = extension_scheduler(second, spec)
    second.load_state_dict(later[0])
    second_scheduler.load_state_dict(later[1])
    new_optimizer.step()
    new_scheduler.step()
    second.step()
    second_scheduler.step()
    assert second.param_groups[0]["lr"] == new_optimizer.param_groups[0]["lr"]


def bindings():
    old = dict(model={"size": 75}, recipe={"eos_gamma": 0}, world_size=8,
               data_index="/data", data_index_sha256="datahash", source_root="/old",
               sources={"/old/scripts/research/run_diffusion_continuation.py": "old",
                        "/old/model.py": "same"},
               training=dict(max_steps=10000, warmup_steps=1000, global_batch=512,
                             scheduler="warmup_cosine_0.01_to_1_to_0.1", optimizer_reset=True))
    new = copy.deepcopy(old)
    new.update(source_root="/new", sources={"/new/scripts/research/run_diffusion_continuation.py": "new",
                                          "/new/model.py": "same"}, extension=extension())
    new["training"].update(max_steps=200000, scheduler="resume_rewarm_cosine", optimizer_reset=False)
    return old, new


def test_only_declared_extension_is_accepted():
    old, new = bindings()
    validate_extension(new, old, extension())


@pytest.mark.parametrize("field", ["recipe", "model", "world_size", "data_index_sha256", "batch", "source"])
def test_unrelated_changes_are_rejected(field):
    old, new = bindings()
    if field == "batch":
        new["training"]["global_batch"] = 256
    elif field == "source":
        new["sources"]["/new/model.py"] = "changed"
    else:
        new[field] = "changed"
    with pytest.raises(ValueError):
        validate_extension(new, old, extension())
