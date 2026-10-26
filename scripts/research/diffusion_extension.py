"""Extend completed diffusion training while restoring its full native state.

Activate pllm; python -B -m pytest
scripts/tests/immune_llada/test_diffusion_extension.py -q.
The recorded extension changes only the future schedule and cumulative budget.
"""
import ast
import json
import math
from pathlib import Path

import torch

from scripts.research.bert_mlm_data import sha, write_json


def extension_factor(step, extension):
    """Global-step schedule: unchanged old phase, continuous rewarm, long cosine."""
    start, end = extension["start_step"], extension["end_step"]
    warmup = extension["rewarm_steps"]
    initial, final = extension["start_fraction"], extension["final_fraction"]
    if not (0 < extension["original_warmup_steps"] < start < start + warmup < end
            and 0 < initial <= 1 and 0 < final <= 1):
        raise ValueError("Invalid continuation schedule")
    if step <= start:
        old_warmup = extension["original_warmup_steps"]
        if step < old_warmup:
            return .01 + .99 * step / old_warmup
        progress = (step - old_warmup) / (start - old_warmup)
        return initial + (1 - initial) * .5 * (1 + math.cos(math.pi * progress))
    if step < start + warmup:
        return initial + (1 - initial) * (step - start) / warmup
    progress = min(1., (step - start - warmup) / (end - start - warmup))
    return final + (1 - final) * .5 * (1 + math.cos(math.pi * progress))


def extension_scheduler(optimizer, extension):
    extension_factor(extension["start_step"], extension)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: extension_factor(step, extension))


def validate_extension(current, previous, extension):
    """Allow the explicit budget/scheduler transition, retaining all other contracts."""
    changed = {"training", "source_root", "sources", "extension"}
    if ({k: v for k, v in current.items() if k not in changed}
            != {k: v for k, v in previous.items() if k not in changed}):
        raise ValueError("Extension changed model/data/objective/distributed semantics")
    expected = {**previous["training"], "max_steps": extension["end_step"],
                "scheduler": "resume_rewarm_cosine", "optimizer_reset": False}
    if current["training"] != expected:
        raise ValueError("Extension changed an unrelated training setting")
    if (extension["start_step"] != previous["training"]["max_steps"]
            or extension["original_warmup_steps"] != previous["training"]["warmup_steps"]
            or previous["training"]["scheduler"] != "warmup_cosine_0.01_to_1_to_0.1"
            or extension["start_fraction"] != .1):
        raise ValueError("Unsupported original schedule boundary")
    old_root, new_root = Path(previous["source_root"]), Path(current["source_root"])
    for filename, digest in previous["sources"].items():
        relative = Path(filename).relative_to(old_root)
        if str(relative) != "scripts/research/run_diffusion_continuation.py":
            if current["sources"].get(str(new_root / relative)) != digest:
                raise ValueError("Unrelated frozen source changed: " + str(relative))
    extension_factor(extension["start_step"], extension)


def verify_unchanged_update(previous_source, current_source):
    """Reuse the measured memory profile only when its compute path is unchanged."""
    def functions(path):
        tree = ast.parse(Path(path).read_text())
        return {item.name: ast.dump(item, include_attributes=False) for item in tree.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
    old, new = functions(previous_source), functions(current_source)
    for name in ("update", "profile", "new_model", "verify_parent"):
        if old[name] != new[name]:
            raise ValueError("Profiled computation changed: " + name)


def restore_extension(plan, provenance, model, optimizer, scheduler, noise, rank):
    """Use the existing strict model/Adam/scheduler/RNG restore after transition checks."""
    from scripts.research.run_bert_mlm import load_checkpoint, rank_zero, parameter_digest
    extension = plan["extension"]
    checkpoint = Path(extension["checkpoint"])
    previous = json.loads((checkpoint / "run_provenance.json").read_text())

    def verify():
        for filename, expected in extension["checkpoint_bindings"].items():
            if sha(checkpoint / filename) != expected:
                raise ValueError("Continuation checkpoint identity changed: " + filename)
        completed = json.loads((checkpoint.parent / "COMPLETE.json").read_text())
        if completed["status"] != "success" or completed["step"] != extension["start_step"]:
            raise ValueError("Original phase did not complete at the expected step")
        validate_extension(provenance, previous, extension)
        verify_unchanged_update(Path(previous["source_root"]) / "scripts/research/run_diffusion_continuation.py",
                                Path(provenance["source_root"]) / "scripts/research/run_diffusion_continuation.py")
        return True

    rank_zero(verify, rank)
    state = load_checkpoint(checkpoint, model, optimizer, scheduler, noise, previous, rank)
    if state["step"] != extension["start_step"] or state["seen_records"] != state["step"] * plan["training"]["global_batch"]:
        raise ValueError("Restored training/data cursor differs")
    expected_lr = plan["training"]["learning_rate"] * extension_factor(state["step"], extension)
    if any(not math.isclose(group["lr"], expected_lr, rel_tol=1e-12)
           for group in optimizer.param_groups):
        raise ValueError("Continuation would jump at the learning-rate boundary")
    write_json(Path(plan["output_dir"]) / f"extension_restore_rank{rank}.json", dict(
        status="passed", restored_step=state["step"], seen_records=state["seen_records"],
        parameter_sha256=parameter_digest(model), optimizer_exact=True,
        scheduler_state_exact=True, rank_rng_exact=True, first_learning_rate=expected_lr,
        next_learning_rate=plan["training"]["learning_rate"] * extension_factor(state["step"]+1, extension),
        checkpoint=str(checkpoint)))
    return state
