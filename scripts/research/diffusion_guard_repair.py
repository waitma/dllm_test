"""Restore a completed diffusion checkpoint for an explicit guarded training fork.

Activate pllm; use prepare_diffusion_guard_repair.py to build an isolated package.
The original optimizer/RNG/data cursor are restored before applying the new LR cap.
"""
import json
from pathlib import Path

from scripts.research.bert_mlm_data import sha, write_json

OVERLAYS = {
    "scripts/research/run_diffusion_continuation.py",
    "scripts/research/diffusion_training_guard.py",
    "scripts/research/diffusion_guard_repair.py",
    "scripts/research/launch_diffusion_guard_repair.py",
    "scripts/research/prepare_diffusion_guard_repair.py",
    "scripts/tests/immune_llada/test_diffusion_training_guard.py",
    "scripts/tests/immune_llada/test_diffusion_continuation_update.py",
}


def validate_repair(current, previous, repair):
    allowed = {"training", "source_root", "sources", "repair"}
    if ({k: v for k, v in current.items() if k not in allowed}
            != {k: v for k, v in previous.items() if k not in allowed}):
        raise ValueError("Repair changed model/data/objective/distributed semantics")
    training = dict(current["training"])
    guard = training.pop("validation_lr_guard", None)
    if not guard or training != previous["training"]:
        raise ValueError("Repair may only add the validation LR guard")
    if not 0 < repair["initial_lr_scale"] <= 1:
        raise ValueError("Repair cannot increase the initial LR")
    old_root, new_root = Path(previous["source_root"]), Path(current["source_root"])
    old = {str(Path(p).relative_to(old_root)): digest for p, digest in previous["sources"].items()}
    new = {str(Path(p).relative_to(new_root)): digest for p, digest in current["sources"].items()}
    if not set(old).issubset(new):
        raise ValueError("Repair removed frozen source")
    for name in set(old) | set(new):
        if old.get(name) != new.get(name) and name not in OVERLAYS:
            raise ValueError("Unrelated frozen source changed: " + name)


def restore_repair(plan, provenance, model, optimizer, scheduler, noise, rank):
    from scripts.research.run_bert_mlm import load_checkpoint, rank_zero
    repair = plan["repair"]
    checkpoint = Path(repair["checkpoint"])
    previous = json.loads((checkpoint / "run_provenance.json").read_text())
    validate_repair(provenance, previous, repair)

    def verify():
        if sha(checkpoint / "checkpoint_receipt.json") != repair["receipt_sha256"]:
            raise ValueError("Repair checkpoint identity changed")
        receipt = json.loads((checkpoint / "checkpoint_receipt.json").read_text())
        if (receipt["world_size"] != provenance["world_size"]
                or receipt["step"] != repair["start_step"]):
            raise ValueError("Repair checkpoint step/world differs")
    rank_zero(verify, rank)
    state = load_checkpoint(checkpoint, model, optimizer, scheduler, noise, previous, rank)
    if state["step"] != repair["start_step"]:
        raise ValueError("Repair restored the wrong step")
    write_json(Path(plan["output_dir"]) / f"repair_restore_rank{rank}.json", dict(
        status="passed", restored_step=state["step"], checkpoint=str(checkpoint),
        model_optimizer_scheduler_rng_exact=True,
        restored_learning_rates=[group["lr"] for group in optimizer.param_groups],
        initial_lr_scale=repair["initial_lr_scale"]))
    return state
