"""Build an isolated, hash-bound diffusion LR guard repair; never launch training.

Activate pllm; USE_TF=0 python -B -m scripts.research.prepare_diffusion_guard_repair
--checkpoint ABS --original-plan ABS --output ABS --checks ABS.
The output includes a smoke-first eight-GPU script. Historical bundles are read-only.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil

from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.diffusion_guard_repair import OVERLAYS, validate_repair

ROOT = Path(__file__).resolve().parents[2]


def prepare(checkpoint, original_plan, output, checks):
    for path in (checkpoint, original_plan, output, checks):
        if not path.is_absolute():
            raise ValueError("Use absolute paths")
    acceptance = json.loads(checks.read_text())
    if acceptance["status"] != "passed":
        raise ValueError("CPU regression checks have not passed")
    for name in OVERLAYS:
        if acceptance["sources"].get(name) != sha(ROOT / name):
            raise ValueError("Unchecked repair source: " + name)
    old = json.loads(original_plan.read_text())
    if sha(original_plan) != json.loads((original_plan.parent / "direct_launch.json").read_text())["plan_sha256"]:
        raise ValueError("Original plan identity changed")
    previous = json.loads((checkpoint / "run_provenance.json").read_text())
    receipt = json.loads((checkpoint / "checkpoint_receipt.json").read_text())
    if (receipt["status"] != "complete" or receipt["world_size"] != 8
            or len(receipt["rank_parameter_sha256"]) != 8
            or len(set(receipt["rank_parameter_sha256"])) != 1):
        raise ValueError("Expected complete eight-rank checkpoint")
    for name, expected in receipt["files"].items():
        if sha(checkpoint / name) != expected:
            raise ValueError("Checkpoint changed: " + name)
    for name, expected in previous["sources"].items():
        if sha(name) != expected:
            raise ValueError("Original source changed: " + name)
    for name in ("model", "training", "recipe", "data_index", "data_index_sha256",
                 "source_root", "sources", "extension", "parent_checkpoint", "parent_model_sha256"):
        if old.get(name) != previous.get(name):
            raise ValueError("Checkpoint differs from original plan: " + name)
    output.mkdir(parents=True, exist_ok=False)
    bundle, source = output / "launch", output / "launch/source"
    old_source = Path(previous["source_root"])
    for name in previous["sources"]:
        target = source / Path(name).relative_to(old_source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    for name in OVERLAYS:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    training = {**previous["training"], "validation_lr_guard": dict(
        factor=.5, patience=3, min_lr=2e-6, relative_threshold=.001)}
    repair = dict(schema="immune.diffusion.lr_guard_repair.v1", checkpoint=str(checkpoint),
        receipt_sha256=sha(checkpoint / "checkpoint_receipt.json"),
        start_step=receipt["step"], initial_lr_scale=.5,
        reason="Persistent fixed-validation regression; controlled LR reduction, no objective change")
    sources = {str(p): sha(p) for p in source.rglob("*.py")}
    proposed = {**previous, "training": training, "source_root": str(source),
                "sources": sources, "repair": repair}
    validate_repair(proposed, previous, repair)
    plan = {**old, "training": training, "source_root": str(source), "sources": sources,
        "repair": repair, "output_dir": str(output / "train"), "smoke_output": str(output / "smoke"),
        "stage": "validation_lr_guard_repair", "phase": "train",
        "gates": [*old.get("gates", []), {"path": str(checks), "sha256": sha(checks)}],
        "created_utc": datetime.now(timezone.utc).isoformat()}
    for name in ("resume_checkpoint", "resume_contract"):
        plan.pop(name, None)
    if plan.get("wandb"):
        plan["wandb"] = {**plan["wandb"], "run_id": "diffguard-" + output.name,
                         "name": "Diffusion validation LR guard " + output.name}
    write_json(bundle / "plan.json", plan)
    write_json(bundle / "direct_launch.json", {"plan_sha256": sha(bundle / "plan.json")})
    launcher = source / "scripts/research/launch_diffusion_guard_repair.py"
    command = f'torchrun --standalone --nproc_per_node=8 {shlex.quote(str(launcher))} --bundle {shlex.quote(str(bundle))}'
    env = ROOT.parents[1] / "conda/envs/pllm"
    script = ("#!/usr/bin/env bash\nset -euo pipefail\n"
              f'export PATH={shlex.quote(str(env / "bin"))}:"$PATH"\n'
              f'cd {shlex.quote(str(source))}\n'
              "export USE_TF=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
              + command + " --phase smoke\n" + command + " --phase train\n")
    (bundle / "train.sh").write_text(script)
    (bundle / "train.sh").chmod(0o755)
    write_json(output / "preparation.json", dict(status="prepared", training_started=False,
        checkpoint=str(checkpoint), files_verified=len(receipt["files"]),
        original_plan_sha256=sha(original_plan), original_sources_unchanged=True,
        cpu_checks=str(checks), bundle=str(bundle), gpu_qualification="pending"))
    print(json.dumps({"bundle": str(bundle), "training_started": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "original-plan", "output", "checks"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.checkpoint, args.original_plan, args.output, args.checks)
