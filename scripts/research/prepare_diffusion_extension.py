"""Freeze and enqueue the requested cumulative 200k diffusion/edit continuation.

Activate pllm; USE_TF=0 python -B -m scripts.research.prepare_diffusion_extension
--phase smoke [--enqueue], then --phase train --enqueue after the smoke passes.
Uses the retained eight-GPU allocation and never submits a new platform job.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.diffusion_extension import validate_extension, verify_unchanged_update

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "output/bert_diffusion_edit_v7_from287000_20261019"
RUN = ROOT / "output/bert_diffusion_edit_v7_from287000_200k_20261020"
POOL = ROOT / "output/immune_training_container_c20250601_20261007"
OVERLAYS = ("scripts/research/run_diffusion_continuation.py",
            "scripts/research/diffusion_extension.py",
            "scripts/research/prepare_diffusion_extension.py",
            "scripts/tests/immune_llada/test_diffusion_extension.py",
            "scripts/tests/immune_llada/test_diffusion_continuation_data.py",
            "scripts/tests/immune_llada/test_diffusion_continuation_update.py")


def prepare(phase, enqueue=False):
    checkpoint = ORIGINAL / "train/checkpoint-10000"
    previous = json.loads((checkpoint / "run_provenance.json").read_text())
    receipt = json.loads((checkpoint / "checkpoint_receipt.json").read_text())
    if receipt["status"] != "complete" or receipt["step"] != 10000 or receipt["world_size"] != 8:
        raise ValueError("Expected the completed eight-rank checkpoint-10000")
    if len(set(receipt["rank_parameter_sha256"])) != 1:
        raise ValueError("Previous rank parameters differ")
    for filename, digest in receipt["files"].items():
        if sha(checkpoint / filename) != digest:
            raise ValueError("Checkpoint changed: " + filename)
    for filename, digest in previous["sources"].items():
        if sha(filename) != digest:
            raise ValueError("Original frozen source changed: " + filename)
    old_package = json.loads((ORIGINAL / "packages.json").read_text())["train"]
    old_bundle = Path(old_package["bundle"])
    old_launch = json.loads((old_bundle / "direct_launch.json").read_text())
    if sha(old_bundle / "plan.json") != old_launch["plan_sha256"]:
        raise ValueError("Original launch changed")
    old_plan = json.loads((old_bundle / "plan.json").read_text())
    state = json.loads((checkpoint / "state.json").read_text())
    best = checkpoint.parent / f"checkpoint-{state['best_step']}"
    best_receipt = json.loads((best / "checkpoint_receipt.json").read_text())
    if best_receipt["status"] != "complete" or sha(best / "model.safetensors") != best_receipt["files"]["model.safetensors"]:
        raise ValueError("Inherited best checkpoint changed")
    extension = dict(schema="immune.diffusion_edit.extension.v1", checkpoint=str(checkpoint),
        checkpoint_bindings={name: sha(checkpoint / name) for name in (
            "checkpoint_receipt.json", "run_provenance.json", "state.json", "scheduler.pt")},
        start_step=10000, end_step=200000, rewarm_steps=1000, original_warmup_steps=1000,
        start_fraction=.1, final_fraction=.1, inherited_best_checkpoint=str(best),
        inherited_best_receipt_sha256=sha(best / "checkpoint_receipt.json"),
        original_wandb_url="https://wandb.ai/pressure_jian/dllm-immune/runs/diffedit287k1019",
        authorization="User requested at least 200000 cumulative diffusion/edit optimizer steps")
    training = {**previous["training"], "max_steps": 200000,
                "scheduler": "resume_rewarm_cosine", "optimizer_reset": False}
    RUN.mkdir(exist_ok=True)
    registry_path = RUN / "packages.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    if phase in registry:
        raise ValueError("Phase already prepared; inspect its existing package")
    bundle = ROOT / f"output/v5_full_binding_20261007/launch_diffusion_edit287000_extend200k_{phase}_20261020"
    bundle.mkdir(exist_ok=False)
    source = bundle / "source"
    old_source = Path(previous["source_root"])
    for path in old_source.rglob("*.py"):
        target = source / path.relative_to(old_source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for filename in OVERLAYS:
        target = source / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / filename, target)
    sources = {str(p): sha(p) for p in source.rglob("*.py")}
    proposed = {**previous, "training": training, "source_root": str(source),
                "sources": sources, "extension": extension}
    validate_extension(proposed, previous, extension)
    verify_unchanged_update(old_source / "scripts/research/run_diffusion_continuation.py",
                            source / "scripts/research/run_diffusion_continuation.py")
    gates = list(old_plan["gates"])
    checks = RUN / "cpu_checks.json"
    if phase == "smoke":
        tests = [source / name for name in OVERLAYS if "/tests/" in name]
        result = subprocess.run([sys.executable, "-I", "-B", "-c",
            "import sys;sys.path.insert(0,sys.argv[1]);import pytest;raise SystemExit(pytest.main(sys.argv[2:]+['-q']))",
            str(source), *map(str, tests)], cwd="/tmp", capture_output=True, text=True)
        (RUN / "cpu_checks.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("Frozen extension tests failed; inspect cpu_checks.log")
        write_json(checks, dict(status="passed", source_root=str(source),
            overlay_sha256={name: sha(source / name) for name in OVERLAYS},
            unchanged_profiled_update=True, checkpoint_files_verified=len(receipt["files"]),
            log=str(RUN / "cpu_checks.log")))
    else:
        smoke_source = Path(registry["smoke"]["source"])
        for filename in OVERLAYS:
            if sha(source / filename) != sha(smoke_source / filename):
                raise ValueError("Code changed after continuation smoke: " + filename)
        smoke = Path(registry["smoke"]["output"])
        done = json.loads((smoke / "COMPLETE.json").read_text())
        if (done["status"] != "passed" or done["world_size"] != 8
                or done["starting_step"] != 10000 or done["extension"] != extension
                or done["training"] != training):
            raise ValueError("Continuation smoke did not pass the exact proposed transition")
        for rank in range(8):
            for filename in (f"extension_restore_rank{rank}.json", f"smoke_rank{rank}.json"):
                path = smoke / filename
                if json.loads(path.read_text())["status"] != "passed":
                    raise ValueError("Incomplete rank continuation verification")
                gates.append(dict(path=str(path), sha256=sha(path)))
        gates.append(dict(path=str(smoke / "COMPLETE.json"), sha256=sha(smoke / "COMPLETE.json")))
    if json.loads(checks.read_text())["status"] != "passed":
        raise ValueError("CPU checks failed")
    gates.append(dict(path=str(checks), sha256=sha(checks)))
    output = RUN / phase
    plan = {**old_plan, "phase": "extension_smoke" if phase == "smoke" else "train",
            "stage": "extend200k_"+phase, "source_root": str(source), "sources": sources,
            "output_dir": str(output), "training": training, "extension": extension,
            "gates": gates, "created_utc": datetime.now(timezone.utc).isoformat()}
    if phase == "smoke":
        for key in ("wandb", "wandb_vendor", "vendor_files"):
            plan.pop(key, None)
    else:
        plan["wandb"] = {**old_plan["wandb"], "run_id": "diffedit287k200k1020",
                         "name": "V7-BERT287k-diffusion-edit-balanced-continue-to-200k"}
    write_json(bundle / "plan.json", plan)
    write_json(bundle / "direct_launch.json", dict(plan_sha256=sha(bundle / "plan.json"),
        runtime_dir=str(RUN / (phase+"_runtime"))))
    launcher = source / "scripts/research/launch_v5_direct.py"
    ticket = dict(bundle=str(bundle), launcher_sha256=sha(launcher),
        purpose="User-authorized continuation from diffusion10000 to cumulative200000: "+phase)
    write_json(bundle / "ticket.json", ticket)
    registry[phase] = dict(bundle=str(bundle), source=str(source), output=str(output),
                           ticket=str(bundle / "ticket.json"))
    write_json(registry_path, registry)
    if enqueue:
        pool_state = json.loads((POOL / "state.json").read_text())
        if (pool_state["state"] != "standby" or time.time()-pool_state["time"] > 60
                or list((POOL / "running").glob("*.json")) or list((POOL / "pending").glob("*.json"))
                or any((POOL / name).exists() for name in ("STOP", "STOP_NOW"))):
            raise ValueError("Pool is not ready; prepared package was not enqueued")
        name = f"diffusion_edit287000_extend200k_{phase}_20261020.json"
        if (POOL / "finished" / name).exists():
            raise ValueError("Ticket was already attempted")
        write_json(POOL / "pending" / name, ticket)
        (POOL / "PAUSE_TRAINING").unlink(missing_ok=True)
    print(json.dumps(dict(**registry[phase], enqueued=enqueue,
                         start_step=10000, target_step=200000), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "train"), required=True)
    parser.add_argument("--enqueue", action="store_true")
    args = parser.parse_args()
    prepare(args.phase, args.enqueue)
