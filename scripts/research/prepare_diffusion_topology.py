"""Repackage a frozen fresh-BERT diffusion run for four/eight ranks, without training.

Activate pllm; python -B -m scripts.research.prepare_diffusion_topology
--reference-bundle ABS --checks ABS --world-size 4 --output ABS.
Keeps data/recipe/optimizer settings; new source identity requires a new GPU profile.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil

from scripts.research.launch_diffusion_bert_restart import digest, production_world_size

ROOT = Path(__file__).resolve().parents[2]
OVERLAYS = ("scripts/research/launch_diffusion_bert_restart.py",
            "scripts/research/verify_diffusion_bert_restart.py",
            "scripts/research/prepare_diffusion_topology.py",
            "scripts/tests/immune_llada/test_diffusion_topology.py",
            "scripts/tests/immune_llada/test_diffusion_continuation_update.py")


def prepare(reference, checks, world, output):
    if not all(p.is_absolute() for p in (reference, checks, output)):
        raise ValueError("Use absolute paths")
    previous = json.loads((reference / "plan.json").read_text())
    if digest(reference / "plan.json") != json.loads((reference / "direct_launch.json").read_text())["plan_sha256"]:
        raise ValueError("Reference plan changed")
    if (previous["parent_step"] != 287000 or not previous["training"]["optimizer_reset"]
            or any(k in previous for k in ("repair", "extension", "resume_checkpoint"))):
        raise ValueError("Only a fresh BERT287000 experiment can change topology here")
    acceptance = json.loads(checks.read_text())
    if acceptance["status"] != "passed":
        raise ValueError("CPU checks must pass first")
    for name in OVERLAYS:
        if acceptance["sources"].get(name) != digest(ROOT / name):
            raise ValueError("Unchecked topology source: " + name)
    for group in ("sources", "vendor_files"):
        for name, expected in previous.get(group, {}).items():
            if digest(Path(name)) != expected:
                raise ValueError("Reference dependency changed: " + name)
    for gate in previous["gates"]:
        if digest(Path(gate["path"])) != gate["sha256"]:
            raise ValueError("Reference gate changed")
    plan = {**previous, "world_size": world}
    production_world_size(plan)
    output.mkdir(parents=True, exist_ok=False)
    bundle, source = output / "launch", output / "launch/source"
    for name in previous["sources"]:
        target = source / Path(name).relative_to(previous["source_root"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    for name in OVERLAYS:
        shutil.copy2(ROOT / name, source / name)
    plan.update(source_root=str(source), sources={str(p): digest(p) for p in source.rglob("*.py")},
                output_dir=str(output / "train"), smoke_output=str(output / "smoke"),
                profile_receipt=str(output / "max_shape_profile.json"),
                created_utc=datetime.now(timezone.utc).isoformat(),
                gates=[*previous["gates"], dict(path=str(checks), sha256=digest(checks))],
                wandb={**previous["wandb"], "run_id": "diffbert-" + output.name,
                       "name": f"BERT287k CDR diffusion {world}GPU " + output.name})
    def write(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n")
    write(bundle / "plan.json", plan)
    write(bundle / "direct_launch.json", dict(plan_sha256=digest(bundle / "plan.json")))
    shell = (reference.parent / "launch.sh").read_text()
    old_world = production_world_size(previous)
    shell = shell.replace(str(reference.parent), str(output))
    shell = shell.replace(f"nproc_per_node={old_world}", f"nproc_per_node={world}")
    shell = shell.replace(f"{old_world} A100", f"{world} A100")
    shell = shell.replace(f"{old_world}-rank", f"{world}-rank")
    shell = shell.replace("eight-rank", f"{world}-rank")
    (output / "launch.sh").write_text(shell)
    (output / "launch.sh").chmod(0o755)
    (bundle / "train.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec bash "
                                    + shlex.quote(str(output / "launch.sh")) + ' "$@"\n')
    (bundle / "train.sh").chmod(0o755)
    write(output / "preparation.json", dict(status="prepared", reference_bundle=str(reference),
        reference_plan_sha256=digest(reference / "plan.json"), plan_sha256=digest(bundle / "plan.json"),
        world_size=world, training_unchanged=plan["training"] == previous["training"],
        recipe_unchanged=plan["recipe"] == previous["recipe"], data_index=plan["data_index"],
        data_index_sha256=plan["data_index_sha256"], profile="pending", production_smoke="pending",
        formal_training_started=False, cloud_submitted=False))
    print(json.dumps(dict(bundle=str(bundle), world_size=world, formal_training_started=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.reference_bundle, args.checks, args.world_size, args.output)
