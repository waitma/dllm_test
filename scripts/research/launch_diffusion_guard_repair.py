"""Run the frozen eight-GPU repair smoke test or guarded diffusion continuation.

Activate pllm; torchrun --nproc_per_node=8 THIS_FILE --bundle ABS --phase smoke
then repeat with --phase train. Training requires the same package's smoke receipt.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(bundle, phase):
    launch = json.loads((bundle / "direct_launch.json").read_text())
    if digest(bundle / "plan.json") != launch["plan_sha256"]:
        raise ValueError("Repair plan changed")
    plan = json.loads((bundle / "plan.json").read_text())
    source = Path(plan["source_root"])
    if Path(__file__).resolve() != source / "scripts/research/launch_diffusion_guard_repair.py":
        raise ValueError("Execute the frozen repair launcher")
    for group in ("sources", "vendor_files"):
        for path, expected in plan.get(group, {}).items():
            if digest(path) != expected:
                raise ValueError("Frozen repair dependency changed: " + path)
    for gate in plan.get("gates", []):
        if digest(gate["path"]) != gate["sha256"]:
            raise ValueError("Repair gate changed")
    if phase == "train":
        smoke = Path(plan["smoke_output"])
        done = json.loads((smoke / "COMPLETE.json").read_text())
        if (done["status"] != "passed" or done["world_size"] != 8
                or done["training"] != plan["training"] or done["recipe"] != plan["recipe"]
                or done["starting_step"] != plan["repair"]["start_step"]):
            raise ValueError("This repair has not passed eight-rank smoke/replay")
        provenance = json.loads((smoke / "run_provenance.json").read_text())
        if provenance["sources"] != plan["sources"] or provenance["repair"] != plan["repair"]:
            raise ValueError("Smoke checked a different repair")
        for rank in range(8):
            for name in (f"smoke_rank{rank}.json", f"repair_restore_rank{rank}.json"):
                if json.loads((smoke / name).read_text())["status"] != "passed":
                    raise ValueError("Incomplete rank smoke/restore")
    else:
        plan["output_dir"] = plan["smoke_output"]
        plan.pop("wandb", None)
    plan["phase"] = phase
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "train"), required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    plan = verify(args.bundle.resolve(), args.phase)
    source = Path(plan["source_root"])
    sys.path.insert(0, str(source))
    if plan.get("wandb_vendor"):
        sys.path.insert(0, plan["wandb_vendor"])
    os.environ.update(USE_TF="0", IMMUNE_SDPA_BACKEND="efficient", OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    from scripts.research import run_diffusion_continuation as runtime
    if not Path(runtime.__file__).resolve().is_relative_to(source):
        raise ValueError("Wrong repair runtime imported")
    if args.resume:
        if args.phase != "train" or args.resume.resolve().parent != Path(plan["output_dir"]).resolve():
            raise ValueError("Resume must refer to this repair run")
        plan["resume_checkpoint"] = str(args.resume.resolve())
    if args.verify_only:
        checkpoint = Path(plan["repair"]["checkpoint"])
        if digest(checkpoint / "checkpoint_receipt.json") != plan["repair"]["receipt_sha256"]:
            raise ValueError("Repair checkpoint identity changed")
        receipt = json.loads((checkpoint / "checkpoint_receipt.json").read_text())
        for name, expected in receipt["files"].items():
            if digest(checkpoint / name) != expected:
                raise ValueError("Repair checkpoint changed: " + name)
        runtime.data_gate(plan)
        runtime.verify_parent(plan)
        print(json.dumps(dict(status="passed", phase=args.phase,
                             checkpoint_files=len(receipt["files"]), training_started=False)))
        return
    import torch
    rank, world, local = [int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")]
    if world != 8 or torch.cuda.device_count() != 8:
        raise ValueError("Eight allocated GPUs required")
    torch.cuda.set_device(local)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        runtime.main(plan, torch.device("cuda", local), rank, world)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
