"""Launch the frozen fresh-from-BERT stage after its configured rank replay.

Activate pllm; torchrun --nproc_per_node=4 THIS_FILE --bundle ABS --phase smoke
then --phase train. For CPU preflight use --phase smoke --verify-only.
The rank count must match plan.world_size (legacy bundles default to eight).
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

from hashlib import sha256


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def production_world_size(plan):
    world = plan.get("world_size", 8)
    if type(world) is not int or world not in (4, 8) or plan["training"]["global_batch"] % world:
        raise ValueError("Expected four or eight ranks dividing the global batch")
    return world


def verify_smoke(plan):
    world = production_world_size(plan)
    smoke = Path(plan["smoke_output"])
    done = json.loads((smoke / "COMPLETE.json").read_text())
    if (done["status"] != "passed" or done["world_size"] != world or done["starting_step"] != 0
            or done["training"] != plan["training"] or done["recipe"] != plan["recipe"]):
        raise ValueError(f"Missing fresh {world}-rank smoke")
    provenance = json.loads((smoke / "run_provenance.json").read_text())
    if provenance["world_size"] != world:
        raise ValueError("Smoke provenance checked another world size")
    for key in ("sources", "data_index_sha256", "parent_model_sha256"):
        if provenance[key] != plan[key]:
            raise ValueError("Smoke checked another package")
    for rank in range(world):
        if json.loads((smoke / f"smoke_rank{rank}.json").read_text())["status"] != "passed":
            raise ValueError("Missing rank replay")


def verify_profile(plan, profile, plan_sha256):
    world = production_world_size(plan)
    records = plan["training"]["global_batch"] // world
    views = records * plan["training"]["complementary_views"]
    micro = plan["training"]["micro_batch_views"]
    chunks = [min(micro, views - offset) for offset in range(0, views, micro)]
    if (profile["status"] != "passed" or profile["plan_sha256"] != plan_sha256
            or profile["production_world_size"] != world
            or profile["per_rank_records"] != records or profile["per_rank_views"] != views
            or profile["micro_views"] != chunks or profile["max_length"] != plan["model"]["max_length"]):
        raise ValueError("Current package/topology has not passed the local GPU profile")


def verify(bundle, phase):
    if digest(bundle / "plan.json") != json.loads((bundle / "direct_launch.json").read_text())["plan_sha256"]:
        raise ValueError("Frozen plan changed")
    plan = json.loads((bundle / "plan.json").read_text())
    production_world_size(plan)
    source = Path(plan["source_root"])
    if Path(__file__).resolve() != source / "scripts/research/launch_diffusion_bert_restart.py":
        raise ValueError("Execute the frozen restart launcher")
    if (any(name in plan for name in ("repair", "extension", "resume_checkpoint"))
            or plan["parent_step"] != 287000 or not plan["training"]["optimizer_reset"]):
        raise ValueError("Fresh diffusion must start from BERT, without old diffusion state")
    for group in ("sources", "vendor_files"):
        for path, expected in plan.get(group, {}).items():
            if digest(path) != expected:
                raise ValueError("Frozen dependency changed: " + path)
    for gate in plan["gates"]:
        if digest(gate["path"]) != gate["sha256"]:
            raise ValueError("Restart qualification changed")
    if phase == "train":
        verify_smoke(plan)
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
        raise ValueError("Wrong restart runtime imported")
    if args.resume:
        if args.phase != "train" or args.resume.resolve().parent != Path(plan["output_dir"]).resolve():
            raise ValueError("Resume must belong to this new BERT-initialized run")
        plan["resume_checkpoint"] = str(args.resume.resolve())
    if args.verify_only:
        runtime.data_gate(plan)
        runtime.verify_parent(plan)
        print(json.dumps(dict(status="passed", training_started=False, diffusion_start_step=0,
                              world_size=production_world_size(plan))))
        return
    profile = json.loads(Path(plan["profile_receipt"]).read_text())
    verify_profile(plan, profile, digest(args.bundle / "plan.json"))
    import torch
    rank, world, local = [int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")]
    expected_world = production_world_size(plan)
    if world != expected_world or torch.cuda.device_count() != expected_world:
        raise ValueError(f"{expected_world} allocated GPUs required")
    torch.cuda.set_device(local)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        runtime.main(plan, torch.device("cuda", local), rank, world)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
