"""Run a hash-bound BERT profile or training package with eight torchrun ranks.

Run: torchrun --standalone --nproc_per_node=8 THIS_FILE --bundle ABS
The persistent pool uses an identical copy named launch_v5_direct.py.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    launch = json.loads((bundle / "direct_launch.json").read_text())
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(bundle / "plan.json") != launch["plan_sha256"]:
        raise ValueError("Launch plan changed")
    plan = json.loads((bundle / "plan.json").read_text())
    source = bundle / "source"
    if Path(__file__).resolve() != source / "scripts/research/launch_v5_direct.py":
        raise ValueError("Execute the frozen launcher")
    for file, expected in plan["sources"].items():
        if digest(file) != expected:
            raise ValueError("Frozen code changed: " + file)
    for file, expected in plan.get("vendor_files", {}).items():
        if digest(file) != expected:
            raise ValueError("Isolated W&B dependency changed: " + file)
    sys.path.insert(0, str(source))
    if plan.get("wandb_vendor"):
        sys.path.insert(0, plan["wandb_vendor"])
    os.environ.update(USE_TF="0", IMMUNE_SDPA_BACKEND="efficient", OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    import torch
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 8 or torch.cuda.device_count() != 8:
        raise ValueError("This recipe requires one node with eight GPUs")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        if plan["phase"] == "profile":
            from scripts.research.profile_bert_mlm import main as run
        elif plan["phase"] in ("smoke", "train"):
            from scripts.research.run_bert_mlm import main as run
        else:
            raise ValueError("Unknown BERT launch phase")
        run(plan, device, rank, world)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
