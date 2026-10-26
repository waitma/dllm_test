"""Execute a frozen conditional diffusion/edit package on the retained 8-GPU pool.

Activate pllm; torchrun --nproc_per_node=8 FROZEN_LAUNCHER --bundle ABS.
The pool's transport copy is named source/scripts/research/launch_v5_direct.py.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    args=parser.parse_args()
    bundle=args.bundle.resolve()
    digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    launch=json.loads((bundle/"direct_launch.json").read_text())
    if digest(bundle/"plan.json")!=launch["plan_sha256"]:
        raise ValueError("Frozen plan changed")
    plan=json.loads((bundle/"plan.json").read_text())
    source=bundle/"source"
    if Path(__file__).resolve()!=source/"scripts/research/launch_v5_direct.py":
        raise ValueError("Execute the frozen transport launcher")
    for key in ("sources","vendor_files"):
        for path,expected in plan.get(key,{}).items():
            if digest(path)!=expected:
                raise ValueError("Frozen asset changed: "+path)
    for item in plan.get("gates",[]):
        if digest(item["path"])!=item["sha256"]:
            raise ValueError("Launch gate changed")
    sys.path.insert(0,str(source))
    if plan.get("wandb_vendor"):
        sys.path.insert(0,plan["wandb_vendor"])
    os.environ.update(USE_TF="0",IMMUNE_SDPA_BACKEND="efficient",OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    import torch
    rank,world,local=[int(os.environ[key]) for key in ("RANK","WORLD_SIZE","LOCAL_RANK")]
    if world!=8 or torch.cuda.device_count()!=8:
        raise ValueError("Eight allocated GPUs required")
    torch.cuda.set_device(local)
    torch.distributed.init_process_group("nccl",timeout=timedelta(minutes=30))
    try:
        from scripts.research.run_diffusion_continuation import main as run
        run(plan,torch.device("cuda",local),rank,world)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


if __name__=="__main__":
    main()
