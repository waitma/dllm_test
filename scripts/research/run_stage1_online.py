"""Run joint GPU qualification, online replay smoke, then full-corpus pretraining.

Activate pllm; submit the frozen --phase online entrypoint.sh on eight A100 80GB.
Completed qualifications are reused; a complete checkpoint resumes automatically.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path
import time

import torch

from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.run_bert_mlm import rank_zero
from scripts.research.run_stage1_training import main as train, profile
from scripts.research.stage1_online_teacher import OnlineTeacher


def qualified_receipt(path, plan, phase):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (value.get("status") != "passed" or value.get("qualification") != plan["qualification"]
            or value.get("world_size") != plan["world_size"]):
        raise ValueError(f"Existing online {phase} receipt is incompatible: {path}")
    if phase == "profile" and (value.get("scope") != "online_teacher_and_student"
            or not isinstance(value.get("micro_batch"), int) or value["micro_batch"] < 1):
        raise ValueError("Online training requires a joint teacher/student profile")
    if phase == "smoke":
        for rank in range(plan["world_size"]):
            detail = json.loads((path.parent / f"smoke_rank{rank}.json").read_text())
            online = detail.get("online_cache") or {}
            if (detail.get("status") != "passed" or not detail.get("corruption_exact")
                    or not online.get("pilot_targets_exact") or not online.get("student_rng_unchanged")):
                raise ValueError("Online replay smoke did not qualify all ranks")
    return value


def archive_incomplete(path):
    if path.exists():
        path.rename(path.with_name(path.name + f".abandoned-{time.time_ns()}"))


def completed_training(root, plan):
    path = root / "train/COMPLETE.json"
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    provenance = json.loads((root / "train/run_provenance.json").read_text())
    if (receipt.get("status") != "success" or receipt.get("step") != plan["training"]["max_steps"]
            or receipt.get("world_size") != plan["world_size"]
            or provenance.get("online_launch_sha256") != plan["online_launch_sha256"]):
        raise ValueError("Existing online training completion belongs to another run")
    return True


def main(plan, device, rank, world, resume=None):
    root = Path(plan["output_dir"])
    rank_zero(lambda: root.mkdir(parents=True, exist_ok=True), rank)
    if rank_zero(lambda: completed_training(root, plan), rank):
        if rank == 0:
            print(json.dumps(dict(event="stage1_already_complete", output=str(root / "train"))), flush=True)
        return
    with OnlineTeacher(plan, device, rank, root / "teacher_logs") as online:
        for phase in ("profile", "smoke"):
            receipt_path = root / phase / "COMPLETE.json"
            receipt = rank_zero(lambda: qualified_receipt(receipt_path, plan, phase), rank)
            if receipt is None:
                rank_zero(lambda: archive_incomplete(root / phase), rank)
                stage = deepcopy(plan)
                stage.update(phase=phase, output_dir=str(root / phase))
                if phase == "profile":
                    profile(stage, device, rank, world, online=online)
                else:
                    stage["training"].update(max_steps=3, global_batch=8, micro_batch=1, warmup_steps=1, loader_workers=0)
                    train(stage, device, rank, world, online=online)
                gc.collect(); torch.cuda.empty_cache()
                torch.distributed.barrier()
                receipt = rank_zero(lambda: qualified_receipt(receipt_path, plan, phase), rank)
            if rank == 0:
                print(json.dumps(dict(event="stage1_online_qualification", phase=phase, status="passed")), flush=True)
        profile_receipt = rank_zero(lambda: qualified_receipt(root / "profile/COMPLETE.json", plan, "profile"), rank)
        training = deepcopy(plan)
        training.update(phase="train", output_dir=str(root / "train"))
        training["training"]["micro_batch"] = profile_receipt["micro_batch"]
        training["gates"] = {p: str(root / p / "COMPLETE.json") for p in ("profile", "smoke")}
        training["gate_hashes"] = {p: sha(path) for p, path in training["gates"].items()}
        rank_zero(lambda: write_json(root / "training_plan.json", training), rank)

        def checkpoint_to_resume():
            latest = root / "train/latest.json"
            if resume == "latest" or (resume is None and latest.exists()):
                return json.loads(latest.read_text())["checkpoint"]
            if resume is not None:
                return resume
            # Failed before the first committed checkpoint: restart from BERT,
            # retaining the append-only teacher cache and old logs for diagnosis.
            archive_incomplete(root / "train")
            return None
        checkpoint = rank_zero(checkpoint_to_resume, rank)
        if rank == 0:
            print(json.dumps(dict(event="stage1_online_training", global_batch=training["training"]["global_batch"],
                micro_batch=training["training"]["micro_batch"], resume=checkpoint,
                teacher_quality="unvalidated_exploratory", output=training["output_dir"])), flush=True)
        train(training, device, rank, world, resume=checkpoint, online=online)
