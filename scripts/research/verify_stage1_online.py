"""Verify the real online teacher and one real BERT+KD update on a local GPU.

Activate pllm; USE_TF=0 IMMUNE_SDPA_BACKEND=efficient python -B -m
scripts.research.verify_stage1_online --bundle ABS_BUNDLE --output ABS_NEW_DIR.
This bounded single-GPU check is not an eight-rank memory/resume qualification.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import time

import torch

from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.run_stage1_distillation import make_optimizer, optimizer_step, prepare_microbatch
from scripts.research.run_stage1_training import new_model
from scripts.research.stage1_corpus import CachedCorpus
from scripts.research.stage1_online_teacher import OnlineTeacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    device = torch.device("cuda", 0)
    plan = json.loads((args.bundle / "plan.json").read_text())
    plan["online_launch_sha256"] = sha(args.bundle / "plan.json")
    manifest = json.loads(Path(plan["cache_manifest"]).read_text())
    with sqlite3.connect(manifest["database"]) as connection:
        row = connection.execute("SELECT row_index FROM targets WHERE split='train' ORDER BY row_index LIMIT 1").fetchone()[0]
    sample = CachedCorpus(plan["data_index"], manifest)[row]
    started = time.monotonic()
    report = dict(scope="local_real_weight_integration_only", world_size=1, gpu=torch.cuda.get_device_name(device),
        frozen_plan_sha256=plan["online_launch_sha256"], sample_row=row, submitted=False)
    with OnlineTeacher(plan, device, 0, args.output / "teacher_logs") as online:
        model = new_model(plan, device)
        optimizer = make_optimizer(model, plan["training"]["backbone_lr"], plan["training"]["pair_lr"],
                                   plan["training"]["weight_decay"])
        rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
        unresolved = [(sample[0], sample[1], None)]
        first, counts = online.resolve(unresolved, model.config)
        second, repeated = online.resolve(unresolved, model.config)
        for key, value in sample[2]["tensors"].items():
            assert torch.equal(value, first[0][2]["tensors"][key]), key
            assert torch.equal(value, second[0][2]["tensors"][key]), key
        assert torch.equal(rng[0], torch.get_rng_state())
        assert torch.equal(rng[1], torch.cuda.get_rng_state(device))
        assert repeated["generated"] == 0 and repeated["online_hits"] == 1
        noise = torch.Generator().manual_seed(52)
        micro = prepare_microbatch(model, [sample[1]], [first[0][2]], noise, device)
        metrics = optimizer_step(model, optimizer, [micro], precision="bf16", intra_weight=.05, inter_weight=.05)
        assert all(v > 0 for v in metrics["bias_gradient_norms"])
        assert metrics["backbone_gradient_norm"] > 0 and metrics["pair_gradient_norm"] > 0
        report.update(status="passed", first=counts, repeated=repeated, pilot_target_tensors_exact=True,
            student_rng_unchanged=True, student_update=metrics, elapsed_seconds=time.monotonic()-started,
            utc=datetime.now(timezone.utc).isoformat(), teacher_cache_root=plan["online_teacher"]["cache_root"])
    report["worker_closed"] = online.process is None
    write_json(args.output / "verification.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
