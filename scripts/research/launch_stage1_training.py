"""Verify and execute an immutable Stage 1 launch package.

Activate pllm; run frozen THIS_FILE --bundle ABS --verify-only for CPU preflight.
GPU execution uses torchrun --standalone --nproc_per_node=8 and the same arguments.
Use --resume latest to restore complete state within the identical frozen run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bound_json(path, digest):
    if sha(path) != digest:
        raise ValueError("Bound file changed: " + str(path))
    return json.loads(Path(path).read_text())


def verify_data(plan):
    index = bound_json(plan["data_index"], plan["data_index_sha256"])
    binding = bound_json(index["binding"], index["binding_sha256"])
    bound_json(index["manifest"], index["manifest_sha256"])
    if binding["model"] != plan["model"] or binding["manifest_sha256"] != index["manifest_sha256"]:
        raise ValueError("Data/model binding differs")
    if bound_json(binding["config_path"], binding["config_sha256"]) != plan["model"]:
        raise ValueError("Data layout changed")
    layout = Path(plan["source_root"])/"dllm/pipelines/immune_llada/fixed_chain/layout.py"
    if sha(layout) != binding["layout_sha256"]:
        raise ValueError("Frozen collator differs from accepted data layout")
    for path, digest in binding["evidence"].items():
        bound_json(path, digest)
    expected = {s["path"]: s for s in binding["shards"]}
    if set(expected) != {s["path"] for s in index["shards"]}:
        raise ValueError("Shard set differs")
    def check(shard):
        reference = expected[shard["path"]]
        if (any(shard[k] != reference[k] for k in ("sha256", "resolved_path", "records", "source", "split"))
                or str(Path(shard["path"]).resolve()) != shard["resolved_path"]
                or sha(shard["path"]) != shard["sha256"] or sha(shard["index"]) != shard["index_sha256"]):
            raise ValueError("Shard/offset binding changed: " + shard["path"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check, index["shards"]))
    for reference in index["files"].values():
        if sha(reference["path"]) != reference["sha256"]:
            raise ValueError("Validation reference changed")
    for split, count in index["records"].items():
        if sum(s["records"] for s in index["shards"] if s["split"] == split) != count:
            raise ValueError("Corpus counts differ")
    return dict(status="passed", shards=len(index["shards"]), records=index["records"])


def readiness(plan, manifest):
    """Return explicit missing qualifications; never silently downgrade formal runs."""
    failures = []
    if plan["phase"] == "online":
        if (plan.get("teacher_quality_policy") != "exploratory_unvalidated"
                or plan.get("cache_coverage_policy") != "generate_every_eligible_miss_before_update"
                or plan.get("online_teacher", {}).get("mode") != "on_demand_write_through"):
            failures.append("online teacher: missing explicit exploratory/generation policy")
        if manifest["counts"].get("valid/cached", 0) < plan["minimum_valid_caches"]:
            failures.append("teacher validation: insufficient fixed held-out cached records")
        # Joint GPU profile + actual online DDP replay are performed by the
        # online driver before it enters training. Initial train coverage is
        # irrelevant because every eligible cache miss must be resolved.
        return failures
    for phase in (("profile",) if plan["phase"] == "smoke" else ("profile", "smoke") if plan["phase"] == "train" else ()):
        path = Path(plan["gates"][phase])
        if not path.is_file():
            failures.append(f"{phase}: missing {path}")
            continue
        if not plan.get("gate_hashes", {}).get(phase):
            failures.append(f"{phase}: receipt was not available when this package was frozen; rebuild the package")
        elif sha(path) != plan["gate_hashes"][phase]:
            failures.append(f"{phase}: bound receipt changed")
        evidence = json.loads(path.read_text())
        if evidence.get("status") != "passed" or evidence.get("qualification") != plan["qualification"] or evidence.get("world_size") != plan["world_size"]:
            failures.append(f"{phase}: receipt incompatible with frozen model/runtime/teacher")
        if phase == "profile" and plan["phase"] == "train" and evidence.get("micro_batch") != plan["training"]["micro_batch"]:
            failures.append("profile: selected microbatch must be frozen into a new training package")
    if plan["phase"] == "train":
        quality = plan["quality"]
        if not quality:
            failures.append("teacher quality: no cache-bound evaluation receipt")
        else:
            report = bound_json(quality["path"], quality["sha256"])
            if (report.get("schema") != "immune.stage1.teacher_quality.v1" or report.get("status") != "passed"
                    or report.get("cache_manifest_sha256") != plan["cache_manifest_sha256"]
                    or not report.get("evaluation_report") or not report.get("evaluation_report_sha256")):
                failures.append("teacher quality: incompatible evaluation receipt")
            elif sha(report["evaluation_report"]) != report["evaluation_report_sha256"]:
                failures.append("teacher quality: evaluation report changed")
        if manifest["coverage"]["train"] < plan["minimum_train_cache_coverage"]:
            failures.append("teacher coverage: training coverage below frozen minimum")
        if manifest["counts"].get("valid/cached", 0) < plan["minimum_valid_caches"]:
            failures.append("teacher validation: insufficient held-out cached records")
    return failures


def verify(bundle, *, full_data=True):
    launch = json.loads((bundle/"launch.json").read_text())
    plan = bound_json(bundle/"plan.json", launch["plan_sha256"])
    source = Path(plan["source_root"])
    if Path(__file__).resolve() != source/"scripts/research/launch_stage1_training.py":
        raise ValueError("Execute the frozen launcher")
    for path, digest in plan["sources"].items():
        if sha(path) != digest:
            raise ValueError("Frozen source changed: " + path)
    parent = Path(plan["parent_checkpoint"])
    receipt = bound_json(parent/"checkpoint_receipt.json", plan["parent_receipt_sha256"])
    if receipt["status"] != "complete" or receipt["step"] != plan["parent_step"] or len(set(receipt["rank_parameter_sha256"])) != 1:
        raise ValueError("Parent checkpoint not complete")
    for name in ("model.safetensors", "research_config.json", "immune_config.json", "vocabulary.json", "state.json", "run_provenance.json"):
        if sha(parent/name) != receipt["files"][name]:
            raise ValueError("Parent checkpoint changed")
    if receipt["files"]["model.safetensors"] != plan["parent_model_sha256"]:
        raise ValueError("Wrong BERT parent")
    manifest = bound_json(plan["cache_manifest"], plan["cache_manifest_sha256"])
    if (manifest["schema"] != "immune.stage1.cache_index.v1"
            or manifest["data_index_sha256"] != plan["data_index_sha256"]
            or manifest["teacher_contract"] != plan["teacher_contract"]
            or manifest["max_pairs_per_record"] > plan["max_pairs_per_record"]
            or sha(manifest["database"]) != manifest["database_sha256"]):
        raise ValueError("Teacher cache index differs")
    gate = verify_data(plan) if full_data else None
    if plan["phase"] == "online":
        online = plan["online_teacher"]
        weights = Path(online["model_dir"])
        contract = plan["teacher_contract"]
        references = {"config.json": contract["config_sha256"], **contract["weight_sha256"]}
        def check_teacher(item):
            name, digest = item
            if sha(weights/name) != digest:
                raise ValueError("Online teacher weight/config changed: " + name)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(check_teacher, references.items()))
        if not Path(online["python"]).is_file():
            raise FileNotFoundError(online["python"])
        worker = source/"scripts/research/stage1_teacher_worker.py"
        runtime = subprocess.run([online["python"], "-I", "-B", str(worker), "--plan", str(bundle/"plan.json"),
            "--verify-runtime"], capture_output=True, text=True, timeout=120)
        if runtime.returncode:
            raise ValueError("Online teacher runtime preflight failed: " + runtime.stderr[-4000:])
        runtime_report = json.loads(runtime.stdout)
        if runtime_report.get("status") != "passed":
            raise ValueError("Online teacher runtime preflight did not pass")
        plan["online_launch_sha256"] = launch["plan_sha256"]
    report = dict(data=gate, cache_counts=manifest["counts"], cache_coverage=manifest["coverage"],
                  blockers=readiness(plan, manifest))
    if plan["phase"] == "online":
        report.update(teacher_runtime=runtime_report, teacher_quality_policy=plan["teacher_quality_policy"],
            runtime_checks=["joint_teacher_student_memory_profile", "eight_rank_online_cache_and_resume_smoke"])
    return plan, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", help="complete checkpoint directory or latest")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if args.verify_only:
        _, report = verify(bundle)
        report["status"] = "blocked" if report["blockers"] else "passed"
        (bundle/"preflight.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
        if report["blockers"]:
            raise SystemExit(2)
        return
    source = bundle/"source"
    sys.path.insert(0, str(source))
    os.environ.update(USE_TF="0", IMMUNE_SDPA_BACKEND="efficient", OMP_NUM_THREADS="1",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import torch
    from scripts.research.run_bert_mlm import rank_zero
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != 8 or torch.cuda.device_count() != 8:
        raise ValueError("Stage 1 requires one node with eight GPUs")
    torch.cuda.set_device(local)
    # Cache misses can contain long chains; one rank may wait while peers run
    # teacher inference. Keep RPC failure handling separate from DDP timeout.
    phase = json.loads((bundle/"plan.json").read_text())["phase"]
    torch.distributed.init_process_group("nccl", timeout=timedelta(hours=6) if phase == "online" else timedelta(minutes=30))
    lock = None
    try:
        def exclusive():
            nonlocal lock
            lock = (bundle/"run.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rank_zero(exclusive, rank)
        plan, report = rank_zero(lambda: verify(bundle), rank)
        if report["blockers"]:
            raise ValueError("Stage 1 is not qualified: " + "; ".join(report["blockers"]))
        if world != plan["world_size"]:
            raise ValueError("Frozen world size differs")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        device = torch.device("cuda", local)
        from scripts.research.run_stage1_training import main as train, profile
        resume = args.resume
        if resume == "latest" and plan["phase"] != "online":
            resume = rank_zero(lambda: json.loads((Path(plan["output_dir"])/"latest.json").read_text())["checkpoint"], rank)
        if plan["phase"] == "online":
            from scripts.research.run_stage1_online import main as online_train
            online_train(plan, device, rank, world, resume=resume)
        elif plan["phase"] == "profile":
            if resume:
                raise ValueError("Profile does not resume a training checkpoint")
            profile(plan, device, rank, world)
        else:
            train(plan, device, rank, world, resume=resume)
        torch.distributed.barrier()
    finally:
        if lock is not None:
            lock.close()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
