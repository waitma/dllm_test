"""Freeze a new diffusion stage initialized only from the retained BERT287000.

Activate pllm; USE_TF=0 python -B -m scripts.research.prepare_diffusion_bert_restart
--reference-plan ABS --audit ABS --checks ABS --output ABS. Does not launch a job.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil

from scripts.research.bert_mlm_data import receptor_key, sha, write_json
from scripts.research.diffusion_continuation_data import source_lengths
from scripts.research.diffusion_source_sampling import POLICY, source_quotas
from scripts.research.diffusion_cdr_tasks import default_cdr_tasks

ROOT = Path(__file__).resolve().parents[2]
OVERLAYS = ["run_diffusion_continuation.py", "diffusion_continuation_data.py",
            "diffusion_source_sampling.py", "diffusion_training_guard.py",
            "prepare_diffusion_bert_restart.py", "launch_diffusion_bert_restart.py",
            "verify_diffusion_bert_restart.py", "diffusion_continuation_objective.py",
            "diffusion_continuation_eval.py", "diffusion_continuation_losses.py",
            "diffusion_cdr_tasks.py", "diffusion_cdr_eval.py"]


def prepare(reference_plan, audit, checks, output, cdr_tasks=False, cdr_audit=None):
    for path in (reference_plan, audit, checks, output):
        if not path.is_absolute():
            raise ValueError("Use absolute paths")
    previous = json.loads(reference_plan.read_text())
    report, acceptance = json.loads(audit.read_text()), json.loads(checks.read_text())
    if report["status"] != "complete" or acceptance["status"] != "passed":
        raise ValueError("Incomplete data audit or CPU checks")
    if report["index_sha256"] != previous["data_index_sha256"]:
        raise ValueError("Audit checked different data")
    if cdr_tasks:
        if cdr_audit is None or not cdr_audit.is_absolute():
            raise ValueError("CDR training requires an absolute CDR coverage audit")
        coverage = json.loads(cdr_audit.read_text())
        if coverage["status"] != "passed" or coverage["index_sha256"] != report["index_sha256"]:
            raise ValueError("CDR coverage audit differs from training data")
        if coverage["annotation_source_sha256"] != sha(ROOT / "scripts/research/diffusion_cdr_tasks.py"):
            raise ValueError("CDR annotation logic changed after coverage audit")
    for name in OVERLAYS:
        if acceptance["sources"].get("scripts/research/" + name) != sha(ROOT / "scripts/research" / name):
            raise ValueError("Unchecked restart source: " + name)
    if any(any(k.startswith("error_") and v for k, v in row["counts"].items())
           for row in report["results"]):
        raise ValueError("Data audit found invalid records")
    parent = Path(previous["parent_checkpoint"])
    receipt = json.loads((parent / "checkpoint_receipt.json").read_text())
    if receipt["status"] != "complete" or receipt["step"] != 287000:
        raise ValueError("Expected the original BERT287000 checkpoint")
    for name in ("model.safetensors", "research_config.json", "vocabulary.json", "immune_config.json"):
        if sha(parent / name) != receipt["files"][name]:
            raise ValueError("Parent checkpoint changed: " + name)
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / "launch"
    source = bundle / "source"
    for name, expected in previous["sources"].items():
        if sha(name) != expected:
            raise ValueError("Reference frozen source changed: " + name)
        target = source / Path(name).relative_to(previous["source_root"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, target)
    for name in OVERLAYS:
        shutil.copy2(ROOT / "scripts/research" / name, source / "scripts/research" / name)
    for name in ("test_diffusion_source_sampling.py", "test_diffusion_continuation_data.py",
                 "test_diffusion_training_guard.py", "test_diffusion_continuation_update.py",
                 "test_diffusion_cdr_tasks.py", "test_diffusion_continuation_objective.py"):
        target = source / "scripts/tests/immune_llada" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts/tests/immune_llada" / name, target)
    # Retain all train bytes. Remove only exposed examples from the new validation
    # file, preserve the historical 889-example file and old experiment identity.
    index = json.loads(Path(previous["data_index"]).read_text())
    old_valid = index["files"]["token_valid"]
    rows = [json.loads(line) for line in Path(old_valid["path"]).read_text().splitlines()]
    held = {receptor_key(row["record"]) for row in rows}
    exposed = set()
    for row in report["results"]:
        if row["split"] == "train":
            with open(row["receptor_keys"]) as stream:
                exposed.update(line.strip() for line in stream if line.strip() in held)
    kept = [row for row in rows if receptor_key(row["record"]) not in exposed]
    valid = bundle / "token_valid_disjoint.jsonl"
    valid.write_text("".join(json.dumps(row) + "\n" for row in kept))
    index["files"]["token_valid"] = dict(path=str(valid), sha256=sha(valid), records=len(kept))
    index["diffusion_validation_filter"] = dict(original=old_valid,
        excluded_records=len(rows)-len(kept), reason="Exact observed receptor identity occurs in training",
        original_index_sha256=previous["data_index_sha256"], audit=str(audit), audit_sha256=sha(audit))
    write_json(bundle / "index.json", index)
    sampling = dict(alpha=.75, amplification_cap=4., period=65536)
    lengths = source_lengths(index)
    quotas = source_quotas(list(lengths.values()), **sampling)
    training = {**previous["training"], "max_steps": 200000,
        "scheduler": "warmup_cosine_0.01_to_1_to_0.1", "optimizer_reset": True,
        "sampler": POLICY, "source_sampling": sampling}
    plan = {**previous, "phase": "train", "stage": "fresh_diffusion_from_bert287000",
        "source_root": str(source), "sources": {str(p): sha(p) for p in source.rglob("*.py")},
        "training": training, "output_dir": str(output / "train"), "smoke_output": str(output / "smoke"),
        "data_index": str(bundle / "index.json"), "data_index_sha256": sha(bundle / "index.json"),
        "gates": [dict(path=str(p), sha256=sha(p)) for p in (audit, checks)],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile_receipt": str(output / "max_shape_profile.json")}
    for name in ("repair", "extension", "resume_checkpoint", "resume_contract"):
        plan.pop(name, None)
    if cdr_tasks:
        plan["recipe"] = {**previous["recipe"], "cdr_tasks": default_cdr_tasks()}
        plan["stage"] = "fresh_diffusion_from_bert287000_fixed_framework_cdr"
        plan["gates"].append(dict(path=str(cdr_audit), sha256=sha(cdr_audit)))
    plan["wandb"] = {**plan["wandb"], "run_id": "diffbert-" + output.name,
                     "name": "BERT287k fresh diffusion capped sources " + output.name}
    write_json(bundle / "plan.json", plan)
    write_json(bundle / "direct_launch.json", dict(plan_sha256=sha(bundle / "plan.json")))
    launch = source / "scripts/research/launch_diffusion_bert_restart.py"
    cmd = f"torchrun --standalone --nproc_per_node=8 {shlex.quote(str(launch))} --bundle {shlex.quote(str(bundle))}"
    (bundle / "train.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
        + f"cd {shlex.quote(str(source))}\n"
        + "export USE_TF=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
        + cmd + " --phase smoke\n" + cmd + " --phase train\n")
    (bundle / "train.sh").chmod(0o755)
    write_json(output / "preparation.json", dict(status="prepared", training_started=False,
        parent_checkpoint=str(parent), parent_step=287000, diffusion_start_step=0,
        optimizer_restored=False, data_records=index["records"], valid_records=len(kept),
        old_valid_records=len(rows), excluded_validation_records=len(rows)-len(kept),
        source_probabilities={s: q / sum(quotas) for s, q in zip(lengths, quotas)},
        source_lengths=lengths, eight_rank_qualification="pending"))
    print(json.dumps(dict(bundle=str(bundle), diffusion_start_step=0, validation_records=len(kept))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-plan", "audit", "checks", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cdr-tasks", action="store_true")
    parser.add_argument("--cdr-audit", type=Path)
    args = parser.parse_args()
    prepare(args.reference_plan, args.audit, args.checks, args.output, args.cdr_tasks, args.cdr_audit)
