"""Run student smoke, teacher pilot, indexing and cache audit in one GPU job.

Activate pllm; use the generated entrypoint.sh or
python -B THIS_FILE --bundle ABS [--verify-only]. GPU phases run sequentially.
Completed smoke is verified and reused; teacher resumes verified caches.
Engineering completion does not automatically approve or start formal training.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def completed_smoke(plan):
    smoke_plan = json.loads(Path(plan["smoke_plan"]).read_text())
    output = Path(smoke_plan["output_dir"])
    receipt_path = output/"COMPLETE.json"
    if not receipt_path.exists():
        return False
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "passed" or receipt.get("world_size") != 8
            or receipt.get("qualification") != smoke_plan["qualification"]):
        raise ValueError("Completed smoke belongs to another configuration or did not pass")
    for rank in range(8):
        report = json.loads((output/f"smoke_rank{rank}.json").read_text())
        if (report.get("status") != "passed" or report.get("corruption_exact") is not True
                or report.get("kd_present") != (rank % 2 == 0)):
            raise ValueError("Smoke rank report invalid")
    return True


class ConsoleSummary:
    """Aggregate known routine output; pass through diagnostics and unknown lines."""

    def __init__(self, interval=60., *, emit=print, clock=time.monotonic):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Log interval must be positive and finite")
        self.interval, self.emit, self.clock = interval, emit, clock
        self.last = clock()
        self.counts = dict(cached_records=0, pairs=0, hash_checks=0, teachers_loaded=0)
        self.loading = 0
        self.metrics = None
        self.dirty = False

    def send(self, value):
        self.emit(json.dumps(value, ensure_ascii=False))

    def flush(self, *, final=False):
        now = self.clock()
        if self.dirty and (final or now-self.last >= self.interval):
            self.send(dict(event="progress_summary", final=final, **self.counts,
                           **({"training": self.metrics} if self.metrics else {})))
            self.last, self.dirty = now, False

    def feed(self, line):
        text = line.strip()
        if not text:
            return
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        if isinstance(value, dict) and (value.get("error") or value.get("status") in {"failed", "error"}):
            self.emit(text)
        elif isinstance(value, dict) and value.get("status") == "cached" and "request" in value:
            self.counts["cached_records"] += 1
            self.counts["pairs"] += value.get("pairs", 0)
            self.dirty = True
        elif isinstance(value, dict) and value.get("event") == "loading_teacher":
            self.loading += 1
            if self.loading == 1:
                self.send(dict(event="loading_teachers", revision=value.get("revision")))
        elif isinstance(value, dict) and value.get("event") == "teacher_loaded":
            self.counts["teachers_loaded"] += 1
            self.dirty = True
        elif isinstance(value, dict) and "step" in value and "loss" in value:
            self.metrics = {k: value[k] for k in ("step", "loss", "mlm", "intra_kl", "inter_kl",
                "kd_coverage", "backbone_lr", "pair_lr", "grad_norm", "seconds") if k in value}
            self.dirty = True
        elif isinstance(value, dict) and value.get("schema") == "immune.stage1.cache_audit.v1":
            self.send({k: value[k] for k in ("status", "cached_records", "coverage", "repeat_max_abs",
                "symmetry_max_abs", "formal_training_ready", "formal_training_blockers") if k in value})
        elif text.endswith(": OK") and text.startswith("/"):
            self.counts["hash_checks"] += 1
            self.dirty = True
        elif text.startswith(("Loading weights:", "Loading checkpoint shards:")) and "%|" in text:
            # Known progress bars only. Tracebacks and unexpected output stay visible.
            if any(word in text.lower() for word in ("error", "failed", "warning")):
                self.emit(text)
        else:
            self.emit(text)
        self.flush()


def execute(command):
    command = list(map(str, command))
    interval = float(os.environ.get("STAGE1_LOG_INTERVAL_SECONDS", "60"))
    summary = ConsoleSummary(interval, emit=lambda text: print(text, flush=True))
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace", bufsize=1) as process:
        try:
            for line in process.stdout:
                summary.feed(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            summary.flush(final=True)
    if code:
        raise subprocess.CalledProcessError(code, command)


def sequence(plan, status):
    status("student_smoke")
    if not completed_smoke(plan):
        execute(["bash", plan["smoke_entrypoint"]])
    if not completed_smoke(plan):
        raise ValueError("Student smoke exited without a valid completion receipt")
    status("teacher_pilot")
    # The original teacher cache entrypoint already uses --resume-cache.
    execute(["bash", plan["teacher_entrypoint"], "--verify-repeat"])
    status("cache_index")
    index_dir = Path(plan["cache_index_dir"])
    manifest = index_dir/"manifest.json"
    if not index_dir.exists():
        staging = index_dir.with_name("."+index_dir.name+".partial."+uuid.uuid4().hex)
        execute([plan["python"], "-I", "-B", "-c",
            "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('scripts.research.stage1_corpus',run_name='__main__')",
            plan["source_root"], "index", "--data-index", plan["data_index"],
            "--requests", plan["requests"], "--cache", plan["cache_root"], "--output", staging])
        staged = staging/"manifest.json"
        value = json.loads(staged.read_text())
        value["database"] = str(index_dir/"cache.sqlite")
        write_json(staged, value)
        staging.rename(index_dir)
    if not manifest.is_file():
        raise ValueError("Existing cache index is incomplete: " + str(index_dir))
    indexed = json.loads(manifest.read_text())
    if (indexed["data_index"] != plan["data_index"] or indexed["requests"] != plan["requests"]
            or indexed["data_index_sha256"] != sha(plan["data_index"])
            or indexed["requests_sha256"] != sha(plan["requests"])):
        raise ValueError("Existing index belongs to another corpus/request set")
    status("cache_audit")
    execute([plan["python"], "-I", "-B", plan["audit_script"], "--source", plan["source_root"],
        "--plan", plan["plan_path"], "--manifest", manifest, "--output", plan["audit_output"]])
    report = json.loads(Path(plan["audit_output"]).read_text())
    if report.get("status") != "passed" or report.get("cache_manifest_sha256") != sha(manifest):
        raise ValueError("Missing, stale or failed cache audit")
    return dict(status="checks_complete", engineering_status="passed", smoke="passed",
        audit=str(plan["audit_output"]), audit_sha256=sha(plan["audit_output"]),
        cache_manifest=str(manifest), cache_manifest_sha256=sha(manifest),
        structural_accuracy=report["structural_accuracy"], formal_training_ready=False,
        formal_training_blockers=report["formal_training_blockers"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    launch = json.loads((bundle/"launch.json").read_text())
    if sha(bundle/"plan.json") != launch["plan_sha256"]:
        raise ValueError("Combined check plan changed")
    plan = json.loads((bundle/"plan.json").read_text())
    os.environ["STAGE1_LOG_INTERVAL_SECONDS"] = str(plan.get("log_interval_seconds", 60))
    if Path(__file__).resolve() != Path(plan["runner"]):
        raise ValueError("Use the frozen check runner")
    for path, digest in plan["references"].items():
        if sha(path) != digest:
            raise ValueError("Combined check input changed: " + path)
    with (bundle/"run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(schema="immune.stage1.combined_checks.v1", plan_sha256=launch["plan_sha256"])
        def status(phase, **values):
            state.update(phase=phase, utc=datetime.now(timezone.utc).isoformat(), **values)
            write_json(bundle/("preflight.json" if args.verify_only else "status.json"), state)
            print(json.dumps(state), flush=True)
        try:
            status("preflight", status="running")
            execute(["bash", plan["smoke_entrypoint"], "--verify-only"])
            execute(["bash", plan["teacher_entrypoint"], "--verify-only"])
            if args.verify_only:
                status("preflight", status="passed", gpu_started=False)
                return
            result = sequence(plan, status)
            status("complete", **result)
            write_json(bundle/"COMPLETE.json", state)
        except Exception as error:
            status(state["phase"], status="failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
