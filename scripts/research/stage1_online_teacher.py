"""Student-side online cache lookup and an isolated persistent teacher process.

Activate pllm; run python -B -m pytest scripts/tests/immune_llada/test_stage1_online.py -q.
Only the training rank calls this client; DataLoader workers never own a GPU teacher.
"""
from collections import Counter
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

import torch

from scripts.research.stage1_data import load_teacher_cache, make_teacher_request
from scripts.research.stage1_online_cache import cache_path


class OnlineTeacher:
    def __init__(self, plan, device, rank, log_dir):
        self.plan, self.device, self.rank = plan, device, rank
        self.process = None
        self.log = None
        self.stats = Counter()
        self.last_progress = time.monotonic()
        self.log_dir = Path(log_dir)

    def start(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        plan_path = self.log_dir / f"worker_plan_rank{self.rank}.json"
        plan_path.write_text(json.dumps(self.plan, indent=2) + "\n")
        self.log = (self.log_dir / f"teacher_rank{self.rank}.log").open("a")
        script = Path(self.plan["source_root"]) / "scripts/research/stage1_teacher_worker.py"
        command = [self.plan["online_teacher"]["python"], "-I", "-B", str(script),
            "--plan", str(plan_path), "--device", str(self.device), "--parent-pid", str(os.getpid())]
        environment = dict(os.environ)
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(name, None)
        environment.update(USE_TF="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, text=True, bufsize=1, env=environment)
            result = self.receive("load")
            if result.get("status") != "ready" or result.get("contract") != self.plan["teacher_contract"]:
                raise ValueError("Online teacher handshake differs")
        except BaseException:
            self.close()
            raise
        return self

    def progress(self, operation):
        now = time.monotonic()
        if self.rank == 0 and now - self.last_progress >= 60:
            print(json.dumps(dict(event="online_teacher_progress", operation=operation,
                rank=self.rank, **self.stats)), flush=True)
            self.last_progress = now

    def receive(self, operation):
        deadline = time.monotonic() + self.plan["online_teacher"]["request_timeout_seconds"]
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.close()
                    raise TimeoutError(f"Teacher {operation} timed out; see {self.log_dir}")
                if selector.select(min(10, remaining)):
                    line = self.process.stdout.readline()
                    if not line:
                        raise RuntimeError(f"Teacher exited during {operation}; see {self.log_dir}")
                    result = json.loads(line)
                    if result.get("status") == "error":
                        if result.get("kind") == "cuda_oom":
                            raise torch.cuda.OutOfMemoryError(result["message"])
                        raise RuntimeError(f"Teacher {operation} failed: {result.get('message')}; see {self.log_dir}")
                    return result
                self.progress(operation)

    def call(self, message):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Online teacher is not running")
        self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.process.stdin.flush()
        return self.receive(message["op"])

    def load(self, directory, request):
        cache = load_teacher_cache(directory, request, allow_unvalidated=True)
        if cache["metadata"]["teacher_contract"] != self.plan["teacher_contract"]:
            raise ValueError("Online cache teacher contract differs")
        if len(cache["tensors"]["teacher_pair_indices"]) > self.plan["max_pairs_per_record"]:
            raise ValueError("Online cache exceeds the profiled pair bound")
        return cache

    def resolve(self, rows, config):
        """Complete every eligible cache miss before constructing corrupted inputs."""
        completed = []
        before = self.stats.copy()
        for row, record, cache in rows:
            if cache is not None:
                if cache["metadata"]["teacher_contract"] != self.plan["teacher_contract"]:
                    raise ValueError("Seed-cache teacher contract differs")
                self.stats["seed_hits"] += 1
            else:
                request = make_teacher_request(record, config)
                if not request["chains"]:
                    self.stats["ineligible"] += 1
                else:
                    destination = cache_path(self.plan["online_teacher"]["cache_root"], request["request_sha256"])
                    if destination.exists():
                        cache = self.load(destination, request)
                        self.stats["online_hits"] += 1
                    else:
                        response = self.call(dict(op="ensure", request=request))
                        if response.get("status") not in {"generated", "reused"} or Path(response["directory"]) != destination:
                            raise ValueError("Unexpected teacher cache response")
                        cache = self.load(destination, request)
                        self.stats["generated" if response["status"] == "generated" else "online_hits"] += 1
                        self.stats["teacher_seconds"] += response["seconds"]
            completed.append((row, record, cache))
            self.progress("resolve_batch")
        return completed, {k: self.stats[k] - before[k] for k in
            ("seed_hits", "online_hits", "generated", "ineligible", "teacher_seconds")}

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            try:
                if process.poll() is None:
                    process.stdin.write('{"op":"close"}\n')
                    process.stdin.flush()
                    process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            finally:
                process.stdin.close()
                process.stdout.close()
        if self.log is not None:
            self.log.close()
            self.log = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
