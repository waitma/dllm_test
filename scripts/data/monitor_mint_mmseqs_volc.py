#!/usr/bin/env python3
"""Monitor MINT MMseqs Volc job; auto-resubmit on preemptible kill/failure.

Usage:
    python scripts/data/monitor_mint_mmseqs_volc.py
    python scripts/data/monitor_mint_mmseqs_volc.py --interval 60
    nohup python scripts/data/monitor_mint_mmseqs_volc.py >> data/ppi_task_raw/processed/pipeline_logs/mmseqs_volc_monitor_stdout.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
YAML = PROJECT_ROOT / "train_jobs/mint_string_mmseqs_cluster_c1ie.yml"
CLU50 = PROJECT_ROOT / "data/ppi_task_raw/raw/stringdb_mint/clu50.tsv"
LOG_DIR = PROJECT_ROOT / "data/ppi_task_raw/processed/pipeline_logs"
LOG_PATH = LOG_DIR / "mmseqs_volc_monitor.log"
STATE_PATH = LOG_DIR / "mmseqs_volc_monitor_state.json"
PROCESS_MD = PROJECT_ROOT / "PROJECT_PROCESS.md"
VOLC = "/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh"
VOLC_LOG = LOG_DIR / "mmseqs_cluster_volc.log"

TERMINAL_RESUBMIT = {"Killed", "Failed"}
TERMINAL_DONE = {"Success"}
POLL_STATUSES = {"Initialized", "Queue", "Staging", "Running", "Killing", "Success", "Failed", "Killed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"task_id": None, "resubmit_count": 0}


def save_state(state: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def run_volc(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [VOLC, *args],
        text=True,
        capture_output=True,
        check=False,
    )


def volc_get(task_id: str) -> dict | None:
    proc = run_volc(
        [
            "ml_task",
            "get",
            "-i",
            task_id,
            "--format",
            "JobId=JobId,Status=Status,Preemptible=Preemptible,Elapsed=Elapsed",
            "-o",
            "json",
        ]
    )
    if proc.returncode != 0:
        log(f"volc get failed for {task_id}: {proc.stderr.strip() or proc.stdout.strip()}")
        return None
    match = re.search(r"\[.*\]", proc.stdout, re.S)
    if not match:
        return None
    items = json.loads(match.group())
    return items[0] if items else None


def volc_submit() -> str | None:
    proc = run_volc(["ml_task", "submit", "--conf", str(YAML), "--preemptible"])
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        log(f"volc submit failed: {out.strip()}")
        return None
    match = re.search(r"task_id=(t-[0-9a-z-]+)", out)
    if not match:
        log(f"volc submit ok but no task_id in output: {out.strip()}")
        return None
    task_id = match.group(1)
    info = volc_get(task_id) or {}
    preemptible = info.get("Preemptible")
    log(f"submitted {task_id} Preemptible={preemptible}")
    return task_id


def tail_volc_log(n: int = 2) -> list[str]:
    if not VOLC_LOG.exists():
        return []
    lines = VOLC_LOG.read_text(errors="replace").splitlines()
    return [ln for ln in lines[-n:] if ln.strip()]


def update_project_process(old_id: str | None, new_id: str, status: str) -> None:
    if not PROCESS_MD.exists():
        return
    text = PROCESS_MD.read_text(encoding="utf-8")
    row = (
        f"| {status} | {new_id} | mint_string_mmseqs_cluster_c1ie | "
        f"`{YAML}` |"
    )
    if old_id:
        text = re.sub(
            r"\| [^|]+ \| " + re.escape(old_id) + r" \| mint_string_mmseqs_cluster_c1ie \|[^\n]+\n",
            row + "\n",
            text,
            count=1,
        )
    else:
        # insert after header separator if missing
        marker = "| Status | Task ID | Job Name | YAML |"
        if marker in text and new_id not in text:
            idx = text.index(marker)
            idx = text.index("\n", idx) + 1
            idx = text.index("\n", idx) + 1
            text = text[:idx] + row + "\n" + text[idx:]

    stamp = datetime.now().strftime("%Y-%m-%d (UTC+8, MMseqs monitor auto-resubmit)")
    text = re.sub(
        r"Last updated: .+\n",
        f"Last updated: {stamp}\n",
        text,
        count=1,
    )
    section = f"""
## {datetime.now().strftime("%Y-%m-%d")} Auto-resubmit MMseqs cluster (monitor)

- Monitor detected terminal failure; resubmitted preemptible MMseqs cluster.
- Previous task: `{old_id or "none"}`
- New task: `{new_id}`, initial status `{status}`
"""
    PROCESS_MD.write_text(text.rstrip() + section, encoding="utf-8")


def resubmit(state: dict, reason: str, old_id: str | None) -> str | None:
    log(f"resubmit triggered ({reason}) old={old_id}")
    new_id = volc_submit()
    if not new_id:
        return None
    state["task_id"] = new_id
    state["resubmit_count"] = int(state.get("resubmit_count", 0)) + 1
    state["last_resubmit_utc"] = utc_now()
    save_state(state)
    info = volc_get(new_id) or {}
    update_project_process(old_id, new_id, info.get("Status", "Initialized"))
    return new_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor MMseqs Volc job and auto-resubmit on kill.")
    parser.add_argument("--task-id", default=None, help="Initial Volc task id (default: from state file)")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval seconds")
    parser.add_argument("--once", action="store_true", help="Single check; resubmit if needed then exit")
    args = parser.parse_args()

    state = load_state()
    task_id = args.task_id or state.get("task_id") or "t-20260622112332-njxvr"
    state["task_id"] = task_id
    save_state(state)

    log(f"monitor start task_id={task_id} interval={args.interval}s")

    while True:
        if CLU50.exists():
            log(f"clu50.tsv exists ({CLU50.stat().st_size} bytes) — monitor done")
            return 0

        info = volc_get(task_id)
        if info is None:
            log(f"task {task_id} not reachable; will retry next poll")
        else:
            status = info.get("Status", "unknown")
            elapsed = info.get("Elapsed")
            preemptible = info.get("Preemptible")
            tails = tail_volc_log(1)
            tail_hint = f" log_tail={tails[-1][:120]}" if tails else ""
            log(
                f"task={task_id} status={status} preemptible={preemptible} "
                f"elapsed={elapsed}s{tail_hint}"
            )

            if status in TERMINAL_DONE:
                if CLU50.exists():
                    log("task Success and clu50.tsv present — done")
                    return 0
                log("task Success but clu50.tsv missing — resubmit")
                new_id = resubmit(state, "success_without_clu50", task_id)
                if new_id:
                    task_id = new_id
                elif args.once:
                    return 1
            elif status in TERMINAL_RESUBMIT:
                new_id = resubmit(state, status.lower(), task_id)
                if new_id:
                    task_id = new_id
                elif args.once:
                    return 1

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
