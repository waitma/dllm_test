"""Monitor grammar_v1 ESMC-300M encoder production training.

Polls Volc task status, WandB metrics, checkpoints, and Volc log snippets.
Writes rolling reports to output/grammar_v1_esmc300m/monitor.log.

Usage:
    /vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/python \
        /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/monitor_grammar_v1_esmc300m.py \
        --interval-min 15

    # one-shot check
    .../monitor_grammar_v1_esmc300m.py --once

    # background long-run (example)
    nohup .../monitor_grammar_v1_esmc300m.py --interval-min 15 \
        >> /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v1_esmc300m/monitor_stdout.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
FLOW_PYTHON = Path("/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/python")
TASK_ID = "t-20260622110228-pj2hb"
JOB_NAME = "qwen3_vl_bioseq_grammar_v1_esmc300m"
WANDB_PROJECT = "bioseq-qwen3-vl"
WANDB_RUN_NAME = "grammar_v1_esmc300m"
OUTPUT_DIR = PROJECT_ROOT / "output/grammar_v1_esmc300m"
LOG_PATH = OUTPUT_DIR / "monitor.log"
STATE_PATH = OUTPUT_DIR / "monitor_state.json"
VOLC = "/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def volc_status() -> dict:
    cmd = [
        VOLC,
        "ml_task",
        "list",
        "-n",
        "bioseq",
        "--status",
        "Initialized,Queue,Staging,Running,Killing,Success,Failed,Killed",
        "-o",
        "json",
        "--limit",
        "50",
    ]
    raw = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return {"error": "no_json_in_volc_output", "raw_tail": raw[-500:]}
    items = json.loads(match.group())
    for item in items:
        if item.get("JobId") == TASK_ID or item.get("JobName") == JOB_NAME:
            return item
    return {"error": "task_not_found", "task_id": TASK_ID}


def latest_output_log() -> Path | None:
    wandb_root = OUTPUT_DIR / "wandb/wandb"
    if not wandb_root.exists():
        return None
    runs = sorted(wandb_root.glob("run-*/files/output.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def tail_training_lines(path: Path, n: int = 8) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    interesting = [line for line in lines if "step=" in line or "val_loss" in line or "saved checkpoint" in line]
    pool = interesting if interesting else [line for line in lines if line.strip()]
    return pool[-n:]


def latest_wandb_run_id() -> str | None:
    wandb_root = OUTPUT_DIR / "wandb/wandb"
    if not wandb_root.exists():
        return None
    runs = sorted(wandb_root.glob("run-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in runs:
        match = re.search(r"run-\d+_\d+-(.+)$", run_dir.name)
        if match:
            return match.group(1)
    return None


def wandb_snapshot() -> dict:
    secret = Path("/vepfs-mlp2/c20250601/251105016/.secrets/wandb_api_key")
    env = os.environ.copy()
    if secret.exists():
        env["WANDB_API_KEY"] = secret.read_text().strip()
    run_id = latest_wandb_run_id()
    if run_id is None:
        script = f"""
import json
import wandb
api = wandb.Api()
runs = api.runs("{WANDB_PROJECT}", filters={{"display_name": "{WANDB_RUN_NAME}"}}, order="-created_at", per_page=1)
run = next(iter(runs), None)
if run is None:
    print(json.dumps({{"error": "wandb_run_not_found"}}))
else:
    hist = list(run.scan_history(keys=["train/loss", "train/lr", "train/grad_norm", "perf/samples_per_sec", "_step"], page_size=500))
    latest = hist[-1] if hist else {{}}
    print(json.dumps({{
        "run_id": run.id,
        "state": run.state,
        "points": len(hist),
        "latest": {{k: latest.get(k) for k in ["_step", "train/loss", "train/lr", "train/grad_norm", "perf/samples_per_sec"]}},
    }}))
"""
    else:
        script = f"""
import json
import wandb
api = wandb.Api()
run = api.run("{WANDB_PROJECT}/{run_id}")
hist = list(run.scan_history(keys=["train/loss", "train/lr", "train/grad_norm", "perf/samples_per_sec", "_step"], page_size=500))
latest = hist[-1] if hist else {{}}
print(json.dumps({{
    "run_id": run.id,
    "state": run.state,
    "points": len(hist),
    "latest": {{k: latest.get(k) for k in ["_step", "train/loss", "train/lr", "train/grad_norm", "perf/samples_per_sec"]}},
}}))
"""
    try:
        raw = subprocess.check_output([str(FLOW_PYTHON), "-c", script], text=True, env=env, stderr=subprocess.STDOUT)
        for line in reversed(raw.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"error": "no_json_in_wandb_output", "raw_tail": raw[-300:]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def checkpoint_info() -> dict:
    ckpt_dir = OUTPUT_DIR
    if not ckpt_dir.exists():
        return {"checkpoints": 0}
    ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        return {"checkpoints": 0}
    latest = ckpts[0]
    return {
        "checkpoints": len(ckpts),
        "latest_checkpoint": latest.name,
        "latest_mtime_utc": datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
    }


def volc_log_snippet(task_id: str = TASK_ID, line_limit: int = 120) -> dict:
    cmd = [VOLC, "ml_task", "logs", "-t", task_id, "-i", "worker_0", "-l", str(line_limit)]
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    lines = [line for line in raw.splitlines() if line.strip() and not line.startswith("volc ")]
    step_lines = [line for line in lines if "step=" in line and "loss=" in line]
    error_keywords = (
        "ProcessGroupNCCL",
        "OOM",
        "CUDA out of memory",
        "Traceback",
        "ChildFailedError",
        "FloatingPointError",
        "SIGABRT",
        "watchdog",
    )
    error_lines = [line for line in lines if any(k in line for k in error_keywords)]
    latest_step = None
    if step_lines:
        match = re.search(r"step=(\d+)", step_lines[-1])
        if match:
            latest_step = int(match.group(1))
    return {
        "latest_step_line": step_lines[-1] if step_lines else None,
        "latest_step": latest_step,
        "step_lines_seen": len(step_lines),
        "error_lines": error_lines[-5:] if error_lines else [],
        "tail": lines[-3:] if lines else [],
    }


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_state(state: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def stale_training_alert(status: dict, wb: dict, volc_log: dict, stale_minutes: float) -> str | None:
    if status.get("Status") != "Running":
        return None
    elapsed = status.get("Elapsed")
    if elapsed is None or elapsed < stale_minutes * 60:
        return None

    current_step = None
    if "latest" in wb and isinstance(wb["latest"], dict):
        step_val = wb["latest"].get("_step")
        if step_val is not None:
            current_step = int(step_val)
    if current_step is None:
        current_step = volc_log.get("latest_step")

    prev = load_state()
    prev_step = prev.get("last_step")
    prev_step_ts = prev.get("last_step_ts")
    now_ts = time.time()

    if current_step is not None and current_step != prev_step:
        save_state({"last_step": current_step, "last_step_ts": now_ts, "task_id": TASK_ID})
        return None

    if prev_step_ts is None:
        save_state({"last_step": current_step, "last_step_ts": now_ts, "task_id": TASK_ID})
        if current_step is None and elapsed >= stale_minutes * 60:
            return f"no_training_step_after_{int(elapsed // 60)}min"
        return None

    idle_min = (now_ts - prev_step_ts) / 60.0
    if idle_min >= stale_minutes:
        return f"step_stalled_{int(idle_min)}min_at_step_{prev_step}"
    return None


def is_terminal_failure(status: dict) -> bool:
    return status.get("Status") in {"Failed", "Killed"} or status.get("error") == "task_not_found"


def format_report(stale_minutes: float) -> str:
    status = volc_status()
    log_path = latest_output_log()
    log_tail = tail_training_lines(log_path) if log_path else []
    wb = wandb_snapshot()
    ckpt = checkpoint_info()
    volc_log = volc_log_snippet()

    lines = [
        f"=== grammar_v1_esmc300m monitor @ {utc_now()} ===",
        f"task_id={TASK_ID} preemptible=false",
    ]
    if "error" in status:
        lines.append(f"volc_error={status['error']}")
    else:
        lines.append(
            f"volc_status={status.get('Status')} elapsed={status.get('Elapsed')}s "
            f"start={status.get('Start')}"
        )
        if is_terminal_failure(status):
            lines.append("ALERT=training_terminal_failure")
            if volc_log.get("error_lines"):
                lines.append(f"failure_snippet={' | '.join(volc_log['error_lines'])}")
            elif volc_log.get("tail"):
                lines.append(f"failure_tail={' | '.join(volc_log['tail'])}")

    stale = stale_training_alert(status, wb, volc_log, stale_minutes)
    if stale:
        lines.append(f"ALERT={stale}")

    if volc_log.get("latest_step_line"):
        lines.append(f"volc_latest_step={volc_log['latest_step_line']}")
    if volc_log.get("error_lines") and not is_terminal_failure(status):
        lines.append(f"volc_errors={' | '.join(volc_log['error_lines'])}")

    lines.append(f"wandb={json.dumps(wb, ensure_ascii=False)}")
    lines.append(f"checkpoints={json.dumps(ckpt, ensure_ascii=False)}")
    if log_path:
        lines.append(f"output_log={log_path}")
    for line in log_tail:
        lines.append(f"  | {line}")
    lines.append("")
    return "\n".join(lines)


def append_report(report: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-min", type=float, default=15.0)
    parser.add_argument("--stale-minutes", type=float, default=30.0, help="Alert if no step progress for this long while Running.")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        report = format_report(stale_minutes=args.stale_minutes)
        print(report, flush=True)
        append_report(report)
        if args.once:
            break
        time.sleep(max(args.interval_min, 1.0) * 60.0)


if __name__ == "__main__":
    main()
