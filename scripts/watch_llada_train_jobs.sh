#!/usr/bin/env bash
# Poll LLaDA Volc training jobs every N minutes.
# Default: log status. On Killed/Failed (unless monitor_only/paused), resume from
# latest.pt or resubmit scratch YAML.
#
# Usage:
#   bash scripts/watch_llada_train_jobs.sh                 # one-shot
#   bash scripts/watch_llada_train_jobs.sh --loop 10m      # every 10 minutes
#   bash scripts/watch_llada_train_jobs.sh --daemon 10m    # background loop
#
# State: scripts/llada_train_watchdog_jobs.json  (task_id updated on resubmit)
# Log:   scripts/logs/llada_train_watchdog.log
# Pid:   scripts/logs/llada_train_watchdog.loop.pid  (daemon mode)

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
VOLC="${ROOT}/scripts/volc-no-proxy.sh"
CONFIG="${ROOT}/scripts/llada_train_watchdog_jobs.json"
LOG_DIR="${ROOT}/scripts/logs"
LOG_FILE="${LOG_DIR}/llada_train_watchdog.log"
PID_FILE="${LOG_DIR}/llada_train_watchdog.loop.pid"
LOOP_OUT="${LOG_DIR}/llada_train_watchdog.loop.out"
TERMINAL_STATUSES="Killed,Failed"

mkdir -p "${LOG_DIR}"
chmod +x "${VOLC}" 2>/dev/null || true

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG_FILE}"
}

parse_interval() {
  local interval="${1:-10m}"
  case "${interval}" in
    *m) echo $(( ${interval%m} * 60 )) ;;
    *h) echo $(( ${interval%h} * 3600 )) ;;
    *s) echo "${interval%s}" ;;
    *) echo 600 ;;
  esac
}

run_once() {
  python3 - "${CONFIG}" "${ROOT}" "${TERMINAL_STATUSES}" "${VOLC}" <<'PY'
import json, re, subprocess, sys
from pathlib import Path

config_path, root, terminal_csv, volc = sys.argv[1:5]
terminal = set(terminal_csv.split(","))
cfg = json.loads(Path(config_path).read_text())
changed = False
actions = []


def status_of(task_id: str) -> str:
    """Prefer list -n (stable JSON); fall back to get."""
    for cmd in (
        ["timeout", "30", volc, "ml_task", "list", "-n", task_id,
         "-s", "Queue,Staging,Running,Initialized,Killing,Success,Failed,Killed",
         "--limit", "5"],
        ["timeout", "30", volc, "ml_task", "get", "-i", task_id, "-o", "json"],
    ):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            continue
        i_arr = out.find("[")
        i_obj = out.find("{")
        try:
            if i_arr >= 0 and (i_obj < 0 or i_arr <= i_obj):
                data = json.loads(out[i_arr:])
                if isinstance(data, list) and data:
                    return data[0].get("Status", "UNKNOWN")
            if i_obj >= 0:
                data = json.loads(out[i_obj:])
                if isinstance(data, dict):
                    return data.get("Status", "UNKNOWN")
                if isinstance(data, list) and data:
                    return data[0].get("Status", "UNKNOWN")
        except Exception:
            continue
    return "UNKNOWN"


def submit(yaml_path: str):
    p = subprocess.run(
        ["timeout", "120", volc, "ml_task", "submit", "--conf", yaml_path],
        capture_output=True,
        text=True,
    )
    text = p.stdout + p.stderr
    m = re.search(r"task_id=(t-[^\s,]+)", text)
    return m.group(1) if m else None


for job in cfg["jobs"]:
    name = job["name"]
    task_id = job["task_id"]
    out_dir = Path(root) / job["output_dir"]
    latest = out_dir / "latest.pt"
    resume_yaml = str(Path(root) / job["yaml"])
    scratch_yaml = job.get("scratch_yaml")
    scratch_path = str(Path(root) / scratch_yaml) if scratch_yaml else None
    monitor_only = bool(job.get("monitor_only"))

    if job.get("paused"):
        actions.append(f"PAUSED {name} task={task_id} note={job.get('note', '')}")
        continue

    status = status_of(task_id)
    line = f"{name} task={task_id} status={status}"
    ckpt = "latest.pt" if latest.is_file() else "no_ckpt"

    if status in terminal:
        if monitor_only:
            actions.append(f"MONITOR_TERMINAL {line} {ckpt} (no auto-resubmit)")
            continue
        if latest.is_file():
            yaml_use, mode = resume_yaml, "resume"
        elif scratch_path and Path(scratch_path).is_file():
            yaml_use, mode = scratch_path, "scratch"
        else:
            actions.append(f"RESUBMIT_SKIP {line} reason=no_latest_pt")
            continue

        new_id = submit(yaml_use)
        if not new_id:
            actions.append(f"RESUBMIT_FAIL {line} mode={mode} yaml={yaml_use}")
            continue
        job["task_id"] = new_id
        changed = True
        actions.append(
            f"RESUBMIT_OK {name} old={task_id} new={new_id} mode={mode} "
            f"yaml={Path(yaml_use).name} latest={'yes' if latest.is_file() else 'no'}"
        )
    elif status == "Success":
        actions.append(f"DONE {line}")
    else:
        tag = "MONITOR" if monitor_only else "OK"
        actions.append(f"{tag} {line} {ckpt}")

if changed:
    Path(config_path).write_text(json.dumps(cfg, indent=2) + "\n")

for a in actions:
    print(a)
PY
}

do_check() {
  log "=== watchdog tick ==="
  while IFS= read -r line; do
    log "${line}"
  done < <(run_once)
}

start_daemon() {
  local interval="${1:-10m}"
  local old_pid=""
  if [[ -f "${PID_FILE}" ]]; then
    old_pid=$(cat "${PID_FILE}")
    if kill -0 "${old_pid}" 2>/dev/null; then
      echo "watchdog already running pid=${old_pid}" >&2
      exit 0
    fi
  fi
  nohup bash "$0" --loop "${interval}" >>"${LOOP_OUT}" 2>&1 &
  echo $! >"${PID_FILE}"
  echo "started watchdog pid=$(cat ${PID_FILE}) interval=${interval} log=${LOG_FILE}"
}

MODE="${1:-}"
case "${MODE}" in
  --loop)
    INTERVAL="${2:-10m}"
    SLEEP_SEC=$(parse_interval "${INTERVAL}")
    log "Starting loop interval=${INTERVAL} (${SLEEP_SEC}s)"
    while true; do
      do_check
      sleep "${SLEEP_SEC}"
    done
    ;;
  --daemon)
    start_daemon "${2:-10m}"
    ;;
  *)
    do_check
    ;;
esac
