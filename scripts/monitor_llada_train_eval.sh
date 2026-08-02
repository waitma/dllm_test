#!/usr/bin/env bash
# Monitor LLaDA train + downstream eval Volc jobs and local artifacts.
#
# Usage:
#   bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/monitor_llada_train_eval.sh

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
VOLC="/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh"
CONFIG="${ROOT}/scripts/llada_train_watchdog_jobs.json"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "=== LLaDA monitor ${TS} ==="

python3 - "${CONFIG}" "${VOLC}" << 'PY'
import json, subprocess, sys
from pathlib import Path

config_path, volc = sys.argv[1:3]
cfg = json.loads(Path(config_path).read_text())
py_get = r'''
import json,sys
raw=sys.stdin.read(); i=raw.find('[')
d=json.loads(raw[i:])[0] if i>=0 else {}
print(d.get("JobName","?"), "|", d.get("Status","?"), "| elapsed=", d.get("Elapsed","?"))
'''
for job in cfg["jobs"]:
    tid = job["task_id"]
    name = job["name"]
    try:
        raw = subprocess.check_output(
            [volc, "ml_task", "get", "-i", tid, "-o", "json"],
            stderr=subprocess.DEVNULL,
            timeout=15,
            text=True,
        )
        line = subprocess.check_output(["python3", "-c", py_get], input=raw, text=True).strip()
    except Exception as e:
        line = f"volc_get_failed ({e})"
    print(f"{name} ({tid}): {line}")
PY

for variant in esmc300m_cmp500k_llada esmc600m_cmp500k_llada; do
  out="${ROOT}/output/grammar_v2_${variant}"
  log="${out}/wandb/wandb/latest-run/files/output.log"
  echo "--- ${variant} ---"
  if [[ -f "${out}/best.pt" ]]; then
    ls -lh "${out}/best.pt" "${out}/latest.pt" 2>/dev/null | awk '{print $5, $6, $7, $8, $9}'
  else
    echo "no checkpoint yet"
  fi
  if [[ -f "${log}" ]]; then
    grep -E "step=[0-9]+ loss=" "${log}" 2>/dev/null | tail -1 || true
    grep -E "step=[0-9]+ val_loss=" "${log}" 2>/dev/null | tail -1 || true
  fi
  summary="${ROOT}/output/downstream_generation/grammar_v2_${variant}_downstream_summary.txt"
  if [[ -f "${summary}" ]]; then
    echo "downstream summary:"
    tail -8 "${summary}"
  fi
done
