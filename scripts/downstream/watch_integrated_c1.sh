#!/usr/bin/env bash
# Poll integrated training + C1 downstream eval Volc jobs (non-interactive JSON).
#
# Usage:
#   bash scripts/downstream/watch_integrated_c1.sh           # one-shot
#   bash scripts/downstream/watch_integrated_c1.sh --loop 5m # every 5 min
#
# Reads task ids from output/downstream_generation/eval_integrated_task_ids.tsv
set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
VOLC="/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh"
TRAIN_ID="t-20260710010623-q9rb2"
OUT_DIR="${ROOT}/output/grammar_v2_esmc300m_integrated_llada"
TSV="${ROOT}/output/downstream_generation/eval_integrated_task_ids.tsv"

volc_status() {
  local tid="$1"
  timeout 25 "${VOLC}" ml_task get -i "${tid}" -o json 2>/dev/null | python3 -c "
import json, sys
raw = sys.stdin.read()
i = raw.find('[')
if i < 0:
    print('UNKNOWN')
else:
    print(json.loads(raw[i:])[0].get('Status', 'UNKNOWN'))
" 2>/dev/null || echo "UNKNOWN"
}

tick() {
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) integrated C1 watch ==="
  echo "train ${TRAIN_ID}  status=$(volc_status "${TRAIN_ID}")"
  if [ -f "${OUT_DIR}/best.pt" ]; then
    echo "checkpoint  best.pt  EXISTS ($(du -h "${OUT_DIR}/best.pt" | awk '{print $1}'))"
  elif [ -f "${OUT_DIR}/latest.pt" ]; then
    echo "checkpoint  latest.pt EXISTS (best.pt not yet)"
  elif [ -d "${OUT_DIR}" ]; then
    echo "checkpoint  output dir exists, no .pt yet"
  else
    echo "checkpoint  output dir missing"
  fi
  if [ -f "${TSV}" ]; then
    tail -n +2 "${TSV}" | while IFS=$'\t' read -r task tid _; do
      [ -n "${task}" ] || continue
      log="${ROOT}/output/downstream_generation/eval_integrated_${task}_volc.log"
      st="$(volc_status "${tid}")"
      extra=""
      if [ -f "${log}" ]; then
        extra=" log=$(wc -l < "${log}")L last=$(tail -1 "${log}" | cut -c1-80)"
      fi
      echo "eval  ${task}  ${tid}  ${st}${extra}"
    done
  fi
  echo
}

if [[ "${1:-}" == "--loop" ]]; then
  interval="${2:-10m}"
  case "${interval}" in
    *m) sleep_sec=$(( ${interval%m} * 60 )) ;;
    *h) sleep_sec=$(( ${interval%h} * 3600 )) ;;
    *s) sleep_sec="${interval%s}" ;;
    *) sleep_sec=600 ;;
  esac
  while true; do tick; sleep "${sleep_sec}"; done
else
  tick
fi
