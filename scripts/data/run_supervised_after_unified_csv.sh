#!/usr/bin/env bash
# Wait for unified CSV build, then run supervised grammar shards (CoV-AbDab neutralization).

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
CSV="${ROOT}/data/ppi_task_raw/processed/interaction_records_unified.csv"
LOG_DIR="${ROOT}/data/ppi_task_raw/processed/pipeline_logs"
LOG="${LOG_DIR}/supervised_shards.log"
PYTHON="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python"
TARGET_GB=3.0

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "Waiting for unified CSV (~${TARGET_GB}GB) ..."
while true; do
  if [[ -f "${CSV}" ]]; then
    size_gb=$(awk -v s="$(stat -c%s "${CSV}" 2>/dev/null || echo 0)" 'BEGIN{printf "%.2f", s/1024/1024/1024}')
    if awk -v s="${size_gb}" -v t="${TARGET_GB}" 'BEGIN{exit !(s >= t * 0.98)}'; then
      if ! pgrep -f build_ppi_interaction_csv.py >/dev/null 2>&1; then
        echo "$(date -u +%H:%M:%S) CSV ready (${size_gb}GB), builder exited"
        break
      fi
    fi
    echo "$(date -u +%H:%M:%S) CSV ${size_gb}GB, builder still running..."
  fi
  sleep 30
done

echo "=== build_supervised_grammar_shards $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"${PYTHON}" "${ROOT}/scripts/data/build_supervised_grammar_shards.py" \
  --sources covabdab_neutralization

echo "=== supervised shards complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
