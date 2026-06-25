#!/usr/bin/env bash
# Launch MINT native stringdb splits in background on a large-memory Volc node.
#
# Usage (inside task WebShell, e.g. t-20260623161337-dhlpw):
#   bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/run_mint_stringdb_on_node.sh
#
# Monitor:
#   tail -f /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/pipeline_logs/mint_stringdb_native_bg.log

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
RAW="${ROOT}/data/ppi_task_raw/raw/stringdb_mint"
OUT="${ROOT}/data/ppi_task_raw/processed/mint_string_pretrain_v1"
LOG_DIR="${ROOT}/data/ppi_task_raw/processed/pipeline_logs"
LOG="${LOG_DIR}/mint_stringdb_native_bg.log"
PID_FILE="${LOG_DIR}/mint_stringdb_native_bg.pid"

mkdir -p "${LOG_DIR}" "${OUT}"

if [[ -f "${OUT}/training_filtered.links.txt.gz" && -f "${OUT}/validation.links.txt.gz" ]]; then
  echo "Outputs already exist under ${OUT}; nothing to do."
  ls -lh "${OUT}"/*.txt.gz
  exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "Already running as pid ${old_pid}. Log: ${LOG}"
    exit 0
  fi
fi

echo "=== host=$(hostname) mem=$(free -h | awk '/^Mem:/ {print $2}') cpus=$(nproc) ==="
mem_gib="$(free -g | awk '/^Mem:/ {print $2}')"
if [[ "${mem_gib}" -lt 512 ]]; then
  echo "ERROR: host has ~${mem_gib} GiB RAM; MINT native stringdb needs ~512+ GiB peak." >&2
  echo "Run this inside task WebShell (e.g. t-20260623161337-dhlpw on ml.pni2.28xlarge), not the dev instance." >&2
  exit 1
fi
echo "Starting MINT stringdb native run in background -> ${LOG}"

nohup bash -lc "
  set -euo pipefail
  source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
  export PYTHONUNBUFFERED=1
  if [[ ! -f ${RAW}/protein.sequences.v12.0.fa ]]; then
    gunzip -k ${RAW}/protein.sequences.v12.0.fa.gz
  fi
  gzip -t ${RAW}/protein.physical.links.full.v12.0.txt.gz
  exec python ${ROOT}/scripts/data/run_mint_stringdb_native.py \
    --raw-root ${RAW} \
    --output-dir ${OUT}
" >> "${LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started pid $(cat "${PID_FILE}"). Monitor: tail -f ${LOG}"
