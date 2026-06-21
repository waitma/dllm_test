#!/usr/bin/env bash
# After STRING sequences.gz is valid, run MMseqs cluster then chain to mint pipeline.

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
STRING_DIR="${ROOT}/data/ppi_task_raw/raw/stringdb_mint"
LOG_DIR="${ROOT}/data/ppi_task_raw/processed/pipeline_logs"
GZ="${STRING_DIR}/protein.sequences.v12.0.fa.gz"

mkdir -p "${LOG_DIR}"

echo "Waiting for valid ${GZ} ..."
while true; do
  if [[ -f "${GZ}" ]] && gzip -t "${GZ}" 2>/dev/null; then
    echo "$(date -u +%H:%M:%S) gzip OK"
    break
  fi
  sleep 30
done

bash "${ROOT}/scripts/data/run_mint_mmseqs_cluster.sh" 2>&1 | tee -a "${LOG_DIR}/mmseqs_cluster.log"
