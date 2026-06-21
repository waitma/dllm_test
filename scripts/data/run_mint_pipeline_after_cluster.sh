#!/usr/bin/env bash
# Orchestrate post-mmseqs steps: mint splits -> mint grammar shards (train then valid).
#
# Waits for clu50.tsv, then runs build_mint_string_splits.py and build_mint_grammar_shards.py.

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
STRING_DIR="${ROOT}/data/ppi_task_raw/raw/stringdb_mint"
LOG_DIR="${ROOT}/data/ppi_task_raw/processed/pipeline_logs"
PYTHON="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python"
CLU="${STRING_DIR}/clu50.tsv"

mkdir -p "${LOG_DIR}"

echo "Waiting for ${CLU} ..."
while [[ ! -f "${CLU}" ]]; do
  sleep 60
  echo "$(date -u +%H:%M:%S) still waiting for clu50.tsv"
done

echo "=== build_mint_string_splits $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "${LOG_DIR}/mint_splits.log"
"${PYTHON}" "${ROOT}/scripts/data/build_mint_string_splits.py" 2>&1 | tee -a "${LOG_DIR}/mint_splits.log"

for SPLIT in train valid; do
  echo "=== build_mint_grammar_shards ${SPLIT} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "${LOG_DIR}/mint_grammar_shards.log"
  "${PYTHON}" "${ROOT}/scripts/data/build_mint_grammar_shards.py" --split "${SPLIT}" --force 2>&1 | tee -a "${LOG_DIR}/mint_grammar_shards.log"
done

echo "=== MINT pipeline complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
