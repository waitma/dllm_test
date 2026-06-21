#!/usr/bin/env bash
# Step 2a: MMseqs2 50% clustering on STRING sequences (MINT protocol).
#
# Usage:
#   bash scripts/data/run_mint_mmseqs_cluster.sh

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
STRING_DIR="${ROOT}/data/ppi_task_raw/raw/stringdb_mint"
LOG_DIR="${ROOT}/data/ppi_task_raw/processed/pipeline_logs"
LOG="${LOG_DIR}/mmseqs_cluster.log"
CONDA="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "=== MMseqs cluster start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cd "${STRING_DIR}"

FA="protein.sequences.v12.0.fa"
if [[ ! -f "${FA}" ]]; then
  echo "Decompressing ${FA}.gz ..."
  gunzip -k protein.sequences.v12.0.fa.gz
fi

source /vepfs-mlp2/c20250601/251105016/conda/etc/profile.d/conda.sh
conda activate "${CONDA}"
export PATH="${CONDA}/bin:${PATH}"

if [[ -f clu50.tsv ]]; then
  echo "clu50.tsv already exists, skip cluster"
  exit 0
fi

echo "mmseqs createdb ..."
mmseqs createdb "${FA}" DB100

echo "mmseqs cluster (50% min-seq-id) ..."
mmseqs cluster DB100 clu50 /tmp/mmseqs_string_db100 --min-seq-id 0.50 --remove-tmp-files

echo "mmseqs createtsv ..."
mmseqs createtsv DB100 DB100 clu50 clu50.tsv

echo "=== MMseqs cluster done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
wc -l clu50.tsv
