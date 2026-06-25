#!/usr/bin/env bash
# Step 2a: MMseqs2 50% clustering on STRING sequences (MINT protocol).
#
# Usage:
#   bash scripts/data/run_mint_mmseqs_cluster.sh                 # v12.0 (default)
#   STRING_VERSION=v11.0 bash scripts/data/run_mint_mmseqs_cluster.sh
#
# Output naming:
#   v12.0  -> DB100        + clu50.tsv          (backward compatible)
#   other  -> DB100_<ver>  + clu50.<ver>.tsv

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
STRING_DIR="${ROOT}/data/ppi_task_raw/raw/stringdb_mint"
LOG_DIR="${ROOT}/data/ppi_task_raw/processed/pipeline_logs"
CONDA="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr"

STRING_VERSION="${STRING_VERSION:-v12.0}"
FA="protein.sequences.${STRING_VERSION}.fa"
if [[ "${STRING_VERSION}" == "v12.0" ]]; then
  DB="DB100"
  CLU_TSV="clu50.tsv"
else
  DB="DB100_${STRING_VERSION}"
  CLU_TSV="clu50.${STRING_VERSION}.tsv"
fi
CLU_PREFIX="clu50_${STRING_VERSION}"
MMSEQS_TMP="${STRING_DIR}/mmseqs_tmp_${STRING_VERSION}"
LOG="${LOG_DIR}/mmseqs_cluster_${STRING_VERSION}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "=== MMseqs cluster start $(date -u +%Y-%m-%dT%H:%M:%SZ) version=${STRING_VERSION} ==="
cd "${STRING_DIR}"

if [[ ! -f "${FA}" ]]; then
  echo "Decompressing ${FA}.gz ..."
  gunzip -k "${FA}.gz"
fi

export PATH="${CONDA}/bin:${PATH}"
MMSEQS="${CONDA}/bin/mmseqs"
THREADS="${MMSEQS_THREADS:-$(nproc)}"
mkdir -p "${MMSEQS_TMP}"

if [[ -f "${CLU_TSV}" ]]; then
    echo "${CLU_TSV} already exists, skip cluster"
    exit 0
fi

if [[ "${MMSEQS_CLEAN_TMP:-0}" == "1" ]] && [[ -d "${MMSEQS_TMP}" ]]; then
  echo "Removing partial mmseqs tmp dir ${MMSEQS_TMP}"
  rm -rf "${MMSEQS_TMP}"
  mkdir -p "${MMSEQS_TMP}"
fi

echo "mmseqs createdb ..."
if [[ ! -f "${DB}.dbtype" ]]; then
  "${MMSEQS}" createdb "${FA}" "${DB}"
else
  echo "${DB} already exists, skip createdb"
fi

echo "mmseqs cluster (50% min-seq-id, threads=${THREADS}) ..."
"${MMSEQS}" cluster "${DB}" "${CLU_PREFIX}" "${MMSEQS_TMP}" --min-seq-id 0.50 --remove-tmp-files --threads "${THREADS}"

echo "mmseqs createtsv ..."
"${MMSEQS}" createtsv "${DB}" "${DB}" "${CLU_PREFIX}" "${CLU_TSV}"

echo "=== MMseqs cluster done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
wc -l "${CLU_TSV}"
