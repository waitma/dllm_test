#!/usr/bin/env bash
# Wait for valid physical links, then MINT splits + mint_ppi grammar shards.
#
# Usage:
#   bash scripts/data/run_mint_pipeline_after_links.sh
#   bash scripts/data/run_mint_pipeline_after_links.sh --skip-wait   # links already verified

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
STRING_DIR="${ROOT}/data/ppi_task_raw/raw/stringdb_mint"
LOG_DIR="${ROOT}/data/ppi_task_raw/processed/pipeline_logs"
PYTHON="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python"
LINKS_GZ="${STRING_DIR}/protein.physical.links.full.v12.0.txt.gz"
CLU50="${STRING_DIR}/clu50.tsv"
LOG="${LOG_DIR}/mint_pipeline_after_links.log"

SKIP_WAIT=0
for arg in "$@"; do
  case "${arg}" in
    --skip-wait) SKIP_WAIT=1 ;;
    *) echo "Unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

mkdir -p "${LOG_DIR}"
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${LOG}"; }

wait_for_links() {
  log "waiting for valid ${LINKS_GZ}"
  while true; do
    if [[ -f "${LINKS_GZ}" ]] && gzip -t "${LINKS_GZ}" 2>/dev/null; then
      log "links gzip OK: $(ls -lh "${LINKS_GZ}" | awk '{print $5}')"
      return 0
    fi
    sleep 30
  done
}

if [[ "${SKIP_WAIT}" -eq 0 ]]; then
  wait_for_links
else
  gzip -t "${LINKS_GZ}"
  log "links gzip OK (skip-wait)"
fi

if [[ ! -f "${CLU50}" ]]; then
  log "ERROR: missing ${CLU50}"
  exit 1
fi

log "=== build_mint_string_splits start ==="
"${PYTHON}" "${ROOT}/scripts/data/build_mint_string_splits.py" 2>&1 | tee -a "${LOG_DIR}/mint_splits.log"

for SPLIT in train valid; do
  log "=== build_mint_grammar_shards ${SPLIT} start ==="
  "${PYTHON}" "${ROOT}/scripts/data/build_mint_grammar_shards.py" --split "${SPLIT}" --force 2>&1 | tee -a "${LOG_DIR}/mint_grammar_shards.log"
done

log "=== MINT pipeline after links complete ==="
