#!/usr/bin/env bash
# Orchestrate STRING v11 MINT + actions pipeline after raw downloads complete.
#
# Waits for gzip integrity, submits Volc cluster/splits jobs, then builds actions
# splits once physical validation exists.
#
# Usage:
#   nohup bash scripts/data/run_v11_actions_pipeline.sh \
#     >> data/ppi_task_raw/processed/pipeline_logs/v11_pipeline_orchestrator.log 2>&1 &

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
RAW="${ROOT}/data/ppi_task_raw/raw/stringdb_mint"
LOG_DIR="${ROOT}/data/ppi_task_raw/processed/pipeline_logs"
VOLC="/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh"
CONDA="source activate /vepfs-mlp2/c20250601/251105016/conda/envs/pllm"

SEQ_GZ="${RAW}/protein.sequences.v11.0.fa.gz"
PHYS_GZ="${RAW}/protein.physical.links.full.v11.0.txt.gz"
ACT_GZ="${RAW}/protein.actions.v11.0.txt.gz"
CLU50="${RAW}/clu50.v11.0.tsv"
PHYS_OUT="${ROOT}/data/ppi_task_raw/processed/mint_string_pretrain_v11.0"
ACT_OUT="${ROOT}/data/ppi_task_raw/processed/mint_string_actions_v11.0"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

wait_gz() {
  local path="$1"
  local label="$2"
  log "waiting for ${label}: ${path}"
  while true; do
    if [[ -f "${path}" ]] && gzip -t "${path}" 2>/dev/null; then
      log "${label} gzip OK ($(stat -c '%s' "${path}") bytes)"
      return 0
    fi
    sleep 120
  done
}

wait_file() {
  local path="$1"
  local label="$2"
  log "waiting for ${label}: ${path}"
  while [[ ! -f "${path}" ]]; do
    sleep 120
  done
  log "${label} exists"
}

submit_once() {
  local marker="$1"
  local yaml="$2"
  if [[ -f "${marker}" ]]; then
    log "skip submit (marker exists): ${yaml}"
    return 0
  fi
  log "submitting ${yaml}"
  local out
  out="$("${VOLC}" ml_task submit --conf "${yaml}" 2>&1)" || true
  echo "${out}"
  local tid
  tid="$(echo "${out}" | grep -oE 't-[0-9]{14}-[a-z0-9]+' | head -1 || true)"
  if [[ -n "${tid}" ]]; then
    echo "${tid}" > "${marker}"
    log "submitted ${yaml} task_id=${tid}"
  else
    log "WARN: could not parse task_id from submit output"
  fi
}

mkdir -p "${LOG_DIR}"

# 1) sequences -> MMseqs cluster
wait_gz "${SEQ_GZ}" "sequences"
submit_once "${LOG_DIR}/v11_cluster_submitted.marker" \
  "${ROOT}/train_jobs/mint_string_mmseqs_cluster_v11.yml"

# 2) physical links -> MINT splits (needs clu50 from cluster job)
wait_gz "${PHYS_GZ}" "physical links"
wait_file "${CLU50}" "clu50.v11.0.tsv"
submit_once "${LOG_DIR}/v11_physical_splits_submitted.marker" \
  "${ROOT}/train_jobs/mint_stringdb_splits_v11_g3a48xlarge.yml"

# 3) actions download -> actions splits (needs physical valid for leak guard)
wait_gz "${ACT_GZ}" "actions"
wait_file "${PHYS_OUT}/validation.links.txt.gz" "physical validation links"

if [[ ! -f "${ACT_OUT}/training_filtered.links.txt.gz" ]]; then
  log "building actions splits -> ${ACT_OUT}"
  eval "${CONDA}"
  if [[ ! -f "${RAW}/protein.sequences.v11.0.fa" ]]; then
    gunzip -k "${SEQ_GZ}"
  fi
  python "${ROOT}/scripts/data/build_string_actions_splits.py" \
    --actions-gz "${ACT_GZ}" \
    --sequences-fa "${RAW}/protein.sequences.v11.0.fa" \
    --cluster-tsv "${CLU50}" \
    --physical-valid-links "${PHYS_OUT}/validation.links.txt.gz" \
    --output-dir "${ACT_OUT}"
  log "actions splits done"
else
  log "actions splits already exist under ${ACT_OUT}"
fi

log "v11 pipeline orchestrator finished"
