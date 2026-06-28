#!/usr/bin/env bash
# Idempotent v11 STRING actions grammar data prep (modes only).
#
# Does NOT build mint_ppi shards (v12 binding lives under mint_string_pretrain_v1).
# Does NOT overwrite existing mint_actions Arrow shards.
#
# Intended to run on a high-RAM node (e.g. ml.pni2.28xlarge ~2TB), typically
# launched in the background while GPU training uses the existing grammar cache::
#
#   nohup bash scripts/data/prepare_v11_mint_grammar_data.sh >> .../prepare_v11_bg.log 2>&1 &
#
# Usage:
#   bash scripts/data/prepare_v11_mint_grammar_data.sh

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
CONDA="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr"
PYTHON="${CONDA}/bin/python"
RAW="${ROOT}/data/ppi_task_raw/raw/stringdb_mint"
PHYS_OUT="${ROOT}/data/ppi_task_raw/processed/mint_string_pretrain_v11.0"
ACT_OUT="${ROOT}/data/ppi_task_raw/processed/mint_string_actions_v11.0"
GRAMMAR_DIR="${ROOT}/data/bioseq_grammar_v1"
LOG_DIR="${ROOT}/data/ppi_task_raw/processed/pipeline_logs"
LOG="${LOG_DIR}/prepare_v11_mint_grammar_data.log"

mkdir -p "${LOG_DIR}"
exec >> "${LOG}" 2>&1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "=== prepare_v11_mint_grammar_data (actions-only) start hostname=$(hostname) nproc=$(nproc) ==="
free -h || true

for path in \
  "${RAW}/protein.physical.links.full.v11.0.txt.gz" \
  "${RAW}/protein.actions.v11.0.txt.gz" \
  "${RAW}/clu50.v11.0.tsv"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing prerequisite ${path}" >&2
    exit 1
  fi
done

if [[ ! -f "${RAW}/protein.sequences.v11.0.fa" ]]; then
  log "decompressing protein.sequences.v11.0.fa.gz"
  gunzip -k "${RAW}/protein.sequences.v11.0.fa.gz"
fi

gzip -t "${RAW}/protein.physical.links.full.v11.0.txt.gz"
gzip -t "${RAW}/protein.actions.v11.0.txt.gz"
log "raw gzip integrity OK"

if [[ -f "${PHYS_OUT}/validation.links.txt.gz" ]]; then
  log "using existing v11 physical validation links at ${PHYS_OUT} (for actions split leakage check)"
elif [[ -f "${PHYS_OUT}/training_filtered.links.txt.gz" && -f "${PHYS_OUT}/validation.links.txt.gz" ]]; then
  log "v11 physical splits exist under ${PHYS_OUT}"
else
  log "running v11 physical MINT splits -> ${PHYS_OUT} (splits only; no mint_ppi grammar shards)"
  "${PYTHON}" "${ROOT}/scripts/data/run_mint_stringdb_native.py" \
    --raw-root "${RAW}" \
    --links-gz "${RAW}/protein.physical.links.full.v11.0.txt.gz" \
    --sequences-fa "${RAW}/protein.sequences.v11.0.fa" \
    --cluster-tsv "${RAW}/clu50.v11.0.tsv" \
    --output-dir "${PHYS_OUT}"
fi

if [[ -f "${ACT_OUT}/training_filtered.links.txt.gz" && -f "${ACT_OUT}/validation.links.txt.gz" ]]; then
  log "skip actions splits (outputs exist under ${ACT_OUT})"
else
  log "running actions mode splits -> ${ACT_OUT}"
  "${PYTHON}" "${ROOT}/scripts/data/build_string_actions_splits.py" \
    --actions-gz "${RAW}/protein.actions.v11.0.txt.gz" \
    --sequences-fa "${RAW}/protein.sequences.v11.0.fa" \
    --cluster-tsv "${RAW}/clu50.v11.0.tsv" \
    --physical-valid-links "${PHYS_OUT}/validation.links.txt.gz" \
    --output-dir "${ACT_OUT}"
fi

for split in train valid; do
  shard="${GRAMMAR_DIR}/mint_actions/${split}"
  if [[ -d "${shard}" ]]; then
    log "skip mint_actions/${split} grammar shard (exists; not overwritten)"
    continue
  fi
  log "building mint_actions/${split} grammar shard (first-time only)"
  "${PYTHON}" "${ROOT}/scripts/data/build_mint_grammar_shards.py" \
    --source mint_actions \
    --split "${split}" \
    --mint-dir "${ACT_OUT}" \
    --max-protein-length 1024
done

log "updating grammar manifest with mint sources (does not rebuild existing shards)"
"${PYTHON}" "${ROOT}/scripts/data/build_bioseq_grammar_v1.py" \
  --output-dir "${GRAMMAR_DIR}" \
  --splits train,valid \
  --sources oas,ots,tcr,ppi,mint_ppi,mint_actions

log "=== prepare_v11_mint_grammar_data done ==="
ls -lh "${ACT_OUT}"/*.txt.gz 2>/dev/null || true
test -f "${GRAMMAR_DIR}/manifest.json"
