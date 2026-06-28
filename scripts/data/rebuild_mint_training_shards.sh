#!/usr/bin/env bash
# Rebuild mint_ppi grammar Arrow shards from v12 binding splits only.
#
# mint_actions (v11 modes) is never rebuilt here — existing shards are preserved.
# PPI pairs with either protein longer than --max-protein-length (default 1024) are
# dropped at shard build time (not cropped).
#
# Usage:
#   bash scripts/data/rebuild_mint_training_shards.sh

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
CONDA="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr"
PYTHON="${CONDA}/bin/python"
MINT_PPI_DIR="${ROOT}/data/ppi_task_raw/processed/mint_string_pretrain_v1"
GRAMMAR_DIR="${ROOT}/data/bioseq_grammar_v1"
MINT_ACTIONS_TRAIN="${GRAMMAR_DIR}/mint_actions/train"
MINT_ACTIONS_VALID="${GRAMMAR_DIR}/mint_actions/valid"
MARKER="${GRAMMAR_DIR}/.mint_shards_filter1024_v12v11"
LOG_DIR="${ROOT}/data/ppi_task_raw/processed/pipeline_logs"
LOG="${LOG_DIR}/rebuild_mint_training_shards.log"

mkdir -p "${LOG_DIR}"
exec >> "${LOG}" 2>&1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

if [[ -f "${MARKER}" ]]; then
  log "marker exists (${MARKER}), skip rebuild"
  exit 0
fi

for path in \
  "${MINT_PPI_DIR}/training_filtered.links.txt.gz" \
  "${MINT_PPI_DIR}/validation.links.txt.gz"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${path}" >&2
    exit 1
  fi
done

for path in "${MINT_ACTIONS_TRAIN}" "${MINT_ACTIONS_VALID}"; do
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: mint_actions shard missing (${path}); build v11 actions first, do not overwrite here" >&2
    exit 1
  fi
done

log "=== rebuild mint_ppi only (v12 binding, filter >1024 aa); mint_actions untouched ==="

for split in train valid; do
  log "mint_ppi/${split} from v12 ${MINT_PPI_DIR}"
  "${PYTHON}" "${ROOT}/scripts/data/build_mint_grammar_shards.py" \
    --source mint_ppi \
    --split "${split}" \
    --mint-dir "${MINT_PPI_DIR}" \
    --max-protein-length 1024 \
    --force
done

log "updating grammar manifest (six-source mix; mint_actions paths unchanged)"
"${PYTHON}" "${ROOT}/scripts/data/build_bioseq_grammar_v1.py" \
  --output-dir "${GRAMMAR_DIR}" \
  --splits train,valid \
  --sources oas,ots,tcr,ppi,mint_ppi,mint_actions \
  --ppi-max-protein-length 1024

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

grammar_dir = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/bioseq_grammar_v1")
manifest = json.loads((grammar_dir / "manifest.json").read_text())
rows = {
    f"{d['source']}:{d['split']}": d["rows"]
    for d in manifest["datasets"]
    if d["source"] in ("mint_ppi", "mint_actions")
}
print("manifest rows:", rows)
mint_ppi_train = rows.get("mint_ppi:train", 0)
mint_actions_train = rows.get("mint_actions:train", 0)
if mint_ppi_train > 120_000_000:
    raise SystemExit(f"mint_ppi:train looks like v11 binding ({mint_ppi_train:,}); expected ~96M from v12")
if mint_actions_train < 9_000_000 or mint_actions_train > 10_000_000:
    raise SystemExit(f"mint_actions:train unexpected ({mint_actions_train:,}); expected ~9.2M")
PY

date -u +%Y-%m-%dT%H:%M:%SZ > "${MARKER}"
log "=== rebuild done; wrote ${MARKER} ==="
