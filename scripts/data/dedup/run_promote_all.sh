#!/usr/bin/env bash
# Materialize deduped shards and promote them in place for the integrated mix.
#
# For each source: apply_blocklists.py --promote --force renames
#   <src>/train -> <src>/train_prededup   (raw, kept for provenance/headline)
#   <src>/train_dedup -> <src>/train      (deduped; what training now reads)
#
# Ordered small -> large. mint_ppi is intentionally EXCLUDED here (rebuild in
# flight); run it separately once mint_ppi/train exists. `tcr` (processed_v2) is
# EXCLUDED from the integrated mix (replaced by tcr_piste).

set -uo pipefail
ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
PY="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python"
LOG_DIR="${ROOT}/data/dedup/reports"
cd "${ROOT}"

SOURCES=("${@}")
if [[ ${#SOURCES[@]} -eq 0 ]]; then
  SOURCES=(neutralization ppi tcr_piste oas ots mint_actions nanobody)
fi

for src in "${SOURCES[@]}"; do
  echo "[$(date -u +%H:%M:%S)] promoting ${src}"
  "${PY}" scripts/data/dedup/apply_blocklists.py --source "${src}" --promote --force \
    > "${LOG_DIR}/promote_${src}.json" 2> "${LOG_DIR}/promote_${src}.err"
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[$(date -u +%H:%M:%S)] FAILED ${src} rc=${rc} (see promote_${src}.err)"
  else
    kept=$(grep -o '"kept": [0-9]*' "${LOG_DIR}/promote_${src}.json" | grep -o '[0-9]*')
    dropped=$(grep -o '"dropped": [0-9]*' "${LOG_DIR}/promote_${src}.json" | grep -o '[0-9]*')
    echo "[$(date -u +%H:%M:%S)] done ${src}: kept=${kept} dropped=${dropped}"
  fi
done
echo "[$(date -u +%H:%M:%S)] promote_all finished"
