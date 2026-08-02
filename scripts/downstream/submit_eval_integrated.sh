#!/usr/bin/env bash
# Submit all 8 integrated downstream eval Volc jobs (wait for best.pt, then run).
#
# Usage:
#   bash scripts/downstream/submit_eval_integrated.sh          # submit all
#   bash scripts/downstream/submit_eval_integrated.sh t1 t3    # submit subset
#   EVAL_JOB_PREFIX=eval_integrated_step101300 \
#   EVAL_RECORD_PREFIX=eval_step101300 \
#     bash scripts/downstream/submit_eval_integrated.sh
#
# After all families finish, aggregate:
#   python scripts/downstream/run_all_downstream.py collect \
#     --checkpoint output/grammar_v2_esmc300m_integrated_llada/best.pt
#
# Then backfill docs / paper:
#   python scripts/downstream/fill_ours_from_summary.py \
#     --checkpoint output/grammar_v2_esmc300m_integrated_llada/best.pt --report
set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
VOLC="${VOLC:-${ROOT}/scripts/volc-no-proxy.sh}"
JOB_PREFIX="${EVAL_JOB_PREFIX:-eval_integrated}"
RECORD_PREFIX="${EVAL_RECORD_PREFIX:-${JOB_PREFIX}}"
ALL=(t1 t2 t3 t4 mint ab flab nbbench)

if [ "$#" -gt 0 ]; then
  TASKS=("$@")
else
  TASKS=("${ALL[@]}")
fi

cd "${ROOT}"
LOG="${ROOT}/output/downstream_generation/${RECORD_PREFIX}_submit.log"
TSV="${ROOT}/output/downstream_generation/${RECORD_PREFIX}_task_ids.tsv"
mkdir -p "$(dirname "${LOG}")"
: > "${TSV}.tmp"
echo -e "task\tvolc_id\tsubmitted_utc" >> "${TSV}.tmp"

for task in "${TASKS[@]}"; do
  yml="${ROOT}/eval_jobs/${JOB_PREFIX}_${task}.yml"
  if [ ! -f "${yml}" ]; then
    echo "missing ${yml}" >&2
    exit 1
  fi
  echo "=== submit ${JOB_PREFIX}_${task} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "${LOG}"
  # volc ml_task submit prints task id on stdout; capture last line matching t-*
  out="$("${VOLC}" ml_task submit --conf "${yml}" 2>&1 | tee -a "${LOG}")"
  tid="$(echo "${out}" | grep -Eo 't-[0-9]{14}-[a-z0-9]+|task_id=t-[0-9]{14}-[a-z0-9]+' | sed 's/task_id=//' | tail -1 || true)"
  if [ -n "${tid}" ]; then
    echo -e "${task}\t${tid}\t$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${TSV}.tmp"
    echo "  -> ${tid}"
  else
    echo "  -> submitted (task id not parsed; check log)" >&2
  fi
done
mv "${TSV}.tmp" "${TSV}"
echo "done. task ids: ${TSV}"
