#!/usr/bin/env bash
# Generate per-task-family Volc eval YAMLs for a grammar_v2 LLaDA checkpoint.
#
# Usage:
#   bash scripts/downstream/gen_eval_integrated_ymls.sh
#   bash scripts/downstream/gen_eval_integrated_ymls.sh grammar_v2_esmc300m_integrated_llada_7l
#   bash scripts/downstream/gen_eval_integrated_ymls.sh \
#     grammar_v2_esmc300m_integrated_llada_7l_step101300 \
#     eval_integrated_step101300
#   EVAL_FORCE_STABLE=1 EVAL_TASK_NAME_SUFFIX=retry2-stable \
#     bash scripts/downstream/gen_eval_integrated_ymls.sh \
#       grammar_v2_esmc300m_integrated_llada_7l_step189000 \
#       eval_integrated_step189000_retry2_stable
#   # then submit e.g.:
#   bash scripts/downstream/submit_eval_integrated.sh
#
set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
JOBS="${ROOT}/eval_jobs"
RUN="${1:-grammar_v2_esmc300m_integrated_llada}"
JOB_PREFIX="${2:-eval_integrated}"
# Short tag for TaskName uniqueness when RUN has a suffix (e.g. _7l).
RUN_TAG="${RUN#grammar_v2_}"
RUN_TAG="${RUN_TAG%_llada}"
RUN_TAG="${RUN_TAG//_/-}"
FORCE_STABLE="${EVAL_FORCE_STABLE:-0}"
TASK_NAME_SUFFIX="${EVAL_TASK_NAME_SUFFIX:-}"

# task-key : ActiveDeadlineSeconds (must exceed checkpoint wait + eval runtime)
declare -A DEADLINE=(
  [t1]=691200
  [t2]=691200
  [t3]=691200
  [t4]=777600
  [mint]=777600
  [ab]=777600
  [flab]=648000
  [nbbench]=864000
)
WAIT_ATTEMPTS=120  # ckpt already on disk for retrain evals; keep a short wait

write_yml() {
  local task="$1"
  local out="${JOBS}/${JOB_PREFIX}_${task}.yml"
  local deadline="${DEADLINE[$task]:-86400}"
  local task_name="eval_${RUN_TAG}_${task}"
  if [[ -n "${TASK_NAME_SUFFIX}" ]]; then
    task_name="${task_name}_${TASK_NAME_SUFFIX}"
  fi
  local preemptible="true"
  # AB and FLAb are long, stateful multi-stage families.  Platform preemption
  # near the end discards the active substage, so generate stable jobs for them.
  if [[ "${FORCE_STABLE}" == "1" || "${task}" == "ab" || "${task}" == "flab" ]]; then
    preemptible="false"
  fi
  cat > "${out}" <<EOF
TaskName: "${task_name}"
Description: "Downstream eval (post-LLaDA globalfeat) of ${RUN} for family '${task}', via scripts/downstream/run_all_downstream.py. Humanization skipped. Waits briefly for best.pt, then runs + aggregates."

Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250601/251105016"
    VepfsId: "vepfs-cnbj2c98dea54433"
    SubPath: "c20250601/251105016"

Entrypoint: |
  set -euo pipefail
  ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
  source activate "\${ENV}"
  export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
  export LD_LIBRARY_PATH=\${ENV}/lib:\${LD_LIBRARY_PATH:-}
  export PATH=\${ENV}/bin:\${PATH}
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=8
  export CUDA_VISIBLE_DEVICES=0
  export HF_ENDPOINT=https://hf-mirror.com
  export SKIP_HUMANIZATION=1

  cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
  OUT_DIR=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/${RUN}
  CKPT=""
  for attempt in \$(seq 1 ${WAIT_ATTEMPTS}); do
    if [ -f "\${OUT_DIR}/best.pt" ]; then CKPT="\${OUT_DIR}/best.pt"; break; fi
    if [ -f "\${OUT_DIR}/latest.pt" ]; then CKPT="\${OUT_DIR}/latest.pt"; break; fi
    echo "\$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for checkpoint in \${OUT_DIR} (attempt \${attempt})"
    sleep 60
  done
  if [ -z "\${CKPT}" ]; then echo "timeout waiting for checkpoint in \${OUT_DIR}"; exit 1; fi

  mkdir -p /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation
  LOG=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_${RUN}_${task}_volc.log
  echo "=== eval ${RUN} ${task} start \$(date -u +%Y-%m-%dT%H:%M:%SZ) ckpt=\${CKPT} ===" | tee "\${LOG}"
  python scripts/downstream/run_all_downstream.py run --only ${task} \\
      --checkpoint "\${CKPT}" --keep-going 2>&1 | tee -a "\${LOG}"
  echo "=== eval ${RUN} ${task} done \$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "\${LOG}"

Tags: [bioseq, grammar_v2, llada, integrated, downstream, eval, ${task}, globalfeat, 1gpu, ${RUN_TAG}]
ImageUrl: "cr-mlp-cn-beijing.cr.volces.com/public/airgen:v1"
Framework: "Custom"
ResourceQueueName: "c20250601"
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 1
    Flavor: "ml.pni2.3xlarge"
ActiveDeadlineSeconds: ${deadline}
DelayExitTimeSeconds: 0
AccessType: "Private"
Preemptible: ${preemptible}
Priority: 6
EOF
  echo "wrote ${out} (TaskName=${task_name})"
}

for task in t1 t2 t3 t4 mint ab flab nbbench; do
  write_yml "${task}"
done
echo "done: 8 eval YAMLs for RUN=${RUN}, JOB_PREFIX=${JOB_PREFIX} under ${JOBS}/"
