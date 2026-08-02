#!/usr/bin/env bash
# Generate grammar-v2 LLaDA downstream / pairing-metrics Volc YAMLs under eval_jobs/.
#
# Usage:
#   bash scripts/downstream/gen_eval_grammar_v2_llada_ymls.sh

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
JOBS="${ROOT}/eval_jobs"

write_downstream_yml() {
  local variant="$1"
  local tag="$2"
  local out="${JOBS}/eval_grammar_v2_${variant}_downstream.yml"
  cat > "${out}" <<EOF
TaskName: "eval_grammar_v2_${variant}_downstream"
Description: "Single-GPU downstream eval for grammar_v2 ${tag} + LLaDA: CDR-H1/H2/H3 10-fold + OAS holdout500 light pairing (prompt3, n=8) + ImmunoMatch metrics. Waits for training checkpoint."

Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250601/251105016"
    VepfsId: "vepfs-cnbj2c98dea54433"
    SubPath: "c20250601/251105016"

Entrypoint: |
  set -euo pipefail
  cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
  source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
  export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
  export OMP_NUM_THREADS=8
  export CUDA_VISIBLE_DEVICES=0

  OUT_DIR=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_${variant}
  CKPT=""
  for attempt in \$(seq 1 2880); do
    if [ -f "\${OUT_DIR}/best.pt" ]; then CKPT="\${OUT_DIR}/best.pt"; break; fi
    if [ -f "\${OUT_DIR}/latest.pt" ]; then CKPT="\${OUT_DIR}/latest.pt"; break; fi
    echo "\$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for checkpoint in \${OUT_DIR} (attempt \${attempt})"
    sleep 60
  done
  if [ -z "\${CKPT}" ]; then echo "timeout waiting for checkpoint in \${OUT_DIR}"; exit 1; fi

  mkdir -p /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation
  LOG=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_grammar_v2_${variant}_volc.log
  echo "=== eval start \$(date -u +%Y-%m-%dT%H:%M:%SZ) ckpt=\${CKPT} ===" | tee "\${LOG}"
  bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh ${variant} "\${CKPT}" 2>&1 | tee -a "\${LOG}"
  echo "=== eval done \$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "\${LOG}"

Tags: [bioseq, grammar_v2, llada, downstream, eval, ${tag}, cmp500k, cdr, light_pairing, 1gpu]
ImageUrl: "cr-mlp-cn-beijing.cr.volces.com/public/airgen:v1"
Framework: "Custom"
ResourceQueueName: "c20250601"
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 1
    Flavor: "ml.pni2.3xlarge"
ActiveDeadlineSeconds: 259200
DelayExitTimeSeconds: 0
AccessType: "Private"
Preemptible: true
Priority: 6
EOF
  echo "wrote ${out}"
}

write_pairing_metrics_yml() {
  local variant="$1"
  local tag="$2"
  local out="${JOBS}/eval_grammar_v2_${variant}_pairing_metrics.yml"
  cat > "${out}" <<EOF
TaskName: "eval_grammar_v2_${variant}_pairing"
Description: "ImmunoMatch + ANARCI metrics only for saved LLaDA ${tag} light-pairing CSV (CDR already done)."

Storages:
  - Type: "Vepfs"
    MountPath: "/vepfs-mlp2/c20250601/251105016"
    VepfsId: "vepfs-cnbj2c98dea54433"
    SubPath: "c20250601/251105016"

Entrypoint: |
  set -euo pipefail
  cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
  source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
  export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
  export OMP_NUM_THREADS=8
  export CUDA_VISIBLE_DEVICES=0

  LOG=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_grammar_v2_${variant}_pairing_volc.log
  echo "=== pairing metrics start \$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "\${LOG}"
  bash scripts/downstream/run_grammar_v2_llada_pairing_metrics_only.sh ${variant} 2>&1 | tee -a "\${LOG}"
  echo "=== pairing metrics done \$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "\${LOG}"

Tags: [bioseq, grammar_v2, llada, downstream, eval, ${tag}, pairing, immunomatch, 1gpu]
ImageUrl: "cr-mlp-cn-beijing.cr.volces.com/public/airgen:v1"
Framework: "Custom"
ResourceQueueName: "c20250601"
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 1
    Flavor: "ml.pni2.3xlarge"
ActiveDeadlineSeconds: 86400
DelayExitTimeSeconds: 0
AccessType: "Private"
Preemptible: true
Priority: 6
EOF
  echo "wrote ${out}"
}

write_downstream_yml "esmc300m_cmp500k_llada" "esmc300m"
write_downstream_yml "esmc600m_cmp500k_llada" "esmc600m"
write_pairing_metrics_yml "esmc300m_cmp500k_llada" "300M"
write_pairing_metrics_yml "esmc600m_cmp500k_llada" "600M"
