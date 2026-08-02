#!/usr/bin/env bash
# Generate grammar-v2 cmp500k downstream eval Volc YAMLs.
#
# Usage:
#   bash scripts/downstream/gen_eval_grammar_v2_cmp500k_ymls.sh

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
JOBS="${ROOT}/eval_jobs"

write_yml() {
  local variant="$1"
  local tag="$2"
  local out="${JOBS}/eval_grammar_v2_${variant}_downstream.yml"
  cat > "${out}" <<EOF
TaskName: "eval_grammar_v2_${variant}_downstream"
Description: "Single-GPU downstream eval for grammar_v2 ${tag}: CDR-H1/H2/H3 10-fold + OAS holdout500 light pairing (prompt3, n=8). Uses best.pt."

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

  CKPT=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_${variant}/best.pt
  test -f "\${CKPT}"
  mkdir -p /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation

  LOG=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_grammar_v2_${variant}_volc.log
  echo "=== eval start \$(date -u +%Y-%m-%dT%H:%M:%SZ) ckpt=\${CKPT} ===" | tee "\${LOG}"
  bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh ${variant} 2>&1 | tee -a "\${LOG}"
  echo "=== eval done \$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "\${LOG}"

Tags: [bioseq, grammar_v2, downstream, eval, ${tag}, cmp500k, cdr, light_pairing, 1gpu]
ImageUrl: "cr-mlp-cn-beijing.cr.volces.com/public/airgen:v1"
Framework: "Custom"
ResourceQueueName: "c20250601"
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 1
    Flavor: "ml.pni2.3xlarge"
ActiveDeadlineSeconds: 172800
DelayExitTimeSeconds: 0
AccessType: "Private"
Preemptible: false
Priority: 6
EOF
  echo "wrote ${out}"
}

write_yml "no_encoder_1b_cmp500k" "no_encoder_1b"
write_yml "esmc300m_cmp500k" "esmc300m"
write_yml "esmc600m_cmp500k" "esmc600m"
write_yml "esm2_650m_cmp500k" "esm2_650m"
