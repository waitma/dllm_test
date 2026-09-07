#!/usr/bin/env bash
# Local replacement for the three stopped queue012 repr tasks (T1/T2/T3 only):
#   t-20260907044259-qd52v  eval-v3-allchains-8gpu2m-151000-repr
#   t-20260907044314-mtzn9  eval-v3-genonly-8gpu2m-44000-repr
#   t-20260907044328-wwfjn  eval-v3-bert-1m-105000-repr
# Env setup mirrors the yaml Entrypoint; all three share the single local GPU.
set -euo pipefail

ROOT=/vepfs-mlp2/c20250601/251105016/project/dllm_test
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
source activate "${ENV}"
export PYTHONPATH="${ROOT}"
export LD_LIBRARY_PATH=${ENV}/lib:${LD_LIBRARY_PATH:-}
export PATH=${ENV}/bin:${PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/vepfs-mlp2/c20250601/251105016/conda/cache/huggingface
cd "${ROOT}"

DRIVER_LOG_DIR="${ROOT}/output/downstream_generation"
mkdir -p "${DRIVER_LOG_DIR}"

run_one() {
  local ckpt="$1" tag="$2"
  test -f "${ckpt}/model.safetensors"
  local rc=0
  bash scripts/downstream/run_immune_fusion_repr.sh "${ckpt}" "${tag}" 16 \
    > "${DRIVER_LOG_DIR}/local_${tag}.driver.log" 2>&1 || rc=$?
  echo "[$(date -u +%FT%TZ)] ${tag} exit=${rc}"
}

run_one "${ROOT}/output/protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m/eval_snapshot_151000" \
        ours_fusion_v3_allchains_8gpu2m_151000 &
run_one "${ROOT}/output/protein_esmc_llada270m_diffusion_immune_v3_8gpu_2m/eval_snapshot_44000" \
        ours_fusion_v3_genonly_8gpu2m_44000 &
run_one "${ROOT}/output/protein_esmc_llada270m_bert_immune_v3_1m/eval_snapshot_105000" \
        ours_fusion_v3_bert_1m_105000 &

wait
echo "[$(date -u +%FT%TZ)] all three local repr runs finished"
