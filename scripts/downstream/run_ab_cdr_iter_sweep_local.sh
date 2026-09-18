#!/usr/bin/env bash
# User-authorized local SAb23H2 iteration diagnostic. Does not touch other processes.
# Run: bash scripts/downstream/run_ab_cdr_iter_sweep_local.sh
set -euo pipefail
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate /vepfs-mlp2/c20250601/251105016/conda/envs/pllm
conda activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 USE_TF=0
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/vepfs-mlp2/c20250601/251105016/conda/cache/huggingface
export LD_LIBRARY_PATH=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/lib:${LD_LIBRARY_PATH:-}
CDR_ROOT=/vepfs-mlp2/c20250601/251105016/project/dllm_test
CDR_OUT="${CDR_ROOT}/output/downstream_generation/ab_v5_92000_cdr_iter_20260916"
cd "${CDR_ROOT}"
test ! -e "${CDR_OUT}"
python -u scripts/downstream/run_ab_cdr_iter_sweep.py \
  --checkpoint "${CDR_ROOT}/output/ab_eval_checkpoints/ab_v5_92000_llada_20260915" \
  --out-dir "${CDR_OUT}" --iterations 1 2 4 8 \
  --reference-logs "${CDR_ROOT}/output/downstream_generation/ab_v5_92000_full_20260915/cdr-sab23" \
  2>&1 | tee "${CDR_OUT}.log"
