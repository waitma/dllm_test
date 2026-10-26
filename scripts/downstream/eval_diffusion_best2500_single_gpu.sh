#!/usr/bin/env bash
# One allocated GPU: bash /project_c20250601/251105016/project/dllm_test/scripts/downstream/eval_diffusion_best2500_single_gpu.sh
# CPU-only integrity check: append --verify-only. All 15 tasks execute serially.
set -euo pipefail

EVAL_ROOT=/project_c20250601/251105016/project/dllm_test
EVAL_RUN="$EVAL_ROOT/output/downstream_generation/diffusion_edit_v7_75m_best2500_single_gpu_20261022"
EVAL_ENV=/project_c20250601/251105016/conda/envs/protenix_abtcr
source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
conda activate "$EVAL_ENV"

export PYTHONPATH="$EVAL_ROOT"
export LD_LIBRARY_PATH="$EVAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 USE_TF=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/project_c20250601/251105016/conda/cache/huggingface
export IMMUNOMATCH_KAPPA_PATH="$EVAL_ROOT/data/downstream/immunomatch/immunomatch-kappa"
export IMMUNOMATCH_LAMBDA_PATH="$EVAL_ROOT/data/downstream/immunomatch/immunomatch-lambda"
# Preserve the scheduler's first assigned device, including a GPU UUID.
EVAL_VISIBLE_GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${EVAL_VISIBLE_GPU%%,*}"

cd "$EVAL_ROOT"
exec > >(tee -a "$EVAL_RUN/entrypoint.log") 2>&1
printf 'Checkpoint: diffusion/edit best2500\nOutput: %s\nVisible GPU: %s\n' "$EVAL_RUN" "$CUDA_VISIBLE_DEVICES"
exec "$EVAL_ENV/bin/python" -B "$EVAL_RUN/driver.py" --run "$EVAL_RUN" "$@"
