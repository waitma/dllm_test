#!/usr/bin/env bash
# Run one frozen-probe audit on a scheduler-allocated GPU; first argument is task.
set -euo pipefail
source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
conda activate /project_c20250601/251105016/conda/envs/protenix_abtcr
export PYTHONPATH=/project_c20250601/251105016/project/dllm_test
export LD_LIBRARY_PATH=/project_c20250601/251105016/conda/envs/protenix_abtcr/lib:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 USE_TF=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/project_c20250601/251105016/conda/cache/huggingface
EVAL_ASSIGNED_GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${EVAL_ASSIGNED_GPU%%,*}"
cd /project_c20250601/251105016/project/dllm_test
python -B scripts/downstream/audit_ophiuchus_probes.py --out /project_c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_probe_audit_20261026 --task "$1" --device cuda 2>&1 | tee /project_c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_probe_audit_20261026/"$1".log
