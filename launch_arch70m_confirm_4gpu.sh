#!/usr/bin/env bash
# Run: bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_confirm_4gpu.sh --gpus 0,1,2,3
set -euo pipefail
ARCH_CONFIRM4_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/conda/etc/profile.d/conda.sh
conda activate "${ARCH_CONDA_PREFIX:-/mnt/project-storage/fset-rN4kfh/251105016/conda/envs/pllm}"
export PYTHONPATH="$ARCH_CONFIRM4_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ARCH70M_PROJECT_ROOT="$ARCH_CONFIRM4_ROOT"
export USE_TF=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 GIT_TERMINAL_PROMPT=0
unset ARCH70M_FROZEN_SOURCE
cd "$ARCH_CONFIRM4_ROOT"
exec "${ARCH_PYTHON:-$ARCH_CONFIRM4_ROOT/output/arch70m_env/bin/python}" -B "$ARCH_CONFIRM4_ROOT/arch70m_confirm_4gpu.py" "$@"
