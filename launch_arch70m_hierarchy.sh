#!/usr/bin/env bash
# Run: bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_hierarchy.sh --gpu 0
set -euo pipefail
ARCH_HIERARCHY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/conda/etc/profile.d/conda.sh
conda activate "${ARCH_CONDA_PREFIX:-/mnt/project-storage/fset-rN4kfh/251105016/conda/envs/pllm}"
export PYTHONPATH="$ARCH_HIERARCHY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 GIT_TERMINAL_PROMPT=0
unset ARCH70M_FROZEN_SOURCE
cd "$ARCH_HIERARCHY_ROOT"
exec "$ARCH_HIERARCHY_ROOT/output/arch70m_env/bin/python" -B -m architecture_hierarchy.runner "$@"
