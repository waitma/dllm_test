#!/usr/bin/env bash
# Eight independent single-GPU runs, seed 43 only.
# Run: bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_8gpu.sh --gpus 0,1,2,3,4,5,6,7
set -euo pipefail
ARCH_LAUNCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/conda/etc/profile.d/conda.sh
conda activate "${ARCH_CONDA_PREFIX:-/mnt/project-storage/fset-rN4kfh/251105016/conda/envs/pllm}"
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 GIT_TERMINAL_PROMPT=0
export ARCH70M_PROJECT_ROOT="$ARCH_LAUNCH_ROOT"
unset ARCH70M_FROZEN_SOURCE
export PYTHONPATH="$ARCH_LAUNCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
ARCH_PREVIEW=0
for ARCH_ARG in "$@"; do
    case "$ARCH_ARG" in --dry-run|--help|-h) ARCH_PREVIEW=1 ;; esac
done
if (( ! ARCH_PREVIEW )) && [[ ! -f "$ARCH_LAUNCH_ROOT/output/arch70m_env/.ready" ]]; then
    bash "$ARCH_LAUNCH_ROOT/experiments/arch70m/setup.sh"
fi
ARCH_PYTHON="${ARCH_PYTHON:-$ARCH_LAUNCH_ROOT/output/arch70m_env/bin/python}"
exec "$ARCH_PYTHON" -B "$ARCH_LAUNCH_ROOT/arch70m_seed_validation.py" "$@"
