#!/usr/bin/env bash
# bash /absolute/worktree/experiments/arch70m/run.sh --stage screen --gpus 0,1,2,3
set -euo pipefail
ARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/conda/etc/profile.d/conda.sh
conda activate "${ARCH_CONDA_PREFIX:-/mnt/project-storage/fset-rN4kfh/251105016/conda/envs/pllm}"
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$ARCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ARCH_ROOT"
if [[ ! -f "$ARCH_ROOT/output/arch70m_env/.ready" ]]; then
    bash "$ARCH_ROOT/experiments/arch70m/setup.sh"
fi
exec "$ARCH_ROOT/output/arch70m_env/bin/python" -B -m experiments.arch70m.suite "$@"
