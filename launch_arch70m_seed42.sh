#!/usr/bin/env bash
# Run: bash /absolute/path/launch_arch70m_seed42.sh --gpu 0
set -euo pipefail
ARCH_SEED42_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/project-storage/fset-rN4kfh/251105016/conda/envs/pllm
export PYTHONPATH="$ARCH_SEED42_ROOT" USE_TF=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/arch70m_seed42_mpl
unset ARCH70M_FROZEN_SOURCE
cd "$ARCH_SEED42_ROOT"
exec "$ARCH_SEED42_ROOT/output/arch70m_env/bin/python" -B -m architecture_seed_repeat.runner "$@"
