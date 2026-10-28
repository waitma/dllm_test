#!/usr/bin/env bash
# Create an exploration-only environment; never install into the active pllm environment.
set -euo pipefail
ARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/conda/etc/profile.d/conda.sh
conda activate "${ARCH_CONDA_PREFIX:-/mnt/project-storage/fset-rN4kfh/251105016/conda/envs/pllm}"
mkdir -p "$ARCH_ROOT/output"
exec 9>"$ARCH_ROOT/output/environment.lock"
flock 9
if [[ ! -x "$ARCH_ROOT/output/arch70m_env/bin/python" ]]; then
    python -m venv --system-site-packages "$ARCH_ROOT/output/arch70m_env"
fi
"$ARCH_ROOT/output/arch70m_env/bin/python" -m pip install --disable-pip-version-check --index-url https://pypi.org/simple --timeout 120 --retries 2 'matplotlib==3.10.7'
"$ARCH_ROOT/output/arch70m_env/bin/python" -m pip freeze > "$ARCH_ROOT/output/arch70m_environment.txt"
touch "$ARCH_ROOT/output/arch70m_env/.ready"
