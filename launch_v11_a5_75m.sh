#!/usr/bin/env bash
# V11 A5 75M, full V8 data, same 1B training settings. Run this absolute script.
set -euo pipefail
umask 077
V11_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
V11_PY="${V11_PYTHON:-/mnt/project-storage/fset-rN4kfh/251105016/conda/envs/pllm/bin/python}"
V11_RUN="$V11_ROOT/output/v11_a5_75m_300k_20261105"
V11_MODE=train
V11_RESUME=""
while (($#)); do
    case "$1" in
        --run-root|--resume)
            (($# >= 2)) || { echo "Missing argument: $1" >&2; exit 2; }
            if [[ "$1" == --run-root ]]; then V11_RUN="$2"; else V11_RESUME="$2"; fi
            shift 2 ;;
        --verify-only) V11_MODE=verify; shift ;;
        --dry-run) V11_MODE=dry; shift ;;
        --help|-h)
            echo 'Usage: launch_v11_a5_75m.sh [--verify-only | --dry-run] [--run-root ABS] [--resume ABS_V11_CHECKPOINT]'
            echo 'Default: one node, 8 GPUs, micro16 x accumulation4 x 8 = global512, 300000 steps.'
            echo 'A fresh run refuses an existing train directory; resume is explicit. No job submission is performed.'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$V11_RUN" == /* ]] || { echo '--run-root must be absolute' >&2; exit 2; }
[[ -z "$V11_RESUME" || "$V11_RESUME" == /* ]] || { echo '--resume must be absolute' >&2; exit 2; }
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 IMMUNE_SDPA_BACKEND=efficient
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_X_DISABLE_VIEWER=true
unset PYTHONPATH PYTHONHOME
export PATH="$(dirname -- "$V11_PY"):$PATH"
export CONDA_PREFIX="$(dirname -- "$(dirname -- "$V11_PY")")"
"$V11_PY" -I -B "$V11_ROOT/v11_a5/prepare.py" --run-root "$V11_RUN"
V11_BUNDLE="$V11_RUN/bundle"
V11_FROZEN="$V11_BUNDLE/source/v11_a5/launch.py"
V11_COMMAND=("$V11_PY" -I -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max_restarts=0 --no_python
    "$V11_PY" -I -B "$V11_FROZEN" --bundle "$V11_BUNDLE")
[[ -z "$V11_RESUME" ]] || V11_COMMAND+=(--resume "$V11_RESUME")
if [[ "$V11_MODE" == dry ]]; then
    printf '%q ' "${V11_COMMAND[@]}"; printf '\n'; exit 0
fi
if [[ "$V11_MODE" == verify ]]; then
    exec "$V11_PY" -I -B "$V11_FROZEN" --bundle "$V11_BUNDLE" --verify-only
fi
exec 9>"$V11_RUN/launch.lock"
flock -n 9 || { echo 'This V11 run is already active.' >&2; exit 2; }
if [[ -e "$V11_RUN/train" && -z "$V11_RESUME" ]]; then
    echo 'Training output exists: select its V11 checkpoint with --resume; refusing overwrite.' >&2; exit 2
fi
if [[ ! -e "$V11_RUN/train" && -n "$V11_RESUME" ]]; then
    echo 'Resume output is missing.' >&2; exit 2
fi
export NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET=Socket
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_DEBUG=WARN NCCL_VERSION=2.31.2
[[ -z "${LD_PRELOAD:-}" ]] || { echo 'V11 requires LD_PRELOAD to be initially empty.' >&2; exit 2; }
V11_LIBRARY="$V11_BUNDLE/runtime/libnccl.so.2"
export LD_PRELOAD="$V11_LIBRARY"
"$V11_PY" -I -B - "$V11_LIBRARY" <<'NCCL'
import ctypes,json,sys
from pathlib import Path
import torch
expected=Path(sys.argv[1]).resolve()
loaded={Path(x.split()[-1]).resolve() for x in Path('/proc/self/maps').read_text().splitlines() if '/libnccl.so' in x}
v=ctypes.c_int();code=ctypes.CDLL(str(expected)).ncclGetVersion(ctypes.byref(v))
if code or v.value!=23102 or loaded!={expected}:raise RuntimeError('V11 NCCL runtime differs')
print(json.dumps(dict(stage='V11_nccl_verified',runtime=v.value)),flush=True)
NCCL
exec > >(tee -a "$V11_RUN/entrypoint.log") 2>&1
exec "${V11_COMMAND[@]}"
