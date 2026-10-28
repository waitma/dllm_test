#!/usr/bin/env bash
# Four independent single-GPU trials. Run: bash /absolute/worktree/launch_arch70m_4gpu.sh
# Kept outside the frozen model/trainer source set so completed A0/A1 remain comparable.
set -euo pipefail
ARCH_LAUNCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARCH_STAGE=screen
ARCH_GPU_LIST="${ARCH_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
ARCH_DRY_RUN=0
ARCH_EXTRA_ARGS=()

usage() {
    cat <<'USAGE'
Usage: bash /absolute/worktree/launch_arch70m_4gpu.sh [options]

  --stage screen|confirm    Default: screen (A0-A7; matching completed trials skip).
  --gpus 0,1,2,3           Exactly four distinct GPU IDs or UUIDs.
                           Default: ARCH_GPUS, then CUDA_VISIBLE_DEVICES, then 0,1,2,3.
  --machine-index N        Forwarded to the suite for disjoint multi-machine trials.
  --machine-count N        Number of machines sharing the frozen experiment outputs.
  --dry-run                Print the command without checking GPUs or starting jobs.
  --help                   Show this help.

Other suite arguments (such as --output-root, --trials, --wandb, --no-push) are forwarded.
Each GPU runs its own architecture trial, including training and generation/probe evaluation.
Confirm requires all eight screening trials to have completed. Data format stays V8.
USAGE
}

while (($#)); do
    case "$1" in
        --stage|--gpus)
            if (($# < 2)); then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
            if [[ "$1" == --stage ]]; then ARCH_STAGE="$2"; else ARCH_GPU_LIST="$2"; fi
            shift 2 ;;
        --stage=*) ARCH_STAGE="${1#*=}"; shift ;;
        --gpus=*) ARCH_GPU_LIST="${1#*=}"; shift ;;
        --dry-run) ARCH_DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) ARCH_EXTRA_ARGS+=("$1"); shift ;;
    esac
done

case "$ARCH_STAGE" in
    screen|confirm) ;;
    *) printf 'Unsupported stage: %s (use screen or confirm)\n' "$ARCH_STAGE" >&2; exit 2 ;;
esac
IFS=',' read -r -a ARCH_GPU_IDS <<< "$ARCH_GPU_LIST"
if ((${#ARCH_GPU_IDS[@]} != 4)) || [[ "$ARCH_GPU_LIST" == *, ]]; then
    printf 'Exactly four GPU IDs are required; received: %s\n' "$ARCH_GPU_LIST" >&2
    exit 2
fi
declare -A ARCH_GPU_SEEN=()
for ARCH_GPU_ID in "${ARCH_GPU_IDS[@]}"; do
    if [[ -z "$ARCH_GPU_ID" || "$ARCH_GPU_ID" == *[[:space:]]* || -n "${ARCH_GPU_SEEN[$ARCH_GPU_ID]:-}" ]]; then
        printf 'GPU IDs must be distinct and nonempty: %s\n' "$ARCH_GPU_LIST" >&2
        exit 2
    fi
    ARCH_GPU_SEEN["$ARCH_GPU_ID"]=1
done

ARCH_COMMAND=(bash "$ARCH_LAUNCH_ROOT/experiments/arch70m/run.sh" --stage "$ARCH_STAGE" --gpus "$ARCH_GPU_LIST" "${ARCH_EXTRA_ARGS[@]}")
if ((ARCH_DRY_RUN)); then
    printf '%q ' "${ARCH_COMMAND[@]}"
    printf '\n'
    exit 0
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi is unavailable; run this script on a four-GPU worker.\n' >&2
    exit 2
fi
if ! ARCH_GPU_QUERY="$(nvidia-smi --id="$ARCH_GPU_LIST" --query-gpu=uuid --format=csv,noheader)"; then
    printf 'Cannot access all four requested GPUs: %s. No trials were launched.\n' "$ARCH_GPU_LIST" >&2
    exit 2
fi
# Some drivers silently omit nonexistent IDs while returning success.
mapfile -t ARCH_GPU_UUIDS <<< "$ARCH_GPU_QUERY"
if ((${#ARCH_GPU_UUIDS[@]} != 4)); then
    printf 'Only %s of four requested GPUs are visible. No trials were launched.\n' "${#ARCH_GPU_UUIDS[@]}" >&2
    exit 2
fi
declare -A ARCH_UUID_SEEN=()
for ARCH_GPU_UUID in "${ARCH_GPU_UUIDS[@]}"; do
    if [[ "$ARCH_GPU_UUID" != GPU-* || -n "${ARCH_UUID_SEEN[$ARCH_GPU_UUID]:-}" ]]; then
        printf 'Four distinct physical GPUs are required. No trials were launched.\n' >&2
        exit 2
    fi
    ARCH_UUID_SEEN["$ARCH_GPU_UUID"]=1
done
export GIT_TERMINAL_PROMPT=0
printf 'Four independent single-GPU trials: stage=%s GPUs=%s\n' "$ARCH_STAGE" "$ARCH_GPU_LIST"
exec "${ARCH_COMMAND[@]}"
