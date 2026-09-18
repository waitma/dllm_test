#!/usr/bin/env bash
# Light-chain pairing only (diffusion ckpts), known-reference-length by default.
# Reference length is an explicit task condition (user decision 2026-09-13).
# Only the first three light residues are visible; the remainder is generated.
# Use prior explicitly for the separate training-length-prior diagnostic.
#
# Usage: bash run_immune_fusion_pairing.sh <fusion_ckpt_dir> <tag> [heavy_batch] [length_mode]
set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
CKPT="${1:?usage: $0 <fusion_ckpt_dir> <tag> [heavy_batch] [length_mode]}"
TAG="${2:?usage: $0 <fusion_ckpt_dir> <tag> [heavy_batch] [length_mode]}"
HEAVY_BS="${3:-2}"
LENGTH_MODE="${4:-reference}"
# AirGen run/heavy2light.sh uses 124; this does not establish the paper's exact run.
MAX_ITER="${PAIR_MAX_ITER:-124}"

OUT="${ROOT}/output/downstream_generation"
PREFIX="${OUT}/${TAG}_light_pairing_holdout500_prompt3_len${LENGTH_MODE}_iter${MAX_ITER}"
LOG="${OUT}/eval_${TAG}_pairing_len${LENGTH_MODE}_iter${MAX_ITER}.log"
HOLDOUT_CSV="${ROOT}/data/downstream/comp_chain/test_data_oas_holdout.csv"

test -f "${CKPT}/model.safetensors"
mkdir -p "${OUT}"
cd "${ROOT}"

echo "=== fusion pairing ${TAG} length_mode=${LENGTH_MODE} start $(date -u +%Y-%m-%dT%H:%M:%SZ) ckpt=${CKPT} ===" | tee "${LOG}"

python -u -m downstream.grammar.light_chain_pairing \
  --csv-path "${HOLDOUT_CSV}" \
  --checkpoint-path "${CKPT}" \
  --output-csv "${PREFIX}.csv" \
  --metrics-json "${PREFIX}_metrics.json" \
  --device cuda \
  --heavy-batch-size "${HEAVY_BS}" \
  --num-seqs 8 \
  --light-prompt-tokens 3 \
  --light-length-mode "${LENGTH_MODE}" \
  --sampling-strategy gumbel_argmax \
  --max-iter "${MAX_ITER}" \
  --seed 42 \
  2>&1 | tee -a "${LOG}"

echo "$(date -Is) pairing leakage diagnostic" | tee -a "${LOG}"
python -u scripts/downstream/pairing_leakage_diagnostic.py \
  --csv "${PREFIX}_n8.csv" \
  --out-json "${PREFIX}_leakage_diagnostic.json" \
  2>&1 | tee -a "${LOG}"

echo "=== fusion pairing ${TAG} done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "${LOG}"
