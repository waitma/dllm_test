#!/usr/bin/env bash
# Light-chain pairing only (diffusion ckpts), with the target-length-leakage fix.
#
# The previous round built the grammar record from the reference light chain, so the
# number of masked residue slots equalled the reference length. That leaked the answer's
# length and turned pairing into near-reconstruction (100% length match, ~0.95 identity,
# diversity 0.10 vs the paper's 0.335). light_length_mode=prior draws the length from the
# OAS-train histogram instead, which is the grammar-side analogue of the official AirGen
# protocol (fixed 128-slot light buffer, model emits its own <eos>).
#
# Usage: bash run_immune_fusion_pairing.sh <fusion_ckpt_dir> <tag> [heavy_batch] [length_mode]
set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
CKPT="${1:?usage: $0 <fusion_ckpt_dir> <tag> [heavy_batch] [length_mode]}"
TAG="${2:?usage: $0 <fusion_ckpt_dir> <tag> [heavy_batch] [length_mode]}"
HEAVY_BS="${3:-2}"
LENGTH_MODE="${4:-prior}"
# AirGen run/zero_shot_test.sh passes --max_iter 124 for light pairing (the argparse
# default 32 is not what the paper used).
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
