#!/usr/bin/env bash
# Light-chain pairing prompt-token sweep for a grammar_v2 checkpoint.
#
# Runs OAS holdout500 heavy->light generation + generation_eval scoring at two
# prompt settings on the SAME checkpoint so de-novo vs prompt3 is a fair
# apples-to-apples comparison (identical weights, only the light prompt differs):
#   * light-prompt-tokens=0  -> de-novo (no light seed; isolates whether the
#                               100% chain-match is purely a prompt3 artifact)
#   * light-prompt-tokens=3  -> prompt3 (same protocol as the headline table)
#
# Usage:
#   bash scripts/downstream/run_grammar_v2_pairing_prompt_sweep.sh <variant> [ckpt]
# e.g.
#   bash scripts/downstream/run_grammar_v2_pairing_prompt_sweep.sh esmc300m_cmp500k_llada
set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
VARIANT="${1:?usage: $0 <variant> [ckpt]}"
CKPT_ARG="${2:-}"
OUT_DIR="${ROOT}/output/downstream_generation"
HOLDOUT_CSV="${ROOT}/data/downstream/comp_chain/test_data_oas_holdout.csv"

if [[ -n "${CKPT_ARG}" ]]; then
  CKPT="${CKPT_ARG}"
else
  OUT="${ROOT}/output/grammar_v2_${VARIANT}"
  if [[ -f "${OUT}/best.pt" ]]; then CKPT="${OUT}/best.pt"; else CKPT="${OUT}/latest.pt"; fi
fi
if [[ ! -f "${CKPT}" ]]; then echo "checkpoint missing: ${CKPT}" >&2; exit 1; fi

PREFIX="grammar_v2_${VARIANT}"
SUMMARY="${OUT_DIR}/${PREFIX}_pairing_prompt_sweep_summary.txt"
mkdir -p "${OUT_DIR}"
: > "${SUMMARY}"
echo "$(date -Is) variant=${VARIANT} ckpt=${CKPT}" | tee -a "${SUMMARY}"

run_pairing() {
  local prompt_tokens="$1"
  local tag="$2"
  local pair_out="${OUT_DIR}/${PREFIX}_light_pairing_holdout500_${tag}"
  local log="${pair_out}.log"
  echo "$(date -Is) starting pairing ${tag} (prompt-tokens=${prompt_tokens}) -> ${log}" | tee -a "${SUMMARY}"
  python -u -m downstream.grammar.light_chain_pairing \
    --csv-path "${HOLDOUT_CSV}" \
    --checkpoint-path "${CKPT}" \
    --output-csv "${pair_out}.csv" \
    --metrics-json "${pair_out}_metrics.json" \
    --device cuda \
    --heavy-batch-size 4 \
    --num-seqs 8 \
    --light-prompt-tokens "${prompt_tokens}" \
    --sampling-strategy gumbel_argmax \
    --max-iter 32 \
    > "${log}" 2>&1
  local exit_code=$?
  echo "EXIT=${exit_code}" >> "${log}"
  grep -E "Gen Immunomatch Mean|Ref Immunomatch Mean|chain_match|Saved comp_chain" "${log}" | tee -a "${SUMMARY}" || true
  echo "$(date -Is) finished pairing ${tag} exit=${exit_code}" | tee -a "${SUMMARY}"
  return "${exit_code}"
}

cd "${ROOT}"
run_pairing 0 denovo
run_pairing 3 prompt3_sameckpt

echo "$(date -Is) pairing prompt sweep done for ${VARIANT}" | tee -a "${SUMMARY}"
