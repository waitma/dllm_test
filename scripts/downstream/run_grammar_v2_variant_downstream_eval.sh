#!/usr/bin/env bash
# Run grammar-v2 downstream eval (CDR H1/H2/H3 10-fold + light pairing 500) for one variant.
#
# Usage:
#   bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh esmc300m
#   bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh esmc600m
#   bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh no_encoder_qwen0_6b
#   bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh esm2_650m
#   bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh esmc300m_fixalign
#   bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh esmc600m_fixalign
#   bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh no_encoder_1b_cmp500k
#   bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh esmc300m_cmp500k
#   bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh esmc600m_cmp500k
#   bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh esm2_650m_cmp500k
#   bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh esmc300m_cmp500k_llada
#   bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh esmc600m_cmp500k_llada

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
VARIANT="${1:?usage: $0 <esmc300m|esmc600m|no_encoder_qwen0_6b|esm2_650m|esmc300m_fixalign|esmc600m_fixalign|no_encoder_1b_cmp500k|esmc300m_cmp500k|esmc600m_cmp500k|esm2_650m_cmp500k|esmc300m_cmp500k_llada|esmc600m_cmp500k_llada>}"
OUT_DIR="${ROOT}/output/downstream_generation"
if [[ "${VARIANT}" == "esm2_650m" ]]; then
  CKPT="${ROOT}/output/grammar_v2_esm2_650m/best.pt"
elif [[ "${VARIANT}" == "esmc300m_fixalign" ]]; then
  CKPT="${ROOT}/output/grammar_v2_esmc300m_fixalign/best.pt"
elif [[ "${VARIANT}" == "esmc600m_fixalign" ]]; then
  CKPT="${ROOT}/output/grammar_v2_esmc600m_fixalign/best.pt"
elif [[ "${VARIANT}" == *_cmp500k_llada ]]; then
  OUT="${ROOT}/output/grammar_v2_${VARIANT}"
  if [[ -f "${OUT}/best.pt" ]]; then
    CKPT="${OUT}/best.pt"
  elif [[ -f "${OUT}/latest.pt" ]]; then
    CKPT="${OUT}/latest.pt"
  else
    CKPT="${OUT}/best.pt"
  fi
elif [[ "${VARIANT}" == *_cmp500k ]]; then
  CKPT="${ROOT}/output/grammar_v2_${VARIANT}/best.pt"
else
  CKPT="${ROOT}/output/grammar_v2_${VARIANT}/latest.pt"
fi
if [[ "${2:-}" != "" ]]; then
  CKPT="${2}"
fi
PREFIX="grammar_v2_${VARIANT}"
HOLDOUT_CSV="${ROOT}/data/downstream/comp_chain/test_data_oas_holdout.csv"
PAIR_OUT="${OUT_DIR}/${PREFIX}_light_pairing_holdout500_prompt3"
HUMAN_OUT="${OUT_DIR}/${PREFIX}_humanization"
SUMMARY="${OUT_DIR}/${PREFIX}_downstream_summary.txt"

if [[ ! -f "${CKPT}" ]]; then
  echo "checkpoint missing: ${CKPT}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
: > "${SUMMARY}"
echo "$(date -Is) variant=${VARIANT} ckpt=${CKPT}" | tee -a "${SUMMARY}"

run_cdr() {
  local mode="$1"
  mkdir -p "${OUT_DIR}"
  local log="${OUT_DIR}/${PREFIX}_${mode}_10fold.log"
  echo "$(date -Is) starting CDR ${mode} -> ${log}" | tee -a "${SUMMARY}"
  python -u -m downstream.grammar.cdr_infill \
    --test-set "${ROOT}/data/downstream/cdr_infilling/sabdab/${mode}" \
    --mode "${mode}" \
    --checkpoint-path "${CKPT}" \
    --device cuda \
    --num-folds 10 \
    --sampling-strategy argmax \
    --max-iter 4 \
    > "${log}" 2>&1
  local exit_code=$?
  echo "EXIT=${exit_code}" >> "${log}"
  grep -E "Average AAR|Standard deviation" "${log}" | tee -a "${SUMMARY}" || true
  echo "$(date -Is) finished CDR ${mode} exit=${exit_code}" | tee -a "${SUMMARY}"
  return "${exit_code}"
}

run_pairing() {
  mkdir -p "${OUT_DIR}"
  local log="${PAIR_OUT}.log"
  echo "$(date -Is) starting light pairing 500 -> ${log}" | tee -a "${SUMMARY}"
  python -u -m downstream.grammar.light_chain_pairing \
    --csv-path "${HOLDOUT_CSV}" \
    --checkpoint-path "${CKPT}" \
    --output-csv "${PAIR_OUT}.csv" \
    --metrics-json "${PAIR_OUT}_metrics.json" \
    --device cuda \
    --heavy-batch-size 4 \
    --num-seqs 8 \
    --light-prompt-tokens 3 \
    --sampling-strategy gumbel_argmax \
    --max-iter 32 \
    > "${log}" 2>&1
  local exit_code=$?
  echo "EXIT=${exit_code}" >> "${log}"
  echo "$(date -Is) finished light pairing exit=${exit_code}" | tee -a "${SUMMARY}"
  return "${exit_code}"
}

run_humanization() {
  mkdir -p "${OUT_DIR}"
  local log="${HUMAN_OUT}_n8.log"
  echo "$(date -Is) starting humanization 28-pair -> ${log}" | tee -a "${SUMMARY}"
  python -u -m downstream.grammar.humanization \
    --checkpoint-path "${CKPT}" \
    --output-csv "${HUMAN_OUT}.csv" \
    --metrics-json "${HUMAN_OUT}_n8_metrics.json" \
    --device cuda \
    --num-seqs 8 \
    --sampling-strategy gumbel_argmax \
    --max-iter 32 \
    > "${log}" 2>&1
  local exit_code=$?
  echo "EXIT=${exit_code}" >> "${log}"
  grep -E "Overall heavy FR AAR|Saved humanization" "${log}" | tee -a "${SUMMARY}" || true
  echo "$(date -Is) finished humanization exit=${exit_code}" | tee -a "${SUMMARY}"
  return "${exit_code}"
}

cd "${ROOT}"
for mode in cdrh1 cdrh2 cdrh3; do
  run_cdr "${mode}"
done
run_pairing
if [[ "${SKIP_HUMANIZATION:-0}" == "1" ]]; then
  echo "$(date -Is) SKIP_HUMANIZATION=1 — skipping humanization" | tee -a "${SUMMARY}"
else
  run_humanization
fi

echo "$(date -Is) all downstream eval done for ${VARIANT}" | tee -a "${SUMMARY}"
