#!/usr/bin/env bash
# Run grammar-v1 downstream eval (CDR H1/H2/H3 10-fold + light pairing 500) for one variant.
#
# Usage:
#   bash scripts/downstream/run_grammar_variant_downstream_eval.sh esmc300m
#   bash scripts/downstream/run_grammar_variant_downstream_eval.sh esmc600m
#   bash scripts/downstream/run_grammar_variant_downstream_eval.sh no_encoder_38m

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
VARIANT="${1:?usage: $0 <esmc300m|esmc600m|no_encoder_38m>}"
OUT_DIR="${ROOT}/output/downstream_generation"
CKPT="${ROOT}/output/grammar_v1_${VARIANT}/latest.pt"
PREFIX="grammar_v1_${VARIANT}"
HOLDOUT_CSV="${ROOT}/data/downstream/comp_chain/test_data_oas_holdout.csv"
PAIR_OUT="${OUT_DIR}/${PREFIX}_light_pairing_holdout500_prompt3"
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
    --device cuda \
    --heavy-batch-size 4 \
    --num-seqs 8 \
    --light-prompt-tokens 3 \
    --sampling-strategy gumbel_argmax \
    --max-iter 32 \
    --metrics-json "${PAIR_OUT}_metrics.json" \
    > "${log}" 2>&1
  local exit_code=$?
  echo "EXIT=${exit_code}" >> "${log}"
  if [[ -f "${PAIR_OUT}_metrics.json" ]]; then
    python - <<PY | tee -a "${SUMMARY}"
import json
from pathlib import Path
m = json.loads(Path("${PAIR_OUT}_metrics.json").read_text())
keys = [
    "gen_immunomatch_mean", "gen_better_than_ref_pct", "gen_chain_consistency_pct",
    "gen_diversity", "gen_v_gene_match_pct",
]
for k in keys:
    if k in m:
        print(f"{k}={m[k]}")
PY
  fi
  echo "$(date -Is) finished light pairing exit=${exit_code}" | tee -a "${SUMMARY}"
  return "${exit_code}"
}

cd "${ROOT}"
for mode in cdrh1 cdrh2 cdrh3; do
  run_cdr "${mode}"
done
run_pairing

echo "$(date -Is) all downstream eval done for ${VARIANT}" | tee -a "${SUMMARY}"
