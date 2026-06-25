#!/usr/bin/env bash
# Wait for GPU, then run grammar_v1 SAbDab CDR-H1/H2/H3 10-fold eval.
# Usage: bash scripts/downstream/run_grammar_cdr_heavy_10fold.sh [cdrh1|cdrh2|cdrh3|all]

set -euo pipefail
ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
CKPT="${ROOT}/output/grammar_v1_esmc300m/latest.pt"
OUT_DIR="${ROOT}/output/downstream_generation"
MODES=("$@")
if [[ ${#MODES[@]} -eq 0 ]]; then
  MODES=(cdrh1 cdrh2)
fi

while pgrep -f "downstream.grammar.light_chain_pairing.*holdout500" >/dev/null 2>&1; do
  echo "$(date -Is) waiting for light_chain_pairing to release GPU..."
  sleep 120
done

for mode in "${MODES[@]}"; do
  log="${OUT_DIR}/grammar_v1_esmc300m_${mode}_10fold.log"
  echo "$(date -Is) starting ${mode} -> ${log}"
  conda run --no-capture-output -n protenix_abtcr python -u -m downstream.grammar.cdr_infill \
    --test-set "${ROOT}/data/downstream/cdr_infilling/sabdab/${mode}" \
    --mode "${mode}" \
    --checkpoint-path "${CKPT}" \
    --device cuda \
    --num-folds 10 \
    --sampling-strategy argmax \
    --max-iter 4 \
    > "${log}" 2>&1
  echo "EXIT=$?" >> "${log}"
  echo "$(date -Is) finished ${mode}"
done
