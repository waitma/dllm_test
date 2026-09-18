#!/usr/bin/env bash
# GPU worker: real-weight feedback gate, full pairing, then existing IM/ANARCI scorer.
# Usage: bash scripts/downstream/run_pairing_encoder_sync.sh 0  (or 3)
set -euo pipefail
PAIR_PROMPT="${1:?expected prompt length 0 or 3}"
case "${PAIR_PROMPT}" in 0|3) ;; *) exit 2 ;; esac
PAIR_ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
PAIR_CKPT="${PAIR_ROOT}/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913"
PAIR_INPUT="${PAIR_ROOT}/data/downstream/comp_chain/test_data_oas_holdout.csv"
PAIR_OUT="${PAIR_ROOT}/output/downstream_generation/ab_v5_49000_pairing_encoder_sync_20260916/pair-p${PAIR_PROMPT}-cfg0"
cd "${PAIR_ROOT}"
test -f "${PAIR_CKPT}/model.safetensors"
mkdir -p "${PAIR_OUT}"
python -m pytest -q -p no:cacheprovider scripts/tests/bioseq/test_pairing_generation_protocol.py scripts/tests/bioseq/test_sampling_bioseq.py
python -u scripts/downstream/preflight_pairing_encoder_feedback.py \
  --checkpoint "${PAIR_CKPT}" --csv "${PAIR_INPUT}" --prompt "${PAIR_PROMPT}" \
  --output "${PAIR_OUT}/encoder_feedback_gate.json" \
  2>&1 | tee "${PAIR_OUT}/encoder_feedback_gate.log"
python -u -m downstream.grammar.light_chain_pairing \
  --checkpoint-path "${PAIR_CKPT}" --csv-path "${PAIR_INPUT}" \
  --output-csv "${PAIR_OUT}/holdout500.csv" --device cuda --heavy-batch-size 2 \
  --num-seqs 8 --light-prompt-tokens "${PAIR_PROMPT}" --cfg-scale 0 \
  --light-length-mode reference --max-iter 124 --sampling-strategy gumbel_argmax \
  --temperature 1 --seed 42 --skip-eval \
  2>&1 | tee "${PAIR_OUT}/holdout500_generation.log"
python -u scripts/downstream/score_pairing_generation.py \
  --csv "${PAIR_OUT}/holdout500_n8.csv" --metrics "${PAIR_OUT}/holdout500_metrics.json" \
  --num-seqs 8 2>&1 | tee "${PAIR_OUT}/holdout500_scoring.log"
