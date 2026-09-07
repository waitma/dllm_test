#!/usr/bin/env bash
# Diffusion-only generation: AB CDR (SAbDab + SAb23H2) + OAS light pairing + T4 Setting B.
# Usage: bash run_immune_fusion_gen.sh <ckpt_dir> <tag> [heavy_batch] [t4_batch]
#
# GEN_STAGES selects which stages run (comma-separated: cdr,pairing,t4; default all).
# The three stages write disjoint outputs, so splitting them across parallel one-GPU
# jobs turns a ~3h40m serial run into ~1h45m wall clock (measured: CDR 79 min,
# pairing 106 min, T4 28 min). Pairing in particular should not wait behind CDR.
set -euo pipefail
GEN_STAGES="${GEN_STAGES:-cdr,pairing,t4}"
stage_enabled() { [[ ",${GEN_STAGES}," == *",$1,"* ]]; }

# Decoding-step budgets. AirGen's run/zero_shot_test.sh (the settings the paper actually
# used, which differ from the scripts' argparse defaults) passes --max_iter 2 for CDR
# infilling and --max_iter 124 for light pairing. AAR is a per-position metric, so extra
# iterations hurt it: each committed token conditions the rest and a wrong commit
# propagates. Measured on the official Ophiuchus ckpt, SAbDab CDR-H3 falls monotonically
# 42.00 -> 41.43 -> 40.96 -> 40.70 as max_iter goes 1 -> 2 -> 4 -> 8, and SAb23H2 at
# max_iter=2 reproduces the paper to within 0.10 pp. Outputs are tagged with the value so
# runs at different budgets never overwrite each other.
CDR_MAX_ITER="${CDR_MAX_ITER:-2}"
PAIR_MAX_ITER="${PAIR_MAX_ITER:-124}"

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
CKPT="${1:?usage: $0 <fusion_ckpt_dir> <tag>}"
TAG="${2:?usage: $0 <fusion_ckpt_dir> <tag>}"
OUT="${ROOT}/output/downstream_generation"
# Per-stage log so parallel stage jobs never clobber each other's log.
if [[ "${GEN_STAGES}" == "cdr,pairing,t4" ]]; then
  LOG="${OUT}/eval_${TAG}_gen.log"
else
  LOG="${OUT}/eval_${TAG}_gen_${GEN_STAGES//,/-}.log"
fi
PREFIX="${OUT}/${TAG}"
HOLDOUT_CSV="${ROOT}/data/downstream/comp_chain/test_data_oas_holdout.csv"
EVAL_JSON="${ROOT}/downstream/benchmark/data/tcr_generation_bench/eval_conditional.json"
SABDAB="${ROOT}/data/downstream/cdr_infilling/sabdab"
SAB23="${ROOT}/data/downstream/cdr_infilling/sab23h2_converted"
HEAVY_BS="${3:-2}"
T4_BS="${4:-8}"

test -f "${CKPT}/model.safetensors"
mkdir -p "${OUT}"
echo "=== fusion gen ${TAG} start $(date -u +%Y-%m-%dT%H:%M:%SZ) ckpt=${CKPT} ===" | tee "${LOG}"
cd "${ROOT}"

run_cdr_sabdab() {
  local mode="$1"
  local log="${PREFIX}_${mode}_10fold_iter${CDR_MAX_ITER}.log"
  echo "$(date -Is) CDR SAbDab ${mode} (max_iter=${CDR_MAX_ITER})" | tee -a "${LOG}"
  python -u -m downstream.grammar.cdr_infill \
    --test-set "${SABDAB}/${mode}" \
    --mode "${mode}" \
    --checkpoint-path "${CKPT}" \
    --device cuda \
    --num-folds 10 \
    --sampling-strategy argmax \
    --max-iter "${CDR_MAX_ITER}" \
    > "${log}" 2>&1
  grep -E "Average AAR|Standard deviation" "${log}" | tee -a "${LOG}" || true
}

run_cdr_sab23() {
  local mode="$1"
  local log="${PREFIX}_sab23h2_${mode}_iter${CDR_MAX_ITER}.log"
  echo "$(date -Is) CDR SAb23H2 ${mode} (max_iter=${CDR_MAX_ITER})" | tee -a "${LOG}"
  python -u -m downstream.grammar.cdr_infill \
    --test-set "${SAB23}" \
    --mode "${mode}" \
    --checkpoint-path "${CKPT}" \
    --device cuda \
    --num-folds 1 \
    --sampling-strategy argmax \
    --max-iter "${CDR_MAX_ITER}" \
    > "${log}" 2>&1
  grep -E "Average AAR|Standard deviation" "${log}" | tee -a "${LOG}" || true
}

if stage_enabled cdr; then
  for mode in cdrh1 cdrh2 cdrh3; do
    run_cdr_sabdab "${mode}"
  done
  for mode in cdrh1 cdrh2 cdrh3 cdrl1 cdrl2 cdrl3; do
    run_cdr_sab23 "${mode}"
  done
fi

if stage_enabled pairing; then

# light_length_mode=prior draws the light length from the OAS-train histogram, so
# the target length is no longer taken from this row's reference. The old
# `reference` mode leaked it (100% length match, ~0.95 identity to the reference).
PAIR_LENGTH_MODE="${PAIR_LENGTH_MODE:-prior}"
PAIR_OUT="${PREFIX}_light_pairing_holdout500_prompt3_len${PAIR_LENGTH_MODE}_iter${PAIR_MAX_ITER}"
echo "$(date -Is) light pairing holdout500 prompt3 (light_length_mode=${PAIR_LENGTH_MODE} max_iter=${PAIR_MAX_ITER})" | tee -a "${LOG}"
python -u -m downstream.grammar.light_chain_pairing \
  --csv-path "${HOLDOUT_CSV}" \
  --checkpoint-path "${CKPT}" \
  --output-csv "${PAIR_OUT}.csv" \
  --metrics-json "${PAIR_OUT}_metrics.json" \
  --device cuda \
  --heavy-batch-size "${HEAVY_BS}" \
  --num-seqs 8 \
  --light-prompt-tokens 3 \
  --light-length-mode "${PAIR_LENGTH_MODE}" \
  --sampling-strategy gumbel_argmax \
  --max-iter "${PAIR_MAX_ITER}" \
  --seed 42 \
  2>&1 | tee -a "${LOG}"

echo "$(date -Is) pairing leakage diagnostic" | tee -a "${LOG}"
python -u scripts/downstream/pairing_leakage_diagnostic.py \
  --csv "${PAIR_OUT}_n8.csv" \
  --out-json "${PAIR_OUT}_leakage_diagnostic.json" \
  2>&1 | tee -a "${LOG}"

fi

if stage_enabled t4; then

T4_HELD="${PREFIX}_t4_held20.jsonl"
T4_UNSEEN="${PREFIX}_t4_unseen.jsonl"
echo "$(date -Is) T4 Setting B held20" | tee -a "${LOG}"
python -u -m downstream.grammar.tcr_generation \
  --mode conditional \
  --checkpoint "${CKPT}" \
  --eval-json "${EVAL_JSON}" \
  --eval-set held20 \
  -k 100 \
  --out "${T4_HELD}" \
  --device cuda \
  --seed 42 \
  --batch-size "${T4_BS}" \
  --max-iter 32 \
  --sampling-strategy gumbel_argmax \
  2>&1 | tee -a "${LOG}"

echo "$(date -Is) T4 Setting B unseen" | tee -a "${LOG}"
python -u -m downstream.grammar.tcr_generation \
  --mode conditional \
  --checkpoint "${CKPT}" \
  --eval-json "${EVAL_JSON}" \
  --eval-set unseen \
  -k 100 \
  --out "${T4_UNSEEN}" \
  --device cuda \
  --seed 42 \
  --batch-size "${T4_BS}" \
  --max-iter 32 \
  --sampling-strategy gumbel_argmax \
  2>&1 | tee -a "${LOG}"

cd "${ROOT}/downstream/benchmark"
python -u tcr_generation_bench/run.py --setting B --method file \
  --eval-set held20 --samples-file "${T4_HELD}" --tag "${TAG}_t4_held20" \
  2>&1 | tee -a "${LOG}"
python -u tcr_generation_bench/run.py --setting B --method file \
  --eval-set all --samples-file "${T4_UNSEEN}" --tag "${TAG}_t4_unseen" \
  2>&1 | tee -a "${LOG}"

fi

echo "=== fusion gen ${TAG} stages=${GEN_STAGES} done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "${LOG}"
