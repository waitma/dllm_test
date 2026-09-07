#!/usr/bin/env bash
# Model-native pairing score: p(L|H) − p(L) on OAS holdout.
# ImmunoMatch generation pairing stays in run_immune_fusion_pairing.sh.
#
# Usage: bash run_pairing_pll.sh <fusion_ckpt_dir> <tag> [batch_size] [max_pairs]
set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
CKPT="${1:?usage: $0 <fusion_ckpt_dir> <tag> [batch_size] [max_pairs]}"
TAG="${2:?usage: $0 <fusion_ckpt_dir> <tag> [batch_size] [max_pairs]}"
BS="${3:-4}"
MAX_PAIRS="${4:-0}"
HOLDOUT_CSV="${ROOT}/data/downstream/comp_chain/test_data_oas_holdout.csv"
OUT="${ROOT}/output/downstream_generation"
JSON="${OUT}/${TAG}_pairing_pll.json"
LOG="${OUT}/eval_${TAG}_pairing_pll.log"

test -f "${CKPT}/model.safetensors"
test -f "${HOLDOUT_CSV}"
mkdir -p "${OUT}"
cd "${ROOT}"

echo "=== pairing PLL ${TAG} start $(date -u +%Y-%m-%dT%H:%M:%SZ) ckpt=${CKPT} ===" | tee "${LOG}"
python -u scripts/downstream/score_pairing_pll.py \
  --checkpoint "${CKPT}" \
  --csv "${HOLDOUT_CSV}" \
  --output-json "${JSON}" \
  --device cuda \
  --batch-size "${BS}" \
  --max-pairs "${MAX_PAIRS}" \
  2>&1 | tee -a "${LOG}"
echo "=== pairing PLL ${TAG} done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "${LOG}"
