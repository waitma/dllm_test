#!/usr/bin/env bash
# Frozen fusion representation suite: T1 MLP retrain + T2 clustering + T3 few-shot.
# Usage: bash run_immune_fusion_repr.sh <ckpt_dir> <tag> [embed_batch_size]
set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
CKPT="${1:?usage: $0 <fusion_ckpt_dir> <tag> [embed_bs]}"
TAG="${2:?usage: $0 <fusion_ckpt_dir> <tag> [embed_bs]}"
EMBED_BS="${3:-16}"
T1_COMPLETION_MANIFEST="${T1_COMPLETION_MANIFEST:-${ROOT}/data/prepared/immune_v5_receptor_completion/dataset_manifest.json}"
T1_CDR3_FORMAT="${T1_CDR3_FORMAT:-junction}"
T1_TRACK="${T1_TRACK:-cdr3b}"
T1_MAX_LENGTH="${T1_MAX_LENGTH:-1024}"
SPEC="grammar:decoder:global:${CKPT}"
BENCH="${ROOT}/downstream/benchmark"
OUT="${ROOT}/output/downstream_generation"

# REPR_STAGES selects which probes run (comma-separated: t1,t2,t3; default all).
# T1 dominates the cost: on 8B it is ~113 min (463k pairs of frozen-feature extraction)
# versus ~5 min for T2 and ~8 min for T3. Sweeping many checkpoints is therefore only
# affordable for t2,t3, so the stage switch exists to run those alone.
REPR_STAGES="${REPR_STAGES:-t1,t2,t3}"
stage_enabled() { [[ ",${REPR_STAGES}," == *",$1,"* ]]; }

if [[ "${REPR_STAGES}" == "t1,t2,t3" ]]; then
  LOG="${OUT}/eval_${TAG}.log"
else
  LOG="${OUT}/eval_${TAG}_${REPR_STAGES//,/-}.log"
fi

test -f "${CKPT}/model.safetensors"
mkdir -p "${OUT}"
echo "=== fusion repr ${TAG} stages=${REPR_STAGES} start $(date -u +%Y-%m-%dT%H:%M:%SZ) ckpt=${CKPT} ===" | tee "${LOG}"

if stage_enabled t1; then
cd "${ROOT}"
python -u downstream/benchmark/tcr_binding/run_retrained_ours.py \
  --checkpoint "${CKPT}" \
  --tag "${TAG}" \
  --neg-source AS \
  --track "${T1_TRACK}" \
  --fold all \
  --eval-set all \
  --seed 0 \
  --completion-manifest "${T1_COMPLETION_MANIFEST}" \
  --cdr3-format "${T1_CDR3_FORMAT}" \
  --max-length "${T1_MAX_LENGTH}" \
  --embedding-batch-size "${EMBED_BS}" \
  --feature-chunk-size 2048 \
  --head-hidden 256 \
  --dropout 0.3 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --head-batch-size 512 \
  --max-epochs 200 \
  --patience 15 \
  --val-fraction 0.1 2>&1 | tee -a "${LOG}"
fi

cd "${BENCH}"
if stage_enabled t2; then
echo "=== T2 embed-threshold ${TAG} $(date -u +%H:%M:%SZ) ===" | tee -a "${LOG}"
python -u tcr_clustering/run.py --method embed-threshold \
  --embedder "${SPEC}" --tag "${TAG}" 2>&1 | tee -a "${LOG}"

echo "=== T2 K-means embed-bench ${TAG} $(date -u +%H:%M:%SZ) ===" | tee -a "${LOG}"
python -u tcr_clustering/run_embed_bench.py \
  --embedder "${SPEC}" --tag "${TAG}" --algos kmeans 2>&1 | tee -a "${LOG}"
fi

if stage_enabled t3; then
echo "=== T3 paper6 ${TAG} $(date -u +%H:%M:%SZ) ===" | tee -a "${LOG}"
python -u tcr_representation/run_paper6.py --method embed \
  --embedder "${SPEC}" --columns cdr3b cdr3a --tag "${TAG}" 2>&1 | tee -a "${LOG}"

echo "=== T3 broad ${TAG} $(date -u +%H:%M:%SZ) ===" | tee -a "${LOG}"
python -u tcr_representation/run.py --method embed \
  --embedder "${SPEC}" --columns cdr3b cdr3a --tag "${TAG}" 2>&1 | tee -a "${LOG}"
fi

echo "=== fusion repr ${TAG} stages=${REPR_STAGES} done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "${LOG}"
