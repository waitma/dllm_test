#!/usr/bin/env bash
# Reproduce the ESM2 grammar-v2 step-1 NaN with REAL 2-rank DDP on a single GPU.
# Both ranks share cuda:0 (LOCAL_RANK=0). Full data, per-rank noise seeds.
#
# Run:
#   bash scripts/debug/ddp2_esm2_repro.sh
set -uo pipefail
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test

DATA_DIR=/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/bioseq_grammar_v1
ESM2_DIR=/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esm2/esm2_t33_650M_UR50D
OUT=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/debug_ddp2_esm2

export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29571
export WORLD_SIZE=2
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

common_args=(
  examples/bioseq/train_qwen3_vl_bioseq_ddp.py
  --grammar-data-dir "${DATA_DIR}"
  --model-type esm2
  --encoder-path "${ESM2_DIR}"
  --sources oas,ots,tcr,ppi
  --oas-weight 3.9 --ots-weight 3.6 --ppi-weight 1.4 --tcr-weight 1.0
  --batch-size 1 --grad-accum 2
  --max-steps 60
  --max-sequence-length 2112 --max-position-embeddings 2304
  --num-hidden-layers 28 --num-attention-heads 16 --intermediate-size 5120
  --dropout 0.1 --qk-norm --gradient-checkpointing
  --lr 1e-4 --encoder-lr 2e-5 --warmup-steps 1000 --warmup-init-lr 1e-7
  --lr-scheduler cosine --min-lr-ratio 0.1 --grad-clip 1.0 --bf16
  --find-unused-parameters --num-workers 0
  --log-interval 1 --val-interval 0 --save-interval 0
  --resume none --device cuda
  --output-dir "${OUT}" --wandb-mode disabled --debug-ddp-timing
)

mkdir -p "${OUT}"
RANK=1 LOCAL_RANK=0 python "${common_args[@]}" > "${OUT}/rank1.log" 2>&1 &
R1=$!
RANK=0 LOCAL_RANK=0 python "${common_args[@]}" > "${OUT}/rank0.log" 2>&1 &
R0=$!
echo "launched rank0=$R0 rank1=$R1; logs in ${OUT}"
wait $R0; rc0=$?
wait $R1; rc1=$?
echo "rank0 exit=$rc0 rank1 exit=$rc1"
