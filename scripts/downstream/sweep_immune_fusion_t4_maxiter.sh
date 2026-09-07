#!/usr/bin/env bash
# Why: T4 Setting-B is the only generation stage whose decoding budget was never swept.
# run_immune_fusion_gen.sh hardcodes --max-iter 32 (no official TCRT5 counterpart to align
# to), and its T4 outputs are NOT iter-tagged, so two budgets would silently clobber each
# other. CDR and pairing already have their answers: AAR falls monotonically with more
# iterations (official ckpt SAbDab H3 42.00/41.43/40.96/40.70 at iter 1/2/4/8) and pairing
# 32 -> 124 moved us only +0.002..0.003. T4 scores d_edit / seq-recovery over a *set* of
# designs rather than per-position recovery, so its response to the budget is untested.
#
# Every artifact here is iter-tagged, and each (iter, eval-set) is skipped when its
# metrics.json already exists, so a preempted run resumes instead of restarting.
#
# Usage: bash sweep_immune_fusion_t4_maxiter.sh <ckpt_dir> <tag> [iters] [t4_batch]
set -uo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
CKPT="${1:?usage: $0 <ckpt_dir> <tag> [iters] [t4_batch]}"
TAG="${2:?usage: $0 <ckpt_dir> <tag> [iters] [t4_batch]}"
ITERS="${3:-8 32 64 128}"
T4_BS="${4:-8}"

OUT="${ROOT}/output/downstream_generation"
PREFIX="${OUT}/${TAG}"
EVAL_JSON="${ROOT}/downstream/benchmark/data/tcr_generation_bench/eval_conditional.json"
BENCH_OUT="${ROOT}/downstream/benchmark/outputs/tcr_generation_bench/setting_B"
LOG="${OUT}/eval_${TAG}_t4_maxiter_sweep.log"

test -f "${CKPT}/model.safetensors"
mkdir -p "${OUT}"
cd "${ROOT}"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${LOG}"; }
: > "${LOG}"
log "START T4 max_iter sweep tag=${TAG} iters=[${ITERS}] ckpt=${CKPT}"

# held20 uses --eval-set held20 on both sides; unseen generates --eval-set unseen and is
# scored with --eval-set all, matching run_immune_fusion_gen.sh so numbers stay comparable.
for it in ${ITERS}; do
  for split in held20 unseen; do
    bench_tag="${TAG}_t4_${split}_iter${it}"
    if [[ -f "${BENCH_OUT}/${bench_tag}/metrics.json" ]]; then
      log "SKIP ${split} iter=${it} (metrics.json exists)"
      continue
    fi
    jsonl="${PREFIX}_t4_${split}_iter${it}.jsonl"
    if [[ ! -s "${jsonl}" ]]; then
      log "GEN  ${split} iter=${it}"
      python -u -m downstream.grammar.tcr_generation \
        --mode conditional \
        --checkpoint "${CKPT}" \
        --eval-json "${EVAL_JSON}" \
        --eval-set "${split}" \
        -k 100 \
        --out "${jsonl}" \
        --device cuda \
        --seed 42 \
        --batch-size "${T4_BS}" \
        --max-iter "${it}" \
        --sampling-strategy gumbel_argmax \
        >> "${LOG}" 2>&1 || { log "  FAIL gen ${split} iter=${it}"; continue; }
    else
      log "REUSE ${split} iter=${it} designs (${jsonl})"
    fi
    score_set="held20"; [[ "${split}" == "unseen" ]] && score_set="all"
    log "BENCH ${split} iter=${it} (--eval-set ${score_set})"
    ( cd "${ROOT}/downstream/benchmark" && python -u tcr_generation_bench/run.py \
        --setting B --method file \
        --eval-set "${score_set}" \
        --samples-file "${jsonl}" \
        --tag "${bench_tag}" ) >> "${LOG}" 2>&1 || log "  FAIL bench ${split} iter=${it}"
  done
done

log "=== SWEEP TABLE (bioseq_unseen_common, the only cross-model T4 slice) ==="
TAG="${TAG}" ITERS="${ITERS}" BENCH_OUT="${BENCH_OUT}" python - <<'PY' 2>&1 | tee -a "${LOG}"
import json, os

tag = os.environ["TAG"]
bench = os.environ["BENCH_OUT"]
print(f"{'split':8} {'iter':>5} {'d_edit':>8} {'seq_rec':>8} {'bleu':>8} {'div_edit':>9} {'n':>5}")
for split in ("held20", "unseen"):
    for it in os.environ["ITERS"].split():
        p = f"{bench}/{tag}_t4_{split}_iter{it}/metrics.json"
        if not os.path.isfile(p):
            print(f"{split:8} {it:>5} {'-':>8}")
            continue
        m = json.load(open(p))
        summary = m.get("summary", m)
        row = summary.get("bioseq_unseen_common") or summary.get("overall") or {}
        f = lambda k: (f"{row[k]:.4f}" if isinstance(row.get(k), (int, float)) else "-")
        print(f"{split:8} {it:>5} {f('d_edit'):>8} {f('seq_recovery'):>8} "
              f"{f('char_bleu'):>8} {f('diversity_edit'):>9} {str(row.get('n_pmhc','-')):>5}")
PY
log "END T4 max_iter sweep"
