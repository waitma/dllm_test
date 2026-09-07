#!/usr/bin/env bash
# TCR CDR3b infilling diagnostic -- sweep how much context the model needs.
#
# Tests RESULTS.md 0.0(h): if the model learned TCR sequence but ignores the
# epitope, no-epitope infilling should sit clearly above the context-blind PWM
# baseline. Sweeping the masked width turns that into a curve: a model relying
# only on local neighbours degrades fast as the window grows.
#
# mask_width 0 masks everything but the two C/F anchors, i.e. unconditional
# generation at a fixed length -- the far end of the same axis.
set -euo pipefail

ROOT=/vepfs-mlp2/c20250601/251105016/project/dllm_test
PY=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python
OUTDIR="${ROOT}/downstream/benchmark/outputs/tcr_generation_bench/infill_diagnostic"
N="${N:-2000}"
WIDTHS="${WIDTHS:-1 2 3 5 0}"

declare -A CKPTS=(
  [270m_diff_42000]="${ROOT}/output/protein_esmc_llada270m_diffusion_immune/checkpoint-42000"
  [8b_diff_45000]="${ROOT}/output/protein_esmc_llada8b_diffusion_immune/checkpoint-45000"
)
TAGS="${TAGS:-270m_diff_42000 8b_diff_45000}"

mkdir -p "${OUTDIR}"
cd "${ROOT}"

for tag in ${TAGS}; do
  ckpt="${CKPTS[$tag]}"
  for w in ${WIDTHS}; do
    out="${OUTDIR}/pred_${tag}_w${w}.jsonl"
    echo "=== ${tag} mask_width=${w} (n=${N}) ==="
    ${PY} -u -m downstream.grammar.tcr_generation \
      --mode infill --mask-width "${w}" -n "${N}" \
      --checkpoint "${ckpt}" --out "${out}" 2>&1 | grep -Ev "absl|oneDNN|TensorFlow|AVX"
    ${PY} -u "${ROOT}/downstream/benchmark/scripts/t4_infill_diagnostic.py" \
      --pred "${out}" --tag "${tag}_w${w}" 2>&1 | grep -Ev "absl|oneDNN|TensorFlow|AVX"
  done
done

echo "=== all done -> ${OUTDIR} ==="
