#!/usr/bin/env bash
# Re-run ONLY the mint格 of T3 (deep + broad) on the current best.pt snapshot of
# grammar_v2_esmc300m_mint_llada, using the PROJ_GUIDE headline口径
# (post-LLaDA global whole-feature), then rebuild both _summary.json.
#
# Default: refuses while mint training (t-20260707105105-9x4hq) is still
# Running/Queue/Staging. Use FORCE=1 to evaluate the latest mid-training ckpt
# (freezes best.pt -> best.t3snap_<ts>.pt so deep+broad share one provenance).
set -euo pipefail

JOB_ID="${MINT_JOB_ID:-t-20260707105105-9x4hq}"
VOLC=/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
CKPT=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_mint_llada/best.pt
TAG=ours_globalfeat_esmc300m_mint_llada

export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
export LD_LIBRARY_PATH="${ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT=https://hf-mirror.com
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark

if [ "${FORCE:-0}" != "1" ]; then
  st="$(timeout 30 bash "$VOLC" ml_task get --id "$JOB_ID" --output json 2>/dev/null \
        | python3 -c "import sys,json,re
raw=sys.stdin.read(); m=re.search(r'[\[{]',raw)
d=json.loads(raw[m.start():]) if m else []
d=d[0] if isinstance(d,list) and d else {}
print(d.get('Status','UNKNOWN'))" || echo UNKNOWN)"
  echo "mint training ${JOB_ID} status = ${st}"
  if [ "$st" = "Running" ] || [ "$st" = "Queue" ] || [ "$st" = "Staging" ]; then
    echo "REFUSING: mint still training (status=${st}); best.pt is a moving target."
    echo "Re-run after it is Success, or set FORCE=1 to override."
    exit 2
  fi
fi

# Freeze the current best.pt into a snapshot (cp -p preserves mtime) so that
# deep + broad both read the SAME bytes even if training overwrites best.pt
# mid-run, and the recorded provenance sha256 is stable. When training is
# already done this is simply a harmless copy of the final ckpt.
SNAP_TS="$(date -u +%Y%m%dT%H%M%SZ)"
SNAP="${CKPT%.pt}.t3snap_${SNAP_TS}.pt"
echo "=== T3 mint re-run start $(date -u +%FT%TZ) ==="
echo "freezing best.pt (mtime=$(stat -c %y "$CKPT")) -> ${SNAP}"
cp -p "$CKPT" "$SNAP"
echo "snapshot sha256: $(sha256sum "$SNAP" | cut -d' ' -f1)"

"$ENV/bin/python" tcr_representation/run_paper6.py --method embed \
  --embedder "grammar:decoder:global:${SNAP}" --columns cdr3b cdr3a --tag "$TAG"
"$ENV/bin/python" tcr_representation/run.py --method embed \
  --embedder "grammar:decoder:global:${SNAP}" --columns cdr3b cdr3a --tag "$TAG"
"$ENV/bin/python" scripts/summarize_tcr_representation_paper6.py
"$ENV/bin/python" scripts/summarize_tcr_representation.py
echo "=== T3 mint re-run done $(date -u +%FT%TZ) | snapshot=${SNAP} ==="
echo "Next: update mint rows in downstream/downstream.md + benchmark/RESULTS.md from the new fewshot.json (checkpoint field carries the snapshot path/mtime/sha)."
