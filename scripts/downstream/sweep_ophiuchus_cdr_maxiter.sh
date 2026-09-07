#!/usr/bin/env bash
# Why: our Ophiuchus-Ab CDR rerun lands ~2.7 pp below the paper on H3 (40.89 vs 43.55)
# while H1/H2 match, and the official run/zero_shot_test.sh shows the paper did NOT use
# the argparse defaults (its SAb23H2 line passes --max_iter 2, pairing passes --max_iter 124
# and gumbel_argmax). AAR is a per-position metric, so iterative decoding can *hurt* it:
# each committed token conditions the rest, and a wrong commit propagates. max_iter=1 is the
# pure one-shot marginal argmax, which maximises expected per-position accuracy.
#
# This sweeps max_iter for the official checkpoint to find the setting that reproduces
# Table 2 (H1 75.50 / H2 70.18 / H3 43.55) and Table 1 (SAb23H2).
#
# Usage: bash sweep_ophiuchus_cdr_maxiter.sh [modes] [iters]
set -uo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
ENV="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr"
PY="$ENV/bin/python"
HERE="$ROOT/downstream/ophiuchus_eval"
OUT="$ROOT/output/downstream_generation/ophiuchus_ab/maxiter_sweep"
SAB="$ROOT/data/downstream/cdr_infilling/sabdab"
SAB23="$ROOT/data/downstream/cdr_infilling/sab23h2/SAb-23-H2-Ab"

MODES="${1:-cdrh3 cdrh1 cdrh2}"
ITERS="${2:-1 2 4 8}"

CKPT="${OPHIUCHUS_AB_CKPT:-}"
if [[ -z "$CKPT" ]]; then
  if [[ -f /c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt ]]; then
    CKPT="/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt"
  else
    CKPT="$ROOT/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt"
  fi
fi

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
mkdir -p "$OUT"
SUM="$OUT/_sweep_summary.log"

log() { echo "$(date -Is) $*" | tee -a "$SUM"; }
log "START max_iter sweep  modes=[${MODES}] iters=[${ITERS}] ckpt=${CKPT}"
log "paper targets: SAbDab H1=75.50 H2=70.18 H3=43.55 | SAb23H2 L1..H3=81.1/80.1/73.7/74.8/68.6/36.8"

for it in ${ITERS}; do
  for mode in ${MODES}; do
    out="$OUT/sabdab_${mode}_iter${it}.json"
    if [[ -f "$out" ]]; then log "SKIP sabdab ${mode} iter=${it} (exists)"; continue; fi
    log "RUN sabdab ${mode} max_iter=${it}"
    "$PY" -u "$HERE/cdr_sabdab.py" \
      --test-set "${SAB}/${mode}" --mode "${mode}" \
      --checkpoint-path "$CKPT" \
      --sampling-strategy argmax --max-iter "${it}" --cfg-scale 0.0 \
      --output "$out" > "$OUT/sabdab_${mode}_iter${it}.log" 2>&1
    if [[ $? -eq 0 ]]; then
      log "  DONE sabdab ${mode} iter=${it} AAR=$($PY -c "import json;print(round(json.load(open('$out'))['average_aar_all_folds'],2))")"
    else
      log "  FAIL sabdab ${mode} iter=${it}"
    fi
  done
done

# SAb23H2 is 60 antibodies and runs in under a minute per CDR, so sweep it fully.
for it in ${ITERS}; do
  out="$OUT/sab23h2_iter${it}.json"
  if [[ -f "$out" ]]; then log "SKIP sab23h2 iter=${it} (exists)"; continue; fi
  log "RUN sab23h2 max_iter=${it}"
  "$PY" -u "$HERE/cdr_sab23h2.py" \
    --test-set "$SAB23" \
    --checkpoint-path "$CKPT" \
    --sampling-strategy argmax --max-iter "${it}" --cfg-scale 0.0 \
    --output "$out" > "$OUT/sab23h2_iter${it}.log" 2>&1
  [[ $? -eq 0 ]] && log "  DONE sab23h2 iter=${it}" || log "  FAIL sab23h2 iter=${it}"
done

log "=== SWEEP TABLE ==="
"$PY" - <<'EOF' 2>&1 | tee -a "$SUM"
import json, glob, os, re
out=os.environ.get("OUT") or "/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/maxiter_sweep"
paper={"cdrh1":75.50,"cdrh2":70.18,"cdrh3":43.55}
rows={}
for p in sorted(glob.glob(f"{out}/sabdab_*_iter*.json")):
    m=re.match(r"sabdab_(\w+)_iter(\d+)\.json", os.path.basename(p))
    if not m: continue
    mode,it=m.group(1),int(m.group(2))
    rows.setdefault(mode,{})[it]=json.load(open(p))["average_aar_all_folds"]
for mode in sorted(rows):
    tgt=paper.get(mode)
    print(f"SAbDab {mode}  paper={tgt}")
    for it in sorted(rows[mode]):
        v=rows[mode][it]
        d=f"  delta={v-tgt:+.2f}" if tgt else ""
        print(f"   max_iter={it:3d}  AAR={v:6.2f}{d}")
EOF
log "END max_iter sweep"
