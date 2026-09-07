#!/usr/bin/env bash
# Rebuild the Kong et al. vintage SAbDab 10-fold split used by Ophiuchus-Ab Table 2.
#
# Why this exists
# ---------------
# `data/downstream/cdr_infilling/sabdab/` is a post-2022 SAbDab snapshot: 3,320 rows /
# 3,131 unique PDB entries, of which 1,079 rows are 8xxx/9xxx codes deposited after
# 2022-11. The paper cites "SAbDab, Kong et al., 3,127 complexes after filtering", and
# Kong et al.'s frozen summary contains exactly one 8xxx entry. Those newer complexes
# have longer CDR-H3 (15.23 vs 14.56 mean), which depresses H3 AAR while leaving the
# fixed-length H1/H2 untouched -- matching the discrepancy we measured (H1/H2 reproduce,
# H3 sits 2.1-2.7 pp low, and a max_iter sweep only moved it ~0.5 pp).
#
# Kong et al. never released the processed data. dyMEAN's only GitHub release (v1.0.0)
# ships checkpoints; MEAN has zero releases. What they do ship in-repo is the frozen
# summary and the split script, and the split is deterministic, so the fold partition is
# reproducible. Sequence data comes from the local `data/sabdab.zip`, which is already
# MEAN pipeline output and has exactly the paper's 3,127 records.
#
# Network use: only the two MEAN helper files below. No sequence data is downloaded.
#
# Usage: bash scripts/downstream/build_sabdab_kong_split.sh [--force]
set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
ENV="/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr"
PY="$ENV/bin/python"
SRC_ZIP="$ROOT/data/sabdab.zip"
OUT="$ROOT/data/downstream/cdr_infilling/sabdab_kong"
PROXY="http://100.68.162.212:3128"
SUMMARY_URL="https://raw.githubusercontent.com/THUNLP-MT/MEAN/main/summaries/sabdab_summary.tsv"
SPLIT_URL="https://raw.githubusercontent.com/THUNLP-MT/MEAN/main/data/split.py"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

export PATH="$ENV/bin:$PATH"
test -f "$SRC_ZIP" || { echo "missing $SRC_ZIP"; exit 1; }
command -v mmseqs >/dev/null || { echo "mmseqs not on PATH (needed by split.py)"; exit 1; }
mkdir -p "$OUT"
cd "$OUT"

# ---------------------------------------------------------------- 1. MEAN helpers
# GitHub needs the lab proxy; volc must never be proxied, but this is plain HTTPS.
fetch() {  # fetch <url> <dest>
  local url="$1" dest="$2"
  if [[ -f "$dest" && $FORCE -eq 0 ]]; then echo "have $dest"; return; fi
  echo "downloading $dest"
  env https_proxy="$PROXY" http_proxy="$PROXY" curl -sSLf -o "$dest" "$url"
}
fetch "$SUMMARY_URL" sabdab_summary_2022-11-12.tsv
fetch "$SPLIT_URL"   mean_split.py

# ---------------------------------------------------- 2. sequences from the local zip
if [[ ! -f sabdab_all.json || $FORCE -eq 1 ]]; then
  echo "extracting sequences from $SRC_ZIP"
  tmpx="$(mktemp -d)"
  unzip -o -j -q "$SRC_ZIP" 'sabdab/*.json' -d "$tmpx"
  cat "$tmpx/train.json" "$tmpx/valid.json" "$tmpx/test.json" > sabdab_all.json
  rm -rf "$tmpx"
fi
n_all=$(wc -l < sabdab_all.json)
echo "sabdab_all.json: ${n_all} records (paper reports 3,127)"
[[ "$n_all" == "3127" ]] || echo "WARNING: expected 3127 records, got ${n_all}"

# ------------------------------------------------------------- 3. vintage audit
"$PY" - <<'PY'
import csv, json, collections
mean_pdbs = set()
with open('sabdab_summary_2022-11-12.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        mean_pdbs.add(row['pdb'].strip().lower())
rows = [json.loads(l) for l in open('sabdab_all.json') if l.strip()]
ours = {r['pdb'].lower() for r in rows}
pref = collections.Counter(p[0] for p in ours)
print(f"  frozen MEAN summary: {len(mean_pdbs)} PDB entries (retrieved 2022-11-12)")
print(f"  sabdab_all.json    : {len(rows)} rows / {len(ours)} unique PDB")
print(f"  inside the snapshot: {len(ours & mean_pdbs)} / {len(ours)}")
print(f"  PDB first-char dist: {dict(sorted(pref.items()))}")
post = sum(1 for p in ours if p[0] in '89')
print(f"  post-2022 (8xxx/9xxx): {post}  <- must be ~0 for the Kong vintage")
PY

# ------------------------------------------------------- 4. deterministic 10-fold
# split.py clusters {cdr}_seq with `mmseqs cluster --min-seq-id 0.4`, shuffles clusters
# under seed 2022, then assigns folds cluster-wise so no cluster spans folds.
# filter 111 = must have heavy + light + antigen. Table 2 is heavy-only, so cdrh1/2/3.
for cdr in cdrh1 cdrh2 cdrh3; do
  if [[ -f "$cdr/fold_9/test.json" && $FORCE -eq 0 ]]; then echo "have $cdr folds"; continue; fi
  echo "=== splitting $cdr ==="
  rm -rf ./tmp "$cdr"
  "$PY" mean_split.py --data sabdab_all.json --out_dir "./$cdr" \
    --cdr "$cdr" --filter 111 --k_fold 10 --seed 2022 \
    | grep -E "Valid entries|Number of clusters|cluster number|fold data split"
  rm -rf ./tmp
  # Zero-shot eval reads only test.json; drop the per-fold train/valid (110MB -> 18MB).
  rm -f "$cdr"/fold_*/train.json "$cdr"/fold_*/valid.json
done

# ------------------------------------------------------------------- 5. verify
echo "=== verification ==="
for cdr in cdrh1 cdrh2 cdrh3; do
  folds=$(ls -d "$cdr"/fold_* 2>/dev/null | wc -l)
  total=$(cat "$cdr"/fold_*/test.json | wc -l)
  printf "  %-6s folds=%s  test rows total=%s\n" "$cdr" "$folds" "$total"
  [[ "$total" == "$n_all" ]] || echo "    WARNING: expected ${n_all} rows across folds"
done
"$PY" - <<'PY'
import json
d = json.loads(open('cdrh3/fold_0/test.json').readline())
need = ['heavy_chain_seq', 'light_chain_seq', 'cdrh3_seq', 'cdrh3_pos']
missing = [k for k in need if k not in d]
print(f"  harness schema: {'OK' if not missing else 'MISSING ' + str(missing)}")
PY
echo "  size: $(du -sh . | cut -f1)"
echo
echo "Paper Table 2 targets: H1 75.50 / H2 70.18 / H3 43.55"
echo "Evaluate with:"
echo "  python downstream/ophiuchus_eval/cdr_sabdab.py \\"
echo "    --test-set ${OUT}/cdrh3 --mode cdrh3 --checkpoint-path <ckpt> \\"
echo "    --sampling-strategy argmax --max-iter 4 --cfg-scale 0.0"
