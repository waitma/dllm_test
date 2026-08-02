#!/usr/bin/env bash
# Poll the mint resume-training Volc job and emit a sentinel line when it
# leaves the Running state, so the T3 mint格 can be re-run on the final best.pt.
# Heartbeat lines are intentionally distinct from the sentinel so an output
# watcher can trigger only on completion.
set -uo pipefail

JOB_ID="${1:-t-20260707105105-9x4hq}"
CKPT=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_mint_llada/best.pt
VOLC=/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh
INTERVAL="${2:-600}"      # seconds between polls
MAX_ITERS="${3:-300}"     # safety cap (300 * 600s = ~50h)

status_of() {
  # The volc wrapper prints version-notice lines to stdout before the JSON, so
  # slice from the first '[' / '{' before parsing.
  timeout 30 bash "$VOLC" ml_task get --id "$JOB_ID" --output json 2>/dev/null \
    | python3 -c "import sys,json,re
raw=sys.stdin.read()
m=re.search(r'[\[{]', raw)
try:
    d=json.loads(raw[m.start():]) if m else []
    d=d[0] if isinstance(d,list) and d else (d if isinstance(d,dict) else {})
    print(d.get('Status','UNKNOWN'))
except Exception:
    print('UNKNOWN')"
}

echo "=== watch mint training ${JOB_ID} start $(date -u +%FT%TZ) (interval=${INTERVAL}s) ==="
for i in $(seq 1 "$MAX_ITERS"); do
  st="$(status_of)"
  mt="$(stat -c %y "$CKPT" 2>/dev/null || echo NA)"
  echo "heartbeat $(date -u +%FT%TZ) iter=${i} status=${st} best_mtime=${mt}"
  case "$st" in
    Success|Failed|Killed)
      echo "MINT_TRAINING_DONE status=${st} at=$(date -u +%FT%TZ) best_mtime=${mt}"
      exit 0
      ;;
  esac
  sleep "$INTERVAL"
done
echo "MINT_TRAINING_DONE status=TIMEOUT_MAXITERS at=$(date -u +%FT%TZ)"
