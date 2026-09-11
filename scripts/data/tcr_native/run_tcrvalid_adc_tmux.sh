#!/usr/bin/env bash
# Detached TCR-VALID ADC extract. Safe to re-run: resumes from inventory/done markers.
set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project"
PROXY="${ROOT}/.cursor/skills/https-proxy/scripts/with-proxy.sh"
PY="/vepfs-mlp2/c20250601/251105016/miniforge3/bin/python3"
SCRIPT="${ROOT}/dllm_test/scripts/data/tcr_native/extract_tcrvalid_adc.py"
OUT="${ROOT}/dllm_test/data/tcr_bulk_raw/tcrvalid"
LOG="${OUT}/extract.log"
SESSION="tcrvalid-adc"

mkdir -p "${OUT}" "${OUT}/rearrangements" "${OUT}/pairs"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists; not starting a second copy"
  tmux ls
  exit 0
fi

INNER="${OUT}/_tmux_inner.sh"
cat > "${INNER}" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd '${ROOT}'
export PYTHONUNBUFFERED=1
echo "[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] tmux job start session=${SESSION} pid=\$\$" >> '${LOG}'
'${PROXY}' '${PY}' -u '${SCRIPT}' run --mode pairs --pair-every 1 >> '${LOG}' 2>&1
rc=\$?
echo "[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] tmux job exit rc=\${rc}" >> '${LOG}'
exit "\${rc}"
EOF
chmod +x "${INNER}"

tmux new-session -d -s "${SESSION}" -n extract
tmux set-option -t "${SESSION}" remain-on-exit on
tmux send-keys -t "${SESSION}:extract" "bash '${INNER}'" C-m

echo "started tmux session ${SESSION}"
echo "log: ${LOG}"
echo "attach: tmux attach -t ${SESSION}"
echo "status: cat ${OUT}/status.json"
