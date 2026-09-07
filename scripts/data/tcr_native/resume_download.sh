#!/usr/bin/env bash
# Resume-until-complete downloader.
#
# Zenodo through the lab proxy drops the connection every few minutes
# ("curl: (18) transfer closed with N bytes remaining"), so a single curl -C -
# is not enough: it has to be retried until the file reaches its full size.
#
# Usage: resume_download.sh URL OUT_PATH EXPECTED_BYTES [MAX_TRIES]
set -uo pipefail

URL="$1"
OUT="$2"
EXPECTED="$3"
MAX_TRIES="${4:-200}"

PROXY_WRAPPER="/vepfs-mlp2/c20250601/251105016/project/.cursor/skills/https-proxy/scripts/with-proxy.sh"

size_of() { [ -f "$OUT" ] && stat -c %s "$OUT" || echo 0; }

for try in $(seq 1 "$MAX_TRIES"); do
    have=$(size_of)
    if [ "$have" -ge "$EXPECTED" ]; then
        echo "[resume] complete: $have bytes after $((try - 1)) retries"
        exit 0
    fi
    pct=$((100 * have / EXPECTED))
    echo "[resume] try $try/$MAX_TRIES  have ${have}/${EXPECTED} bytes (${pct}%)"
    "$PROXY_WRAPPER" curl -sS -L \
        --retry 3 --retry-delay 5 --retry-all-errors \
        --speed-time 90 --speed-limit 2048 \
        -C - "$URL" -o "$OUT"
    sleep 5
done

have=$(size_of)
echo "[resume] GAVE UP after $MAX_TRIES tries: ${have}/${EXPECTED} bytes"
exit 1
