#!/usr/bin/env bash
# Fetch the two TCR sources that were on the source wishlist but never landed on
# disk: the VDJdb 2026-06-03 GitHub release (disk only had 2025-12-29) and Zenodo
# record 11208211 (OTS/TCRLang datasets + paired model weights).
#
# Zenodo throttles and drops long connections, so every transfer goes through
# aria2c with --continue and is re-runnable. Files with a known-good md5 are
# skipped on re-run.
set -uo pipefail

PROXY="http://100.68.162.212:3128"
DATA="/vepfs-mlp2/c20250601/251105016/project/dllm_test/data"
LOG="${DATA}/ots_tcrlang/raw/download.log"

TCR_DIR="${DATA}/tcr"
OTS_DIR="${DATA}/ots_tcrlang/raw"
mkdir -p "${TCR_DIR}" "${OTS_DIR}"

# path|md5 (empty md5 = size check only)|url
JOBS=(
"${TCR_DIR}/vdjdb-2026-06-03.zip||https://github.com/antigenomics/vdjdb-db/releases/download/2026-06-03/vdjdb-2026-06-03.zip"
"${OTS_DIR}/OTS_CoherenceCode.tar.gz|53900bf6864a5f4a2cd07876698eba8c|https://zenodo.org/api/records/11208211/files/OTS_CoherenceCode.tar.gz/content"
"${OTS_DIR}/TCRLang_Datasets.tar.gz|f2d5cbdcbc518c7f4b73f27515ecb37d|https://zenodo.org/api/records/11208211/files/TCRLang_Datasets.tar.gz/content"
"${OTS_DIR}/tcrlang-weights.tar.gz|8bbade5ba653096a490467cbc68b4034|https://zenodo.org/api/records/11208211/files/tcrlang-weights.tar.gz/content"
)

echo "=== download start $(date -Is) ===" | tee -a "${LOG}"
rc_all=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r out md5 url <<<"${job}"
    name="$(basename "${out}")"

    if [[ -s "${out}" && -n "${md5}" ]]; then
        have="$(md5sum "${out}" | awk '{print $1}')"
        if [[ "${have}" == "${md5}" ]]; then
            echo "[skip] ${name} already verified" | tee -a "${LOG}"
            continue
        fi
    fi

    echo "[get ] ${name}" | tee -a "${LOG}"
    aria2c \
        --all-proxy="${PROXY}" \
        --continue=true \
        --max-tries=20 \
        --retry-wait=10 \
        --timeout=120 \
        --max-connection-per-server=4 \
        --split=4 \
        --min-split-size=5M \
        --summary-interval=30 \
        --console-log-level=warn \
        --dir="$(dirname "${out}")" \
        --out="${name}" \
        "${url}" 2>&1 | tee -a "${LOG}"

    if [[ ! -s "${out}" ]]; then
        echo "[FAIL] ${name} not written" | tee -a "${LOG}"
        rc_all=1
        continue
    fi
    if [[ -n "${md5}" ]]; then
        have="$(md5sum "${out}" | awk '{print $1}')"
        if [[ "${have}" == "${md5}" ]]; then
            echo "[ok  ] ${name} md5 ${have}" | tee -a "${LOG}"
        else
            echo "[FAIL] ${name} md5 ${have} != ${md5}" | tee -a "${LOG}"
            rc_all=1
        fi
    else
        echo "[ok  ] ${name} $(stat -c%s "${out}") bytes" | tee -a "${LOG}"
    fi
done

echo "=== download end $(date -Is) rc=${rc_all} ===" | tee -a "${LOG}"
exit "${rc_all}"
