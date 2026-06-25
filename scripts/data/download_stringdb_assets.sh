#!/usr/bin/env bash
# Download STRING-DB assets for MINT-style PPI pretraining and (optionally)
# functional / regulatory edges.
#
# MINT minimal set (default): the *full* physical links file + sequences.
# These two files are all `run_mint_stringdb_native.py` needs (clu50.tsv is
# generated locally by MMseqs2, NOT downloaded).
#
# Usage:
#   bash .../download_stringdb_assets.sh                       # v12.0 MINT minimal
#   bash .../download_stringdb_assets.sh --version v11.0       # v11.0 MINT minimal
#   bash .../download_stringdb_assets.sh --version v11.0 --with-actions
#                                                              # + all-species protein.actions (mode/action edges)
#   bash .../download_stringdb_assets.sh --with-detailed       # + ~190GB per-channel detailed file
#                                                              #   (NOT needed for MINT; has NO mode/action)
#
# Files land under data/ppi_task_raw/raw/stringdb_mint/

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/stringdb_mint"
BASE="https://stringdb-downloads.org/download"
VERSION="v12.0"
WITH_DETAILED=0
WITH_ACTIONS=0

while [[ $# -gt 0 ]]; do
  case "${1}" in
    --version) VERSION="${2}"; shift 2 ;;
    --with-detailed) WITH_DETAILED=1; shift ;;
    --with-actions) WITH_ACTIONS=1; shift ;;
    *) echo "Unknown arg: ${1}" >&2; exit 2 ;;
  esac
done

# Known full-file byte sizes for integrity checks (0 = skip size check).
phys_full_size() {
  case "${1}" in
    v12.0) echo 15528028374 ;;
    v11.0) echo 33237086717 ;;
    *) echo 0 ;;
  esac
}
sequences_size() {
  case "${1}" in
    v11.0) echo 5526372370 ;;
    *) echo 0 ;;
  esac
}
actions_size() {
  case "${1}" in
    v11.0) echo 12858366570 ;;
    *) echo 0 ;;
  esac
}

mkdir -p "${ROOT}"
cd "${ROOT}"

verify_gz() {
  local name="$1"
  local expected_size="${2:-0}"
  if [[ ! -f "${name}" ]]; then
    return 1
  fi
  if ! gzip -t "${name}" 2>/dev/null; then
    echo "corrupt gzip: ${name}"
    return 1
  fi
  if [[ "${expected_size}" -gt 0 ]]; then
    local actual
    actual="$(stat -c '%s' "${name}")"
    if [[ "${actual}" -ne "${expected_size}" ]]; then
      echo "size mismatch for ${name}: have ${actual}, expected ${expected_size}"
      return 1
    fi
  fi
  return 0
}

download() {
  local name="$1"
  local expected_size="${2:-0}"
  if verify_gz "${name}" "${expected_size}"; then
    echo "exists and verified: ${name}"
    return 0
  fi
  if [[ -f "${name}" ]]; then
    local backup="${name}.corrupt_$(date +%Y%m%d_%H%M%S)"
    echo "backing up bad file -> ${backup}"
    mv "${name}" "${backup}"
  fi
  echo "downloading: ${name}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -c -o "${name}" "${BASE}/${name}"
  else
    curl -L --retry 5 --continue-at - -o "${name}" "${BASE}/${name}"
  fi
  if ! verify_gz "${name}" "${expected_size}"; then
    echo "ERROR: downloaded file failed integrity check: ${name}" >&2
    exit 1
  fi
}

echo "=== STRING ${VERSION} MINT minimal download ==="

# --- MINT pretraining minimal set (physical interaction) ---
download "protein.sequences.${VERSION}.fa.gz" "$(sequences_size "${VERSION}")"
download "protein.physical.links.full.${VERSION}.txt.gz" "$(phys_full_size "${VERSION}")"

# --- Optional: mode/action edges (activation / inhibition / catalysis / ...) ---
# STRING actions files only exist up to v11.0; v12 replaced them with the
# 'regulatory' network (API network_type=regulatory), not a bulk channel file.
if [[ "${WITH_ACTIONS}" -eq 1 ]]; then
  if [[ "${VERSION}" == "v11.0" ]]; then
    download "protein.actions.${VERSION}.txt.gz" "$(actions_size "${VERSION}")"
  else
    echo "skip: protein.actions only published up to v11.0 (requested ${VERSION})"
  fi
fi

# --- Optional: per-channel detailed subscores (NOT needed for MINT) ---
# Detailed = combined_score broken into 7 evidence channels. It does NOT contain
# mode/action; do not download unless you specifically need channel subscores.
if [[ "${WITH_DETAILED}" -eq 1 ]]; then
  download "protein.links.detailed.${VERSION}.txt.gz"
else
  echo "skip: protein.links.detailed.${VERSION}.txt.gz (pass --with-detailed; ~190GB, no mode/action)"
fi

echo "Done. Next steps (MINT-style):"
echo "  1) gunzip -k protein.sequences.${VERSION}.fa.gz"
echo "  2) mmseqs createdb protein.sequences.${VERSION}.fa DB100 && \\"
echo "     mmseqs cluster DB100 clu50 /tmp/mmseqs --min-seq-id 0.50 --remove-tmp-files && \\"
echo "     mmseqs createtsv DB100 DB100 clu50 clu50.${VERSION}.tsv"
echo "  3) python scripts/data/run_mint_stringdb_native.py \\"
echo "       --links-gz protein.physical.links.full.${VERSION}.txt.gz \\"
echo "       --sequences-fa protein.sequences.${VERSION}.fa \\"
echo "       --cluster-tsv clu50.${VERSION}.tsv \\"
echo "       --output-dir data/ppi_task_raw/processed/mint_string_pretrain_${VERSION}"
