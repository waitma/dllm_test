#!/usr/bin/env bash
# Download STRING-DB v12 assets for MINT-style PPI pretraining and functional edges.
#
# Usage:
#   bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/download_stringdb_assets.sh
#   bash .../download_stringdb_assets.sh --with-detailed   # adds ~190GB detailed links (functional channels)
#
# Files land under data/ppi_task_raw/raw/stringdb_mint/

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/stringdb_mint"
BASE="https://stringdb-downloads.org/download"
WITH_DETAILED=0

for arg in "$@"; do
  case "${arg}" in
    --with-detailed) WITH_DETAILED=1 ;;
    *) echo "Unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

mkdir -p "${ROOT}"
cd "${ROOT}"

download() {
  local name="$1"
  if [[ -f "${name}" ]]; then
    echo "exists: ${name}"
    return 0
  fi
  echo "downloading: ${name}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -o "${name}" "${BASE}/${name}"
  else
    curl -L --retry 5 --continue-at - -o "${name}" "${BASE}/${name}"
  fi
}

# --- MINT pretraining (physical interaction) ---
download "protein.sequences.v12.0.fa.gz"
download "protein.physical.links.full.v12.0.txt.gz"
download "protein.physical.links.v12.0.txt.gz"

# --- Functional relation channels ---
# STRING does NOT ship per-channel files like protein.activation.links.v12.0.txt.gz.
# Channel subscores live in protein.links.detailed.v12.0.txt.gz (~190GB).
# Map dominant channel -> grammar relation in build_ppi_interaction_csv.py.
if [[ "${WITH_DETAILED}" -eq 1 ]]; then
  download "protein.links.detailed.v12.0.txt.gz"
else
  echo "skip: protein.links.detailed.v12.0.txt.gz (pass --with-detailed to fetch ~190GB functional channel file)"
fi

echo "Done. Next steps:"
echo "  1) gunzip -k protein.sequences.v12.0.fa.gz && mmseqs cluster ... -> clu50.tsv"
echo "  2) python scripts/data/build_mint_string_splits.py"
