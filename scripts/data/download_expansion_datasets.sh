#!/usr/bin/env bash
# Download / refresh datasets for grammar_v1 expansion (official splits only).
#
# Usage:
#   bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/download_expansion_datasets.sh
#   bash .../download_expansion_datasets.sh --tcr-only
#   bash .../download_expansion_datasets.sh --ppi-only

set -euo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
TCR_RAW="${ROOT}/data/tcr"
TCR_BENCH="${ROOT}/data/ppi_task_raw/raw/nat_methods_tcr_benchmark"
PISTE_DIR="${ROOT}/data/ppi_task_raw/raw/piste_tcr_epitope_hla"

MODE="${1:-all}"

download() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "${out}")"
  if [[ -f "${out}" ]]; then
    echo "exists: ${out}"
    return 0
  fi
  echo "downloading: ${url} -> ${out}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -o "$(basename "${out}")" -d "$(dirname "${out}")" "${url}"
  else
    curl -L --retry 5 --continue-at - -o "${out}" "${url}"
  fi
}

clone_if_missing() {
  local url="$1"
  local dir="$2"
  if [[ -d "${dir}/.git" ]]; then
    echo "exists: ${dir}"
    return 0
  fi
  mkdir -p "$(dirname "${dir}")"
  git clone --depth 1 "${url}" "${dir}"
}

tcr_downloads() {
  echo "=== TCR expansion ==="
  # PISTE (already cloned under piste_tcr_epitope_hla; refresh if shallow)
  clone_if_missing "https://github.com/Armilius/PISTE.git" "${PISTE_DIR}/PISTE"

  # Nat Methods 2025 benchmark (figshare) — processed train/test with seen/unseen epitope splits
  mkdir -p "${TCR_BENCH}"
  echo "Nat Methods 2025 TCR benchmark: download manually or via figshare API"
  echo "  DOI: https://doi.org/10.6084/m9.figshare.27020455"
  echo "  Target dir: ${TCR_BENCH}"

  # VDJdb latest (if older than local 2025-12-29 snapshot)
  download "https://vdjdb.cdr3.net/database/2025-12-29/vdjdb.zip" "${TCR_RAW}/vdjdb-latest.zip"

  # IEDB tcell (large archive; optional)
  if [[ ! -f "${TCR_RAW}/iedb_tcell_full.zip" ]]; then
    echo "skip IEDB (large): place iedb_tcell_full.zip under ${TCR_RAW} manually"
  fi

  # TDC TCR-epitope (Therapeutics Data Commons)
  mkdir -p "${ROOT}/data/ppi_task_raw/raw/tdc_tcr_epitope"
  echo "TDC: export TCREpitopeBinding to CSV under data/ppi_task_raw/raw/tdc_tcr_epitope/"
  echo "  pip install PyTDC && python -c \"from tdc.single_pred import TCRpMHCBinding; data=TCRpMHCBinding(name='Weber et al.'); data.get_data().to_csv('...')\""
}

ppi_downloads() {
  echo "=== PPI expansion ==="
  bash "${ROOT}/scripts/data/download_stringdb_assets.sh"
  echo "Next: mmseqs cluster + build_mint_string_splits.py"
}

case "${MODE}" in
  --tcr-only) tcr_downloads ;;
  --ppi-only) ppi_downloads ;;
  all|*) tcr_downloads; ppi_downloads ;;
esac

echo "Done. Run audit: python ${ROOT}/scripts/data/audit_training_data_scale.py"
