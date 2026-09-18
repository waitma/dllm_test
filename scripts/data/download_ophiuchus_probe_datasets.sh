#!/usr/bin/env bash
# Download the two public probe datasets used by Ophiuchus-Ab Table 5 / Fig 4:
#   1. CurrAb HD-Flu-CoV classification CSV (Zenodo 14661302)
#   2. Desautels m396 in-silico ΔΔG supplementary zip (bioRxiv 10.1101/2020.04.03.024885)
# GDPa1 is gated on Hugging Face and is NOT fetched here; see
# data/downstream/PROBE_DATA.md.
# Uses the lab HTTPS proxy. Safe to re-run.
set -uo pipefail

ROOT="/vepfs-mlp2/c20250601/251105016/project/dllm_test"
PROXY="${LAB_HTTPS_PROXY:-http://100.68.162.212:3128}"
export http_proxy="$PROXY" https_proxy="$PROXY" HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
unset ALL_PROXY all_proxy 2>/dev/null || true

# Zenodo file CDN rejects Chrome <144 with a fake "unusual traffic from your
# network" 403. Chrome/120 was the whole failure mode, not the URL or the IP.
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
SPEC_RAW="$ROOT/data/downstream/specificity/raw"
INS_RAW="$ROOT/data/downstream/in_silico/raw"
INS_OFF="$INS_RAW/official"
SPEC_OFF="$SPEC_RAW/official"
mkdir -p "$SPEC_RAW" "$SPEC_OFF" "$INS_RAW" "$INS_OFF"

md5of() { md5sum "$1" 2>/dev/null | awk '{print $1}'; }

fetch() {
  local out="$1"
  local url="$2"
  local want_md5="${3:-}"
  local max_time="${4:-600}"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" && -n "$want_md5" && "$(md5of "$out")" == "$want_md5" ]]; then
    echo "SKIP $(basename "$out") md5 ok"
    return 0
  fi
  echo "GET  $(basename "$out")"
  echo "     $url"
  if curl -fL --retry 8 --retry-delay 8 --retry-all-errors \
      --connect-timeout 30 --max-time "$max_time" \
      -A "$UA" -o "$out" "$url"; then
    if [[ -n "$want_md5" ]]; then
      local got
      got="$(md5of "$out")"
      if [[ "$got" != "$want_md5" ]]; then
        echo "WARN md5 mismatch $(basename "$out") got=$got want=$want_md5"
        return 1
      fi
    fi
    ls -lh "$out"
    return 0
  fi
  echo "FAIL $(basename "$out")"
  return 1
}

echo "=== $(date -Is) start ophiuchus probe dataset download ==="

rc=0
LOCAL_SPEC="$ROOT/data/downstream/specificity/hd-0_flu-1_cov-2.csv"
# CurrAb classification-datasets.tar.gz, 5.4 MB, md5 from Zenodo API.
# Needs Chrome/144+ UA. Chrome/120 gets a fake "unusual traffic" 403.
if ! fetch \
    "$SPEC_OFF/classification-datasets.tar.gz" \
    "https://zenodo.org/api/records/14661302/files/classification-datasets.tar.gz/content" \
    "e5049ecdac31020b6ac45e684b38e87a" \
    600; then
  if ! fetch \
      "$SPEC_OFF/classification-datasets.tar.gz" \
      "https://zenodo.org/records/14661302/files/classification-datasets.tar.gz?download=1" \
      "e5049ecdac31020b6ac45e684b38e87a" \
      600; then
    if [[ -s "$LOCAL_SPEC" ]]; then
      echo "WARN Zenodo 14661302 still 403; keep local CurrAb table at $LOCAL_SPEC"
    else
      echo "FAIL CurrAb official tar and no local table"
      rc=1
    fi
  fi
fi
if [[ -s "$SPEC_OFF/classification-datasets.tar.gz" ]]; then
  cp -f "$SPEC_OFF/classification-datasets.tar.gz" "$SPEC_RAW/classification-datasets.tar.gz"
fi

# m396 WT sequences (already fetched once; keep resumable).
if fetch \
  "$INS_OFF/rcsb_pdb_2G75.fasta" \
  "https://www.rcsb.org/fasta/entry/2G75" \
  "" \
  60; then
  cp -f "$INS_OFF/rcsb_pdb_2G75.fasta" "$INS_RAW/2G75.fasta"
else
  echo "WARN RCSB 2G75 fasta download failed"
fi

# Desautels supplementary zip. bioRxiv rate-limits; try a few known Highwire paths.
DES_ZIP="$INS_OFF/024885_file02.zip"
if [[ ! -s "$DES_ZIP" && -s "$INS_RAW/024885_file02.zip" ]]; then
  cp -f "$INS_RAW/024885_file02.zip" "$DES_ZIP"
fi
if [[ ! -s "$DES_ZIP" ]]; then
  ok=0
  for url in \
    "https://www.biorxiv.org/content/biorxiv/early/2020/04/10/2020.04.03.024885/DC1/embed/media-1.zip" \
    "https://www.biorxiv.org/content/10.1101/2020.04.03.024885v1.supplementary-material" \
    "https://www.biorxiv.org/highwire/filestream/218930/field_highwire_adjunct_files/0/024885_file02.zip" \
    "https://www.biorxiv.org/highwire/filestream/218930/field_highwire_adjunct_files/1/024885_file02.zip"
  do
    if fetch "$DES_ZIP" "$url" "" 900; then
      # Reject HTML error pages.
      if file "$DES_ZIP" | grep -qi 'zip\|gzip\|compress'; then
        ok=1
        break
      fi
      echo "WARN not a zip: $(file "$DES_ZIP")"
      rm -f "$DES_ZIP"
    fi
    sleep 8
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "FAIL Desautels supplementary zip"
    rc=1
  fi
else
  echo "SKIP $(basename "$DES_ZIP") already present ($(du -h "$DES_ZIP" | awk '{print $1}'))"
fi
if [[ -s "$DES_ZIP" ]]; then
  cp -f "$DES_ZIP" "$INS_RAW/024885_file02.zip"
fi

echo "=== $(date -Is) download done rc=$rc ==="
exit "$rc"
