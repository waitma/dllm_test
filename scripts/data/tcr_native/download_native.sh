#!/usr/bin/env bash
# Download native full-length TCR Fv sources into data/tcr_native/.
# Resumable + md5-verified. Uses the lab HTTPS proxy.
set -u
ROOT=/vepfs-mlp2/c20250601/251105016/project/dllm_test
DEST=$ROOT/data/tcr_native
export http_proxy="http://100.68.162.212:3128"
export https_proxy="http://100.68.162.212:3128"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

# "outpath|md5|url"
MANIFEST=$(cat <<'EOF'
10x/donor1_all_contig_annotations.csv|6430a0286fd9cb693bf880f8d55a7c13|https://zenodo.org/api/records/6952657/files/vdj_v1_hs_aggregated_donor1_all_contig_annotations.csv/content
10x/donor2_all_contig_annotations.csv|b581477de193a155aad1d48af84bf7e5|https://zenodo.org/api/records/6952657/files/vdj_v1_hs_aggregated_donor2_all_contig_annotations.csv/content
10x/donor3_all_contig_annotations.csv|0883bc7115c247fb00334a7b3cf8ca6d|https://zenodo.org/api/records/6952657/files/vdj_v1_hs_aggregated_donor3_all_contig_annotations.csv/content
10x/donor4_all_contig_annotations.csv|b5971d538782b66b329ae708f3751391|https://zenodo.org/api/records/6952657/files/vdj_v1_hs_aggregated_donor4_all_contig_annotations.csv/content
10x/donor1_binarized_matrix.csv|48c1693b9b8a1608519639023395b3d5|https://zenodo.org/api/records/6952657/files/vdj_v1_hs_aggregated_donor1_binarized_matrix.csv/content
10x/donor2_binarized_matrix.csv|0c216150eedce01bd7d9d76a401219c3|https://zenodo.org/api/records/6952657/files/vdj_v1_hs_aggregated_donor2_binarized_matrix.csv/content
10x/donor3_binarized_matrix.csv|ae7bdf80f48cc9fea037c88c2604e00d|https://zenodo.org/api/records/6952657/files/vdj_v1_hs_aggregated_donor3_binarized_matrix.csv/content
10x/donor4_binarized_matrix.csv|c2db252e8694c7791363e0e764c96e5f|https://zenodo.org/api/records/6952657/files/vdj_v1_hs_aggregated_donor4_binarized_matrix.csv/content
10x/donor1_clonotypes.csv|2428c5d59a9af399ee063da8ad57928e|https://zenodo.org/api/records/6952657/files/vdj_v1_hs_aggregated_donor1_clonotypes.csv/content
10x/donor2_clonotypes.csv|f7a9a10620b6dd23d2cb270717e1b389|https://zenodo.org/api/records/6952657/files/vdj_v1_hs_aggregated_donor2_clonotypes.csv/content
10x/donor3_clonotypes.csv|aa03e221203de5432795239208e1fbf5|https://zenodo.org/api/records/6952657/files/vdj_v1_hs_aggregated_donor3_clonotypes.csv/content
10x/donor4_clonotypes.csv|800f404e7527441c5578a2bb6a123f52|https://zenodo.org/api/records/6952657/files/vdj_v1_hs_aggregated_donor4_clonotypes.csv/content
minervina/libs_aggregate_rev.zip|6087540fd1d5562b29ffed37c582f634|https://zenodo.org/api/records/6232103/files/libs_aggregate_rev.zip/content
covidvac/02_dex_annotated_cd8.h5ad|bc6f364346e5e28169f1e34d3542585b|https://zenodo.org/api/records/15691612/files/02_dex_annotated_cd8.h5ad/content
EOF
)

md5of() { md5sum "$1" 2>/dev/null | awk '{print $1}'; }

echo "=== download start $(date -u +%FT%TZ) ==="
FAIL=0
while IFS='|' read -r rel md5 url; do
  [ -z "$rel" ] && continue
  out="$DEST/$rel"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(md5of "$out")" = "$md5" ]; then
    echo "SKIP (md5 ok) $rel"
    continue
  fi
  echo "GET $rel"
  for attempt in 1 2 3; do
    curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 --max-time 3600 \
         -o "$out" "$url" 2>>"$DEST/_meta/download.log"
    got=$(md5of "$out")
    if [ "$got" = "$md5" ]; then
      echo "OK   $rel  ($got)"
      break
    fi
    echo "RETRY $attempt md5 mismatch $rel got=$got want=$md5"
    [ "$attempt" = 3 ] && { echo "FAILED $rel"; FAIL=1; }
    sleep 5
  done
done <<< "$MANIFEST"
echo "=== download done $(date -u +%FT%TZ) FAIL=$FAIL ==="
echo "DOWNLOAD_EXIT $FAIL"
