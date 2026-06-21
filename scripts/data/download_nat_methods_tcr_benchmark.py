#!/usr/bin/env python3
"""Download Nat Methods 2025 TCR benchmark from figshare."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

OUT_DIR = Path(
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/nat_methods_tcr_benchmark"
)
LOG = Path(
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/pipeline_logs/nat_methods_download.log"
)

# Figshare article 27020455 — file list via public API
FIGSHARE_ARTICLE = "27020455"
API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = msg.rstrip() + "\n"
    print(msg, flush=True)
    with LOG.open("a") as handle:
        handle.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    import urllib.request

    log(f"Fetching figshare metadata {API}")
    with urllib.request.urlopen(API, timeout=120) as resp:
        meta = json.loads(resp.read().decode())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = meta.get("files") or []
    log(f"Found {len(files)} files on figshare")

    manifest = []
    for item in files:
        name = item.get("name", "unknown")
        url = item.get("download_url")
        size = item.get("size", 0)
        target = args.output_dir / name
        if target.exists() and target.stat().st_size > 0:
            log(f"exists: {name}")
            manifest.append({"name": name, "path": str(target), "status": "exists"})
            continue
        if not url:
            log(f"skip (no url): {name}")
            continue
        log(f"downloading: {name} ({size} bytes)")
        urlretrieve(url, target)
        manifest.append({"name": name, "path": str(target), "status": "downloaded"})
        if name.endswith(".zip") and zipfile.is_zipfile(target):
            log(f"extracting: {name}")
            with zipfile.ZipFile(target) as zf:
                zf.extractall(args.output_dir / name.replace(".zip", ""))

    out = args.output_dir / "manifest.json"
    out.write_text(json.dumps({"article": FIGSHARE_ARTICLE, "files": manifest}, indent=2) + "\n")
    log(f"wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
