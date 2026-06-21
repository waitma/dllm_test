#!/usr/bin/env python3
"""Real-time monitor for BioSeq data pipeline jobs.

Example::

    python scripts/data/monitor_data_pipeline.py
    python scripts/data/monitor_data_pipeline.py --watch --interval 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
PROCESSED = PROJECT_ROOT / "data/ppi_task_raw/processed"
LOG_DIR = PROCESSED / "pipeline_logs"

JOBS = {
    "unified_csv": {
        "log": PROCESSED / "build_unified_csv.log",
        "output": PROCESSED / "interaction_records_unified.csv",
        "target_gb": 3.0,
        "pattern": "build_ppi_interaction_csv",
    },
    "string_sequences_redownload": {
        "log": LOG_DIR / "string_sequences_redownload.log",
        "output": PROJECT_ROOT / "data/ppi_task_raw/raw/stringdb_mint/protein.sequences.v12.0.fa.gz",
        "pattern": "aria2c",
    },
    "mmseqs_cluster": {
        "log": LOG_DIR / "mmseqs_cluster.log",
        "output": PROJECT_ROOT / "data/ppi_task_raw/raw/stringdb_mint/clu50.tsv",
        "pattern": "mmseqs cluster",
    },
    "mint_splits": {
        "log": LOG_DIR / "mint_splits.log",
        "output": PROJECT_ROOT / "data/ppi_task_raw/processed/mint_string_pretrain_v1/manifest.json",
        "pattern": "build_mint_string_splits",
    },
    "tcr_piste_shards": {
        "log": LOG_DIR / "tcr_piste_shards.log",
        "output": PROJECT_ROOT / "data/bioseq_grammar_v1/tcr_piste/train",
        "pattern": "build_tcr_grammar_shards",
    },
    "mint_grammar_shards": {
        "log": LOG_DIR / "mint_grammar_shards.log",
        "output": PROJECT_ROOT / "data/bioseq_grammar_v1/mint_ppi/train",
        "pattern": "build_mint_grammar_shards",
    },
    "nat_methods_download": {
        "log": LOG_DIR / "nat_methods_download.log",
        "output": PROJECT_ROOT / "data/ppi_task_raw/raw/nat_methods_tcr_benchmark",
        "pattern": "download_nat_methods",
    },
}


def file_gb(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_size / (1024**3)


def tail_lines(path: Path, n: int = 3) -> list[str]:
    if not path.exists():
        return []
    try:
        text = path.read_text(errors="replace")
        return [line for line in text.splitlines()[-n:] if line.strip()]
    except OSError:
        return []


def pgrep(pattern: str) -> list[str]:
    try:
        out = subprocess.check_output(["pgrep", "-af", pattern], text=True, stderr=subprocess.DEVNULL)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except subprocess.CalledProcessError:
        return []


def job_status(name: str, cfg: dict) -> dict:
    output = Path(cfg["output"])
    log_path = Path(cfg["log"])
    running = bool(pgrep(cfg["pattern"]))
    done = output.exists() and (output.is_dir() or output.stat().st_size > 0)
    size_gb = file_gb(output) if output.is_file() else None
    target = cfg.get("target_gb")
    progress = None
    if size_gb is not None and target:
        progress = min(100.0, round(100 * size_gb / target, 1))
    return {
        "name": name,
        "running": running,
        "done": done,
        "output": str(output),
        "size_gb": round(size_gb, 3) if size_gb is not None else None,
        "progress_pct": progress,
        "log_tail": tail_lines(log_path),
    }


def snapshot() -> dict:
    rows = [job_status(name, cfg) for name, cfg in JOBS.items()]
    return {"timestamp": datetime.utcnow().isoformat() + "Z", "jobs": rows}


def print_snapshot(data: dict) -> None:
    print(f"\n=== Pipeline monitor @ {data['timestamp']} ===")
    for job in data["jobs"]:
        state = "RUNNING" if job["running"] else ("DONE" if job["done"] else "IDLE")
        extra = ""
        if job.get("size_gb") is not None:
            extra = f" size={job['size_gb']}GB"
        if job.get("progress_pct") is not None:
            extra += f" progress={job['progress_pct']}%"
        print(f"[{state:7}] {job['name']:22}{extra}")
        for line in job.get("log_tail") or []:
            print(f"         | {line[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--json-out", type=Path, default=PROCESSED / "pipeline_monitor.json")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        data = snapshot()
        print_snapshot(data)
        args.json_out.write_text(json.dumps(data, indent=2) + "\n")
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
