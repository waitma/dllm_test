#!/usr/bin/env python3
"""Submit humanization CSV sequences to the public BioPhi humanness endpoint."""

from __future__ import annotations

import argparse
import json
import re
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

BIOPHI_BASE = "https://biophi.dichlab.org"
HUMANNESS_URL = f"{BIOPHI_BASE}/humanization/humanness/"


def csv_to_fasta_batches(
    csv_path: Path,
    output_dir: Path,
    batch_size: int = 50,
) -> tuple[list[Path], int]:
    df = pd.read_csv(csv_path)
    required = {"pdb_id", "variant_idx", "generated_heavy", "generated_light"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_paths: list[Path] = []
    for batch_idx, start in enumerate(range(0, len(df), batch_size)):
        chunk = df.iloc[start : start + batch_size]
        batch_path = output_dir / f"batch_{batch_idx:02d}.fa"
        with batch_path.open("w") as handle:
            for row_idx, row in chunk.iterrows():
                antibody_id = f"{row['pdb_id']}_v{int(row['variant_idx'])}_row{int(row_idx)}"
                handle.write(f">{antibody_id}_VH\n{row['generated_heavy']}\n")
                handle.write(f">{antibody_id}_VL\n{row['generated_light']}\n")
        batch_paths.append(batch_path)
    return batch_paths, len(df)


def submit_batch(
    batch_path: Path,
    *,
    scheme: str = "imgt",
    cdr_definition: str = "imgt",
    threshold: str = "relaxed",
    timeout: int = 600,
) -> dict:
    with batch_path.open("rb") as handle:
        response = requests.post(
            HUMANNESS_URL,
            data={
                "input_mode": "bulk",
                "scheme": scheme,
                "cdr_definition": cdr_definition,
                "min_subjects": threshold,
            },
            files={"sequence_files[]": (batch_path.name, handle, "application/octet-stream")},
            allow_redirects=False,
            timeout=timeout,
        )

    if response.status_code not in {302, 303}:
        raise RuntimeError(
            f"BioPhi submit failed for {batch_path.name}: HTTP {response.status_code}. "
            "The public endpoint may be temporarily unavailable; retry later.\n"
            f"{response.text[:500]}"
        )

    location = response.headers.get("Location", "")
    report_url = urljoin(BIOPHI_BASE, location)
    match = re.search(r"/report/([0-9a-f-]{36})", report_url)
    if not match:
        raise RuntimeError(f"Could not parse task id from redirect: {report_url}")
    task_id = match.group(1)
    return {
        "task_id": task_id,
        "report_url": report_url if report_url.endswith("/") else f"{report_url}/",
        "xls_url": f"{BIOPHI_BASE}/humanization/humanness/report/{task_id}/oasis.xls",
    }


def wait_for_xls(xls_url: str, *, poll_seconds: float = 5.0, timeout_seconds: float = 900.0) -> bytes:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        response = requests.get(xls_url, timeout=60)
        if response.status_code == 200 and response.content[:2] == b"PK":
            return response.content
        last_error = f"HTTP {response.status_code}, len={len(response.content)}"
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for {xls_url}: {last_error}")


def parse_oasis_xls(content: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(content))


def summarize_oasis(df: pd.DataFrame) -> dict:
    def mean_col(name: str) -> float:
        if name not in df.columns:
            raise KeyError(f"Missing column {name!r}; available={df.columns.tolist()}")
        return float(df[name].mean())

    overall = mean_col("OASis Identity")
    heavy = mean_col("Heavy OASis Identity")
    light = mean_col("Light OASis Identity")
    germline = mean_col("Germline Content")
    heavy_germline = mean_col("Heavy Germline Content")
    light_germline = mean_col("Light Germline Content")
    return {
        "n_antibodies": int(len(df)),
        "threshold": str(df["Threshold"].iloc[0]) if "Threshold" in df.columns else "relaxed",
        "scheme": "imgt",
        "cdr_definition": "imgt",
        "oasis_identity_mean": overall,
        "oasis_identity_percent": overall * 100.0,
        "heavy_oasis_identity_mean": heavy,
        "heavy_oasis_identity_percent": heavy * 100.0,
        "light_oasis_identity_mean": light,
        "light_oasis_identity_percent": light * 100.0,
        "germline_content_mean": germline,
        "heavy_germline_content_mean": heavy_germline,
        "light_germline_content_mean": light_germline,
    }


def run(args: argparse.Namespace) -> None:
    csv_path = args.csv_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_paths, n_rows = csv_to_fasta_batches(csv_path, output_dir, batch_size=args.batch_size)
    tasks: list[dict] = []
    summary_frames: list[pd.DataFrame] = []

    for batch_idx, batch_path in enumerate(batch_paths):
        print(f"[batch {batch_idx}] submitting {batch_path.name}")
        task = submit_batch(
            batch_path,
            scheme=args.scheme,
            cdr_definition=args.cdr_definition,
            threshold=args.threshold,
        )
        task.update({"batch": batch_idx, "fasta": str(batch_path), "n": min(args.batch_size, n_rows - batch_idx * args.batch_size)})
        print(f"[batch {batch_idx}] task_id={task['task_id']}")
        xls_bytes = wait_for_xls(task["xls_url"])
        xls_path = output_dir / f"batch_{batch_idx:02d}_oasis.xls"
        xls_path.write_bytes(xls_bytes)
        frame = parse_oasis_xls(xls_bytes)
        frame.insert(0, "batch", batch_idx)
        summary_frames.append(frame)
        tasks.append(task)

    combined = pd.concat(summary_frames, ignore_index=True)
    summary_csv = output_dir / f"{csv_path.stem}_biophi_oasis_{args.threshold}_summary.csv"
    combined.to_csv(summary_csv, index=False)
    metrics = summarize_oasis(combined)
    metrics_path = output_dir / "summary_metrics.json"
    tasks_path = output_dir / "tasks.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    tasks_path.write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {summary_csv}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {tasks_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score humanization CSV via public BioPhi humanness endpoint.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to output/downstream_generation/ophiuchus_ab/biophi_public_<csv_stem>_<threshold>/",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--scheme", default="imgt")
    parser.add_argument("--cdr-definition", default="imgt")
    parser.add_argument("--threshold", default="relaxed", choices=["loose", "relaxed", "medium", "strict"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        stem = args.csv_path.stem
        args.output_dir = Path("output/downstream_generation/ophiuchus_ab") / f"biophi_public_{stem}_{args.threshold}"
    run(args)


if __name__ == "__main__":
    main()
