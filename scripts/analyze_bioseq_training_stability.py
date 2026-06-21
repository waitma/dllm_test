#!/usr/bin/env python3
"""Summarize BioSeq train/validation logs and flag unstable loss transitions."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


TRAIN_PATTERN = re.compile(
    r"step=(?P<step>\d+)\s+loss=(?P<loss>\S+).*?"
    r"corrupt_rate=(?P<corrupt_rate>\S+)\s+lr=(?P<lr>\S+)"
)
VAL_PATTERN = re.compile(
    r"step=(?P<step>\d+)\s+val_loss=(?P<loss>\S+).*?"
    r"val_corrupt_rate=(?P<corrupt_rate>\S+)"
)


@dataclass(frozen=True)
class LossPoint:
    step: int
    loss: float
    corruption_rate: float
    lr: float | None = None


@dataclass(frozen=True)
class RunSummary:
    run: str
    log_path: str
    train_points: int
    val_points: int
    initial_train_loss: float | None
    minimum_train_loss: float | None
    final_train_loss: float | None
    best_val_loss: float | None
    best_val_step: int | None
    final_val_loss: float | None
    first_val_spike_step: int | None
    first_val_spike_loss: float | None
    first_nonfinite_train_step: int | None
    median_corruption_rate: float | None


def parse_float(value: str) -> float:
    return float(value.rstrip(","))


def parse_log(path: Path) -> tuple[list[LossPoint], list[LossPoint]]:
    train: list[LossPoint] = []
    val: list[LossPoint] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            match = VAL_PATTERN.search(line)
            if match:
                val.append(
                    LossPoint(
                        step=int(match.group("step")),
                        loss=parse_float(match.group("loss")),
                        corruption_rate=parse_float(match.group("corrupt_rate")),
                    )
                )
                continue
            match = TRAIN_PATTERN.search(line)
            if match:
                train.append(
                    LossPoint(
                        step=int(match.group("step")),
                        loss=parse_float(match.group("loss")),
                        corruption_rate=parse_float(match.group("corrupt_rate")),
                        lr=parse_float(match.group("lr")),
                    )
                )
    return train, val


def first_spike(
    points: list[LossPoint],
    *,
    ratio: float,
    minimum_loss: float,
    window: int,
) -> LossPoint | None:
    finite_history: list[float] = []
    for point in points:
        if not math.isfinite(point.loss):
            continue
        if len(finite_history) >= max(3, window):
            baseline = statistics.median(finite_history[-window:])
            if point.loss >= minimum_loss and point.loss >= baseline * ratio:
                return point
        finite_history.append(point.loss)
    return None


def infer_run_name(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("qwen3_vl_bioseq_"):
            return parent.name
    return path.parent.name


def summarize(path: Path, spike_ratio: float, spike_min_loss: float, spike_window: int) -> RunSummary:
    train, val = parse_log(path)
    finite_train = [point for point in train if math.isfinite(point.loss)]
    finite_val = [point for point in val if math.isfinite(point.loss)]
    best_val = min(finite_val, key=lambda point: point.loss) if finite_val else None
    spike = first_spike(val, ratio=spike_ratio, minimum_loss=spike_min_loss, window=spike_window)
    nonfinite = next((point for point in train if not math.isfinite(point.loss)), None)
    corruption_rates = [point.corruption_rate for point in train if math.isfinite(point.corruption_rate)]
    return RunSummary(
        run=infer_run_name(path),
        log_path=str(path),
        train_points=len(train),
        val_points=len(val),
        initial_train_loss=train[0].loss if train else None,
        minimum_train_loss=min((point.loss for point in finite_train), default=None),
        final_train_loss=train[-1].loss if train else None,
        best_val_loss=best_val.loss if best_val else None,
        best_val_step=best_val.step if best_val else None,
        final_val_loss=val[-1].loss if val else None,
        first_val_spike_step=spike.step if spike else None,
        first_val_spike_loss=spike.loss if spike else None,
        first_nonfinite_train_step=nonfinite.step if nonfinite else None,
        median_corruption_rate=statistics.median(corruption_rates) if corruption_rates else None,
    )


def discover_logs(paths: Iterable[Path]) -> list[Path]:
    logs: list[Path] = []
    for path in paths:
        if path.is_file():
            logs.append(path)
        elif path.is_dir():
            logs.extend(path.rglob("output.log"))
    return sorted(set(logs))


def fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="output.log files or directories containing them")
    parser.add_argument("--spike-ratio", type=float, default=2.0)
    parser.add_argument("--spike-min-loss", type=float, default=8.0)
    parser.add_argument("--spike-window", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a Markdown table")
    args = parser.parse_args()

    logs = discover_logs(args.paths)
    if not logs:
        raise SystemExit("no output.log files found")
    summaries = [
        summarize(path, args.spike_ratio, args.spike_min_loss, args.spike_window)
        for path in logs
    ]
    if args.json:
        print(json.dumps([asdict(summary) for summary in summaries], indent=2, allow_nan=True))
        return

    print(
        "| run | initial train | min train | best val (step) | final val | "
        "first val spike | first nonfinite | median corrupt |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        best_val = (
            f"{fmt(summary.best_val_loss)} ({fmt(summary.best_val_step)})"
            if summary.best_val_loss is not None
            else "-"
        )
        spike = (
            f"{fmt(summary.first_val_spike_loss)} ({fmt(summary.first_val_spike_step)})"
            if summary.first_val_spike_loss is not None
            else "-"
        )
        print(
            f"| {summary.run} | {fmt(summary.initial_train_loss)} | "
            f"{fmt(summary.minimum_train_loss)} | {best_val} | "
            f"{fmt(summary.final_val_loss)} | {spike} | "
            f"{fmt(summary.first_nonfinite_train_step)} | "
            f"{fmt(summary.median_corruption_rate)} |"
        )


if __name__ == "__main__":
    main()
