"""Monitor and audit the authorized v5 92000 Specificity budget experiments.

Run in protenix_abtcr, after activating pllm, with CUDA_VISIBLE_DEVICES='':
  python scripts/downstream/monitor_ab_specificity_budgets.py --wait
Without --wait, perform one read-only producer check and write monitor artifacts.
Never submits, cancels, restarts producers, or changes their files. Publishes
only the owned RESULTS marker and terminal rows/log marker in PROJECT_PROCESS.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from downstream.grammar.ab_features import ROOT, file_sha256, write_json
from downstream.grammar.ab_probes import load_specificity


BASE = ROOT / "output/downstream_generation"
JOBS = {300: "t-20260915143055-j547p", 400: "t-20260915143055-pgzq6",
        500: "t-20260915143055-xzftx"}
SOURCE_HASH = "276981b93993375f1e6f270a9199bd487a410b515354aa16069d66b17619bb0e"
CHECKPOINT_HASH = "2f003a5b42a15498a7d6f2fc51414cad717f96a9cb405dca2388fcef0de3039c"
SCORE_KEYS = ("accuracy", "f1_macro", "mcc")
CURVE_KEYS = ("train_loss", "test_loss", *SCORE_KEYS,
              "lr_first_step", "lr_last_step", "lr_next_step")
RESULT_START = "<!-- ab92000-budget-results:start -->"
RESULT_END = "<!-- ab92000-budget-results:end -->"
PROCESS_START = "<!-- ab92000-budget-terminal:start -->"
PROCESS_END = "<!-- ab92000-budget-terminal:end -->"


def read(path):
    return json.loads(path.read_text())


def directory(epochs):
    return BASE / f"ab_v5_92000_specificity_ep{epochs}_20260915"


def close(actual, expected):
    if not (math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-10)):
        raise ValueError(f"Numeric mismatch: {actual} != {expected}")


def audit(epochs, reference_manifest, frame, folds):
    """Recompute saved prediction/curve metrics; do not repeat model inference."""
    folder = directory(epochs)
    manifest = read(folder / "run_manifest.json")
    result = read(folder / "metrics.json")
    for key in ("runner_sha256", "data_sha256", "sample_ids", "runtime_versions"):
        if manifest[key] != reference_manifest[key]:
            raise ValueError(f"{epochs}: manifest mismatch {key}")
    for key, value in manifest["args"].items():
        if key not in ("epochs", "out_dir") and value != reference_manifest["args"][key]:
            raise ValueError(f"{epochs}: changed argument {key}")
    assert manifest["args"]["epochs"] == result["selected_epoch"] == epochs
    assert result["protocol"] == "native_ab_fixed_last_head_5fold_v1"
    assert result["feature_manifest"]["checkpoint_sha256"] == CHECKPOINT_HASH
    assert file_sha256(folder / "run_manifest.json") == result["run_manifest_sha256"]
    assert file_sha256(folder / "features.pt") == SOURCE_HASH
    assert result["feature_manifest"] == read(directory(200) / "metrics.json")["feature_manifest"]
    source_files = [folder / name for name in
                    ("run_manifest.json", "metrics.json", "features.pt", "epoch_mean.json", "oof_predictions.csv")]
    scores, curves, predictions = [], [], []
    for k, (train, test) in enumerate(folds):
        part = folder / f"fold_{k}"
        curve = read(part / "epoch_metrics.json")
        history = read(part / "train_history.json")
        selection = read(part / "selection.json")
        assert [row["epoch"] for row in curve] == list(range(1, epochs + 1))
        assert [row["epoch"] for row in history] == list(range(1, epochs + 1))
        assert selection["selected_epoch"] == epochs and selection["seed"] == 42 + k
        assert selection["rule"] == "fixed_last_no_test_selection"
        assert selection["train_ids"] == frame.name.iloc[train].tolist()
        assert selection["test_ids"] == frame.name.iloc[test].tolist()
        assert curve[-1]["lr_next_step"] == 0
        assert len(list(part.glob("diagnostic_head_epoch_*.pt"))) == epochs // 20
        head = torch.load(part / "selected_head.pt", map_location="cpu", weights_only=False)
        last = torch.load(part / f"diagnostic_head_epoch_{epochs:03d}.pt", map_location="cpu", weights_only=False)
        assert head["selected_epoch"] == last["epoch"] == epochs
        assert head["selection_rule"] == "fixed_last" and head["seed"] == 42 + k
        assert head["state_dict"].keys() == last["state_dict"].keys()
        assert all(torch.equal(v, last["state_dict"][n]) for n, v in head["state_dict"].items())
        pred = pd.read_csv(part / "predictions.csv")
        assert pred.sample_id.tolist() == frame.name.iloc[test].tolist()
        assert np.array_equal(pred.y_true, frame.label.iloc[test])
        assert (pred.fold == k).all()
        assert np.array_equal(pred.y_pred, pred[[f"logit_{i}" for i in range(3)]].to_numpy().argmax(1))
        score = {"accuracy": accuracy_score(pred.y_true, pred.y_pred),
                 "f1_macro": f1_score(pred.y_true, pred.y_pred, average="macro"),
                 "mcc": matthews_corrcoef(pred.y_true, pred.y_pred)}
        stored = result["folds"][k]
        assert stored["fold"] == k and stored["n_train"] == len(train) and stored["n_test"] == len(test)
        for key, value in score.items():
            close(stored[key], value)
            close(curve[-1][key], value)
        for row, train_row in zip(curve, history):
            close(row["train_loss"], train_row["train_loss"])
            assert all(math.isfinite(row[key]) for key in CURVE_KEYS)
        scores.append(score)
        curves.append(curve)
        predictions.append(pred)
        source_files.extend(part / name for name in
                            ("predictions.csv", "epoch_metrics.json", "train_history.json",
                             "selection.json", "selected_head.pt", f"diagnostic_head_epoch_{epochs:03d}.pt"))
    combined = pd.concat(predictions, ignore_index=True)
    oof = pd.read_csv(folder / "oof_predictions.csv")
    pd.testing.assert_frame_equal(combined, oof, check_exact=False, atol=1e-12, rtol=1e-12)
    assert len(oof) == oof.sample_id.nunique() == len(frame) == 4398
    means = read(folder / "epoch_mean.json")
    assert len(means) == epochs
    for i, row in enumerate(means):
        assert row["epoch"] == i + 1
        for key in CURVE_KEYS:
            close(row[key], np.mean([c[i][key] for c in curves]))
    for key in SCORE_KEYS:
        close(result["mean"][key], np.mean([r[key] for r in scores]))
        close(result["sample_sd_ddof1"][key], np.std([r[key] for r in scores], ddof=1))
    return {"status": "passed", "epochs": epochs, "checkpoint_step": 92000,
            "metrics": result["mean"], "sample_sd_ddof1": result["sample_sd_ddof1"],
            "final_train_loss": means[-1]["train_loss"], "final_eval_loss": means[-1]["test_loss"],
            "diagnostic_min_eval_loss_epoch": min(means, key=lambda r: r["test_loss"])["epoch"],
            "source_sha256": {str(p): file_sha256(p) for p in source_files},
            "scope": "Saved predictions and fold/curve aggregates; no backbone or head inference rerun"}


def remote_states():
    states = {}
    for epochs, job in JOBS.items():
        proc = subprocess.run(["bash", str(ROOT / "scripts/volc-no-proxy.sh"), "ml_task", "get",
                               "-i", job, "-o", "json"], cwd=ROOT, capture_output=True, text=True, timeout=25)
        if proc.returncode:
            raise RuntimeError(f"Status CLI failed for {job}: {proc.stderr[-300:]}")
        payloads = [line for line in proc.stdout.splitlines() if line.startswith('[{"')]
        records = json.loads(payloads[-1])
        record = next(r for r in records if r["JobId"] == job)
        assert record["JobName"] == f"ab-v5-92000-specificity-ep{epochs}-20260915"
        states[str(epochs)] = record
    return states


def replace_owned(text, start, end, body):
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("Owned Markdown markers missing or duplicated; refusing to overwrite")
    left, tail = text.split(start, 1)
    _, right = tail.split(end, 1)
    return left + start + "\n" + body.strip("\n") + "\n" + end + right


def save_if_unchanged(path, before, after):
    """Mechanical publication with concurrent-edit guard; keep all unowned text."""
    if before == after:
        return
    temp = path.with_name(path.name + f".ab92000-{os.getpid()}.tmp")
    with temp.open("x") as handle:
        handle.write(after)
    if path.read_text() != before:
        temp.unlink()
        raise RuntimeError(f"Concurrent Markdown edit: {path}; publication stopped")
    temp.replace(path)


def publish(status, verified, out_dir, *, result_path=None, process_path=None):
    """Never publish an epoch result before both audit pass and platform Success."""
    result_path = result_path or ROOT / "downstream/benchmark/RESULTS.md"
    process_path = process_path or ROOT / "PROJECT_PROCESS.md"
    before = result_path.read_text()
    if before.count(RESULT_START) != 1 or before.count(RESULT_END) != 1:
        raise ValueError("RESULTS publication markers missing")
    existing = before.split(RESULT_START, 1)[1].split(RESULT_END, 1)[0]
    kept = [line for line in existing.strip("\n").splitlines()
            if not any(line.startswith(f"| Ours v5 step92000 {e}轮 |") for e in JOBS)]
    for epochs in JOBS:
        if epochs in verified and status["jobs"][str(epochs)]["Status"] == "Success":
            row = verified[epochs]
            cells = [f"{row['final_train_loss']:.6f}", f"{row['final_eval_loss']:.6f}",
                     *(f"{row['metrics'][k]:.6f}" for k in SCORE_KEYS)]
        else:
            cells = ["待完整验收", "—", "—", "—", "—"]
        kept.append(f"| Ours v5 step92000 {epochs}轮 | " + " | ".join(cells) + " |")
    after = replace_owned(before, RESULT_START, RESULT_END, "\n".join(kept))
    save_if_unchanged(result_path, before, after)
    terminal = {e: record for e, record in status["jobs"].items()
                if record["Status"] in ("Success", "Failed", "Killed")}
    if not terminal:
        return
    before = process_path.read_text()
    body = []
    for epochs, record in sorted(terminal.items(), key=lambda item: int(item[0])):
        report = out_dir / f"ep{epochs}_audit.json"
        evidence = f"[完整验收]({report})" if int(epochs) in verified else "未通过完整验收，不收数"
        body.append(f"- `{record['JobId']}` / {epochs}轮：**{record['Status']}**；End={record.get('End')}；{evidence}。")
    body.append("- 数字只进入 RESULTS §0.6a 的训练预算诊断表；未按曲线最优轮选模。其余任务与原始产物未修改。")
    after = replace_owned(before, PROCESS_START, PROCESS_END, "\n".join(body))
    ids = {r["JobId"] for r in terminal.values()}
    lines, section = [], ""
    for line in after.splitlines():
        if line.startswith("## "):
            section = line
        if (section in ("## Active Volc Training Tasks", "## Active Volc Evaluation Tasks")
                and line.startswith("|") and any(f"`{job}`" in line for job in ids)):
            continue
        lines.append(line)
    after = "\n".join(lines) + "\n"
    if before != after:
        stamp = status["updated_utc"].replace("+00:00", "Z")
        after = re.sub(r"(?m)^> Last updated: .*", f"> Last updated: {stamp}", after, count=1)
        for title in ("## Active Volc Training Tasks", "## Active Volc Evaluation Tasks"):
            start = after.index(title)
            end = after.find("\n## ", start + 1)
            if end == -1:
                end = len(after)
            section = re.sub(r"(?m)^Last updated: .*", f"Last updated: {stamp}（92000预算实验终态同步；其他任务保持各自查询快照）",
                             after[start:end], count=1)
            after = after[:start] + section + after[end:]
        save_if_unchanged(process_path, before, after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--extra-job", help="Additional authorized budget: EPOCHS:TASK_ID")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=BASE / "ab_v5_92000_specificity_budget_audit_20260915")
    args = parser.parse_args()
    if args.extra_job:
        epoch, job = args.extra_job.split(":", 1)
        if int(epoch) in JOBS or int(epoch) == 200:
            parser.error("Budget already registered")
        JOBS[int(epoch)] = job
    monitor_hash = file_sha256(Path(__file__))
    if args.poll_seconds < 10:
        parser.error("poll interval must be at least 10 seconds")
    frame, _, _, folds = load_specificity()
    reference = read(directory(200) / "run_manifest.json")
    verified = {200: audit(200, reference, frame, folds)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "ep200_audit.json", verified[200])
    while True:
        status = {"updated_utc": datetime.now(timezone.utc).isoformat(), "complete": False,
                  "monitor_sha256": monitor_hash}
        try:
            states = remote_states()
            status["jobs"] = states
            status["progress"] = {}
            for epochs in JOBS:
                status["progress"][str(epochs)] = {
                    p.parent.name: len(read(p)) for p in sorted(directory(epochs).glob("fold_*/epoch_metrics.json"))}
                if epochs not in verified and (directory(epochs) / "metrics.json").exists():
                    verified[epochs] = audit(epochs, reference, frame, folds)
                    write_json(args.out_dir / f"ep{epochs}_audit.json", verified[epochs])
            status["audited_epochs"] = sorted(verified)
            status["complete"] = len(verified) == len(JOBS) + 1 and all(r["Status"] == "Success" for r in states.values())
            if any(r["Status"] in ("Failed", "Killed") for r in states.values()):
                status["attention_required"] = "A producer ended without Success; no automatic retry"
            publish(status, verified, args.out_dir)
            if status["complete"]:
                write_json(args.out_dir / "comparison.json", {
                    **status, "rows": [verified[e] for e in sorted(verified)],
                    "interpretation": "Independent fixed-last budgets, not checkpoints of one 500-epoch run; diagnostic, not blind selection"})
        except (json.JSONDecodeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            # Producers can be between file creation and their completed write.
            status["complete"] = False
            status["retryable_read_error"] = repr(exc)
        except Exception as exc:
            status["complete"] = False
            status["attention_required"] = repr(exc)
        write_json(args.out_dir / "status.json", status)
        print(json.dumps(status), flush=True)
        if status["complete"] or status.get("attention_required") or not args.wait:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
