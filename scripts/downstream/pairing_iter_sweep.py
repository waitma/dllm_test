"""Run the fixed-checkpoint, no-prefix pairing step diagnostic.

Activate pllm; use protenix_abtcr for GPU workers. From the repository root:
  python scripts/downstream/pairing_iter_sweep.py prepare
  python scripts/downstream/pairing_iter_sweep.py run --steps 8
  python scripts/downstream/pairing_iter_sweep.py summarize
Workers are submitted separately through eval_jobs, not launched by prepare.
Production generation/scoring are reused unchanged. Existing runs are never overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from downstream.grammar.ab_features import ROOT, file_sha256, write_json

STEPS = (8, 16, 32, 64, 96, 128)
TAG = "ab_v5_49000_pairing_iters_p0_20260916"
OUT = ROOT / "output/downstream_generation" / TAG
REFERENCE = ROOT / "output/downstream_generation/ab_v5_49000_pairing_encoder_sync_20260916/pair-p0-cfg0"
FIXED_KEYS = (
    "checkpoint_path", "checkpoint_sha256", "input_csv_sha256", "dataset_len",
    "start_index", "max_samples", "heavy_batch_size", "num_seqs", "sampling_strategy",
    "temperature", "cfg_scale", "light_prompt_tokens", "light_length_mode", "seed",
    "protocol_version", "encoder_state", "source_sha256", "cfg_condition", "cfg_formula",
)
SCORE_KEYS = (
    "gen_immunomatch_mean", "ref_immunomatch_mean", "gen_better_ratio", "gen_valid_rate",
    "overall_chain_match_rate", "overall_v_gene_match_rate", "overall_j_gene_match_rate",
    "overall_v_gene_family_match_rate", "overall_j_gene_family_match_rate", "diversity_mean",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_result(directory, reference_manifest, reference_rows, steps):
    """Check all candidates and denominators; do not discard invalid sequences."""
    directory = Path(directory)
    manifest = read_json(directory / "holdout500_n8_manifest.json")
    metrics = read_json(directory / "holdout500_metrics.json")
    for key in FIXED_KEYS:
        if manifest[key] != reference_manifest[key]:
            raise ValueError(f"iter{steps}: changed {key}")
    if manifest["max_iter"] != steps:
        raise ValueError("Wrong iteration count")
    for key, value in metrics["evaluation_protocol"].items():
        if key in manifest and value != manifest[key]:
            raise ValueError(f"Metrics/manifest disagree: {key}")
    if metrics["evaluation_protocol"].get("max_iter") != steps:
        raise ValueError("Metrics iteration count missing or wrong")
    rows = read_csv(directory / "holdout500_n8.csv")
    n = manifest["dataset_len"] * manifest["num_seqs"]
    details = metrics["detailed_results"]
    if not (len(rows) == len(reference_rows) == len(details) == metrics["total_pairs"] == n):
        raise ValueError("Incomplete candidate/annotation count")
    identities = Counter()
    for index, (row, ref, detail) in enumerate(zip(rows, reference_rows, details)):
        for key in ("h_sequence", "raw_l_sequence", "_input_row_idx", "variant_idx"):
            if row[key] != ref[key]:
                raise ValueError(f"Candidate identity/order mismatch at {index}: {key}")
        identities[(row["_input_row_idx"], int(row["variant_idx"]))] += 1
        if detail["index"] != index:
            raise ValueError("Annotation order mismatch")
        if int(row["target_light_length"]) != len(row["raw_l_sequence"]):
            raise ValueError("Reference-length allocation mismatch")
        if int(row["generated_light_length"]) != len(row["gen_l_sequence"]):
            raise ValueError("Generated length metadata mismatch")
        if int(row["light_prompt_tokens"]) != 0 or float(row["cfg_scale"]) != 0:
            raise ValueError("Not a no-prefix CFG0 run")
        if row["generation_protocol_version"] != "5" or row["light_length_mode"] != "reference":
            raise ValueError("Wrong generation protocol")
    expected = {(r["_input_row_idx"], j) for r in reference_rows for j in range(manifest["num_seqs"])}
    if set(identities) != expected or any(v != 1 for v in identities.values()):
        raise ValueError("Missing/duplicated candidate identities")
    if len({r["h_sequence"] for r in rows}) != manifest["dataset_len"]:
        raise ValueError("Wrong heavy group count")
    for key in SCORE_KEYS:
        if not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
            raise ValueError(f"Invalid score: {key}")
    for key in ("chain", "v_gene", "j_gene", "v_gene_family", "j_gene_family"):
        value = sum(bool(d[f"{key}_match"]) for d in details) / n
        if not math.isclose(value, metrics[f"overall_{key}_match_rate"], abs_tol=1e-12):
            raise ValueError(f"Wrong all-candidate denominator: {key}")
    if sum(bool(d["gen_valid"]) for d in details) / n != metrics["gen_valid_rate"]:
        raise ValueError("Wrong validity denominator")
    from scripts.downstream.pairing_leakage_diagnostic import audit
    if audit(directory / "holdout500_n8.csv") != metrics["generation_audit"]:
        raise ValueError("Generation audit differs")
    return {"max_iter": steps, "n_candidates": n, **{k: metrics[k] for k in SCORE_KEYS},
            "metrics": str(directory / "holdout500_metrics.json"),
            "metrics_sha256": file_sha256(directory / "holdout500_metrics.json"),
            "csv_sha256": file_sha256(directory / "holdout500_n8.csv"),
            "manifest_sha256": file_sha256(directory / "holdout500_n8_manifest.json")}


def verify_inputs(plan, *, weights=False):
    for path, digest in plan["pinned_files"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"Pinned source/data/reference changed: {path}")
    if weights:
        ref = plan["reference_manifest"]
        if file_sha256(Path(ref["checkpoint_path"]) / "model.safetensors") != ref["checkpoint_sha256"]:
            raise ValueError("Retained checkpoint changed")


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    ref = read_json(REFERENCE / "holdout500_n8_manifest.json")
    for key, expected in {"dataset_len": 500, "num_seqs": 8, "light_prompt_tokens": 0,
                          "cfg_scale": 0, "protocol_version": 5, "max_iter": 124}.items():
        if ref[key] != expected:
            raise ValueError(f"Unexpected reference configuration: {key}")
    reference_result = validate_result(REFERENCE, ref, read_csv(REFERENCE / "holdout500_n8.csv"), 124)
    pinned = dict(ref["source_sha256"])
    pinned[ref["csv_path"]] = ref["input_csv_sha256"]
    paths = [Path(__file__), ROOT / "scripts/downstream/preflight_pairing_encoder_feedback.py",
             ROOT / "scripts/downstream/score_pairing_generation.py",
             ROOT / "scripts/downstream/pairing_leakage_diagnostic.py",
             ROOT / "downstream/grammar/ab_features.py"]
    paths += list((ROOT / "downstream/comp_chain/eval_scripts").glob("*.py"))
    paths += [REFERENCE / name for name in ("holdout500_n8.csv", "holdout500_metrics.json", "holdout500_n8_manifest.json")]
    jobs = [ROOT / f"eval_jobs/eval_{TAG}_iter{s}.yml" for s in STEPS]
    for path in paths + jobs:
        pinned[str(path.resolve())] = file_sha256(path)
    plan = {"created": now(), "iterations": STEPS, "reference_manifest": ref,
            "reference_dir": str(REFERENCE), "reference_result": reference_result,
            "pinned_files": pinned, "jobs": [str(p) for p in jobs],
            "protocol": "no_prefix_pairing_step_diagnostic_v1", "seed": 42,
            "selection": "report every step; no test-set-selected headline",
            "seed_limitation": "same seed, different step budgets consume different RNG streams",
            "paper_reference": {"model": "Ophiuchus-Ab", "prompt": 0, "immunomatch": 0.701,
                                "source": "/root/oph_paper/oph.pdf Table 3", "length_condition": "native generation"}}
    verify_inputs(plan, weights=True)
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / "experiment_manifest.json", plan)
    summarize()
    print(f"Prepared {len(STEPS)} independent single-GPU jobs; no submission performed.", flush=True)


def summarize():
    plan = read_json(OUT / "experiment_manifest.json")
    with (OUT / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify_inputs(plan)
        reference_rows = read_csv(Path(plan["reference_dir"]) / "holdout500_n8.csv")
        result = {"updated": now(), "complete": False, "complete_means": "artifacts validated; not a platform-status query",
                  "reference_iter124": plan["reference_result"], "paper_reference": plan["paper_reference"],
                  "results": {}, "pending": [], "errors": {}, "states": {}}
        for steps in STEPS:
            directory = OUT / f"iter{steps}"
            state = directory / "status.json"
            if state.exists():
                result["states"][str(steps)] = read_json(state)
            if not (directory / "holdout500_metrics.json").exists():
                result["pending"].append(steps)
                continue
            try:
                result["results"][str(steps)] = validate_result(directory, plan["reference_manifest"], reference_rows, steps)
            except Exception as exc:
                result["errors"][str(steps)] = str(exc)
        result["complete"] = len(result["results"]) == len(STEPS) and not result["errors"]
        write_json(OUT / "summary.json", result)
        print(f"Summary: {len(result['results'])}/{len(STEPS)} validated, errors={result['errors']}", flush=True)
        return result


def run(steps):
    plan = read_json(OUT / "experiment_manifest.json")
    verify_inputs(plan, weights=True)
    ref = plan["reference_manifest"]
    directory = OUT / f"iter{steps}"
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    state = {"started": now(), "pid": os.getpid(), "host": socket.gethostname(), "max_iter": steps}

    def status(phase, **fields):
        state.update(phase=phase, updated=now(), elapsed_seconds=time.monotonic() - started, **fields)
        write_json(directory / "status.json", state)

    def command(phase, args):
        status(phase)
        print(f"{now()} iter{steps}: {phase}", flush=True)
        with (directory / f"{phase}.log").open("w") as log:
            subprocess.run([sys.executable, "-u", *args], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    try:
        command("feedback_gate", ["scripts/downstream/preflight_pairing_encoder_feedback.py",
                "--checkpoint", ref["checkpoint_path"], "--csv", ref["csv_path"], "--prompt", "0",
                "--output", str(directory / "encoder_feedback_gate.json")])
        gate = read_json(directory / "encoder_feedback_gate.json")
        if not all(gate[k] for k in ("passed", "hidden_reference_invariance", "encoder_decoder_state_match")):
            raise ValueError("Real-weight feedback gate failed")
        for key in ("checkpoint_sha256", "input_csv_sha256", "source_sha256"):
            if gate[key] != ref[key]:
                raise ValueError(f"Gate identity changed: {key}")
        verify_inputs(plan)
        command("generation", ["-m", "downstream.grammar.light_chain_pairing",
                "--checkpoint-path", ref["checkpoint_path"], "--csv-path", ref["csv_path"],
                "--output-csv", str(directory / "holdout500.csv"), "--device", "cuda",
                "--heavy-batch-size", "2", "--num-seqs", "8", "--light-prompt-tokens", "0",
                "--cfg-scale", "0", "--light-length-mode", "reference", "--max-iter", str(steps),
                "--sampling-strategy", "gumbel_argmax", "--temperature", "1", "--seed", "42", "--skip-eval"])
        verify_inputs(plan)
        command("scoring", ["scripts/downstream/score_pairing_generation.py",
                "--csv", str(directory / "holdout500_n8.csv"),
                "--metrics", str(directory / "holdout500_metrics.json"), "--num-seqs", "8"])
        verify_inputs(plan, weights=True)
        validate_result(directory, ref, read_csv(REFERENCE / "holdout500_n8.csv"), steps)
        status("complete")
        summarize()
    except BaseException:
        status("failed", error=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--steps", type=int, choices=STEPS)
    args = parser.parse_args()
    if args.action == "run" and args.steps is None:
        parser.error("run requires --steps")
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run(args.steps)
    else:
        summarize()


if __name__ == "__main__":
    main()
