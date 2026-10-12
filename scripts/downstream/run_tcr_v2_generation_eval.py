"""Official T4 adapter for v2 fixed-canvas checkpoints.

Historical ``run_tcr_native_generation.py`` still requires the v5
generation_preflight gate, ``BetaGenerationProtocol.profile`` (length prior),
and ``n_valid == 100``. Those checks reject empty CDR3s that the v2
full-beta-EOS extractor is required to keep. This runner reuses
``V5BetaSampler`` (already dispatches to ``FixedBetaGenerationProtocol``) and
the official scorer. It is not a training-code change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from downstream.grammar.ab_features import file_sha256, write_json
from downstream.grammar.tcr_generation_v5 import V5BetaSampler
from scripts.downstream.prepare_tcr_native_jobs import CKPT, OUT, ROOT, TAG
from scripts.downstream.run_tcr_native_generation import (
    AUDIT,
    AUDIT_SHA,
    BENCH,
    EVAL,
    overlap_flags,
    output_for,
    score_result,
)
from scripts.downstream.tcr_eval_pin import PIN


def identity(sampler):
    files = [
        Path(__file__),
        ROOT / "downstream/grammar/tcr_generation_v5.py",
        ROOT / "downstream/grammar/tcr_generation_v2.py",
        ROOT / "dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py",
        ROOT / "downstream/grammar/common.py",
        BENCH / "tcr_generation_bench/scoring.py",
        BENCH / "common/metrics.py",
        BENCH / "data/tcr_generation_bench/bioseq_unseen_pmhc.json",
        BENCH / "data/tcr_generation_bench/train_binders.csv",
        BENCH / "tcr_generation_bench/result_io.py",
    ]
    if file_sha256(AUDIT) != AUDIT_SHA:
        raise ValueError("v5 overlap audit identity changed")
    audit = json.loads(AUDIT.read_text())
    if any(audit["integrity_errors"].values()) or audit["eval_sha256"] != file_sha256(EVAL):
        raise ValueError("overlap audit does not cover this evaluation file")
    return {
        "checkpoint": str(CKPT),
        "checkpoint_sha256": PIN.sha,
        "global_step": PIN.step,
        "protocol": sampler.protocol.provenance,
        "eval_sha256": file_sha256(EVAL),
        "overlap_audit_sha256": AUDIT_SHA,
        "code_and_scoring_sha256": {str(path): file_sha256(path) for path in files},
        "length_protocol": "v2_full_beta_eos_then_cdr3",
        "empty_cdr3_policy": "retain_on_disk_scorer_drops_empty_strings",
    }


def assert_v2_checkpoint():
    if file_sha256(CKPT / "model.safetensors") != PIN.sha:
        raise ValueError("not the approved checkpoint")
    step = json.loads((CKPT / "trainer_state.json").read_text())["global_step"]
    if int(step) != PIN.step:
        raise ValueError(f"not step {PIN.step}")


def prepare_output(output, ident, restart_incomplete=False):
    """Archive failed output only on explicit retry; never overwrite results."""
    if output.exists():
        previous = json.loads((output / "run_manifest.json").read_text())
        if previous.get("identity", {}).get("checkpoint_sha256") != ident["checkpoint_sha256"]:
            raise ValueError("Existing T4 output belongs to another checkpoint")
        if previous.get("status") == "success" or (output / "metrics.json").exists():
            raise FileExistsError("Completed T4 results exist; refusing to replace them")
        if not restart_incomplete:
            raise FileExistsError("Incomplete T4 output exists; use --restart-incomplete")
        from datetime import datetime, timezone
        archive = output.with_name(output.name + ".incomplete_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        output.rename(archive)
    output.mkdir(parents=True, exist_ok=False)


def run(task, restart_incomplete=False):
    assert_v2_checkpoint()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    sampler = V5BetaSampler(CKPT)
    if not sampler.fixed_v2:
        raise ValueError("refusing a non-v2 checkpoint; use run_tcr_native_generation.py")
    ident = identity(sampler)
    output = output_for(task)
    prepare_output(output, ident, restart_incomplete)
    (OUT / f"t4-{task}").mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "identity": ident,
        "dataset": task,
        "paired_generation": False,
        "output": str(output),
        "runner": str(Path(__file__)),
        "tag": TAG,
    }
    write_json(output / "run_manifest.json", manifest)
    try:
        entries = {
            key: value
            for key, value in json.loads(EVAL.read_text()).items()
            if value["category"] == task
        }
        expected = 14 if task == "benchmark14" else 20
        if len(entries) != expected:
            raise ValueError(f"{task} entry count {len(entries)} != {expected}")
        designs, greedy = {}, {}
        greedy_length = sampler.protocol.lengths(42, "greedy", 1)[0]
        with (output / "designs.jsonl").open("x") as handle:
            for key, entry in entries.items():
                raw = sampler.generate(entry["epitope"], entry["mhc_pseudo"], key)
                control, checks = sampler.sample_records(
                    [sampler.protocol.record(
                        entry["epitope"], entry["mhc_pseudo"], greedy_length, "greedy:" + key
                    )],
                    seed=42,
                    strategy="argmax",
                )
                record = {
                    "pmhc": key,
                    "epitope": entry["epitope"],
                    "mhc_pseudo": entry["mhc_pseudo"],
                    "sequences": [item["sequence"] for item in raw],
                    "greedy": control[0]["sequence"],
                    "candidates": raw,
                    "greedy_raw": control[0],
                    "greedy_checks": checks,
                    "n_requested": 100,
                    "n_raw": len(raw),
                    "n_valid": sum(bool(item["valid"]) for item in raw),
                    "v5_overlap": overlap_flags(key),
                    "protocol": "tcr_v2_full_beta_eos_then_cdr3",
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                designs[key] = record["sequences"]
                greedy[key] = record["greedy"]
                print(
                    f"generated {key}: {record['n_valid']}/100 valid CDR3, "
                    f"greedy_valid={bool(control[0]['valid'])}",
                    flush=True,
                )
        result = score_result(task, designs, greedy, entries, ident)
        write_json(output / "metrics.json", result)
        manifest["status"] = "success"
        manifest["n_targets"] = len(entries)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(output / "run_manifest.json", manifest)
        write_json(OUT / f"t4-{task}/run_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["benchmark14", "held20"], required=True)
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    run(args.task, args.restart_incomplete)


if __name__ == "__main__":
    main()
