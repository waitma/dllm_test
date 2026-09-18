"""Gated 49000 beta-only conditional generation, with raw candidates retained.

Activate pllm then protenix_abtcr. Run --task preflight, then --task benchmark14
or --task held20. Each target uses 100 stochastic candidates and one separately
labelled argmax control. Never overwrites an existing experiment directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from downstream.grammar.ab_features import file_sha256, write_json
from downstream.grammar.tcr_generation_v5 import BetaGenerationProtocol, V5BetaSampler
from scripts.downstream.prepare_tcr_native_jobs import ROOT, CKPT, OUT, TAG
from scripts.downstream.tcr_eval_pin import PIN
from scripts.downstream.run_tcr_native_repr import assert_checkpoint, EXPECTED_SHA
from downstream.benchmark.tcr_generation_bench.result_io import prepare_metrics_for_json

BENCH = ROOT / "downstream/benchmark"
EVAL = BENCH / "data/tcr_generation_bench/eval_conditional.json"
AUDIT = BENCH / "outputs/tcr_generation_bench/audit_v5_overlap_20260913/full_scan_v2/report.json"
AUDIT_SHA = "3848531613f0108e6c3978906809fc5eedb9d1afbfa810dbaf1b5dea10a6c072"
DEFAULT_GATE = OUT / "generation_preflight_v2/passed.json"
LEGACY_RUNNER_SHA = "9cae845fa6b24d7cb66fab4e596eb41093da3077086db751566cb5a1364bb792"


def output_for(task):
    return BENCH / "outputs/tcr_generation_bench/setting_B" / f"ours_{TAG}_{task}_pmhc_beta_v2"


def identity():
    protocol = BetaGenerationProtocol()
    files = [Path(__file__), ROOT / "downstream/grammar/tcr_generation_v5.py",
        ROOT / "dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py",
        ROOT / "downstream/grammar/common.py", BENCH / "tcr_generation_bench/scoring.py",
        BENCH / "common/metrics.py", BENCH / "data/tcr_generation_bench/bioseq_unseen_pmhc.json",
        BENCH / "data/tcr_generation_bench/train_binders.csv",
        BENCH / "tcr_generation_bench/result_io.py"]
    if file_sha256(AUDIT) != AUDIT_SHA:
        raise ValueError("v5 overlap audit identity changed")
    audit = json.loads(AUDIT.read_text())
    if any(audit["integrity_errors"].values()) or audit["eval_sha256"] != file_sha256(EVAL):
        raise ValueError("overlap audit does not cover this evaluation file")
    return {"checkpoint": str(CKPT), "checkpoint_sha256": EXPECTED_SHA, "global_step": PIN.step,
        "protocol": protocol.provenance, "eval_sha256": file_sha256(EVAL),
        "overlap_audit_sha256": AUDIT_SHA, "code_and_scoring_sha256": {str(p): file_sha256(p) for p in files}}


def preflight(path=DEFAULT_GATE):
    if path.exists():
        raise FileExistsError(path)
    assert_checkpoint()
    ident = identity()
    sampler = V5BetaSampler(CKPT)
    p = sampler.protocol
    entries = json.loads(EVAL.read_text())
    for key, entry in entries.items():
        for length in p.lengths(42, key, 100):
            p.record(entry["epitope"], entry["mhc_pseudo"], length, key)
    # A/G placeholder perturbation must yield identical decoder predictions:
    # both target streams must be hidden, including the first ESMC forward.
    entry = next(iter(entries.values()))
    args = (entry["epitope"], entry["mhc_pseudo"])
    outputs = []
    for char in ("A", "G"):
        records = [p.record(*args, length, str(i), placeholder=char) for i, length in enumerate([12, 82])]
        generated, checks = sampler.sample_records(records, seed=42, max_iter=4, strategy="argmax")
        if not all(x["valid"] for x in generated):
            raise ValueError("invalid GPU generation tokens")
        outputs.append(generated)
    if outputs[0] != outputs[1]:
        raise ValueError("target placeholder leaked into predictions")
    # Exercise the production strategy and full 32-step ceiling.
    generated, checks = sampler.sample_records([p.record(*args, 12, "stochastic")], seed=42)
    if not all(x["valid"] for x in generated):
        raise ValueError("invalid stochastic production-path tokens")
    # Exercise the formerly missing real scorer -> strict writer -> reader path.
    # Synthetic designs here are a gate fixture, not a quality evaluation.
    sample_key = next(k for k, e in entries.items() if e["category"] == "benchmark14"
                      and k != "RVRAYTYSK_HLA-A*03:01")
    sample_entry = entries[sample_key]
    fixture = score_result("benchmark14", {sample_key: ["CASSF", "CATF"] * 50},
        {sample_key: "CASSF"}, {sample_key: sample_entry}, ident)
    fixture_path = path.parent / "strict_json_roundtrip.json"
    write_json(fixture_path, fixture)
    if json.loads(fixture_path.read_text()) != fixture:
        raise ValueError("strict scoring JSON roundtrip failed")
    write_json(path, {"status": "passed", "identity": ident, "quality_evaluation": False,
        "strict_scoring_json_roundtrip": True, "fixture_sha256": file_sha256(fixture_path),
        "placeholder_invariant": True, "all_steps_fixed_context_invariant": True,
        "n_inputs_layout_checked": 3400, "max_core_length_tested": 82,
        "gpu": torch.cuda.get_device_name(0), "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
    print("TCR_V5_BETA_GENERATION_GPU_GATE_PASSED", flush=True)


def overlap_flags(pmhc):
    groups = json.loads(AUDIT.read_text())["per_pmhc"][pmhc]["groups"]
    return {f"{split}_{field}_seen": any(value[field] > 0 for key, value in groups.items()
            if key.startswith(split + "/"))
        for split in ("train", "valid") for field in ("epitope_targets", "beta_references", "pair_references")}


def score_result(task, designs, greedy, entries, ident):
    from downstream.benchmark.tcr_generation_bench import scoring
    result = scoring.score_conditional(designs, entries, k=100, greedy_map=greedy)
    for key in list(result["summary"]):
        if key.startswith("bioseq_"):
            result["summary"]["historical_" + key] = result["summary"].pop(key)
    for key, item in result["per_pmhc"].items():
        item["historical_bioseq_unseen"] = item.pop("bioseq_unseen")
        item["v5_overlap"] = overlap_flags(key)
    allowed = [key for key in entries if key != scoring.RESERVED_SIMULATION_PMHC]
    for seen in (False, True):
        members = [key for key in allowed if overlap_flags(key)["train_epitope_targets_seen"] == seen]
        if members:
            result["summary"]["v5_train_epitope_" + ("seen" if seen else "unseen")] = scoring._macro(result["per_pmhc"], members)
    if task == "benchmark14":
        result["summary"]["paper_sparse13"] = scoring._macro(result["per_pmhc"], allowed)
    result.update(identity=ident, dataset=task, candidate_count=100, greedy_count=1,
        input_condition="epitope_and_provided_34mer_mhc_no_reference_tcr",
        interpretation="sequence_similarity_not_experimental_binding_validation")
    conditioning = scoring.load_conditioning_by_epitope()
    missing = {k for k, e in entries.items() if not conditioning.get(e["epitope"])}
    return prepare_metrics_for_json(result, missing)


def verify_recovery_identity(source_identity, current_identity):
    """Permit only this audited runner/JSON-boundary repair, not sampler drift."""
    old, new = dict(source_identity), dict(current_identity)
    old_files = dict(old.pop("code_and_scoring_sha256"))
    new_files = dict(new.pop("code_and_scoring_sha256"))
    if old != new:
        raise ValueError("recovery checkpoint/protocol/data identity mismatch")
    runner = str(ROOT / "scripts/downstream/run_tcr_native_generation.py")
    if old_files.pop(runner) != LEGACY_RUNNER_SHA:
        raise ValueError("not the approved serialization-failure runner")
    new_files.pop(runner)
    new_files.pop(str(BENCH / "tcr_generation_bench/result_io.py"))
    if old_files != new_files:
        raise ValueError("recovery sampler/scoring/reference code drift")


def recover(task, source, gate_path=DEFAULT_GATE):
    """CPU-only rescore of complete saved generation, never invoke the sampler."""
    from scripts.downstream.audit_tcr_49000_generation_results import validate_raw
    ident = identity()
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "passed" or gate["identity"] != ident:
        raise ValueError("recovery requires the repaired generation gate")
    original = json.loads((source / "run_manifest.json").read_text())
    legacy_gate_path = OUT / "generation_preflight/passed.json"
    legacy_gate = json.loads(legacy_gate_path.read_text())
    if (original["status"] != "failed" or original["dataset"] != task
            or original["paired_generation"] is not False
            or "Out of range float values" not in original.get("error", "")
            or legacy_gate["status"] != "passed" or original["identity"] != legacy_gate["identity"]):
        raise ValueError("source is not the known gated serialization failure")
    verify_recovery_identity(original["identity"], ident)
    entries, raw, designs, greedy = validate_raw(task, source)
    source_files = [source / "run_manifest.json", source / "designs.jsonl", legacy_gate_path]
    provenance = {"source": str(source.resolve()), "generation_identity": original["identity"],
        "source_sha256": {str(p): file_sha256(p) for p in source_files},
        "mode": "cpu_rescore_saved_sequences_no_regeneration"}
    result = score_result(task, designs, greedy, entries, ident)
    result["recovery"] = provenance
    output = output_for(task)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "identity": ident, "dataset": task,
        "paired_generation": False, "output": str(output), "recovery": provenance,
        "gate": str(gate_path), "gate_sha256": file_sha256(gate_path)}
    try:
        shutil.copyfile(source / "designs.jsonl", output / "designs.jsonl")
        if file_sha256(output / "designs.jsonl") != provenance["source_sha256"][str(source / "designs.jsonl")]:
            raise ValueError("saved designs changed during recovery")
        write_json(output / "metrics.json", result)
        manifest["status"] = "success"
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(output / "run_manifest.json", manifest)
    print("TCR_SAVED_GENERATION_RESCORE_SUCCESS", output, flush=True)


def run(task, gate_path=DEFAULT_GATE):
    assert_checkpoint()
    ident = identity()
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "passed" or gate["identity"] != ident:
        raise ValueError("input/source/checkpoint changed after gate")
    output = output_for(task)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "identity": ident, "dataset": task,
        "paired_generation": False, "output": str(output),
        "gate": str(gate_path), "gate_sha256": file_sha256(gate_path)}
    write_json(output / "run_manifest.json", manifest)
    try:
        entries = {key: value for key, value in json.loads(EVAL.read_text()).items() if value["category"] == task}
        assert len(entries) == (14 if task == "benchmark14" else 20)
        # Only these condition fields ever enter the generator.
        conditions = {key: {field: value[field] for field in ("epitope", "mhc_pseudo")}
            for key, value in entries.items()}
        sampler = V5BetaSampler(CKPT)
        designs, greedy = {}, {}
        dist = sampler.protocol.profile["region_distributions"]["tcr_beta"]["CDR3"]
        modal_length = int(max(dist, key=dist.get))
        with (output / "designs.jsonl").open("x") as fh:
            for key, condition in conditions.items():
                raw = sampler.generate(condition["epitope"], condition["mhc_pseudo"], key)
                control, checks = sampler.sample_records([sampler.protocol.record(
                    condition["epitope"], condition["mhc_pseudo"], modal_length, "greedy:" + key)],
                    seed=42, strategy="argmax")
                record = {"pmhc": key, **condition, "sequences": [x["sequence"] for x in raw],
                    "greedy": control[0]["sequence"], "candidates": raw, "greedy_raw": control[0],
                    "n_requested": 100, "n_raw": len(raw), "n_valid": sum(x["valid"] for x in raw),
                    "v5_overlap": overlap_flags(key)}
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                # Keep all raw attempts on disk; do not replace rejected candidates.
                if record["n_valid"] != 100 or not control[0]["valid"]:
                    raise ValueError("invalid candidates retained; scoring blocked rather than silently filtered")
                designs[key], greedy[key] = record["sequences"], record["greedy"]
                print(f"generated {key}: {record['n_valid']}/100 valid", flush=True)
        result = score_result(task, designs, greedy, entries, ident)
        write_json(output / "metrics.json", result)
        manifest["status"] = "success"
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(output / "run_manifest.json", manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["preflight", "benchmark14", "held20"], required=True)
    p.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    p.add_argument("--rescore-from", type=Path)
    args = p.parse_args()
    if args.task == "preflight":
        if args.rescore_from:
            p.error("preflight cannot rescore")
        preflight(args.gate)
    elif args.rescore_from:
        recover(args.task, args.rescore_from, args.gate)
    else:
        run(args.task, args.gate)


if __name__ == "__main__":
    main()
