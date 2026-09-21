#!/usr/bin/env python3
"""Read-only fixed-canvas audit of manifest-listed prepared semantic JSONL.

Run after activating conda pllm (CPU only, no dataset index/cache or model load)::

    python -B scripts/debug/audit_v2_canvas_data.py \
        --dataset-dir data/prepared/immune_v6_binding_only \
        --dataset-dir data/prepared/immune_v3_heterotypic \
        --output-dir logs/v2_canvas_data_audit --timeout-seconds 570

The fast length calculation learns each task/role layout from the real renderer
using independent short fixtures, then verifies real rows against that renderer.
Oversize rows are counted, never cropped, filtered, or written back. Grammar
length is reported only where receptor canvases fit; rejected receptor rows do
not have a valid fixed-canvas rendering. Reports checkpoint after every shard.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dllm.pipelines.immune_llada.data.esm_encoding import HuggingFaceEsmTokenizerAdapter
from dllm.pipelines.immune_llada.data.grammar import (
    FIXED_HEAVY_DECODER_SLOTS,
    FIXED_HEAVY_ENCODER_LENGTH,
    FIXED_LIGHT_DECODER_SLOTS,
    FIXED_LIGHT_ENCODER_LENGTH,
    GrammarRenderer,
    GrammarTokenizer,
    TOKEN_CLASS_CHAIN_EOS,
    TOKEN_CLASS_RESIDUE,
    _build_per_chain_encoder_inputs,
)
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, VALID_PROTEIN_CHARS

SCHEMA = "immune_llada.semantic.v1"
ROLE_KIND = {
    "antibody_heavy": "heavy", "nanobody_vhh": "heavy", "tcr_beta": "heavy",
    "antibody_light": "light", "tcr_alpha": "light",
}
SLOTS = {"heavy": FIXED_HEAVY_DECODER_SLOTS, "light": FIXED_LIGHT_DECODER_SLOTS}
ENCODER = {"heavy": FIXED_HEAVY_ENCODER_LENGTH, "light": FIXED_LIGHT_ENCODER_LENGTH}
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def location(path, line, lengths, row):
    return {"shard": str(path), "line": line, "lengths": lengths,
            "roles": row["chain_roles"], "task_type": row["task_type"]}


def new_stats():
    return {
        "rows_checked": 0, "roles": {}, "record_sources": Counter(),
        "task_types": Counter(), "oversize_receptor_rows": 0,
        "rendered_oversize_receptor_rows": 0,
        "grammar_length_checked_rows": 0, "grammar_gt_1024_rows": 0,
        "max_grammar_length": 0, "incompatible_rows_union": 0,
        "renderer_mapping_issue_rows": 0, "examples": {},
    }


def make_plan(row, renderer):
    # Unique fixture lengths identify exactly which input chains the renderer
    # selects, reorders, drops, or duplicates. No prepared sequence is modified.
    roles = row["chain_roles"]
    fixture = {"chains": ["A" * (i + 1) for i in range(len(roles))],
               "chain_roles": roles, "task_type": row["task_type"], "source": "audit_fixture"}
    out = renderer.encode(BioSeqRecord.from_prepared_dict(fixture))
    groups = Counter(c for c, k in zip(out["position_ids_chain"], out["token_class_ids"])
                     if k == TOKEN_CLASS_RESIDUE)
    emitted = [(count - 1, out["chain_role_map"][c]) for c, count in sorted(groups.items())]
    overhead = sum(k not in {TOKEN_CLASS_RESIDUE, TOKEN_CLASS_CHAIN_EOS}
                   for k in out["token_class_ids"])
    selected = Counter(i for i, _ in emitted)
    issues = []
    if selected != Counter(range(len(roles))):
        issues.append("renderer does not emit every prepared chain exactly once")
    for i, kind in emitted:
        expected = ROLE_KIND.get(roles[i].strip().lower(), "context")
        if expected != kind:
            issues.append(f"input chain {i}: declared {expected}, rendered {kind}")
    return {"task_type": row["task_type"], "roles": roles, "emitted": emitted,
            "structure_relation_tokens": overhead, "grammar_name": out["grammar_name"],
            "mapping_issues": issues, "rows": 0, "actual_renderer_checks": 0,
            "actual_renderer_rejections": 0}


def verify_row(row, lengths, plan, renderer, tokenizer, expected_length, rejected):
    try:
        out = renderer.encode(BioSeqRecord.from_prepared_dict(row))
    except ValueError as exc:
        if not rejected or "fixed decoder canvas" not in str(exc):
            raise
        plan["actual_renderer_rejections"] += 1
        return
    if rejected:
        raise AssertionError("Expected receptor overflow, but renderer accepted the row")
    if len(out["input_ids"]) != expected_length:
        raise AssertionError((len(out["input_ids"]), expected_length, plan))
    chains, _, _, _ = _build_per_chain_encoder_inputs(
        out["input_ids"], out["token_class_ids"], out["position_ids_chain"],
        out["position_ids_inner"], tokenizer, chain_role_map=out["chain_role_map"],
    )
    expected_encoder = [ENCODER.get(kind, lengths[i] + 2) for i, kind in plan["emitted"]]
    if [len(c) for c in chains] != expected_encoder:
        raise AssertionError((list(map(len, chains)), expected_encoder, plan))
    plan["actual_renderer_checks"] += 1


def scan_dataset(directory, deadline, renderer, tokenizer, save):
    manifest_path = directory / "dataset_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["schema_version"] != SCHEMA or manifest["format"] != "jsonl":
        raise ValueError(f"Unsupported prepared format: {manifest_path}")
    report = {
        "dataset_dir": str(directory), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "schema_version": manifest["schema_version"], "created_at": manifest.get("created_at"),
        "manifest_budget": manifest.get("budget"), "status": "running",
        "manifest_rows": {}, "manifest_shards": {}, "groups": {}, "shards": [], "plans": [],
    }
    plans = {}
    save(report)
    for split in ("train", "valid"):
        info = manifest["splits"][split]
        report["manifest_rows"][split] = info["records"]
        report["manifest_shards"][split] = len(info["shards"])
        if sum(shard["records"] for shard in info["shards"]) != info["records"]:
            raise ValueError(f"Manifest count inconsistency: {split}")
        for shard in info["shards"]:
            path = (directory / shard["path"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError(f"Shard escapes dataset directory: {path}")
            source = shard["source"]  # Registry source, not row.source (papers uses tcr_native).
            group_key = f"{split}/{source}"
            stats = report["groups"].setdefault(group_key, new_stats())
            before = path.stat()
            shard_report = {"path": str(path), "split": split, "source": source,
                            "expected_rows": shard["records"], "rows_checked": 0,
                            "bytes": before.st_size, "complete": False}
            report["shards"].append(shard_report)
            with path.open("rb", buffering=1024 * 1024) as handle:
                for line_number, line in enumerate(handle, 1):
                    if line_number % 1024 == 1 and (STOP or time.monotonic() >= deadline):
                        report["status"] = "partial_time_limit"
                        save(report)
                        return report
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("schema_version") != SCHEMA or row.get("split") != split:
                        raise ValueError(f"Schema/split mismatch: {path}:{line_number}")
                    sequences, roles = row["chains"], row["chain_roles"]
                    if not sequences or len(sequences) != len(roles):
                        raise ValueError(f"Bad chains/roles: {path}:{line_number}")
                    # Prepared inputs are canonical; fail rather than silently
                    # estimating token counts on whitespace or multi-token text.
                    if any(not seq or not set(seq) <= VALID_PROTEIN_CHARS for seq in sequences):
                        raise ValueError(f"Noncanonical sequence: {path}:{line_number}")
                    lengths = list(map(len, sequences))
                    signature = (row["task_type"], tuple(roles))
                    if signature not in plans:
                        plans[signature] = make_plan(row, renderer)
                        report["plans"].append(plans[signature])
                    plan = plans[signature]
                    plan["rows"] += 1
                    oversize = False
                    offending_roles = set()
                    for role, length in zip(roles, lengths):
                        kind = ROLE_KIND.get(role.strip().lower())
                        maximum = SLOTS[kind] - 1 if kind else None
                        role_stats = stats["roles"].setdefault(role, {
                            "chains": 0, "max_length": 0, "residue_limit": maximum,
                            "offending_rows": 0, "offending_chains": 0,
                        })
                        role_stats["chains"] += 1
                        if length > role_stats["max_length"]:
                            role_stats["max_length"] = length
                            role_stats["max_location"] = {"shard": str(path), "line": line_number}
                        if maximum is not None and length > maximum:
                            oversize = True
                            offending_roles.add(role)
                            role_stats["offending_chains"] += 1
                    for role in offending_roles:
                        stats["roles"][role]["offending_rows"] += 1
                    rejected = any(kind in SLOTS and lengths[i] >= SLOTS[kind]
                                   for i, kind in plan["emitted"])
                    grammar_length = None if rejected else plan["structure_relation_tokens"] + sum(
                        SLOTS[kind] if kind in SLOTS else lengths[i] for i, kind in plan["emitted"]
                    )
                    grammar_over = grammar_length is not None and grammar_length > 1024
                    stats["rows_checked"] += 1
                    shard_report["rows_checked"] += 1
                    stats["record_sources"][row["source"]] += 1
                    stats["task_types"][row["task_type"]] += 1
                    stats["oversize_receptor_rows"] += int(oversize)
                    stats["rendered_oversize_receptor_rows"] += int(rejected)
                    stats["grammar_length_checked_rows"] += int(grammar_length is not None)
                    stats["grammar_gt_1024_rows"] += int(grammar_over)
                    stats["incompatible_rows_union"] += int(oversize or rejected or grammar_over)
                    stats["renderer_mapping_issue_rows"] += int(bool(plan["mapping_issues"]))
                    verify = plan["rows"] == 1 or shard_report["rows_checked"] == 1
                    for key, present in (("oversize_receptor", oversize), ("grammar_gt_1024", grammar_over),
                                         ("renderer_mapping", bool(plan["mapping_issues"]))):
                        if present and key not in stats["examples"]:
                            stats["examples"][key] = location(path, line_number, lengths, row)
                            verify = True
                    if grammar_length is not None and grammar_length > stats["max_grammar_length"]:
                        stats["max_grammar_length"] = grammar_length
                        stats["max_grammar_location"] = location(path, line_number, lengths, row)
                        verify = True
                    if verify:
                        verify_row(row, lengths, plan, renderer, tokenizer, grammar_length, rejected)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Shard changed during scan: {path}")
            if shard_report["rows_checked"] != shard["records"]:
                raise ValueError(f"Shard count mismatch: {shard_report}")
            shard_report["complete"] = True
            save(report)
            print(f"{directory.name} {shard['path']}: {shard_report['rows_checked']:,} rows; "
                  f"group role_over={stats['oversize_receptor_rows']:,}, "
                  f"grammar_over={stats['grammar_gt_1024_rows']:,}", flush=True)
    if manifest_path.read_bytes() != manifest_bytes:
        raise RuntimeError("Manifest changed during scan")
    report["status"] = "complete"
    save(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/v2_canvas_data_audit")
    parser.add_argument("--timeout-seconds", type=float, default=570)
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 590:
        parser.error("--timeout-seconds must be in (0, 590]")
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / "logs/v2_canvas_data_audit").resolve()):
        parser.error("output must be inside logs/v2_canvas_data_audit")
    directories = [p.resolve() for p in args.dataset_dir]
    if any(output.is_relative_to(p) for p in directories):
        parser.error("output must not be inside a dataset")
    output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    start = time.monotonic()
    tokenizer_path = ROOT / "model_weights/esmc/ESMC-300M"
    tokenizer = GrammarTokenizer(HuggingFaceEsmTokenizerAdapter.from_pretrained(tokenizer_path))
    alphabet = "".join(sorted(VALID_PROTEIN_CHARS))
    assert len(tokenizer.encode_residues(alphabet)) == len(alphabet)
    assert len(tokenizer.encode_residues("A" * 200)) == 200
    renderer = GrammarRenderer(tokenizer, fixed_receptor_lengths=True)
    report = {
        "read_only_datasets": True, "requested_dataset_dirs": list(map(str, directories)),
        "tokenizer_path": str(tokenizer_path), "tokenizer_one_token_per_residue_verified": True,
        "encoder_lengths_including_cls": ENCODER, "decoder_slots_including_eos": SLOTS,
        "residue_limits": {k: v - 1 for k, v in SLOTS.items()}, "max_grammar_length": 1024,
        "grammar_policy": "Exact for receptor-fitting rows; receptor-rejected rows are not renderable, not truncated",
        "source_grouping": "Manifest registry source; record.source counts retained separately",
        "datasets": [], "status": "running",
    }
    # Hash implementation inputs so a concurrent code edit cannot silently make
    # the on-disk renderer differ from the audit's recorded implementation.
    code_paths = [ROOT / "dllm/pipelines/immune_llada/data" / name
                  for name in ("grammar.py", "esm_encoding.py", "records.py")]
    report["code_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}

    def save(current):
        if not report["datasets"] or report["datasets"][-1]["dataset_dir"] != current["dataset_dir"]:
            report["datasets"].append(current)
        report["elapsed_seconds"] = round(time.monotonic() - start, 3)
        target = output / "audit.json"
        temporary = output / "audit.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(target)

    try:
        for directory in directories:
            current = scan_dataset(directory, start + args.timeout_seconds, renderer, tokenizer, save)
            if current["status"] != "complete":
                break
        report["status"] = "complete" if len(report["datasets"]) == len(directories) and all(
            d["status"] == "complete" for d in report["datasets"]
        ) else "partial_time_limit"
        report["code_unchanged_during_scan"] = all(
            hashlib.sha256(p.read_bytes()).hexdigest() == report["code_sha256"][str(p)] for p in code_paths
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if report["datasets"]:
            save(report["datasets"][-1])
        print(json.dumps({"status": report["status"], "output": str(output / "audit.json"),
                          "elapsed_seconds": round(time.monotonic() - start, 3)}), flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
