"""Audit full immune data multiplicities against an isolated pre-refactor commit.

Run from the repository root with --background for a bounded offline job. Raw
CSV, blocklists and prepared shards are read-only. Digest streams, pinned legacy
sources, progress and reports live in a fresh --work-dir (no automatic resume).
A bounded --max-rows run checks prefixes, not whole-corpus parity.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarRenderer, GrammarTokenizer
from dllm.pipelines.immune_llada.data.preprocessing.filters import build_filters, filter_reason, load_blocklists
from dllm.pipelines.immune_llada.data.preprocessing.pipeline import load_preprocess_config
from dllm.pipelines.immune_llada.data.preprocessing.validators import validate_manifest, validate_prepared_row
from dllm.pipelines.immune_llada.data.registry import parse_sources, source_spec, source_split_path
from dllm.pipelines.immune_llada.data.sources import row_to_record

BASELINE_COMMIT = "7f6f2351702dc307f710bce7228a53eae391b26f"
LEGACY_FILES = {
    "adapters": "dllm/pipelines/bioseq/adapters.py",
    "datasets": "dllm/pipelines/bioseq/datasets.py",
    "records": "dllm/pipelines/qwen3_vl_arch/data/records.py",
    "esm_encoding": "dllm/pipelines/qwen3_vl_arch/data/esm_encoding.py",
    "mixture": "dllm/pipelines/qwen3_vl_arch/data/mixture.py",
    "grammar": "dllm/pipelines/qwen3_vl_arch/data/grammar.py",
    "train": "examples/llada/protein_pretrain_esmc.py",
}
FAILURE_KEYS = (
    "decision_mismatches", "semantic_mismatches", "legacy_record_errors",
    "rendering_mismatches", "collator_mismatches", "count_mismatches",
    "legacy_multiset_differences", "canonical_multiset_differences",
)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def small_file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), **file_stat(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_legacy(snapshot_dir: Path, commit: str) -> SimpleNamespace:
    """Load Git blobs in their own namespace, without the live alias modules.

    Only three data-builder functions and the old dataset class are executed
    from the training script; importing that whole script would load transformers
    and current fusion code, invalidating an independent data-path comparison.
    """
    import torch

    snapshot_dir.mkdir(parents=True, exist_ok=False)
    package_name = "_immune_parity_legacy"
    package = ModuleType(package_name)
    package.__path__ = [str(snapshot_dir)]
    sys.modules[package_name] = package
    identities = {}
    modules = {}
    for name, git_path in LEGACY_FILES.items():
        blob = subprocess.check_output(["git", "--no-pager", "show", f"{commit}:{git_path}"], cwd=_ROOT)
        path = snapshot_dir / f"{name}.py"
        path.write_bytes(blob)
        identities[git_path] = hashlib.sha256(blob).hexdigest()
        if name == "train":
            continue
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    train_source = (snapshot_dir / "train.py").read_text()
    wanted = {"_record_chain_lengths_ok", "with_length_filter", "build_immune_specs", "ImmuneBioSeqDataset"}
    nodes = [node for node in ast.parse(train_source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise RuntimeError("Baseline does not contain the expected pre-refactor data implementation")
    namespace = dict(vars(modules["datasets"]))
    namespace.update({"Dataset": torch.utils.data.Dataset, "BioSeqChain": modules["records"].BioSeqChain,
                      "BioSeqRecord": modules["records"].BioSeqRecord, "logger": logging.getLogger("legacy_parity")})
    extracted = "from __future__ import annotations\n\n" + "\n\n".join(ast.get_source_segment(train_source, node) for node in nodes)
    (snapshot_dir / "extracted_training_data.py").write_text(extracted)
    exec(compile(extracted, str(snapshot_dir / "extracted_training_data.py"), "exec"), namespace)
    return SimpleNamespace(**modules, builder=namespace["build_immune_specs"],
                           wrapper=namespace["ImmuneBioSeqDataset"]([]), hashes=identities)


def legacy_record(legacy: SimpleNamespace, row: dict[str, Any]):
    # Same construction as the historical __getitem__, but fail rather than
    # silently replacing a bad sample with a neighbour during this audit.
    return legacy.records.BioSeqRecord(
        chains=legacy.wrapper._roles_for(row), task_type=row["task_type"],
        source=row.get("source", ""), labels={"relation": row["relation"]} if row.get("relation") else {},
    )


def training_semantics(record) -> dict[str, Any]:
    # Legacy fusion never retained raw regions/metadata/split/identifiers. These
    # new annotations are audited separately by canonical-vs-prepared full rows.
    return {"chains": [chain.sequence for chain in record.chains],
            "chain_roles": [chain.role for chain in record.chains],
            "task_type": record.task_type, "source": record.source,
            "labels": record.labels, "weight": float(record.weight)}


def compare_multiplicity(left: Path, right: Path, work_dir: Path) -> dict[str, Any]:
    """External sort + merge counts every SHA256 occurrence (not set equality)."""
    sorted_paths = []
    for path in (left, right):
        target = path.with_suffix(path.suffix + ".sorted")
        subprocess.run(["sort", "-S", "256M", "-T", str(work_dir), "-o", str(target), str(path)],
                       check=True, env={**os.environ, "LC_ALL": "C"}, timeout=1800)
        sorted_paths.append(target)
    result = {"different_fingerprints": 0, "missing_occurrences": 0, "extra_occurrences": 0, "examples": []}
    def grouped(handle):
        return ((key.strip(), sum(1 for _ in group)) for key, group in itertools.groupby(handle))
    with sorted_paths[0].open() as a, sorted_paths[1].open() as b:
        ia, ib = grouped(a), grouped(b)
        va, vb = next(ia, None), next(ib, None)
        while va is not None or vb is not None:
            if vb is None or (va is not None and va[0] < vb[0]):
                key, na, nb = va[0], va[1], 0
                va = next(ia, None)
            elif va is None or vb[0] < va[0]:
                key, na, nb = vb[0], 0, vb[1]
                vb = next(ib, None)
            else:
                key, na, nb = va[0], va[1], vb[1]
                va, vb = next(ia, None), next(ib, None)
            if na != nb:
                result["different_fingerprints"] += 1
                result["missing_occurrences"] += max(na - nb, 0)
                result["extra_occurrences"] += max(nb - na, 0)
                if len(result["examples"]) < 10:
                    result["examples"].append({"fingerprint": key, "expected": na, "actual": nb})
    for path in sorted_paths:
        path.unlink()
    return result


def compare_rendering(legacy, samples, config, seed: int) -> dict[str, Any]:
    import torch

    old_tok = legacy.grammar.GrammarTokenizer()
    new_tok = GrammarTokenizer()
    old_renderer = legacy.grammar.GrammarRenderer(old_tok, config.max_protein_length, random.Random(seed))
    new_renderer = GrammarRenderer(new_tok, config.max_protein_length, random.Random(seed))
    old_collator = legacy.grammar.GrammarBioSeqCollator(old_tok, config.max_length, config.max_protein_length)
    new_collator = GrammarBioSeqCollator(new_tok, config.max_length, config.max_protein_length)
    counts = Counter(records=len(samples))
    examples = []
    for index in range(0, len(samples), 4):
        group = samples[index:index + 4]
        old_rows, new_rows = [], []
        for raw_index, old_record, new_record in group:
            old_row, new_row = old_renderer.encode(old_record), new_renderer.encode(new_record)
            old_rows.append(old_row)
            new_rows.append(new_row)
            if old_row != new_row:
                counts["rendering_mismatches"] += 1
                if len(examples) < 10:
                    examples.append({"raw_index": raw_index, "fields": [key for key in old_row.keys() | new_row.keys() if old_row.get(key) != new_row.get(key)]})
        old_batch, new_batch = old_collator(old_rows), new_collator(new_rows)
        different = []
        for key in old_batch.keys() | new_batch.keys():
            a, b = old_batch.get(key), new_batch.get(key)
            equal = torch.equal(a, b) and a.dtype == b.dtype if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) else a == b
            if not equal:
                different.append(key)
        counts["batches"] += 1
        if different:
            counts["collator_mismatches"] += 1
            if len(examples) < 10:
                examples.append({"batch": index // 4, "fields": different})
    return {"counts": dict(counts), "examples": examples, "tokenizer": "ESM2 built-in residue vocab", "seed": seed}


def audit_pair(source, split, config, legacy, legacy_spec, manifest, dataset_dir, work_dir,
               max_rows, sample_limit, seed, blocklists, progress):
    raw_path = legacy.datasets._source_split_path(legacy_spec, split)
    before = file_stat(raw_path)
    raw_sha = hashlib.sha256()
    counts = Counter()
    drops = Counter()
    examples = []
    samples = []
    rng = random.Random(seed)
    matching_seen = 0
    filters = build_filters(source, manifest["sources"], blocklists, config.max_protein_length, config.max_length)
    paths = {name: work_dir / f"{split}.{source}.{name}.sha256" for name in ("legacy", "canonical", "prepared_semantics", "prepared_full")}
    with paths["legacy"].open("w") as old_out, paths["canonical"].open("w") as new_out:
        with raw_path.open("rb") as raw:
            def decoded_lines():
                for line in raw:
                    raw_sha.update(line)
                    yield line.decode("utf-8")
            reader = csv.DictReader(decoded_lines())
            rows = reader if max_rows is None else itertools.islice(reader, max_rows)
            for raw_index, row in enumerate(rows, 1):
                counts["raw_rows"] += 1
                old = legacy_spec.row_to_record(row)
                new = row_to_record(source, row, split=split, weight=legacy_spec.weight)
                reason = "schema" if new is None else filter_reason(new, filters)
                if reason:
                    drops[reason] += 1
                    new = None
                counts["legacy_kept"] += int(old is not None)
                counts["canonical_kept"] += int(new is not None)
                if (old is None) != (new is None):
                    counts["decision_mismatches"] += 1
                    if len(examples) < 10:
                        examples.append({"raw_index": raw_index, "kind": "decision", "legacy_kept": old is not None, "canonical_reason": reason, "raw_sha256": fingerprint(row)})
                if new is not None:
                    new_out.write(fingerprint(new.to_dict()) + "\n")
                if old is None:
                    continue
                try:
                    old_record = legacy_record(legacy, old)
                except (ValueError, IndexError, TypeError, KeyError) as exc:
                    counts["legacy_record_errors"] += 1
                    if len(examples) < 10:
                        examples.append({"raw_index": raw_index, "kind": "legacy_record_error", "error": repr(exc)})
                    continue
                old_payload = training_semantics(old_record)
                old_out.write(fingerprint(old_payload) + "\n")
                if new is not None:
                    new_payload = training_semantics(new)
                    if old_payload != new_payload:
                        counts["semantic_mismatches"] += 1
                        if len(examples) < 10:
                            examples.append({"raw_index": raw_index, "kind": "semantics", "legacy": old_payload, "canonical": new_payload})
                    matching_seen += 1
                    sample = (raw_index, old_record, new)
                    if len(samples) < sample_limit:
                        samples.append(sample)
                    else:
                        index = rng.randrange(matching_seen)
                        if index < sample_limit:
                            samples[index] = sample
                if raw_index % 100000 == 0:
                    progress(source=source, split=split, phase="raw", counts=dict(counts))
    if file_stat(raw_path) != before:
        raise RuntimeError(f"Raw file changed during audit: {raw_path}")
    progress(source=source, split=split, phase="prepared", counts=dict(counts))
    shard_identities = []
    expected = manifest["splits"][split]["sources"][source]
    if max_rows is None:
        counts["count_mismatches"] += int(counts["raw_rows"] != expected["raw_rows"])
        limit = None
    else:
        # Prefix comparison is meaningful only if the raw decisions agree.
        limit = counts["canonical_kept"]
    with paths["prepared_semantics"].open("w") as projected, paths["prepared_full"].open("w") as full:
        for shard in manifest["splits"][split]["shards"]:
            if shard["source"] != source:
                continue
            if limit is not None and counts["prepared"] >= limit:
                break
            path = dataset_dir / shard["path"]
            initial = file_stat(path)
            digest = hashlib.sha256()
            shard_count = 0
            with path.open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    if not line.strip():
                        continue
                    record = validate_prepared_row(json.loads(line))
                    shard_count += 1
                    counts["prepared"] += 1
                    projected.write(fingerprint(training_semantics(record)) + "\n")
                    full.write(fingerprint(record.to_dict()) + "\n")
                    if limit is not None and counts["prepared"] >= limit:
                        break
            if file_stat(path) != initial:
                raise RuntimeError(f"Prepared shard changed during audit: {path}")
            if max_rows is None:
                counts["count_mismatches"] += int(shard_count != shard["records"])
            shard_identities.append({"path": str(path), **initial, "sha256": digest.hexdigest(), "hash_scope": "full_file" if max_rows is None else "consumed_prefix", "records": shard_count})
    if max_rows is None:
        counts["count_mismatches"] += int(counts["prepared"] != expected["records"])
    counts["count_mismatches"] += int(not counts["legacy_kept"] == counts["canonical_kept"] == counts["prepared"])
    progress(source=source, split=split, phase="multiset_sort", counts=dict(counts))
    old_cmp = compare_multiplicity(paths["legacy"], paths["prepared_semantics"], work_dir)
    new_cmp = compare_multiplicity(paths["canonical"], paths["prepared_full"], work_dir)
    counts["legacy_multiset_differences"] = old_cmp["different_fingerprints"]
    counts["canonical_multiset_differences"] = new_cmp["different_fingerprints"]
    rendering = compare_rendering(legacy, samples, config, seed)
    counts.update({key: rendering["counts"].get(key, 0) for key in ("rendering_mismatches", "collator_mismatches")})
    result = {"source": source, "split": split, "counts": dict(counts), "canonical_drops": dict(drops),
              "legacy_vs_prepared": old_cmp, "canonical_vs_prepared": new_cmp, "examples": examples,
              "rendering": rendering, "raw": {"path": str(raw_path), **before, "sha256": raw_sha.hexdigest(), "hash_scope": "full_file" if max_rows is None else "consumed_prefix"},
              "shards": shard_identities}
    result["status"] = "failed" if any(counts[key] for key in FAILURE_KEYS) else "passed"
    # Preserve all unsorted digests, including duplicates, for later attribution.
    write_json(work_dir / f"{split}.{source}.json", result)
    return result


def run(args) -> int:
    start = time.time()
    status_path = args.work_dir / "status.json"
    report = {"baseline_commit": args.baseline_commit, "scope": "full" if args.max_rows is None else "prefix_smoke", "pairs": [],
              "limitations": ["Historical prepared manifest has no original input/blocklist hashes; this audit pins current files, not their past contents.",
                              "Old fusion dropped regions/metadata/split/identifiers: training semantics are compared separately from complete new canonical rows.",
                              "Rendering/collator is seeded reservoir-sampled per source/split with ESM2 vocabulary; not a full-corpus model/remap parity or FSDP test.",
                              "No historical runtime retry or eval reservoir cap is applied; any legacy record error fails the audit."]}
    def progress(**values):
        payload = {"status": "running", "pid": os.getpid(), "elapsed_seconds": time.time() - start,
                   "completed_pairs": len(report["pairs"]), **values}
        write_json(status_path, payload)
        print(json.dumps(payload), flush=True)
    try:
        config = load_preprocess_config(args.config)
        selected = parse_sources(args.source) if args.source else [item["name"] for item in config.sources]
        splits = args.split.split(",") if args.split else list(config.splits)
        if not splits or len(splits) != len(set(splits)):
            raise ValueError("Splits must be a nonempty, unique list")
        manifest_path = args.prepared_data_dir / "dataset_manifest.json"
        manifest_identity = small_file_identity(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest)
        configured = {item["name"]: item for item in config.sources}
        if set(manifest["sources"]) != set(configured) or not set(selected) <= set(configured):
            raise ValueError("Config/prepared mix mismatch; subset audits still use the entire original mix")
        if manifest["budget"] != {"max_length": config.max_length, "max_protein_length": config.max_protein_length}:
            raise ValueError("Prepared and reference length budgets differ")
        blocklists = load_blocklists(config.blocklists)
        frozen_inputs = [small_file_identity(args.config), manifest_identity]
        for value in config.blocklists.values():
            if value is not None and str(value).lower() not in {"none", "off", "disabled", "-"}:
                frozen_inputs.append(small_file_identity(Path(value)))
        report["input_identities"] = frozen_inputs
        report["config"] = vars(config)
        report["current_code"] = [small_file_identity(path) for path in sorted((_ROOT / "dllm/pipelines/immune_llada").rglob("*.py"))]
        legacy = load_legacy(args.work_dir / "legacy_snapshot", args.baseline_commit)
        report["legacy_blob_sha256"] = legacy.hashes
        values = {"dataset_args": "+".join(manifest["sources"]), "max_protein_length": config.max_protein_length, "max_length": config.max_length}
        for name, item in configured.items():
            values[f"{name}_dir"] = str(source_spec(name, path=item.get("path")).path)
        # Builder creates lambdas for optional nanobody but doesn't evaluate them.
        values.update({f"{name}_blocklist": value for name, value in config.blocklists.items()})
        specs = {spec.name: spec for spec in legacy.builder(SimpleNamespace(**values))}
        for name, spec in specs.items():
            if float(configured[name].get("weight", 1.0)) != spec.weight:
                raise ValueError(f"Source weight differs from historical builder: {name}")
        csv.field_size_limit(2**31 - 1)
        progress(phase="preflight_complete", total_pairs=len(splits) * len(selected))
        for split in splits:
            for source in selected:
                report["pairs"].append(audit_pair(source, split, config, legacy, specs[source], manifest,
                    args.prepared_data_dir, args.work_dir, args.max_rows, args.sample_limit, args.seed, blocklists, progress))
                write_json(args.work_dir / "report.partial.json", report)
                progress(source=source, split=split, phase="pair_complete", pair_status=report["pairs"][-1]["status"])
        for identity in frozen_inputs + report["current_code"]:
            if small_file_identity(Path(identity["path"])) != identity:
                raise RuntimeError(f"Pinned input/code changed during audit: {identity['path']}")
        totals = Counter()
        for pair in report["pairs"]:
            totals.update(pair["counts"])
        report["totals"] = dict(totals)
        report["status"] = "failed" if any(totals[key] for key in FAILURE_KEYS) else "passed"
        report["elapsed_seconds"] = time.time() - start
        report_path = args.work_dir / "report.json"
        write_json(report_path, report)
        write_json(status_path, {"status": report["status"], "scope": report["scope"], "report": str(report_path), "totals": dict(totals), "elapsed_seconds": report["elapsed_seconds"]})
        print(json.dumps({"status": report["status"], "totals": dict(totals)}), flush=True)
        return 0 if report["status"] == "passed" else 2
    except BaseException as exc:
        write_json(args.work_dir / "report.partial.json", report)
        write_json(status_path, {"status": "error", "error": repr(exc), "elapsed_seconds": time.time() - start})
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_ROOT / "configs/data/immune_v3.yaml")
    parser.add_argument("--prepared-data-dir", type=Path, default=_ROOT / "data/prepared/immune_v3")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--baseline-commit", default=BASELINE_COMMIT)
    parser.add_argument("--source", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--sample-limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--max-seconds", type=int, default=21600)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.sample_limit < 1 or args.max_seconds < 1 or (args.max_rows is not None and args.max_rows < 1):
        parser.error("Limits must be positive")
    args.work_dir = args.work_dir.resolve()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if args.background or args.supervise:
        flag = "--background" if args.background else "--supervise"
        arguments.remove(flag)
        arguments.append("--supervise" if args.background else "--worker")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *arguments]
        if args.background:
            if args.work_dir.exists() and any(args.work_dir.iterdir()):
                parser.error("Use a new empty work directory; never overwrite audit artifacts")
            args.work_dir.mkdir(parents=True, exist_ok=True)
            with (args.work_dir / "run.log").open("x") as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            write_json(args.work_dir / "launch.json", {"supervisor_pid": proc.pid, "command": command, "max_seconds": args.max_seconds})
            print(f"Started bounded parity supervisor PID={proc.pid}, logs={args.work_dir / 'run.log'}")
            return 0
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(args.work_dir / "worker.json", {"pid": child.pid, "command": command})
        try:
            code = child.wait(timeout=args.max_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124
            write_json(args.work_dir / "status.json", {"status": "timed_out", "max_seconds": args.max_seconds})
        write_json(args.work_dir / "exit.json", {"exit_code": code, "finished_at_unix": time.time()})
        return code
    if not args.worker:
        if args.work_dir.exists() and any(args.work_dir.iterdir()):
            parser.error("Use a new empty work directory; never overwrite audit artifacts")
        args.work_dir.mkdir(parents=True, exist_ok=True)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
