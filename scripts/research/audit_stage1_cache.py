"""Audit a completed pilot without treating cache integrity as structural accuracy.

Activate pllm; python -B THIS_FILE --source FROZEN_SOURCE --plan CHECKS_PLAN
--manifest CACHE_MANIFEST --output REPORT. Uses the frozen corpus/cache readers.
This emits an engineering report, never an immune.stage1.teacher_quality receipt.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import sys


def audit(manifest_path, teacher_plan, *, max_pairs=8192, minimum_train_coverage=.1,
          minimum_valid_caches=128):
    from scripts.research.bert_mlm_data import IndexedPreparedDataset, sha
    from scripts.research.stage1_corpus import CacheStore, entries
    from scripts.research.stage1_data import verify_request

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    index = json.loads(Path(manifest["data_index"]).read_text())
    contract = manifest["teacher_contract"]
    if (manifest["schema"] != "immune.stage1.cache_index.v1"
            or sha(manifest["database"]) != manifest["database_sha256"]
            or sha(manifest["data_index"]) != manifest["data_index_sha256"]
            or sha(teacher_plan["requests"]) != manifest["requests_sha256"]
            or Path(manifest["requests"]) != Path(teacher_plan["requests"])
            or manifest["corpus_records"] != index["records"]):
        raise ValueError("Cache manifest input binding differs")
    if (contract["fast_compat_source_sha256"] != teacher_plan["references"][teacher_plan["script"]]
            or not contract["all_parameters_frozen"] or contract["mode"] != "trunk_only"
            or contract["num_loops"] != teacher_plan["num_loops"]
            or contract["base_seed"] != teacher_plan["seed"]
            or contract["pairs_per_chain_pair"] != teacher_plan["pairs_per_chain_pair"]):
        raise ValueError("Cache teacher configuration differs")
    weights = Path(teacher_plan["model_dir"])
    if contract["config_sha256"] != teacher_plan["references"][str(weights/"config.json")]:
        raise ValueError("Teacher config identity differs")
    expected_weights = {Path(path).name: digest for path, digest in teacher_plan["references"].items()
                        if Path(path).parent == weights and path.endswith(".safetensors")}
    if not expected_weights or contract["weight_sha256"] != expected_weights:
        raise ValueError("Teacher weight identity differs")

    datasets = {s: IndexedPreparedDataset(manifest["data_index"], s) for s in ("train", "valid")}
    config = json.loads(Path(index["binding"]).read_text())["model"]
    store = CacheStore(manifest)
    counts, excluded = Counter(), Counter()
    expected_rows, seen_rows = set(), set()
    worker_totals = [0] * teacher_plan["workers"]
    worker_ineligible = [0] * teacher_plan["workers"]
    maximum = 0
    symmetry_max, repeat_max = 0., 0.
    repeat_checked = 0
    try:
        for number, envelope in enumerate(entries(teacher_plan["requests"], manifest["data_index"])):
            split, row, request = envelope["split"], envelope["row_index"], envelope["request"]
            verify_request(request)
            if (split not in datasets or type(row) is not int or not 0 <= row < len(datasets[split])
                    or (split, row) in seen_rows):
                raise ValueError("Invalid or duplicate requested corpus row")
            seen_rows.add((split, row))
            worker = number % teacher_plan["workers"]
            worker_totals[worker] += 1
            record = datasets[split][row]
            if request["record"] != record.to_dict() or request["config"] != config:
                raise ValueError("Request/corpus row/config differs")
            counts[f"{split}/requested"] += 1
            excluded.update(x["reason"] for x in request["excluded"])
            cache = store.lookup(split, row, record)
            if not request["chains"]:
                if cache is not None:
                    raise ValueError("Ineligible record unexpectedly has teacher targets")
                counts[f"{split}/ineligible"] += 1
                worker_ineligible[worker] += 1
                continue
            if cache is None or cache["request"] != request:
                raise ValueError("Eligible request missing or mismatched in cache index")
            expected_rows.add((split, row))
            metadata = cache["metadata"]
            lengths = [len(c["sequence"]) for c in request["chains"]]
            pairs = len(cache["tensors"]["teacher_pair_indices"])
            asym = [i for i, length in enumerate(lengths) for _ in range(length)]
            if (metadata["pair_count"] != pairs or metadata["chain_lengths"] != lengths
                    or metadata["asym_id"] != asym or metadata["teacher_requires_grad"]
                    or metadata["seed"] != (teacher_plan["seed"] + int(request["request_sha256"][:8], 16)) % (2**31)
                    or pairs > max_pairs):
                raise ValueError("Cache chain/count/seed/gradient metadata invalid")
            for field in ("symmetry_max_abs", "repeat_max_abs"):
                value = metadata.get(field)
                if value is None and field == "repeat_max_abs":
                    continue
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1e-5:
                    raise ValueError("Teacher consistency check failed: " + field)
            symmetry_max = max(symmetry_max, metadata["symmetry_max_abs"])
            if metadata.get("repeat_max_abs") is not None:
                repeat_checked += 1
                repeat_max = max(repeat_max, metadata["repeat_max_abs"])
            counts[f"{split}/cached"] += 1
            counts[f"{split}/{record.source}/cached"] += 1
            counts[f"{split}/pairs"] += pairs
            maximum = max(maximum, pairs)
    finally:
        if store.connection is not None:
            store.connection.close()
    with sqlite3.connect(Path(manifest["database"]).as_uri()+"?mode=ro", uri=True) as connection:
        indexed_rows = set(connection.execute("SELECT split,row_index FROM targets"))
    coverage = {s: counts[f"{s}/cached"]/index["records"][s] for s in datasets}
    if (expected_rows != indexed_rows or dict(counts) != manifest["counts"]
            or maximum != manifest["max_pairs_per_record"] or coverage != manifest["coverage"]):
        raise ValueError("Cache index counts/coverage/rows differ from requests")

    peak_memory = []
    for worker in range(teacher_plan["workers"]):
        suffix = f"_worker{worker}" if teacher_plan["workers"] > 1 else ""
        path = Path(teacher_plan["output"])/f"verification{suffix}.json"
        report = json.loads(path.read_text())
        if (report["status"] != "passed" or report["teacher_contract"] != contract
                or report["worker_index"] != worker or report["worker_count"] != teacher_plan["workers"]
                or report["requests_sha256"] != manifest["requests_sha256"]
                or sum(report["counts"].values()) != worker_totals[worker]
                or report["counts"]["mlm_only"] != worker_ineligible[worker]):
            raise ValueError("Teacher worker completion differs")
        if any(report["loading_info"].get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Teacher worker did not strictly load weights")
        peak_memory.append(report.get("peak_cuda_allocated_gb"))
    cached = len(expected_rows)
    blockers = ["No cache-bound evaluation against experimental structures is configured"]
    if coverage["train"] < minimum_train_coverage:
        blockers.append("Training cache coverage is below the configured minimum")
    if counts["valid/cached"] < minimum_valid_caches:
        blockers.append("Insufficient held-out validation caches")
    if repeat_checked != cached:
        blockers.append("Some reused caches have no seeded replay measurement")
    return dict(schema="immune.stage1.cache_audit.v1", status="passed", scope="engineering_cache_integrity",
        cache_manifest=str(manifest_path.resolve()), cache_manifest_sha256=sha(manifest_path),
        counts=dict(counts), coverage=coverage, excluded_chain_reasons=dict(excluded), max_pairs_per_record=maximum,
        symmetry_max_abs=symmetry_max, repeat_max_abs=repeat_max if repeat_checked else None,
        repeat_checked_records=repeat_checked, cached_records=cached, worker_peak_allocated_gib=peak_memory,
        structural_accuracy=dict(status="unvalidated", reason="Experimental reference structures and evaluation protocol are not bound"),
        formal_training_ready=False, formal_training_blockers=blockers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "plan", "manifest", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.source).resolve()))
    from scripts.research.bert_mlm_data import write_json
    plan = json.loads(Path(args.plan).read_text())
    teacher = json.loads(Path(plan["teacher_plan"]).read_text())
    report = audit(args.manifest, teacher, max_pairs=plan["max_pairs_per_record"],
        minimum_train_coverage=plan["minimum_train_cache_coverage"], minimum_valid_caches=plan["minimum_valid_caches"])
    write_json(args.output, report)
    print(json.dumps(report), flush=True)
