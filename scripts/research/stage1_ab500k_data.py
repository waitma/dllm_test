"""Freeze an OAS H/L subset and aligned teacher requests without GPU work.

Activate pllm, then run ``python -B -m scripts.research.stage1_ab500k_data
--data-index ABS_INDEX --output ABS_DIRECTORY``. Use the same arguments with
``--verify-only`` to recheck an existing subset. Counts can be reduced for CPU
fixtures; production defaults are 500000 training rows and 4096 unique valid
pairs. Source JSONL and its original split/row numbers are never modified.
"""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import heapq
import json
from itertools import islice
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from dllm.pipelines.immune_llada.data.records import BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMCollator
from scripts.research.bert_mlm_data import receptor_key, sha, write_json
from scripts.research.stage1_data import CANONICAL, make_teacher_request, verify_request


SCHEMA = "immune.stage1.ab500k_subset.v1"
WORKERS = 16
ROLES = ["antibody_heavy", "antibody_light"]
try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads


def _absolute(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Frozen references must be absolute: " + str(path))
    return path


def _index(path):
    path = _absolute(path)
    index = loads(path.read_bytes())
    if index.get("schema") != "immune.bert_mlm.index.v1":
        raise ValueError("Unsupported parent data index")
    binding_path = _absolute(index["binding"])
    if sha(binding_path) != index["binding_sha256"]:
        raise ValueError("Parent data binding changed")
    binding = loads(binding_path.read_bytes())
    if index.get("manifest_sha256") and sha(_absolute(index["manifest"])) != index["manifest_sha256"]:
        raise ValueError("Parent dataset manifest changed")
    if binding.get("config_path") and sha(_absolute(binding["config_path"])) != binding["config_sha256"]:
        raise ValueError("Parent model configuration changed")
    for split in ("train", "valid"):
        if sum(x["records"] for x in index["shards"] if x["split"] == split) != index["records"][split]:
            raise ValueError("Parent split count differs from shard counts")
    return index, FixedChainConfig(**binding["model"])


def _oas_shards(index, split):
    base = 0
    for shard in index["shards"]:
        if shard["split"] != split:
            continue
        if shard["source"] == "oas":
            yield base, shard
        base += shard["records"]


def _offsets(shard):
    path = _absolute(shard["index"])
    if sha(path) != shard["index_sha256"]:
        raise ValueError("Parent byte offsets changed")
    offsets = np.load(path, mmap_mode="r", allow_pickle=False)
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) != shard["records"]:
        raise ValueError("Invalid parent byte offset array")
    if len(offsets) and (int(offsets[0]) < 0 or np.any(offsets[1:] <= offsets[:-1])):
        raise ValueError("Parent byte offsets must be increasing")
    return offsets


def _scan(index, split):
    """Stream and hash every OAS byte, retaining original indexed row identities."""
    for base, shard in _oas_shards(index, split):
        path, offsets = _absolute(shard["path"]), _offsets(shard)
        before, digest, position, ordinal = path.stat(), hashlib.sha256(), 0, 0
        if shard.get("resolved_path") and str(path.resolve()) != shard["resolved_path"]:
            raise ValueError("Parent shard resolves to a different file")
        with path.open("rb", buffering=4 * 1024 * 1024) as stream:
            for line in stream:
                digest.update(line)
                if ordinal < len(offsets) and int(offsets[ordinal]) < position:
                    raise ValueError("Parent offset is not a JSONL line boundary")
                if ordinal < len(offsets) and int(offsets[ordinal]) == position:
                    yield base + ordinal, loads(line), dict(path=str(path), offset=position)
                    ordinal += 1
                position += len(line)
        after = path.stat()
        if ordinal != len(offsets) or digest.hexdigest() != shard["sha256"]:
            raise ValueError("Parent OAS shard bytes/offsets changed: " + str(path))
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Parent OAS shard changed while reading")


def _selected(index, split, rows):
    """Read sorted selected offsets; no full records are retained between yields."""
    consumed = 0
    for base, shard in _oas_shards(index, split):
        left = int(np.searchsorted(rows, base))
        right = int(np.searchsorted(rows, base + shard["records"]))
        if left == right:
            continue
        offsets, path = _offsets(shard), _absolute(shard["path"])
        with path.open("rb", buffering=4 * 1024 * 1024) as stream:
            for row_index in rows[left:right]:
                offset = int(offsets[int(row_index) - base])
                stream.seek(offset)
                row = loads(stream.readline())
                consumed += 1
                yield int(row_index), row, dict(path=str(path), offset=offset)
    if consumed != len(rows):
        raise ValueError("Subset includes a row outside the OAS parent shards")


def _request_worker(config):
    global _WORKER_CONFIG
    torch.set_num_threads(1)
    _WORKER_CONFIG = FixedChainConfig(**config)


def _request_chunk(chunk):
    return [(row_index, row, origin, make_teacher_request(row, _WORKER_CONFIG))
            for row_index, row, origin in chunk]


def _request_rows(index, split, rows, config, workers):
    """Bound queued records while CPU workers construct exact existing requests."""
    source = iter(_selected(index, split, rows))
    if workers <= 1:
        for row_index, row, origin in source:
            yield row_index, row, origin, make_teacher_request(row, config)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_request_worker, initargs=(asdict(config),)) as pool:
        pending = deque()
        for _ in range(workers * 2):
            chunk = list(islice(source, 128))
            if not chunk:
                break
            pending.append(pool.submit(_request_chunk, chunk))
        while pending:
            # Consume in original row order, independently of worker timing.
            yield from pending.popleft().result()
            chunk = list(islice(source, 128))
            if chunk:
                pending.append(pool.submit(_request_chunk, chunk))


def _eligibility(row, split, config):
    if row.get("source") != "oas_paired" or row.get("split") != split:
        return "source_or_split", None
    if row.get("chain_roles") != ROLES or len(row.get("chains", [])) != 2:
        return "not_native_heavy_light", None
    if row.get("task_type") != "antibody":
        return "not_antibody", None
    metadata = row.get("chain_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Malformed chain metadata")
    for number, (sequence, capacity) in enumerate(zip(row["chains"],
            (config.first_chain_capacity, config.second_chain_capacity))):
        if not isinstance(sequence, str) or not sequence or not set(sequence) <= CANONICAL:
            return "noncanonical_or_empty", None
        if len(sequence) + 1 > capacity:
            return "slot_capacity", None
        meta = metadata.get(str(number), {})
        synthetic = meta.get("synthetic_residue_mask", [])
        if synthetic and (len(synthetic) != len(sequence) or any(x not in (0, 1) for x in synthetic)):
            raise ValueError("Malformed synthetic residue provenance")
        if any(synthetic):
            return "synthetic", None
        if meta.get("sequence_scope", "full") not in {"full", "fv"}:
            return "fragment", None
    # Length prefixes remove delimiter ambiguity; only digests are retained.
    h, light = row["chains"]
    key = hashlib.sha256((str(len(h)) + ":" + h + ":" + light).encode()).digest()
    return None, key


def _score(seed, domain, key):
    return int.from_bytes(hashlib.sha256(str(seed).encode() + b":" + domain.encode() + b":" + key).digest(), "big")


def _select_rows(index, config, train_count, valid_count, seed):
    valid_unique, stats = {}, Counter()
    for row_index, row, _ in _scan(index, "valid"):
        stats["valid_oas_rows"] += 1
        reason, key = _eligibility(row, "valid", config)
        if reason:
            stats["valid_excluded_" + reason] += 1
        else:
            stats["valid_eligible_rows"] += 1
            valid_unique.setdefault(key, row_index)
    if len(valid_unique) < valid_count:
        raise ValueError(f"Need {valid_count} unique eligible valid pairs; found {len(valid_unique)}")
    ordered_keys = sorted(valid_unique, key=lambda key: (_score(seed, "valid", key), key))[:valid_count]
    held = set(ordered_keys)
    valid_rows = np.asarray(sorted(valid_unique[key] for key in ordered_keys), dtype=np.int64)
    stats["valid_unique_eligible_pairs"] = len(valid_unique)
    del valid_unique
    rng, selected, eligible = np.random.default_rng(seed), np.empty(train_count, dtype=np.int64), 0
    previous_log = time.monotonic()
    for row_index, row, _ in _scan(index, "train"):
        stats["train_oas_rows"] += 1
        reason, key = _eligibility(row, "train", config)
        if reason:
            stats["train_excluded_" + reason] += 1
            continue
        if key in held:
            stats["train_excluded_fixed_valid_exact_pair"] += 1
            continue
        eligible += 1
        if eligible <= train_count:
            selected[eligible - 1] = row_index
        else:
            replacement = int(rng.integers(eligible))
            if replacement < train_count:
                selected[replacement] = row_index
        if time.monotonic() - previous_log >= 60:
            print(json.dumps(dict(phase="select", scanned=stats["train_oas_rows"], eligible=eligible)), flush=True)
            previous_log = time.monotonic()
    if eligible < train_count:
        raise ValueError(f"Need {train_count} eligible train rows after valid exclusion; found {eligible}")
    stats["train_eligible_after_fixed_valid_exclusion"] = eligible
    return {"train": np.sort(selected), "valid": valid_rows}, dict(stats)


def _reference(path, count):
    return dict(path=str(path), sha256=sha(path), records=count, bytes=path.stat().st_size)


def _line(value):
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _pilot_choice(candidates, limit):
    groups = [sorted((-score, ordinal, offset, size, worker) for score, ordinal, offset, size, worker in candidates[key])
              for key in sorted(candidates)]
    positions, selected = [0] * len(groups), []
    while len(selected) < limit:
        changed = False
        for number, group in enumerate(groups):
            if positions[number] < len(group) and len(selected) < limit:
                selected.append(group[positions[number]])
                positions[number] += 1
                changed = True
        if not changed:
            break
    return sorted(selected, key=lambda item: item[1])


def build_subset(args):
    """Create subset.json last, after source hashing and exact request alignment."""
    train_count, valid_count, seed = int(args.train_count), int(args.valid_count), int(args.seed)
    workers = int(getattr(args, "workers", 1))
    if min(train_count, valid_count) <= 0 or seed < 0:
        raise ValueError("Subset counts must be positive and seed nonnegative")
    if workers < 1:
        raise ValueError("CPU worker count must be positive")
    output = Path(args.output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Subset output must be empty: " + str(output))
    index_path = Path(args.data_index).absolute()
    initial_index_sha = sha(index_path)
    index, config = _index(index_path)
    torch.set_num_threads(1)
    print(json.dumps(dict(phase="select", source="oas", counts=dict(train=train_count, valid=valid_count))), flush=True)
    rows, stats = _select_rows(index, config, train_count, valid_count, seed)
    references = {}
    for split in ("train", "valid"):
        path = output / (split + "_rows.npy")
        np.save(path, rows[split], allow_pickle=False)
        references[split + "_rows"] = _reference(path, len(rows[split]))
    total = train_count + valid_count
    pilot_count = min(int(getattr(args, "pilot_count", 2048)), total)
    if pilot_count < 1:
        raise ValueError("Pilot count must be positive")
    candidates = defaultdict(list)
    shard_paths = [output / f"requests.{i:02d}.jsonl" for i in range(WORKERS)]
    pilot_paths = [output / f"pilot_requests.{i:02d}.jsonl" for i in range(WORKERS)]
    shard_counts, pilot_counts = Counter(), Counter()
    length_hist, identities = Counter(), set()
    valid_heavy, valid_light = set(), set()
    for _, row, _ in _selected(index, "valid", rows["valid"]):
        valid_heavy.add(hashlib.sha256(row["chains"][0].encode()).digest())
        valid_light.add(hashlib.sha256(row["chains"][1].encode()).digest())
    ordinal, max_teacher_length, student_max_length = 0, 0, 0
    collator = FullTokenMLMCollator(config)
    collated_lengths = {}
    previous_log = time.monotonic()
    print(json.dumps(dict(phase="requests", records=total)), flush=True)
    with ExitStack() as stack:
        requests = stack.enter_context((output / "requests.jsonl").open("xb"))
        token_valid = stack.enter_context((output / "token_valid.jsonl").open("xb"))
        shards = [stack.enter_context(path.open("xb")) for path in shard_paths]
        for split in ("train", "valid"):
            for row_index, row, origin, request in _request_rows(index, split, rows[split], config, workers):
                reason, key = _eligibility(row, split, config)
                if reason:
                    raise ValueError("Selected row became ineligible: " + reason)
                if len(request["chains"]) != 2 or request["excluded"] or request["inter_evidence"]:
                    raise ValueError("Selected OAS pair does not map to two complete intra-only teacher chains")
                # The existing request constructor checks exact residues and rendered positions.
                envelope = dict(split=split, row_index=row_index, request=request)
                encoded, worker = _line(envelope), ordinal % WORKERS
                offset = requests.tell()
                requests.write(encoded)
                shards[worker].write(encoded)
                shard_counts[worker] += 1
                lengths = tuple(len(x) for x in row["chains"])
                teacher_length = sum(lengths)
                max_teacher_length = max(max_teacher_length, teacher_length)
                if lengths not in collated_lengths:
                    clean = collator([BioSeqRecord.from_prepared_dict(row)])
                    collated_lengths[lengths] = int(clean["attention_mask"][0].sum())
                student_max_length = max(student_max_length, collated_lengths[lengths])
                bucket = next((bound for bound in (220, 240, 260) if teacher_length < bound), 10000)
                length_hist[f"{split}/lt_{bucket}"] += 1
                score = _score(seed, "pilot", f"{split}:{row_index}".encode())
                item = (-score, ordinal, offset, len(encoded), worker)
                heap = candidates[(worker, split, bucket)]
                if len(heap) < max(1, (pilot_count + WORKERS - 1) // WORKERS):
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
                if split == "train":
                    identities.add(key)
                    stats["selected_train_exact_heavy_overlap_valid"] = stats.get("selected_train_exact_heavy_overlap_valid", 0) + int(hashlib.sha256(row["chains"][0].encode()).digest() in valid_heavy)
                    stats["selected_train_exact_light_overlap_valid"] = stats.get("selected_train_exact_light_overlap_valid", 0) + int(hashlib.sha256(row["chains"][1].encode()).digest() in valid_light)
                else:
                    token_valid.write(_line(dict(record=row, source_group="oas", receptor_key=receptor_key(row),
                                                 origin=origin, split=split, row_index=row_index)))
                ordinal += 1
                if time.monotonic() - previous_log >= 60:
                    print(json.dumps(dict(phase="requests", completed=ordinal, total=total)), flush=True)
                    previous_log = time.monotonic()
    stats["selected_train_unique_exact_pairs"] = len(identities)
    stats["selected_train_duplicate_pair_rows"] = train_count - len(identities)
    stats["length_buckets"] = dict(length_hist)
    pilot_quotas, allocated = Counter(), 0
    while allocated < pilot_count:
        for worker in range(WORKERS):
            if allocated < pilot_count and pilot_quotas[worker] < shard_counts[worker]:
                pilot_quotas[worker] += 1
                allocated += 1
    pilot_selected = []
    for worker in range(WORKERS):
        groups = {key[1:]: heap for key, heap in candidates.items() if key[0] == worker}
        pilot_selected.extend(_pilot_choice(groups, pilot_quotas[worker]))
    pilot_selected.sort(key=lambda item: item[1])
    with ExitStack() as stack:
        source = stack.enter_context((output / "requests.jsonl").open("rb"))
        pilot = stack.enter_context((output / "pilot_requests.jsonl").open("xb"))
        pilot_shards = [stack.enter_context(path.open("xb")) for path in pilot_paths]
        for _, _, offset, length, worker in pilot_selected:
            source.seek(offset)
            encoded = source.read(length)
            pilot.write(encoded)
            pilot_shards[worker].write(encoded)
            pilot_counts[worker] += 1
    for name, count in (("token_valid", valid_count), ("requests", total), ("pilot_requests", len(pilot_selected))):
        references[name] = _reference(output / (name + ".jsonl"), count)
    request_shards = [dict(worker_index=i, path=str(shard_paths[i]), sha256=sha(shard_paths[i]), count=shard_counts[i],
                          bytes=shard_paths[i].stat().st_size, pilot_path=str(pilot_paths[i]),
                          pilot_sha256=sha(pilot_paths[i]), pilot_count=pilot_counts[i]) for i in range(WORKERS)]
    if sha(index_path) != initial_index_sha:
        raise ValueError("Parent data index changed while preparing subset")
    # Recheck binding metadata after the complete source scans and request export.
    _index(index_path)
    result = dict(schema=SCHEMA, data_index=str(index_path), data_index_sha256=initial_index_sha,
                  binding=index["binding"], binding_sha256=index["binding_sha256"], model=asdict(config),
                  counts=dict(train=train_count, valid=valid_count), seed=seed, parent_records=index["records"],
                  files=references, request_shards=request_shards, worker_count=WORKERS,
                  pilot_count=len(pilot_selected), student_max_length=student_max_length,
                  max_teacher_length=max_teacher_length, stats=stats,
                  policies=dict(source="oas_paired; exactly native full/fv H then L; no synthetic residues",
                      valid="seeded exact-H/L-pair hash order; first indexed row for each pair; original valid only",
                      train="seeded reservoir without replacement over eligible original train rows; preserve duplicate-row frequency",
                      exclusion="exclude only exact complete H/L pairs in the frozen selected validation set",
                      request="unchanged make_teacher_request; intra_only; quality unvalidated",
                      pilot="128 per production worker; round-robin split/total-length strata within worker; seeded row hash; original request bytes and worker assignment",
                      interpretation="continued-training comparison; exact pair isolation does not establish single-chain/family isolation or unseen parent pretraining"),
                  verified_sources=[{k: shard[k] for k in ("path", "sha256", "index", "index_sha256", "split", "records")}
                                    for shard in index["shards"] if shard["source"] == "oas"],
                  auxiliary_bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file()))
    for name, ref in references.items():
        result[name], result[name + "_sha256"] = ref["path"], ref["sha256"]
    result["requests_count"] = total
    write_json(output / "subset.json", result)
    print(json.dumps(dict(phase="complete", manifest=str(output / "subset.json"), counts=result["counts"],
                          auxiliary_bytes=result["auxiliary_bytes"], student_max_length=student_max_length)), flush=True)
    return result


def validate_subset_files(path):
    """Check frozen files and row arrays without reconstructing teacher requests.

Launchers pair this with a separately frozen successful full-validation receipt.
This check intentionally does not claim independent source/request alignment.
"""
    path = _absolute(path)
    if path.is_dir():
        path = path / "subset.json"
    subset = loads(path.read_bytes())
    if subset.get("schema") != SCHEMA:
        raise ValueError("Unsupported AB subset schema")
    if sha(_absolute(subset["data_index"])) != subset["data_index_sha256"]:
        raise ValueError("Parent data index changed")
    index, config = _index(subset["data_index"])
    if (subset["model"] != asdict(config) or subset["parent_records"] != index["records"]
            or subset["binding"] != index["binding"] or subset["binding_sha256"] != index["binding_sha256"]):
        raise ValueError("Subset parent binding/model/counts changed")
    if set(subset["counts"]) != {"train", "valid"} or any(
            not isinstance(count, int) or count < 1 for count in subset["counts"].values()):
        raise ValueError("Invalid subset counts")
    expected_sources = [{k: shard[k] for k in ("path", "sha256", "index", "index_sha256", "split", "records")}
                        for shard in index["shards"] if shard["source"] == "oas"]
    if subset["verified_sources"] != expected_sources or subset["worker_count"] != WORKERS:
        raise ValueError("Subset source/worker contract differs")
    for split in ("train", "valid"):
        for _, shard in _oas_shards(index, split):
            source = _absolute(shard["path"])
            if shard.get("resolved_path") and str(source.resolve()) != shard["resolved_path"]:
                raise ValueError("Parent OAS source resolves to a different file")
            if sha(source) != shard["sha256"]:
                raise ValueError("Parent OAS source bytes changed")
            _offsets(shard)
    expected_counts = dict(train_rows=subset["counts"]["train"], valid_rows=subset["counts"]["valid"],
                           token_valid=subset["counts"]["valid"], requests=sum(subset["counts"].values()),
                           pilot_requests=subset["pilot_count"])
    if set(subset["files"]) != set(expected_counts) or subset["requests_count"] != expected_counts["requests"]:
        raise ValueError("Subset artifact set/total count differs")
    for name, ref in subset["files"].items():
        artifact = _absolute(ref["path"])
        if (subset[name] != ref["path"] or subset[name + "_sha256"] != ref["sha256"]
                or ref["records"] != expected_counts[name]):
            raise ValueError("Subset artifact reference/count differs: " + name)
        if sha(artifact) != ref["sha256"] or artifact.stat().st_size != ref["bytes"]:
            raise ValueError("Subset artifact changed: " + name)
    for split in ("train", "valid"):
        ids = np.load(subset[split + "_rows"], mmap_mode="r", allow_pickle=False)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) != subset["counts"][split]:
            raise ValueError("Subset row array/count differs")
        if len(ids) and (ids[0] < 0 or ids[-1] >= index["records"][split] or np.any(ids[1:] <= ids[:-1])):
            raise ValueError("Subset rows are out of bounds or repeated")
        covered = sum(int(np.searchsorted(ids, base + shard["records"])) - int(np.searchsorted(ids, base))
                      for base, shard in _oas_shards(index, split))
        if covered != len(ids):
            raise ValueError("Subset includes a non-OAS parent row")
    if len(subset["request_shards"]) != WORKERS:
        raise ValueError("Need exactly sixteen request partitions")
    total, pilot_total = expected_counts["requests"], 0
    for worker, ref in enumerate(subset["request_shards"]):
        if (ref["worker_index"] != worker or ref["count"] != total // WORKERS + int(worker < total % WORKERS)
                or not 0 <= ref["pilot_count"] <= ref["count"]):
            raise ValueError("Request worker/count differs")
        shard, pilot = _absolute(ref["path"]), _absolute(ref["pilot_path"])
        if sha(shard) != ref["sha256"] or shard.stat().st_size != ref["bytes"]:
            raise ValueError("Worker request artifact changed")
        if sha(pilot) != ref["pilot_sha256"]:
            raise ValueError("Worker pilot artifact changed")
        pilot_total += ref["pilot_count"]
    if pilot_total != subset["pilot_count"] or not 0 < pilot_total <= total:
        raise ValueError("Pilot partition counts differ")
    return dict(status="passed", mode="files_only", manifest=str(path), manifest_sha256=sha(path),
                counts=subset["counts"], requests=total, workers=WORKERS, pilot_count=pilot_total)


def validate_subset(path, *, workers=1):
    """Verify all artifacts, source bytes, exact indexed records and request mapping."""
    path = _absolute(path)
    if path.is_dir():
        path = path / "subset.json"
    subset = loads(path.read_bytes())
    if subset.get("schema") != SCHEMA:
        raise ValueError("Unsupported AB subset schema")
    if sha(_absolute(subset["data_index"])) != subset["data_index_sha256"]:
        raise ValueError("Parent data index changed")
    index, config = _index(subset["data_index"])
    if subset["model"] != asdict(config) or subset["parent_records"] != index["records"]:
        raise ValueError("Subset parent model/counts changed")
    expected_sources = [{k: shard[k] for k in ("path", "sha256", "index", "index_sha256", "split", "records")}
                        for shard in index["shards"] if shard["source"] == "oas"]
    if subset["verified_sources"] != expected_sources or subset["worker_count"] != WORKERS:
        raise ValueError("Subset source/worker contract differs")
    expected_counts = dict(train_rows=subset["counts"]["train"], valid_rows=subset["counts"]["valid"],
                           token_valid=subset["counts"]["valid"], requests=sum(subset["counts"].values()),
                           pilot_requests=subset["pilot_count"])
    if subset["requests_count"] != expected_counts["requests"]:
        raise ValueError("Subset total request count differs")
    for reference in subset["verified_sources"]:
        if sha(_absolute(reference["path"])) != reference["sha256"]:
            raise ValueError("Parent OAS source bytes changed")
    for name, ref in subset["files"].items():
        if ref["records"] != expected_counts[name]:
            raise ValueError("Subset artifact count differs")
        if subset[name] != ref["path"] or subset[name + "_sha256"] != ref["sha256"]:
            raise ValueError("Subset reference aliases disagree")
        if sha(_absolute(ref["path"])) != ref["sha256"]:
            raise ValueError("Subset artifact changed: " + name)
    rows = {split: np.load(subset[split + "_rows"], allow_pickle=False) for split in ("train", "valid")}
    for split, ids in rows.items():
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) != subset["counts"][split]:
            raise ValueError("Subset row array/count differs")
        if len(ids) and (ids[0] < 0 or ids[-1] >= index["records"][split] or np.any(ids[1:] <= ids[:-1])):
            raise ValueError("Subset rows are out of bounds or repeated")
    held = set()
    for _, row, _ in _selected(index, "valid", rows["valid"]):
        reason, key = _eligibility(row, "valid", config)
        if reason or key in held:
            raise ValueError("Validation pairs are ineligible or nonunique")
        held.add(key)
    expected_shards, expected_pilot = [hashlib.sha256() for _ in range(WORKERS)], set()
    pilot_lines = Path(subset["pilot_requests"]).read_bytes().splitlines(keepends=True)
    if len(pilot_lines) != subset["pilot_count"] or len(set(pilot_lines)) != len(pilot_lines):
        raise ValueError("Pilot count or uniqueness differs")
    pending_pilot = set(pilot_lines)
    pilot_hashes, counts, pilot_counts, ordinal = [hashlib.sha256() for _ in range(WORKERS)], Counter(), Counter(), 0
    torch.set_num_threads(1)
    previous_log = time.monotonic()
    with Path(subset["requests"]).open("rb") as requests, Path(subset["token_valid"]).open("rb") as token_valid:
        for split in ("train", "valid"):
            for row_index, row, origin, expected_request in _request_rows(index, split, rows[split], config, workers):
                line = requests.readline()
                envelope = loads(line)
                reason, key = _eligibility(row, split, config)
                if reason or (split == "train" and key in held):
                    raise ValueError("Train/valid exact-pair isolation or eligibility failed")
                if envelope["split"] != split or envelope["row_index"] != row_index:
                    raise ValueError("Request row identity differs from frozen subset")
                request = envelope["request"]
                verify_request(request)
                if request != expected_request:
                    raise ValueError("Request differs from the actual indexed parent record/positions")
                worker = ordinal % WORKERS
                expected_shards[worker].update(line)
                counts[worker] += 1
                if line in pending_pilot:
                    pilot_hashes[worker].update(line)
                    expected_pilot.add(line)
                    pilot_counts[worker] += 1
                if split == "valid":
                    sample = loads(token_valid.readline())
                    if sample != dict(record=row, source_group="oas", receptor_key=receptor_key(row),
                                      origin=origin, split=split, row_index=row_index):
                        raise ValueError("MLM validation record differs from the frozen validation subset")
                ordinal += 1
                if time.monotonic() - previous_log >= 60:
                    print(json.dumps(dict(phase="verify", completed=ordinal, total=subset["requests_count"])), flush=True)
                    previous_log = time.monotonic()
        if requests.readline() or token_valid.readline():
            raise ValueError("Unexpected trailing request/validation records")
    if expected_pilot != pending_pilot:
        raise ValueError("Pilot contains a request outside the full request set")
    if len(subset["request_shards"]) != WORKERS:
        raise ValueError("Need exactly sixteen request partitions")
    for worker, ref in enumerate(subset["request_shards"]):
        if ref["worker_index"] != worker or ref["count"] != counts[worker] or ref["pilot_count"] != pilot_counts[worker]:
            raise ValueError("Request worker/count differs")
        if sha(_absolute(ref["path"])) != ref["sha256"] or expected_shards[worker].hexdigest() != ref["sha256"]:
            raise ValueError("Worker requests do not partition the full request sequence")
        if sha(_absolute(ref["pilot_path"])) != ref["pilot_sha256"] or pilot_hashes[worker].hexdigest() != ref["pilot_sha256"]:
            raise ValueError("Pilot worker assignments differ from production requests")
    return dict(status="passed", manifest=str(path), manifest_sha256=sha(path), counts=subset["counts"],
                requests=ordinal, workers=WORKERS, pilot_count=len(pilot_lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=500000)
    parser.add_argument("--valid-count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8, help="Bounded CPU request-construction workers")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = validate_subset(Path(args.output).absolute() / "subset.json", workers=args.workers)
        if result["manifest"] and loads(Path(result["manifest"]).read_bytes())["data_index"] != str(Path(args.data_index).absolute()):
            raise ValueError("CLI data index differs from the existing subset")
        print(json.dumps(result), flush=True)
    else:
        build_subset(args)


if __name__ == "__main__":
    main()
