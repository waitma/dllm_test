"""Compact, immutable offline targets for the AB500K comparison.

Activate pllm; python -B -m scripts.research.stage1_ab500k_cache index --plan PLAN.
Use ``check --plan PLAN`` to verify every payload before launching training.
This module deliberately imports no student models or Transformers packages.
"""
import argparse
import ctypes
import ctypes.util
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import tempfile
import time

import numpy as np
import torch


CODEC = "immune.structure_teacher.fp16_u16_shuffle_zstd3.v1"
SCHEMA = "immune.stage1.ab500k_cache.v1"
MAGIC = b"ABKD001\0"
FRAME = struct.Struct("<8sQQ32s")
MAX_RAW_BYTES = 16 << 20
WORKER_BUDGET = 8_750_000_000
AUX_BUDGET = 10_000_000_000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + ".partial.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class WorkerTimer:
    """Persist cumulative worker occupancy; flag unobserved tails after SIGKILL."""
    def __init__(self, directory, started, started_utc):
        self.directory, self.started = Path(directory), started
        self.path = self.directory / "timing.json"
        previous = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.previous_seconds = previous.get("accumulated_seconds", 0.)
        self.started_utc = previous.get("start_utc", started_utc)
        self.session_started_utc = started_utc
        self.session_count = previous.get("session_count", 0) + 1
        self.complete = previous.get("timing_complete", True) and previous.get("status", "complete") in {"complete", "failed"}
        self.last_update = 0.

    def snapshot(self, status="running"):
        elapsed = time.monotonic()-self.started
        return dict(schema="immune.stage1.ab500k_worker_timing.v1", status=status,
            start_utc=self.started_utc, end_utc=datetime.now(timezone.utc).isoformat(),
            session_start_utc=self.session_started_utc, session_count=self.session_count,
            session_seconds=elapsed, accumulated_seconds=self.previous_seconds+elapsed,
            timing_complete=self.complete,
            timing_note="complete_observed_sessions" if self.complete else "lower_bound_after_unclean_exit")

    def update(self, status="running", force=False):
        if force or time.monotonic()-self.last_update >= 60:
            atomic_json(self.path, self.snapshot(status))
            self.last_update = time.monotonic()

    def __enter__(self):
        self.update(force=True)
        return self

    def __exit__(self, exc_type, *args):
        self.update("failed" if exc_type else "complete", force=True)
        completion = self.directory / "complete.json"
        if exc_type is None and completion.exists():
            timing = json.loads(self.path.read_text())
            receipt = json.loads(completion.read_text())
            receipt.update({k: timing[k] for k in ("start_utc", "end_utc", "session_start_utc", "session_count",
                "session_seconds", "timing_complete", "timing_note")})
            receipt.update(seconds=timing["accumulated_seconds"], gpu_worker_hours=timing["accumulated_seconds"]/3600)
            atomic_json(completion, receipt)


class Zstd:
    """Use the same system libzstd as the earlier cache feasibility study."""
    def __init__(self):
        name = ctypes.util.find_library("zstd")
        if not name:
            raise RuntimeError("libzstd is required for the bound cache codec")
        self.lib = ctypes.CDLL(name)
        self.lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_compressBound.restype = ctypes.c_size_t
        for name in ("ZSTD_compress", "ZSTD_decompress"):
            fn = getattr(self.lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
            if name == "ZSTD_compress":
                fn.argtypes.append(ctypes.c_int)
            fn.restype = ctypes.c_size_t
        self.lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self.lib.ZSTD_isError.restype = ctypes.c_uint

    def compress(self, value):
        destination = ctypes.create_string_buffer(self.lib.ZSTD_compressBound(len(value)))
        size = self.lib.ZSTD_compress(destination, len(destination), value, len(value), 3)
        if self.lib.ZSTD_isError(size):
            raise ValueError("Zstd compression failed")
        return destination.raw[:size]

    def decompress(self, value, size):
        if not 0 < size <= MAX_RAW_BYTES:
            raise ValueError("Invalid decompressed record size")
        destination = ctypes.create_string_buffer(size)
        actual = self.lib.ZSTD_decompress(destination, size, value, len(value))
        if self.lib.ZSTD_isError(actual) or actual != size:
            raise ValueError("Zstd payload failed decompression")
        return destination.raw


_zstd = None


def compressor():
    global _zstd
    if _zstd is None:
        _zstd = Zstd()
    return _zstd


def verify_request(request):
    if content_sha({k: v for k, v in request.items() if k != "request_sha256"}) != request.get("request_sha256"):
        raise ValueError("Teacher request hash differs")


def validate_record(cache):
    request, metadata, tensors = cache["request"], cache["metadata"], cache["tensors"]
    verify_request(request)
    if metadata["request_sha256"] != request["request_sha256"]:
        raise ValueError("Target/request identity differs")
    positions = tensors["student_positions"]
    expected = [p for chain in request["chains"] for p in chain["student_positions"]]
    pairs, logits = tensors["teacher_pair_indices"], tensors["teacher_logits"]
    if positions.dtype != torch.long or positions.tolist() != expected:
        raise ValueError("Teacher/student mapping differs")
    if (pairs.dtype != torch.long or pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
            or pairs.min() < 0 or pairs.max() >= len(positions)
            or not (pairs[:, 0] < pairs[:, 1]).all() or len(pairs.unique(dim=0)) != len(pairs)):
        raise ValueError("Invalid teacher pairs")
    if logits.shape != (len(pairs), metadata["teacher_contract"]["distogram_bins"]) or not torch.isfinite(logits).all():
        raise ValueError("Invalid teacher logits")
    chain_ids = torch.tensor([c["record_chain"] for c in request["chains"] for _ in c["sequence"]])
    allowed = {tuple(sorted(x["chains"])) for x in request.get("inter_evidence", [])}
    for a, b in chain_ids[pairs].unique(dim=0).tolist():
        if a != b and tuple(sorted((a, b))) not in allowed:
            raise ValueError("Inter-chain target lacks evidence")
    if metadata.get("pair_count", len(pairs)) != len(pairs):
        raise ValueError("Target pair count differs")


def encode_record(cache):
    validate_record(cache)
    arrays, descriptions = [], []
    for name in ("teacher_logits", "teacher_pair_indices", "student_positions"):
        value = cache["tensors"][name].detach().cpu().numpy()
        if name == "teacher_logits":
            with np.errstate(over="ignore", invalid="ignore"):
                value = value.astype("<f2")
            if not np.isfinite(value).all():
                raise ValueError("FP16 conversion overflow")
        else:
            if value.size and (value.min() < 0 or value.max() > 65535):
                raise ValueError("Teacher indices exceed uint16 range")
            value = value.astype("<u2")
        value = np.ascontiguousarray(value)
        arrays.append(value.view(np.uint8).reshape(-1, 2).T.copy().tobytes())
        descriptions.append(dict(name=name, dtype=value.dtype.str, shape=list(value.shape), nbytes=value.nbytes))
    header = canonical(dict(codec=CODEC, metadata=cache["metadata"], request=cache["request"], tensors=descriptions))
    raw = struct.pack("<I", len(header)) + header + b"".join(arrays)
    if len(raw) > MAX_RAW_BYTES:
        raise ValueError("Target record exceeds codec bound")
    packed = compressor().compress(raw)
    return FRAME.pack(MAGIC, len(raw), len(packed), hashlib.sha256(raw).digest()) + packed


def decode_record(blob):
    if len(blob) < FRAME.size:
        raise ValueError("Truncated target frame")
    magic, raw_size, packed_size, digest = FRAME.unpack_from(blob)
    if magic != MAGIC or packed_size != len(blob) - FRAME.size:
        raise ValueError("Invalid target frame")
    raw = compressor().decompress(blob[FRAME.size:], raw_size)
    if hashlib.sha256(raw).digest() != digest:
        raise ValueError("Target checksum differs")
    header_size = struct.unpack_from("<I", raw)[0]
    if not 0 < header_size <= len(raw) - 4:
        raise ValueError("Invalid target header")
    header = json.loads(raw[4:4 + header_size])
    if header["codec"] != CODEC:
        raise ValueError("Unsupported cache codec")
    offset, tensors = 4 + header_size, {}
    expected_names = ["teacher_logits", "teacher_pair_indices", "student_positions"]
    if [x["name"] for x in header["tensors"]] != expected_names:
        raise ValueError("Invalid tensor names")
    for spec in header["tensors"]:
        dtype = "<f2" if spec["name"] == "teacher_logits" else "<u2"
        size = spec["nbytes"]
        shape = spec["shape"]
        if (spec["dtype"] != dtype or any(type(n) is not int or n < 0 for n in shape)
                or int(np.prod(shape)) * 2 != size or offset + size > len(raw)):
            raise ValueError("Invalid tensor descriptor")
        shuffled = np.frombuffer(raw[offset:offset + size], dtype=np.uint8).reshape(2, -1)
        value = np.frombuffer(shuffled.T.copy().tobytes(), dtype=dtype).reshape(shape).copy()
        tensor = torch.from_numpy(value.astype(np.float32 if dtype == "<f2" else np.int64))
        tensors[spec["name"]] = tensor
        offset += size
    if offset != len(raw):
        raise ValueError("Trailing target payload")
    cache = dict(metadata=header["metadata"], request=header["request"], tensors=tensors)
    validate_record(cache)
    return cache


def precision_metrics(reference, restored, temperature=1.):
    reference = reference.double() / temperature
    restored = restored.double() / temperature
    log_p, log_q = reference.log_softmax(-1), restored.log_softmax(-1)
    values = (log_p.exp() * (log_p - log_q)).sum(-1).clamp_min(0)
    return dict(mean_kl=float(values.mean()), max_pair_kl=float(values.max()), pairs=len(values))


def read_index(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def validate_pilot_receipt(pilot, spec, contract):
    if (pilot.get("status") != "passed" or pilot.get("codec") != CODEC
            or not all(math.isfinite(pilot[k]) for k in ("mean_kl", "max_pair_kl"))
            or not 0 <= pilot["mean_kl"] <= 1e-5 or not 0 <= pilot["max_pair_kl"] <= 1e-3
            or pilot["pilot_sha256"] != spec["pilot_sha256"] or pilot["count"] != spec["pilot_count"]
            or pilot["teacher_contract_sha256"] != content_sha(contract)):
        raise ValueError("Worker precision pilot is incomplete or failed")
    measurements = pilot["measurements"]
    repeated = [x["repeat_max_abs"] for x in measurements if x.get("repeat_max_abs") is not None]
    if (len(measurements) != spec["pilot_count"] or len(repeated) < min(2, spec["pilot_count"])
            or any(not math.isfinite(x) or not 0 <= x <= 1e-5 for x in repeated)):
        raise ValueError("Worker seeded pilot replay is incomplete or failed")


def shard_info(directory, verify_data=True):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("schema") != "immune.stage1.ab500k_shard.v1" or manifest.get("codec") != CODEC
            or type(manifest.get("count")) is not int or manifest["count"] <= 0):
        raise ValueError("Invalid committed shard manifest")
    for field, name in (("index_sha256", "index.jsonl"), ("data_sha256", "records.bin")):
        if (name != "records.bin" or verify_data) and sha(directory / name) != manifest[field]:
            raise ValueError("Committed shard checksum differs: " + str(directory))
    entries = list(read_index(directory / "index.jsonl"))
    if len(entries) != manifest["count"]:
        raise ValueError("Shard count differs")
    offset = 0
    for entry in entries:
        if entry["offset"] != offset or entry["length"] <= 0:
            raise ValueError("Invalid shard offsets")
        offset += entry["length"]
    if offset != (directory / "records.bin").stat().st_size:
        raise ValueError("Shard length differs")
    return manifest, entries


class ShardWriter:
    def __init__(self, root, worker_index, contract, *, budget=WORKER_BUDGET,
                 max_records=1024, max_bytes=256 << 20):
        if not 0 <= worker_index < 16 or not 0 < budget <= WORKER_BUDGET:
            raise ValueError("Invalid worker or storage quota")
        self.root = Path(root).resolve()
        self.directory = self.root / f"worker{worker_index:02d}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / ".writer.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise ValueError("Worker already has an active writer") from None
        self.worker_index, self.contract = worker_index, contract
        self.budget, self.max_records, self.max_bytes = budget, max_records, max_bytes
        self.seen, self.shards, self.record_bytes = {}, [], {}
        self.temporary = self.data = self.index = None
        self.pending, self.pending_bytes = [], 0
        try:
            for abandoned in self.directory.glob("shard*.partial.*"):
                shutil.rmtree(abandoned)
            for directory in sorted(self.directory.glob("shard[0-9]*")):
                if not directory.is_dir() or ".partial." in directory.name:
                    continue
                manifest, entries = shard_info(directory)
                if manifest["teacher_contract_sha256"] != content_sha(contract) or manifest["worker_index"] != worker_index:
                    raise ValueError("Committed shard teacher/worker differs")
                for entry in entries:
                    key = (entry["split"], entry["row_index"])
                    if key in self.seen:
                        raise ValueError("Duplicate committed target row")
                    self.seen[key] = entry["request_sha256"]
                    self.record_bytes[key] = entry["length"] + len(canonical(entry)) + 1
                self.shards.append(directory)
            self.committed_bytes = sum(p.stat().st_size for p in self.directory.rglob("*") if p.is_file())
            if self.committed_bytes > budget:
                raise ValueError("Worker storage quota already exceeded")
        except BaseException:
            self.close()
            raise

    def contains(self, envelope):
        key = (envelope["split"], envelope["row_index"])
        if key not in self.seen:
            return False
        if self.seen[key] != envelope["request"]["request_sha256"]:
            raise ValueError("Resumed row has a different request")
        return True

    def append(self, envelope, cache):
        if self.contains(envelope):
            return False
        if cache["request"] != envelope["request"] or cache["metadata"]["teacher_contract"] != self.contract:
            raise ValueError("Target/envelope/teacher mismatch")
        blob = encode_record(cache)
        if self.pending and (len(self.pending) >= self.max_records or self.pending_bytes + len(blob) > self.max_bytes):
            self.commit()
        if len(blob) > self.max_bytes:
            raise ValueError("One target exceeds shard bound")
        entry = dict(split=envelope["split"], row_index=envelope["row_index"],
            request_sha256=cache["request"]["request_sha256"], record_sha256=content_sha(cache["request"]["record"]),
            offset=0 if self.data is None else self.data.tell(), length=len(blob),
            payload_sha256=hashlib.sha256(blob).hexdigest(), pairs=len(cache["tensors"]["teacher_logits"]))
        line = canonical(entry) + b"\n"
        if self.pending and self.pending_bytes + len(blob) + len(line) > self.max_bytes:
            self.commit()
            entry["offset"] = 0
            line = canonical(entry) + b"\n"
        if len(blob) + len(line) > self.max_bytes:
            raise ValueError("One target and index exceed shard bound")
        # Reserve metadata and receipts before any additional bytes are written.
        if self.committed_bytes + self.pending_bytes + len(blob) + len(line) + (2 << 20) > self.budget:
            # Preserve every already accepted target before leaving on quota.
            self.commit()
            raise RuntimeError("Worker cache storage quota exhausted; no targets may be dropped")
        if self.temporary is None:
            self.temporary = Path(tempfile.mkdtemp(prefix=f"shard{len(self.shards):06d}.partial.", dir=self.directory))
            self.data = (self.temporary / "records.bin").open("wb")
            self.index = (self.temporary / "index.jsonl").open("wb")
        self.data.write(blob)
        self.index.write(line)
        self.pending.append(entry)
        self.pending_bytes += len(blob) + len(line)
        self.seen[(entry["split"], entry["row_index"])] = entry["request_sha256"]
        self.record_bytes[(entry["split"], entry["row_index"])] = len(blob) + len(line)
        return True

    def commit(self):
        if not self.pending:
            return
        for stream in (self.data, self.index):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        self.data = self.index = None
        atomic_json(self.temporary / "manifest.json", dict(schema="immune.stage1.ab500k_shard.v1", codec=CODEC,
            teacher_contract_sha256=content_sha(self.contract), worker_index=self.worker_index,
            count=len(self.pending), data_sha256=sha(self.temporary / "records.bin"),
            index_sha256=sha(self.temporary / "index.jsonl")))
        shard_info(self.temporary)
        destination = self.directory / f"shard{len(self.shards):06d}"
        self.temporary.rename(destination)
        fsync_dir(self.directory)
        self.shards.append(destination)
        self.committed_bytes += sum(p.stat().st_size for p in destination.iterdir())
        self.temporary, self.pending, self.pending_bytes = None, [], 0

    def finish(self, expected_count, receipt):
        self.commit()
        if len(self.seen) != expected_count:
            raise ValueError("Worker target coverage is incomplete")
        atomic_json(self.directory / "complete.json", dict(status="passed", worker_index=self.worker_index,
            count=len(self.seen), teacher_contract=self.contract, codec=CODEC, **receipt))

    def close(self):
        for stream in (self.data, self.index):
            if stream is not None:
                stream.close()
        self.data = self.index = None
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def request_binding(subset):
    requests = Path(subset["requests"])
    if sha(requests) != subset["requests_sha256"]:
        raise ValueError("Subset request file changed")
    return requests


def tree_bytes(path):
    path = Path(path)
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def verify_subset_receipt(plan):
    """Bind the one deep CPU audit without importing the student's data stack."""
    receipt_path = Path(plan["subset_validation"])
    if sha(receipt_path) != plan["subset_validation_sha256"]:
        raise ValueError("Subset validation receipt changed")
    receipt = json.loads(receipt_path.read_text())
    subset_path = Path(plan["subset_manifest"])
    subset = json.loads(subset_path.read_text())
    if (receipt.get("status") != "passed" or receipt.get("manifest_sha256") != sha(subset_path)
            or receipt.get("counts") != subset["counts"]):
        raise ValueError("Subset validation does not bind this frozen selection")
    return receipt


def build_cache_index(subset_manifest, cache_root, output):
    subset_manifest, cache_root, output = map(lambda p: Path(p).resolve(), (subset_manifest, cache_root, output))
    subset = json.loads(subset_manifest.read_text())
    requests = request_binding(subset)
    if output.exists() and (output / "manifest.json").exists():
        existing = validate_cache_manifest(output / "manifest.json", verify_payloads=True)
        if existing["subset_manifest_sha256"] != sha(subset_manifest):
            raise ValueError("Existing index belongs to another subset")
        return output / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial.", dir=output.parent))
    connection = sqlite3.connect(temporary / "cache.sqlite")
    try:
        connection.execute("CREATE TABLE targets (split TEXT,row_index INTEGER,record_sha256 TEXT,request_sha256 TEXT,shard TEXT,offset INTEGER,length INTEGER,payload_sha256 TEXT,pairs INTEGER, PRIMARY KEY(split,row_index))")
        connection.execute("CREATE TABLE expected (split TEXT,row_index INTEGER,request_sha256 TEXT,PRIMARY KEY(split,row_index))")
        counts = dict(train=0, valid=0, records=0, pairs=0)
        for env in read_index(requests):
            verify_request(env["request"])
            connection.execute("INSERT INTO expected VALUES (?,?,?)", (env["split"], env["row_index"], env["request"]["request_sha256"]))
            counts[env["split"]] += 1
        if subset.get("counts") and {s: counts[s] for s in ("train", "valid")} != subset["counts"]:
            raise ValueError("Subset request counts differ from frozen selection")
        contracts, shards, worker_receipts, maximum = [], [], [], 0
        partition = subset["request_shards"]
        if len(partition) != 16 or {x["worker_index"] for x in partition} != set(range(16)):
            raise ValueError("Subset must define exactly 16 workers")
        for spec in sorted(partition, key=lambda x: x["worker_index"]):
            directory = cache_root / f"worker{spec['worker_index']:02d}"
            receipt = json.loads((directory / "complete.json").read_text())
            if (receipt["status"] != "passed" or receipt["count"] != spec["count"]
                    or receipt["requests_sha256"] != spec["sha256"]
                    or receipt["subset_manifest_sha256"] != sha(subset_manifest)):
                raise ValueError("Worker completion binding differs")
            pilot_path = directory / "pilot.json"
            if sha(pilot_path) != receipt["pilot_sha256"]:
                raise ValueError("Worker pilot receipt changed")
            pilot = json.loads(pilot_path.read_text())
            validate_pilot_receipt(pilot, spec, receipt["teacher_contract"])
            worker_receipts.append(dict(worker_index=spec["worker_index"], path=str(directory / "complete.json"),
                sha256=sha(directory / "complete.json"), pilot_path=str(pilot_path), pilot_sha256=sha(pilot_path)))
            contract = receipt["teacher_contract"]
            if contracts and contracts[0] != contract:
                raise ValueError("Workers used different teachers")
            contracts.append(contract)
            worker_count = 0
            for directory in sorted(directory.glob("shard[0-9]*")):
                if ".partial." in directory.name:
                    continue
                shard, entries = shard_info(directory)
                if shard["teacher_contract_sha256"] != content_sha(contract):
                    raise ValueError("Shard contract mismatch")
                data_path = directory / "records.bin"
                with data_path.open("rb") as stream:
                    for entry in entries:
                        stream.seek(entry["offset"])
                        blob = stream.read(entry["length"])
                        if hashlib.sha256(blob).hexdigest() != entry["payload_sha256"]:
                            raise ValueError("Target payload checksum differs")
                        cache = decode_record(blob)
                        if (cache["metadata"]["teacher_contract"] != contract
                                or cache["request"]["request_sha256"] != entry["request_sha256"]
                                or content_sha(cache["request"]["record"]) != entry["record_sha256"]):
                            raise ValueError("Target record binding differs")
                        connection.execute("INSERT INTO targets VALUES (?,?,?,?,?,?,?,?,?)", (entry["split"], entry["row_index"],
                            entry["record_sha256"], entry["request_sha256"], str(data_path), entry["offset"], entry["length"], entry["payload_sha256"], entry["pairs"]))
                        counts["records"] += 1
                        counts["pairs"] += entry["pairs"]
                        maximum = max(maximum, entry["pairs"])
                        worker_count += 1
                shards.append(dict(path=str(directory), manifest_sha256=sha(directory / "manifest.json"),
                    data_path=str(data_path), data_sha256=shard["data_sha256"], index_path=str(directory / "index.jsonl"),
                    index_sha256=shard["index_sha256"], count=shard["count"]))
            if worker_count != spec["count"]:
                raise ValueError("Worker committed shard count differs")
        missing = connection.execute("SELECT COUNT(*) FROM expected e LEFT JOIN targets t USING(split,row_index) WHERE t.request_sha256 IS NULL OR t.request_sha256 != e.request_sha256").fetchone()[0]
        extra = connection.execute("SELECT COUNT(*) FROM targets t LEFT JOIN expected e USING(split,row_index) WHERE e.request_sha256 IS NULL").fetchone()[0]
        if missing or extra or counts["records"] != counts["train"] + counts["valid"]:
            raise ValueError("Cache coverage is incomplete or requests differ")
        connection.execute("DROP TABLE expected")
        connection.commit()
        connection.close()
        auxiliary_bytes = tree_bytes(subset_manifest.parent) + tree_bytes(temporary) + tree_bytes(output)
        if auxiliary_bytes + (2 << 20) > AUX_BUDGET:
            raise RuntimeError("Subset files plus temporary/final cache indexes exceed 10 GB auxiliary quota")
        atomic_json(temporary / "manifest.json", dict(schema=SCHEMA, status="passed", quality_status="unvalidated",
            subset_manifest=str(subset_manifest), subset_manifest_sha256=sha(subset_manifest),
            data_index=subset["data_index"], data_index_sha256=subset["data_index_sha256"],
            database=str(output / "cache.sqlite"), database_sha256=sha(temporary / "cache.sqlite"),
            teacher_contract=contracts[0], teacher_contract_sha256=content_sha(contracts[0]), codec=CODEC,
            counts=counts, max_pairs_per_record=maximum, worker_count=16, shards=shards, workers=worker_receipts,
            auxiliary_bytes_before_manifest=auxiliary_bytes))
        with (temporary / "cache.sqlite").open("rb") as stream:
            os.fsync(stream.fileno())
        fsync_dir(temporary)
        if output.exists():
            if list(output.iterdir()):
                raise ValueError("Refusing to replace nonempty cache index output")
            output.rmdir()
        temporary.rename(output)
        fsync_dir(output.parent)
        return output / "manifest.json"
    except BaseException:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_cache_manifest(path, verify_payloads=False):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "passed" or manifest.get("codec") != CODEC:
        raise ValueError("Cache manifest is not a completed supported cache")
    for field in ("subset_manifest", "data_index", "database"):
        if sha(manifest[field]) != manifest[field + "_sha256"]:
            raise ValueError("Cache binding changed: " + field)
    if content_sha(manifest["teacher_contract"]) != manifest["teacher_contract_sha256"]:
        raise ValueError("Teacher contract hash differs")
    subset = json.loads(Path(manifest["subset_manifest"]).read_text())
    requests = request_binding(subset)
    if len(manifest["workers"]) != 16:
        raise ValueError("Cache manifest does not bind all worker receipts")
    for worker in manifest["workers"]:
        if sha(worker["path"]) != worker["sha256"] or sha(worker["pilot_path"]) != worker["pilot_sha256"]:
            raise ValueError("Worker completion or precision receipt changed")
    for shard in manifest["shards"]:
        if sha(Path(shard["path"]) / "manifest.json") != shard["manifest_sha256"]:
            raise ValueError("Shard manifest changed")
        shard_info(shard["path"], verify_data=verify_payloads)
    counts = dict(train=0, valid=0, records=0, pairs=0)
    store = PackedCacheStore(manifest)
    try:
        for envelope in read_index(requests):
            split, row, request = envelope["split"], envelope["row_index"], envelope["request"]
            value = store.entry(split, row)
            if value[1] != request["request_sha256"] or value[0] != content_sha(request["record"]):
                raise ValueError("Indexed target differs from subset")
            if verify_payloads:
                cache = store.lookup(split, row, request["record"])
                if cache["request"] != request:
                    raise ValueError("Decoded request differs from subset")
            counts[split] += 1
            counts["records"] += 1
            counts["pairs"] += value[-1]
        if counts != manifest["counts"] or store.connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0] != counts["records"]:
            raise ValueError("Cache coverage counts differ")
        if subset.get("counts") and {s: counts[s] for s in ("train", "valid")} != subset["counts"]:
            raise ValueError("Cache split sizes differ from frozen subset")
    finally:
        store.close()
    return manifest


class PackedCacheStore:
    def __init__(self, manifest):
        self.manifest = json.loads(Path(manifest).read_text()) if not isinstance(manifest, dict) else manifest
        if self.manifest["schema"] != SCHEMA or self.manifest["status"] != "passed":
            raise ValueError("Uncommitted cache manifest")
        self.connection, self.pid = None, None

    def entry(self, split, row):
        if self.connection is None or self.pid != os.getpid():
            self.close()
            self.connection = sqlite3.connect(Path(self.manifest["database"]).as_uri() + "?mode=ro&immutable=1", uri=True)
            self.pid = os.getpid()
        result = self.connection.execute("SELECT record_sha256,request_sha256,shard,offset,length,payload_sha256,pairs FROM targets WHERE split=? AND row_index=?", (split, row)).fetchone()
        if result is None:
            raise ValueError(f"Required offline teacher target missing: {split}/{row}")
        return result

    def lookup(self, split, row, record):
        expected_record, expected_request, shard, offset, length, digest, _ = self.entry(split, row)
        record = record.to_dict() if hasattr(record, "to_dict") else record
        if content_sha(record) != expected_record:
            raise ValueError("Training record differs from cached row")
        with Path(shard).open("rb") as stream:
            stream.seek(offset)
            blob = stream.read(length)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("Cached record payload changed")
        cache = decode_record(blob)
        if (cache["request"]["request_sha256"] != expected_request
                or cache["metadata"]["teacher_contract"] != self.manifest["teacher_contract"]):
            raise ValueError("Cached request or teacher differs")
        return cache

    def close(self):
        if self.connection is not None:
            self.connection.close()
        self.connection, self.pid = None, None

    def __getstate__(self):
        return dict(manifest=self.manifest, connection=None, pid=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("index", "check"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--verify-payloads", action="store_true", help="check always verifies all payloads")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    verify_subset_receipt(plan)
    path = Path(plan["cache_manifest"])
    if args.command == "index":
        path = build_cache_index(plan["subset_manifest"], plan["cache_root"], path.parent)
        # Index creation has just decoded and checked every target.
        manifest = validate_cache_manifest(path, verify_payloads=False)
    else:
        manifest = validate_cache_manifest(path, verify_payloads=True)
    print(json.dumps(dict(status="passed", manifest=str(path), counts=manifest["counts"], codec=CODEC)), flush=True)


if __name__ == "__main__":
    main()
