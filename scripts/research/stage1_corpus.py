"""Bind sparse teacher caches to immutable V7 row offsets without loading the corpus.

Activate pllm; run `python -B -m scripts.research.stage1_corpus --help`.
`requests` exports bounded, seeded per-source samples as JSONL; `index` verifies
their exact dataset rows and builds a read-only SQLite lookup for student workers.
Missing entries mean MLM-only, and never imply teacher quality approval.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import torch
from torch.utils.data import Dataset

from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from scripts.research.bert_mlm_data import IndexedPreparedDataset, sha, write_json
from scripts.research.stage1_data import content_sha, make_teacher_request, verify_request, load_teacher_cache


def entries(path, index_path):
    """Read streaming envelopes or import the existing provenance-bound pilot."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        return
    pilot = json.loads(path.read_text())
    index = json.loads(Path(index_path).read_text())
    if sha(pilot["source"]) != pilot["source_sha256"]:
        raise ValueError("Pilot source changed")
    offsets, base = None, 0
    for shard in index["shards"]:
        if shard["split"] != "train":
            continue
        if Path(shard["path"]).resolve() == Path(pilot["source"]).resolve():
            offsets = np.load(shard["index"], mmap_mode="r")
            break
        base += shard["records"]
    if offsets is None or len(pilot["requests"]) != len(pilot["origins"]):
        raise ValueError("Pilot source is absent from bound training index")
    for request, origin in zip(pilot["requests"], pilot["origins"]):
        row = int(np.searchsorted(offsets, origin["offset"]))
        if row >= len(offsets) or int(offsets[row]) != origin["offset"]:
            raise ValueError("Pilot record excluded from bound training index")
        yield dict(split="train", row_index=base+row, request=request)


def export_requests(args):
    index = json.loads(Path(args.data_index).read_text())
    config = FixedChainConfig(**json.loads(Path(index["binding"]).read_text())["model"])
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with output.open("x") as stream:
        for split in args.splits:
            dataset = IndexedPreparedDataset(args.data_index, split)
            groups, base = {}, 0
            for shard in index["shards"]:
                if shard["split"] == split:
                    groups.setdefault(shard["source"], []).append((base, shard["records"]))
                    base += shard["records"]
            for source, ranges in sorted(groups.items()):
                # Sampling requests changes cache coverage, never the student sampler.
                order = torch.randperm(sum(n for _, n in ranges), generator=torch.Generator().manual_seed(
                    args.seed + int(content_sha([split, source])[:8], 16)))[:args.per_source]
                for ordinal in order.tolist():
                    for start, size in ranges:
                        if ordinal < size:
                            row_index = start + ordinal
                            break
                        ordinal -= size
                    record = dataset[row_index]
                    request = make_teacher_request(record, config)
                    # General exporter is intra-only. Pairing alone does not approve geometry.
                    envelope = dict(split=split, row_index=row_index, request=request)
                    stream.write(json.dumps(envelope, separators=(",", ":"))+"\n")
                    counts[f"{split}/{source}/records"] += 1
                    counts[f"{split}/{source}/teacher_eligible"] += bool(request["chains"])
    write_json(output.with_suffix(".receipt.json"), dict(schema="immune.stage1.requests.v1",
        data_index=str(Path(args.data_index).resolve()), data_index_sha256=sha(args.data_index),
        requests_sha256=sha(output), counts=dict(counts), seed=args.seed,
        quality_status="unvalidated", inter_policy="explicit_evidence_required; exporter_intra_only"))


def build_index(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    index = json.loads(Path(args.data_index).read_text())
    config = json.loads(Path(index["binding"]).read_text())["model"]
    datasets = {split: IndexedPreparedDataset(args.data_index, split) for split in ("train", "valid")}
    connection = sqlite3.connect(output / "cache.sqlite")
    connection.execute("CREATE TABLE targets (split TEXT, row_index INTEGER, record_sha TEXT, request TEXT, directory TEXT, metadata_sha TEXT, PRIMARY KEY(split,row_index))")
    counts, contracts, maximum = Counter(), [], 0
    try:
        for envelope in entries(args.requests, args.data_index):
            split, row = envelope["split"], envelope["row_index"]
            request = envelope["request"]
            verify_request(request)
            if split not in datasets or not isinstance(row, int) or not 0 <= row < len(datasets[split]):
                raise ValueError("Invalid corpus row binding")
            record = datasets[split][row]
            if request["record"] != record.to_dict() or request["config"] != config:
                raise ValueError("Cache request differs from its bound corpus row/config")
            counts[f"{split}/requested"] += 1
            if not request["chains"]:
                counts[f"{split}/ineligible"] += 1
                continue
            directory = Path(args.cache).resolve() / request["request_sha256"]
            # Unvalidated cache is indexed for engineering; the launcher separately
            # requires a cache-bound scientific quality report for formal training.
            cache = load_teacher_cache(directory, request, allow_unvalidated=True)
            contract = cache["metadata"]["teacher_contract"]
            if contracts and contracts[0] != contract:
                raise ValueError("Mixed teacher contracts")
            if not contracts:
                contracts.append(contract)
            connection.execute("INSERT INTO targets VALUES (?,?,?,?,?,?)", (split, row,
                content_sha(record.to_dict()), json.dumps(request), str(directory), sha(directory / "metadata.json")))
            counts[f"{split}/cached"] += 1
            counts[f"{split}/{record.source}/cached"] += 1
            counts[f"{split}/pairs"] += len(cache["tensors"]["teacher_logits"])
            maximum = max(maximum, len(cache["tensors"]["teacher_logits"]))
        connection.commit()
    finally:
        connection.close()
    if not contracts:
        raise ValueError("No teacher targets; cannot qualify a distillation run")
    write_json(output / "manifest.json", dict(schema="immune.stage1.cache_index.v1",
        database=str(output / "cache.sqlite"), database_sha256=sha(output / "cache.sqlite"),
        data_index=str(Path(args.data_index).resolve()), data_index_sha256=sha(args.data_index),
        requests=str(Path(args.requests).resolve()), requests_sha256=sha(args.requests),
        teacher_contract=contracts[0], counts=dict(counts), max_pairs_per_record=maximum,
        coverage={s: counts[f"{s}/cached"]/index["records"][s] for s in datasets},
        quality_status="unvalidated", corpus_records=index["records"]))


class CacheStore:
    def __init__(self, manifest):
        self.manifest = manifest
        self.connection = None
        self.pid = None

    def lookup(self, split, row, record):
        if self.connection is None or self.pid != os.getpid():
            if self.connection is not None:
                self.connection.close()
            self.connection = sqlite3.connect(Path(self.manifest["database"]).as_uri()+"?mode=ro&immutable=1", uri=True)
            self.pid = os.getpid()
        value = self.connection.execute("SELECT record_sha,request,directory,metadata_sha FROM targets WHERE split=? AND row_index=?", (split, row)).fetchone()
        if value is None:
            return None
        digest, request, directory, metadata_sha = value
        if content_sha(record.to_dict()) != digest or sha(Path(directory)/"metadata.json") != metadata_sha:
            raise ValueError("Bound record/cache metadata changed")
        cache = load_teacher_cache(directory, json.loads(request), allow_unvalidated=True)
        if cache["metadata"]["teacher_contract"] != self.manifest["teacher_contract"]:
            raise ValueError("Teacher contract changed")
        return cache

    def __getstate__(self):
        return dict(manifest=self.manifest, connection=None, pid=None)


class CachedCorpus(Dataset):
    def __init__(self, index, manifest, split="train", rows=None):
        self.dataset = IndexedPreparedDataset(index, split)
        self.store = CacheStore(manifest)
        self.split, self.rows = split, rows

    def __len__(self):
        return len(self.dataset) if self.rows is None else len(self.rows)

    def __getitem__(self, item):
        row = item if self.rows is None else self.rows[item]
        record = self.dataset[row]
        return row, record, self.store.lookup(self.split, row, record)


def identity_batch(rows):
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("requests", "index"):
        child = sub.add_parser(command)
        child.add_argument("--data-index", required=True)
        child.add_argument("--output", required=True)
        if command == "requests":
            child.add_argument("--per-source", type=int, default=128)
            child.add_argument("--splits", nargs="+", choices=("train", "valid"), default=["train", "valid"])
            child.add_argument("--seed", type=int, default=42)
        else:
            child.add_argument("--requests", required=True)
            child.add_argument("--cache", required=True)
    args = parser.parse_args()
    if args.command == "requests" and args.per_source < 1:
        parser.error("per-source must be positive")
    (export_requests if args.command == "requests" else build_index)(args)
