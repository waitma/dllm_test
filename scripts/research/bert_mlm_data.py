"""Freeze byte indices and development examples for complete-epoch BERT training.

Run: python -B -m scripts.research.bert_mlm_data --data-dir ABS --output ABS
The dataset reuses the validated prepared-row reader; shards are never rewritten.
"""
import argparse
from array import array
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import heapq
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler
from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset

ROLES = ("antibody_heavy", "antibody_light", "tcr_beta", "tcr_alpha")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def receptor_key(row):
    """Observed sequence identity, ignoring synthetic template and condition labels."""
    parts = []
    for i, (role, sequence) in enumerate(zip(row["chain_roles"], row["chains"])):
        if role in ROLES:
            synthetic = row.get("chain_metadata", {}).get(str(i), {}).get("synthetic_residue_mask", [])
            observed = "".join(aa for j, aa in enumerate(sequence) if not synthetic or not synthetic[j])
            parts.append((role, observed))
    return hashlib.sha256(json.dumps(sorted(parts)).encode()).hexdigest()


def keep_smallest(heap, score, row, limit):
    # A deterministic secondary key prevents dictionary comparisons on exact duplicates.
    item = (-score, json.dumps(row, sort_keys=True))
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def scan_shard(job):
    directory, entry, output, held_keys = job
    path = Path(directory) / entry["path"]
    try:
        import orjson
        loads = orjson.loads
    except ImportError:
        loads = json.loads
    before = path.stat()
    digest, offsets, reasons = hashlib.sha256(), array("Q"), Counter()
    probe, validation = defaultdict(list), []
    overlap = set()
    counts, position, seen = Counter(), 0, 0
    with path.open("rb", buffering=4*1024*1024) as stream:
        for line in stream:
            digest.update(line)
            offset, position = position, position + len(line)
            if not line.strip():
                continue
            row = loads(line)
            seen += 1
            if entry.get("forbid_nonbinding") and row.get("labels", {}).get("relation") == "nonbinding":
                raise ValueError("Explicit nonbinding row in cleaned pretraining data: " + str(path))
            eligible = 0
            for i, (role, sequence) in enumerate(zip(row["chain_roles"], row["chains"])):
                if role not in ROLES:
                    continue
                capacities = entry.get("receptor_capacities", {})
                if role in capacities and len(sequence) + 1 > capacities[role]:
                    raise ValueError("Receptor exceeds its bound slot capacity: " + str(path))
                synthetic = row.get("chain_metadata", {}).get(str(i), {}).get("synthetic_residue_mask", [])
                if synthetic and (len(synthetic) != len(sequence) or any(x not in (0, 1) for x in synthetic)):
                    raise ValueError("Invalid synthetic provenance: " + str(path))
                n = len(sequence) - sum(synthetic)
                eligible += n
                counts[role] += n
            if not eligible:
                reasons["zero_observed_receptor_residues"] += 1
                continue
            offsets.append(offset)
            key = receptor_key(row)
            if key in held_keys:
                overlap.add(key)
            label = row.get("labels", {}).get("relation")
            score = int(hashlib.sha256((entry["path"] + ":" + str(offset)).encode()).hexdigest(), 16)
            sample = {"record": row, "source_group": entry["source"], "receptor_key": key,
                      "origin": {"path": str(path), "offset": offset}}
            if entry["split"] == "valid":
                keep_smallest(validation, score, sample, 128)
            if label in ("binding", "nonbinding"):
                keep_smallest(probe[label], score, sample, 512 if entry["split"] == "valid" else 256)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or seen != entry["records"]:
        raise ValueError("Shard changed or record count differs: " + str(path))
    index = Path(output) / (entry["split"] + "_" + path.stem + ".npy")
    np.save(index, np.asarray(offsets, dtype=np.uint64))
    return dict(path=str(path), resolved_path=str(path.resolve()),
                source=entry["source"], split=entry["split"], sha256=digest.hexdigest(),
                index=str(index), index_sha256=sha(index), records=len(offsets), original_records=seen,
                excluded=dict(reasons), observed_residues=dict(counts),
                probe={k: v for k, v in probe.items()}, validation=validation, overlap=sorted(overlap))


class IndexedPreparedDataset(PreparedImmuneDataset):
    """Use frozen compact offsets with the existing per-worker lazy JSONL reader."""

    def __init__(self, index_path, split="train"):
        index = json.loads(Path(index_path).read_text())
        self.dataset_dir = Path(index["data_dir"])
        self.split, self.source = split, None
        self._shards, self._cumulative_counts = [], []
        total = 0
        for shard in index["shards"]:
            if shard["split"] != split:
                continue
            if sha(shard["index"]) != shard["index_sha256"]:
                raise ValueError("Frozen offsets changed")
            offsets = np.load(shard["index"], mmap_mode="r")
            if len(offsets) != shard["records"]:
                raise ValueError("Offset count changed")
            self._shards.append((Path(shard["path"]), offsets, shard["source"]))
            total += len(offsets)
            self._cumulative_counts.append(total)
        self._length, self._handles, self._handle_pid = total, {}, os.getpid()


class CompleteEpochBatchSampler(Sampler):
    """Each global shuffled batch is partitioned over ranks without duplication.

    An incomplete final global batch remains an optimizer step. Starting from a
    saved batch index regenerates the permutation without reading skipped rows.
    """

    def __init__(self, size, global_batch, rank, world_size, seed, epoch, start_batch=0, max_batches=None):
        if global_batch % world_size or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed batch")
        if 0 < size % global_batch < world_size:
            raise ValueError("Tail needs at least one record per rank")
        self.order = torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch))
        self.size, self.global_batch = size, global_batch
        self.rank, self.world_size, self.start_batch = rank, world_size, start_batch
        available = math.ceil(size / global_batch) - start_batch
        if available < 0 or start_batch < 0 or (max_batches is not None and max_batches < 0):
            raise ValueError("Invalid epoch cursor or step limit")
        self.batch_count = available if max_batches is None else min(available, max_batches)

    def __iter__(self):
        for batch in range(self.start_batch, self.start_batch + self.batch_count):
            start = batch * self.global_batch
            yield self.order[start:start+self.global_batch][self.rank::self.world_size].tolist()

    def __len__(self):
        return self.batch_count


def step_budget(size, global_batch, max_steps):
    """Count real record visits, including short epoch tails and a partial last epoch."""
    if min(size, global_batch, max_steps) <= 0:
        raise ValueError("Dataset size, batch and training steps must be positive")
    steps_per_epoch = math.ceil(size / global_batch)
    epochs, next_batch = divmod(max_steps, steps_per_epoch)
    seen = epochs * size + next_batch * global_batch
    return dict(steps_per_epoch=steps_per_epoch, completed_epochs=epochs, next_batch=next_batch,
                seen_records=seen, equivalent_epochs=seen/size, tail_records=size % global_batch,
                epochs_started=epochs + bool(next_batch))


def prepare(data_dir, output, binding_path, workers):
    from scripts.research.fixed_chain_data_binding import verify_binding
    binding = json.loads(binding_path.read_text())
    verify_binding(binding_path, sha(binding_path), data_dir, 343, "fixed_chain_full_global", full_hash=False)
    audit = json.loads(Path(binding["audit"]["path"]).read_text())
    expected = {x["path"]: x["sha256"] for x in audit["shard_results"]}
    manifest_path = data_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    results, held_keys = [], set()
    for split in ("valid", "train"):
        jobs = [(str(data_dir), {**x, "split": split}, str(output), held_keys)
                for x in manifest["splits"][split]["shards"]]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(scan_shard, job) for job in jobs]
            for future in as_completed(futures):
                item = future.result()
                if expected[item["path"]] != item["sha256"]:
                    raise ValueError("V5 bytes differ from the full native audit")
                results.append(item)
                print(json.dumps(dict(split=split, completed=len(results), records=item["records"])), flush=True)
        if split == "valid":
            held_keys = {json.loads(text)["receptor_key"] for r in results for h in r["probe"].values() for _, text in h}
    exposed = {key for r in results if r["split"] == "train" for key in r["overlap"]}
    candidates, validation = defaultdict(list), defaultdict(list)
    for item in results:
        for label, heap in item.pop("probe").items():
            for score, text in heap:
                sample = json.loads(text)
                if item["split"] == "valid" and sample["receptor_key"] in exposed:
                    continue
                keep_smallest(candidates[item["split"], item["source"], label], -score, sample,
                              128 if item["split"] == "valid" else 256)
        for score, text in item.pop("validation"):
            keep_smallest(validation[item["source"]], -score, json.loads(text), 128)
        item.pop("overlap")
    files = {}
    def save_rows(name, rows):
        path = output / (name + ".jsonl")
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row)+"\n")
        files[name] = dict(path=str(path), sha256=sha(path), records=len(rows))
    save_rows("token_valid", [json.loads(s) for source in sorted(validation) for _, s in sorted(validation[source])])
    valid_sources = sorted({source for split, source, label in candidates if all(
        len(candidates.get((part, source, target), [])) >= 16
        for part in ("train", "valid") for target in ("binding", "nonbinding"))})
    if not valid_sources:
        raise ValueError("No independent labelled development source with both classes")
    for split in ("train", "valid"):
        rows, seen = [], set()
        for source in valid_sources:
            for label in ("binding", "nonbinding"):
                for _, text in sorted(candidates[split, source, label]):
                    sample = json.loads(text)
                    # Preserve different antigen conditions; remove exact duplicate rows.
                    key = (source, sample["receptor_key"], json.dumps(sample["record"], sort_keys=True))
                    if key not in seen:
                        rows.append(sample)
                        seen.add(key)
        save_rows("probe_" + split, rows)
    result = dict(schema="immune.bert_mlm.index.v1", data_dir=str(data_dir),
                  manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
                  binding=str(binding_path), binding_sha256=sha(binding_path),
                  shards=sorted(results, key=lambda x: (x["split"], x["path"])), files=files,
                  records={s: sum(r["records"] for r in results if r["split"] == s) for s in ("train", "valid")},
                  probe_sources=valid_sources, probe_valid_candidates_exposed_to_train=len(exposed),
                  probe_protocol="V5 valid receptor identities absent from all pretraining rows; per-source binary relation diagnostic; no benchmark test",
                  full_shard_hashes_verified=len(results), max_audited_length=audit["max_total"])
    write_json(output / "index.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("shards", "files")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.data_dir.resolve(), args.output.resolve(), args.binding.resolve(), args.workers)
