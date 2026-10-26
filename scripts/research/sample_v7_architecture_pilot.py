"""Copy a deterministic source-stratified exploratory subset without changing rows.

Activate conda pllm, then run:
python /project_c20250601/251105016/project/dllm_test/scripts/research/sample_v7_architecture_pilot.py --input-dir /absolute/prepared --output-dir /absolute/new_subset
Use --self-test for small stdlib-only fixtures. No training or annotation runs here.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile


def digest(value):
    return hashlib.sha256(value).hexdigest()


def stamp(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def quotas(counts, limit):
    """Give each nonempty source one row, then use proportional largest remainder."""
    counts = {key: value for key, value in sorted(counts.items()) if value}
    budget = min(limit, sum(counts.values()))
    if budget < len(counts):
        raise ValueError("Sample budget must cover every nonempty source")
    result = dict.fromkeys(counts, 1)
    remaining = budget - len(counts)
    capacity = sum(value - 1 for value in counts.values())
    if capacity:
        remainders = {}
        for key, count in counts.items():
            whole, remainder = divmod(remaining * (count - 1), capacity)
            result[key] += whole
            remainders[key] = remainder
        for key in sorted(counts, key=lambda key: (-remainders[key], key))[:budget - sum(result.values())]:
            result[key] += 1
    return result


def sample(input_dir, output_dir, train_records=100000, valid_records=1024, seed=42):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_absolute() or not output_dir.is_absolute():
        raise ValueError("Input and output paths must be absolute")
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("Output must be separate from the source dataset")
    if not 1 <= train_records <= 100000 or not 1 <= valid_records <= 1024:
        raise ValueError("Limits are 1..100000 train and 1..1024 valid records")
    metadata = {}
    metadata_stamps = {}

    def read_metadata(path):
        before = stamp(path)
        raw = path.read_bytes()
        if stamp(path) != before:
            raise ValueError(f"Metadata changed while reading: {path}")
        metadata[path] = raw
        metadata_stamps[path] = before
        return json.loads(raw), {"path": str(path), "sha256": digest(raw)}

    manifest_path = input_dir / "dataset_manifest.json"
    manifest, manifest_ref = read_metadata(manifest_path)
    manifest_ref["training_ready"] = manifest.get("training_ready")
    certificates = {}
    for name in ("audit_certificate", "downstream_scope_certificate"):
        path = input_dir / f"{name}.json"
        if path.exists():
            payload, ref = read_metadata(path)
            ref.update(status=payload.get("status"), pending=payload.get("pending"))
            certificates[name] = ref
    grouped = {}
    seen = set()
    for split in ("train", "valid"):
        groups = defaultdict(list)
        for shard in manifest["splits"][split]["shards"]:
            path = (input_dir / shard["path"]).resolve()
            count = shard["records"]
            if input_dir not in path.parents or path in seen:
                raise ValueError("Shard path escapes input or occurs more than once")
            if shard.get("split", split) != split or type(count) is not int or count < 0:
                raise ValueError("Invalid shard split/count")
            source = shard["source"]
            if not isinstance(source, str) or not source:
                raise ValueError("Each shard needs a source group")
            if not isinstance(shard.get("sha256"), str) or len(shard["sha256"]) != 64:
                raise ValueError("Each materialized shard needs its existing SHA256")
            groups[source].append((path, shard))
            seen.add(path)
        if sum(item[1]["records"] for group in groups.values() for item in group) != manifest["splits"][split]["records"]:
            raise ValueError("Manifest split count differs from shard counts")
        grouped[split] = groups
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()  # Refuse to overwrite any existing output.
    try:
        files, inspected, selected_stamps = {}, [], {}
        provenance_path = output_dir / "provenance.jsonl"
        with provenance_path.open("wb") as provenance:
            provenance_hash = hashlib.sha256()
            for split, limit in (("train", train_records), ("valid", valid_records)):
                counts = {source: sum(shard["records"] for _, shard in group)
                          for source, group in grouped[split].items()}
                allocation = quotas(counts, limit)
                output_path = output_dir / f"{split}.jsonl"
                output_hash, output_count, ranges = hashlib.sha256(), 0, []
                with output_path.open("wb") as output:
                    for source in sorted(allocation):
                        first = output_count + 1
                        rng = random.Random(int(digest(json.dumps([seed, split, source]).encode()), 16))
                        indices = sorted(rng.sample(range(counts[source]), allocation[source]))
                        cursor, offset = 0, 0
                        for path, shard in sorted(grouped[split][source], key=lambda item: str(item[0])):
                            selected = set()
                            while cursor < len(indices) and indices[cursor] < offset + shard["records"]:
                                selected.add(indices[cursor] - offset + 1)
                                cursor += 1
                            offset += shard["records"]
                            if not selected:
                                continue
                            before = stamp(path)
                            selected_stamps[path] = before
                            source_hash, records, copied = hashlib.sha256(), 0, 0
                            with path.open("rb") as handle:
                                for line_number, raw in enumerate(handle, 1):
                                    source_hash.update(raw)
                                    if not raw.strip():
                                        continue
                                    records += 1
                                    if records not in selected:
                                        continue
                                    row = json.loads(raw)
                                    if not isinstance(row, dict) or row.get("split") not in (None, split):
                                        raise ValueError(f"Selected row split mismatch: {path}:{line_number}")
                                    if not raw.endswith(b"\n"):
                                        raise ValueError("Selected rows must end in newline to preserve bytes when concatenated")
                                    output.write(raw)
                                    output_hash.update(raw)
                                    output_count += 1
                                    copied += 1
                                    entry = dict(split=split, output_line=output_count, source_group=source,
                                                 source_shard=str(path), source_line=line_number,
                                                 source_record=records, row_sha256=digest(raw))
                                    encoded = (json.dumps(entry, sort_keys=True) + "\n").encode()
                                    provenance.write(encoded)
                                    provenance_hash.update(encoded)
                            if (stamp(path) != before or records != shard["records"] or
                                    source_hash.hexdigest() != shard["sha256"] or copied != len(selected)):
                                raise ValueError(f"Source changed or manifest hash/count mismatch: {path}")
                            inspected.append(dict(path=str(path), sha256=source_hash.hexdigest(), records=records))
                        ranges.append(dict(source_group=source, start_line=first, end_line=output_count))
                files[split] = dict(path=str(output_path), sha256=output_hash.hexdigest(), records=output_count,
                                    source_counts=allocation, source_ranges=ranges)
        for path, before in selected_stamps.items():
            if stamp(path) != before:
                raise ValueError(f"Source changed after scanning: {path}")
        for path, raw in metadata.items():
            if stamp(path) != metadata_stamps[path] or path.read_bytes() != raw:
                raise ValueError(f"Source metadata changed: {path}")
        for path, raw in metadata.items():
            (output_dir / f"parent_{path.name}").write_bytes(raw)
        result = dict(schema="v7.exploratory_uncertified.v1", status="exploratory_uncertified",
                      formal_training_ready=False, seed=seed,
                      parent=dict(manifest=manifest_ref, certificates=certificates), files=files,
                      provenance=dict(path=str(provenance_path), sha256=provenance_hash.hexdigest()),
                      sampling=dict(method="source_min_one_then_proportional_largest_remainder_uniform_without_replacement",
                                    requested=dict(train=train_records, valid=valid_records),
                                    source_group="parent manifest shard.source", indices="nonblank record ordinal",
                                    inspected_shards=inspected),
                      limitations=["Exploratory subset; no new audit, annotations, deduplication, or functional selection.",
                                   "Original splits and selected row bytes preserved; existing overlap may remain.",
                                   "Rare sources receive at least one row; source proportions may be slightly distorted.",
                                   "Only shards containing selected rows are scanned and checked against parent hashes.",
                                   "Selected rows must be newline terminated; source files must remain unchanged."])
        write_json(output_dir / "sample_manifest.json", result)
        return result
    except BaseException:
        shutil.rmtree(output_dir)
        raise


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prepared = root / "prepared"
        prepared.mkdir()
        manifest = {"splits": {}}
        originals = {}
        for split in ("train", "valid"):
            shards = []
            for source in ("a", "b"):
                path = prepared / f"{split}_{source}.jsonl"
                raw = b"".join((json.dumps(dict(split=split, source=source, value=index)) + "\n").encode() for index in range(8))
                path.write_bytes(raw)
                originals[str(path)] = raw.splitlines(keepends=True)
                shards.append(dict(path=str(path), records=8, source=source, split=split, sha256=digest(raw)))
            manifest["splits"][split] = dict(records=16, shards=shards)
        write_json(prepared / "dataset_manifest.json", manifest)
        left = sample(prepared, root / "left", 6, 4, 42)
        right = sample(prepared, root / "right", 6, 4, 42)
        for split in ("train", "valid"):
            assert left["files"][split]["sha256"] == right["files"][split]["sha256"]
        for line in (root / "left" / "provenance.jsonl").read_text().splitlines():
            item = json.loads(line)
            raw = originals[item["source_shard"]][item["source_line"] - 1]
            copied = Path(left["files"][item["split"]]["path"]).read_bytes().splitlines(keepends=True)[item["output_line"] - 1]
            assert raw == copied and json.loads(copied)["split"] == item["split"]
        changed = prepared / "train_a.jsonl"
        changed.write_bytes(changed.read_bytes().replace(b'"value": 0', b'"value": 9'))
        try:
            sample(prepared, root / "changed", 6, 4)
        except ValueError:
            assert not (root / "changed").exists()
        else:
            raise AssertionError("Changed input was not rejected")
    return {"determinism": True, "original_bytes_and_split": True, "changed_source_rejected": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-records", type=int, default=100000)
    parser.add_argument("--valid-records", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return
    if args.input_dir is None or args.output_dir is None:
        parser.error("--input-dir and --output-dir are required unless --self-test is used")
    result = sample(args.input_dir, args.output_dir, args.train_records, args.valid_records, args.seed)
    print(json.dumps(dict(manifest=str(args.output_dir / "sample_manifest.json"),
                          records={key: value["records"] for key, value in result["files"].items()}), sort_keys=True))


if __name__ == "__main__":
    main()
