"""Audit every indexed diffusion row without rewriting the prepared dataset.

Activate pllm; USE_TF=0 python -B -m scripts.research.audit_diffusion_data
--index ABS --output ABS [--workers 4]. Outputs aggregate counts and identity hashes.
Exact observed receptor overlap is checked; this is not a homology audit.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
try:
    from orjson import loads
except ImportError:
    loads = json.loads

from scripts.research.bert_mlm_data import ROLES, receptor_key, sha, write_json


def scan(job):
    source, split, shards, held, output = job
    counts, observed_hist, signatures, labels, origins = (Counter() for _ in range(5))
    receptor_counts, record_counts = Counter(), Counter()
    errors, files = [], []
    for shard in shards:
        path = Path(shard["path"])
        offsets = np.load(shard["index"], allow_pickle=False)
        if sha(shard["index"]) != shard["index_sha256"]:
            raise ValueError("Changed offset index: " + shard["index"])
        if len(offsets) != shard["records"] or (len(offsets) > 1 and not np.all(offsets[1:] > offsets[:-1])):
            raise ValueError("Invalid indexed offsets")
        digest, cursor, position = hashlib.sha256(), 0, 0
        with path.open("rb", buffering=4 << 20) as stream:
            for line in stream:
                digest.update(line)
                selected = cursor < len(offsets) and position == int(offsets[cursor])
                position += len(line)
                if not selected:
                    continue
                cursor += 1
                row = loads(line)
                counts["records"] += 1
                chains, roles = row["chains"], row["chain_roles"]
                faults = []
                receptor_roles = [r for r in roles if r in ROLES]
                if len(chains) != len(roles) or len(set(receptor_roles)) != len(receptor_roles):
                    faults.append("chain_role_mismatch_or_duplicate")
                # Class II MHC legitimately contains two chains with role 'mhc'.
                counts["records_with_repeated_context_roles"] += int(
                    len(set(roles)) != len(roles))
                if row.get("split") != split:
                    faults.append("split_mismatch")
                observed, present, synthetic_total = 0, [], 0
                for i, (role, seq) in enumerate(zip(roles, chains)):
                    if not seq or not seq.isalpha() or not seq.isupper():
                        faults.append("empty_or_non_residue_sequence")
                    if role not in ROLES:
                        continue
                    mask = row.get("chain_metadata", {}).get(str(i), {}).get("synthetic_residue_mask", [])
                    if mask and (len(mask) != len(seq) or any(x not in (0, 1) for x in mask)):
                        faults.append("synthetic_mask_invalid")
                        continue
                    n = len(seq) - sum(mask)
                    observed += n
                    synthetic_total += sum(mask)
                    if n:
                        present.append(role)
                    else:
                        counts["entirely_synthetic_chains"] += 1
                    cap = 167 if role in ("antibody_heavy", "tcr_beta") else 135
                    if len(seq) + 1 > cap:
                        faults.append("slot_overflow")
                if observed == 0:
                    faults.append("no_observed_receptor")
                for fault in faults:
                    counts["error_" + fault] += 1
                if faults and len(errors) < 10:
                    errors.append(dict(path=str(path), offset=int(offsets[cursor-1]), reasons=faults))
                if "synthetic_mask_invalid" in faults:
                    continue
                counts["observed_receptor_residues"] += observed
                counts["synthetic_receptor_residues"] += synthetic_total
                counts["records_with_synthetic"] += int(synthetic_total > 0)
                counts["records_one_observed_chain"] += int(len(present) == 1)
                observed_hist[observed] += 1
                signatures["+".join(present)] += 1
                labels[str(row.get("labels", {}).get("relation", "absent"))] += 1
                origins[str(row.get("metadata", {}).get("source", row.get("source")))] += 1
                key = receptor_key(row)
                receptor_counts[key] += 1
                counts["rows_with_exact_valid_receptor"] += int(key in held)
                # Keep the observed supervision identity and conditioning sequence;
                # ignore identifiers, synthetic scaffold and ancillary metadata.
                context = sorted((role, seq) for role, seq in zip(roles, chains) if role not in ROLES)
                payload = json.dumps([key, context, row.get("labels", {})], sort_keys=True).encode()
                record_counts[hashlib.sha256(payload).hexdigest()] += 1
        if cursor != len(offsets) or digest.hexdigest() != shard["sha256"]:
            raise ValueError("Changed shard or missing indexed row: " + str(path))
        files.append(dict(path=str(path), sha256=digest.hexdigest(), records=cursor))
    target = Path(output) / f"{split}_{source}_receptor_keys.txt"
    target.write_text("".join(k + "\n" for k in sorted(receptor_counts)))
    return dict(source=source, split=split, counts=dict(counts), files=files,
        observed_length_histogram=dict(sorted(observed_hist.items())),
        observed_chain_signatures=dict(signatures), relation_labels=dict(labels), origins=dict(origins),
        unique_observed_receptors=len(receptor_counts),
        duplicate_observed_receptor_rows=sum(n-1 for n in receptor_counts.values()),
        max_observed_receptor_multiplicity=max(receptor_counts.values(), default=0),
        unique_supervision_context_records=len(record_counts),
        duplicate_supervision_context_rows=sum(n-1 for n in record_counts.values()),
        unique_receptors_overlapping_valid=len(receptor_counts.keys() & held),
        receptor_keys=str(target), error_examples=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.index.is_absolute() or not args.output.is_absolute():
        raise ValueError("Use absolute paths")
    args.output.mkdir(parents=True, exist_ok=False)
    index = json.loads(args.index.read_text())
    grouped = defaultdict(list)
    for shard in index["shards"]:
        grouped[(shard["split"], shard["source"])].append(shard)
    results, held = [], set()
    for split in ("valid", "train"):
        jobs = [(source, split, shards, held, str(args.output))
                for (part, source), shards in grouped.items() if part == split]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(scan, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                write_json(args.output / f"{split}_{result['source']}.json", result)
                print(json.dumps({k: result[k] for k in ("source", "split", "counts", "duplicate_supervision_context_rows")}), flush=True)
        if split == "valid":
            for item in results:
                held.update(Path(item["receptor_keys"]).read_text().splitlines())
    for split, expected in index["records"].items():
        if sum(x["counts"]["records"] for x in results if x["split"] == split) != expected:
            raise ValueError("Audit count mismatch")
    summary = dict(status="complete", index=str(args.index), index_sha256=sha(args.index),
        script_sha256=sha(Path(__file__)), scope="All indexed train and valid rows; exact observed receptor identity; no homology or experimental label verification",
        results=sorted(results, key=lambda x: (x["split"], x["source"])))
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
