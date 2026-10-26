"""Verify CDR annotation coverage over all indexed antibody training/valid rows.

Activate pllm; USE_TF=0 python -B -m scripts.research.audit_diffusion_cdr_annotations
--index ABS --output ABS [--workers 4]. Only reads the bound ASD/OAS shards.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.diffusion_cdr_tasks import cdr_spans


def scan(shard):
    offsets = np.load(shard["index"], allow_pickle=False)
    if sha(shard["index"]) != shard["index_sha256"]:
        raise ValueError("Changed source index")
    cursor, position, counts, digest = 0, 0, Counter(), hashlib.sha256()
    with open(shard["path"], "rb", buffering=4 << 20) as stream:
        for line in stream:
            digest.update(line)
            selected = cursor < len(offsets) and position == int(offsets[cursor])
            position += len(line)
            if not selected:
                continue
            cursor += 1
            row = json.loads(line)
            counts["records"] += 1
            spans = cdr_spans(row)
            counts["eligible_records"] += bool(spans)
            counts["eligible_cdrs"] += len(spans)
            counts["eligible_residues"] += sum(end-start for _, _, start, end in spans)
            counts["six_loop_records"] += len(spans) == 6
            counts["fallback_records"] += not spans
    if cursor != shard["records"] or digest.hexdigest() != shard["sha256"]:
        raise ValueError("Changed shard bytes or selected record count")
    return dict(source=shard["source"], split=shard["split"], path=shard["path"],
                sha256=digest.hexdigest(), counts=dict(counts))


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
    shards = [s for s in index["shards"] if s["source"] in ("oas", "asd_antibody")]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task in as_completed([pool.submit(scan, shard) for shard in shards]):
            result = task.result()
            results.append(result)
            print(json.dumps(dict(completed_shards=len(results), **result["counts"])), flush=True)
    totals = {}
    for source in ("oas", "asd_antibody"):
        totals[source] = {}
        for split in ("train", "valid"):
            total = Counter()
            for row in results:
                if row["source"] == source and row["split"] == split:
                    total.update(row["counts"])
            totals[source][split] = dict(total)
    write_json(args.output / "summary.json", dict(status="passed", index=str(args.index),
        index_sha256=sha(args.index), annotation_source_sha256=sha(Path(__file__).with_name("diffusion_cdr_tasks.py")),
        script_sha256=sha(Path(__file__)), coverage=totals, shards=results,
        scope="All bound ASD/OAS rows; exact complete fully observed antibody regions only"))


if __name__ == "__main__":
    main()
