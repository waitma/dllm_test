"""Merge independently run MINT MLP seeds into the canonical metrics file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from downstream.mint_tasks.extract_embeddings import RESULTS_ROOT, model_cache_name
from downstream.mint_tasks.metrics import calculate_mean_std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol-tag", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    cache_name = model_cache_name(args.model, False, None, args.protocol_tag)
    result_dir = RESULTS_ROOT / args.task / cache_name
    rep_records = []
    metric_values = []
    metadata_reference = None
    for rep in range(args.repetitions):
        path = result_dir / f"{args.task}_{cache_name}_rep{rep}_metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = [row for row in payload if row.get("rep") == float(rep)]
        if len(records) != 1:
            raise RuntimeError(f"expected one rep={rep} record in {path}")
        record = records[0]
        metrics = {
            key: value for key, value in record.items() if key.startswith("best_test_")
        }
        if not metrics:
            raise RuntimeError(f"no best_test metrics in {path}")
        metadata = {
            key: value
            for key, value in record.items()
            if not key.startswith("best_test_") and key != "rep"
        }
        if metadata_reference is None:
            metadata_reference = metadata
        elif metadata != metadata_reference:
            raise RuntimeError(f"repetition metadata mismatch in {path}")
        rep_records.append(record)
        metric_values.append(metrics)

    summary = {**metadata_reference, **calculate_mean_std(metric_values)}
    canonical = result_dir / f"{args.task}_{cache_name}_metrics.json"
    canonical.write_text(
        json.dumps([*rep_records, summary], indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "success",
        "task": args.task,
        "repetitions": args.repetitions,
        "seeds": list(range(args.repetitions)),
        "canonical_metrics": str(canonical),
        "records": args.repetitions + 1,
    }
    (result_dir / "parallel_repetitions_merge.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
