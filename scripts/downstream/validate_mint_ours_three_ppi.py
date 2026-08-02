"""Fail-fast validation and manifest writer for the formal three-PPI run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from downstream.mint_tasks.extract_embeddings import RESULTS_ROOT, model_cache_name


EXPECTED = {
    "HumanPPI": {
        "rows": {"train": 26319, "val": 234, "test": 180},
        "primary_metric": "best_test_Accuracy_mean",
        "paper_comparable": True,
    },
    "YeastPPI": {
        "rows": {"train": 4945, "val": 95, "test": 394},
        "primary_metric": "best_test_Accuracy_mean",
        "paper_comparable": True,
    },
    "Bernett": {
        "rows": {"train": 163192, "val": 59260, "test": 52048},
        "primary_metric": "best_test_AUPRC_mean",
        "paper_comparable": False,
    },
}


def _shape(path: Path) -> list[int]:
    import torch

    tensor = torch.load(path, map_location="cpu")
    return list(tensor.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol-tag", required=True)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--embedding-batch-size", type=int, required=True)
    args = parser.parse_args()

    cache_name = model_cache_name(
        args.model,
        sep_chains=False,
        max_train=None,
        protocol_tag=args.protocol_tag,
    )
    result_dir = RESULTS_ROOT / args.task / cache_name
    shapes = {}
    for split, rows in EXPECTED[args.task]["rows"].items():
        path = result_dir / f"{split}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        shapes[split] = _shape(path)
        if shapes[split][0] != rows:
            raise RuntimeError(f"{args.task}/{split}: {shapes[split][0]} != {rows}")

    metrics_path = result_dir / f"{args.task}_{cache_name}_metrics.json"
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    if len(records) != 4:
        raise RuntimeError(f"expected 3 repetitions plus summary, got {len(records)} records")
    summary = records[-1]
    primary_key = EXPECTED[args.task]["primary_metric"]
    primary_value = float(summary[primary_key])
    if not math.isfinite(primary_value):
        raise RuntimeError(f"non-finite primary metric: {primary_value}")

    required_metadata = {
        "checkpoint_step": 121000,
        "checkpoint_sha256": "51f0eee5fa7b127a0487a2ea8fdbc322bda6f6908ff2662087bd87d4980d4613",
        "max_sequence_tokens": args.max_seq_length,
        "embedding_pair_mode": "joint_record_global_mean",
        "head_hidden_size": 640,
        "head_epochs": 100,
        "experimental_repetitions": 3,
    }
    for index, record in enumerate(records):
        for key, expected in required_metadata.items():
            if record.get(key) != expected:
                raise RuntimeError(
                    f"record {index} metadata {key}: {record.get(key)!r} != {expected!r}"
                )
        provenance = record.get("baseline_provenance", {})
        if provenance.get("paper_comparable") is not EXPECTED[args.task]["paper_comparable"]:
            raise RuntimeError(f"record {index} has incorrect paper_comparable provenance")

    manifest = {
        "status": "success",
        "task": args.task,
        "model": args.model,
        "cache_protocol_tag": args.protocol_tag,
        "max_sequence_tokens_per_chain": args.max_seq_length,
        "embedding_batch_size": args.embedding_batch_size,
        "embedding_pair_mode": "joint_record_global_mean",
        "embedding_shapes": shapes,
        "metrics_path": str(metrics_path),
        "primary_metric": primary_key,
        "primary_value": primary_value,
        "paper_comparable": EXPECTED[args.task]["paper_comparable"],
        "records": len(records),
    }
    out_path = result_dir / "formal_result_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
