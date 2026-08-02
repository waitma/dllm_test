"""Profile worst-case BioSeq MINT embedding batches on a real evaluation GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from downstream.mint_tasks.embedders import build_embedder


def _max_length_sequence(length: int) -> str:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return (alphabet * ((length + len(alphabet) - 1) // len(alphabet)))[:length]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal embedding profile")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    embedder = build_embedder(
        f"grammar:{args.checkpoint}",
        device=args.device,
        sep_chains=False,
        max_protein_length=args.max_length,
    )
    sequence = _max_length_sequence(args.max_length)
    results = []
    for batch_size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        record = {"batch_size": batch_size, "status": "success"}
        try:
            embedding = embedder.embed(
                [[sequence, sequence] for _ in range(batch_size)],
                sep_chains=False,
            )
            torch.cuda.synchronize()
            record["embedding_shape"] = list(embedding.shape)
            del embedding
        except torch.cuda.OutOfMemoryError as exc:
            record["status"] = "oom"
            record["error"] = str(exc)
            torch.cuda.empty_cache()
        record["elapsed_seconds"] = time.perf_counter() - started
        record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        record["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        results.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if record["status"] == "oom":
            break

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "max_length_per_chain": args.max_length,
        "pair_mode": "joint_ppi_record_global_mean",
        "synthetic_chain_lengths": [args.max_length, args.max_length],
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "device_total_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"profile written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
