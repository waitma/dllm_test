"""Extract or merge contiguous embedding shards without changing row order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from downstream.mint_tasks.embedders import build_embedder
from downstream.mint_tasks.extract_embeddings import (
    RESULTS_ROOT,
    embed_dataset,
    model_cache_name,
)
from downstream.mint_tasks.tasks import get_task_datasets


def shard_bounds(total: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError((total, shard_index, num_shards))
    return total * shard_index // num_shards, total * (shard_index + 1) // num_shards


def _result_dir(args) -> Path:
    return RESULTS_ROOT / args.task / model_cache_name(
        args.model,
        sep_chains=False,
        max_train=None,
        protocol_tag=args.protocol_tag,
    )


def extract(args) -> None:
    train, val, test = get_task_datasets(args.task)
    dataset = {"train": train, "val": val, "test": test}[args.split]
    if dataset is None:
        raise RuntimeError(f"{args.task} has no {args.split} split")
    start, end = shard_bounds(len(dataset), args.shard_index, args.num_shards)
    subset = Subset(dataset, range(start, end))
    result_dir = _result_dir(args)
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}.part-{args.shard_index:02d}-of-{args.num_shards:02d}"
    out_path = result_dir / f"{stem}.pt"
    manifest_path = result_dir / f"{stem}.json"
    if out_path.is_file() and manifest_path.is_file() and not args.overwrite:
        print(f"cached shard: {out_path}")
        return

    embedder = build_embedder(
        args.model,
        device=args.device,
        sep_chains=False,
        max_protein_length=args.max_length,
    )
    embeddings = embed_dataset(
        embedder,
        subset,
        args.task,
        sep_chains=False,
        batch_size=args.bs,
        cat=False,
        aligned_max_tokens=args.max_length,
    )
    expected = end - start
    if tuple(embeddings.shape) != (expected, 960):
        raise RuntimeError(f"unexpected shard shape {tuple(embeddings.shape)}")
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(embeddings, tmp)
    tmp.replace(out_path)
    manifest = {
        "status": "success",
        "task": args.task,
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_rows": len(dataset),
        "start_row_inclusive": start,
        "end_row_exclusive": end,
        "rows": expected,
        "shape": list(embeddings.shape),
        "output": str(out_path),
        "model": args.model,
        "protocol_tag": args.protocol_tag,
        "max_sequence_tokens_per_chain": args.max_length,
        "embedding_batch_size": args.bs,
        "pair_mode": "joint_record_global_mean",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def merge(args) -> None:
    result_dir = _result_dir(args)
    pieces = []
    expected_start = 0
    total_rows = None
    for index in range(args.num_shards):
        stem = f"{args.split}.part-{index:02d}-of-{args.num_shards:02d}"
        manifest_path = result_dir / f"{stem}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "success" or manifest["shard_index"] != index:
            raise RuntimeError(f"invalid shard manifest: {manifest_path}")
        if manifest["start_row_inclusive"] != expected_start:
            raise RuntimeError(f"non-contiguous shard: {manifest_path}")
        total_rows = manifest["total_rows"] if total_rows is None else total_rows
        if manifest["total_rows"] != total_rows:
            raise RuntimeError("shards disagree on total_rows")
        tensor = torch.load(manifest["output"], map_location="cpu")
        if list(tensor.shape) != manifest["shape"]:
            raise RuntimeError(f"shape mismatch: {manifest_path}")
        pieces.append(tensor)
        expected_start = manifest["end_row_exclusive"]
    if expected_start != total_rows:
        raise RuntimeError(f"shards cover {expected_start}, expected {total_rows}")
    merged = torch.cat(pieces, dim=0)
    if tuple(merged.shape) != (total_rows, 960):
        raise RuntimeError(f"unexpected merged shape {tuple(merged.shape)}")
    out_path = result_dir / f"{args.split}.pt"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(out_path)
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(merged, tmp)
    tmp.replace(out_path)
    merge_manifest = {
        "status": "success",
        "split": args.split,
        "num_shards": args.num_shards,
        "shape": list(merged.shape),
        "output": str(out_path),
        "row_order": "original_contiguous_order",
    }
    (result_dir / f"{args.split}.merge.json").write_text(
        json.dumps(merge_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(merge_manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("extract", "merge"))
    parser.add_argument("--task", default="Bernett")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol-tag", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--bs", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    (extract if args.mode == "extract" else merge)(args)


if __name__ == "__main__":
    main()
