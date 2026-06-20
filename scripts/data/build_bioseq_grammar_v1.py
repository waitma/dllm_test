"""Build semantic Arrow shards for grammar-v1 BioSeq training.

Example::

    python scripts/data/build_bioseq_grammar_v1.py \
        --output-dir data/bioseq_grammar_v1 --splits train,valid

The builder keeps OAS and OTS pairs, excludes duplicated processed-v2 PPI,
keeps processed-v2 TCR/epitope rows, and canonicalizes STRING pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.data import (  # noqa: E402
    DEFAULT_GRAMMAR_DATA_DIR,
    BioSeqRecord,
    CsvBioSeqSource,
    JsonlSourceConfig,
    ProcessedJsonlSource,
    default_source_configs,
)
from dllm.pipelines.qwen3_vl_arch.data.records import (  # noqa: E402
    BioSeqChain,
    is_valid_protein_sequence,
    normalize_sequence,
)
from dllm.pipelines.qwen3_vl_arch.data.sources import DEFAULT_PPI_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRAMMAR_DATA_DIR)
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--sources", default="oas,ots,tcr,ppi")
    parser.add_argument("--ppi-max-protein-length", type=int, default=1024)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def semantic_row(record: BioSeqRecord, split: str) -> dict[str, Any]:
    return {
        "chains": record.sequences,
        "roles": record.chain_roles,
        "task_type": record.task_type,
        "source": record.source,
        "split": split,
        "relation": str(record.labels.get("relation", "binding")),
        "weight": float(record.weight),
    }


def iter_oas_or_ots(name: str, split: str, limit: int | None) -> Iterator[dict[str, Any]]:
    config = next(
        config
        for config in default_source_configs(split=split, max_records=limit)
        if config.name == name
    )
    source = CsvBioSeqSource(config)
    for record in source.iter_records():
        yield semantic_row(record, split)


def iter_tcr(split: str, limit: int | None) -> Iterator[dict[str, Any]]:
    raw_split = "val" if split in {"valid", "validation"} else split
    source = ProcessedJsonlSource(
        JsonlSourceConfig("processed_v2", PROJECT_ROOT / "data/processed_v2", split=raw_split)
    )
    kept = 0
    for record in source.iter_records():
        if record.source.lower() == "ppi":
            continue
        if not (record.task_type.startswith("tcr") or {"tcr_alpha", "tcr_beta"} & set(record.chain_roles)):
            continue
        yield semantic_row(record, split)
        kept += 1
        if limit is not None and kept >= limit:
            break


def stable_crop(sequence: str, max_length: int, key: str) -> str:
    if len(sequence) <= max_length:
        return sequence
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    start = int.from_bytes(digest, "little") % (len(sequence) - max_length + 1)
    return sequence[start : start + max_length]


def _ppi_pair_key(row: dict[str, Any], seq_a: str, seq_b: str) -> tuple[str, str]:
    raw_ids = row.get("IDs")
    if isinstance(raw_ids, (list, tuple)) and len(raw_ids) >= 2:
        first, second = str(raw_ids[0]), str(raw_ids[1])
    elif isinstance(raw_ids, str) and "|" in raw_ids:
        first, second = raw_ids.split("|", 1)
    else:
        first, second = seq_a, seq_b
    return tuple(sorted((first, second)))


def iter_ppi(split: str, limit: int | None, max_protein_length: int) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise ImportError("Building grammar Arrow data requires the `datasets` package") from exc

    dataset = load_from_disk(str(DEFAULT_PPI_DIR))[split]
    seen: set[tuple[str, str]] = set()
    kept = 0
    for row in dataset:
        org_a = str(row.get("OrgA") or "")
        org_b = str(row.get("OrgB") or "")
        if org_a and org_b and org_a != org_b:
            continue
        seq_a = normalize_sequence(row.get("SeqA"))
        seq_b = normalize_sequence(row.get("SeqB"))
        if not is_valid_protein_sequence(seq_a) or not is_valid_protein_sequence(seq_b):
            continue
        pair_key = _ppi_pair_key(row, seq_a, seq_b)
        if pair_key in seen:
            continue
        seen.add(pair_key)
        seq_a = stable_crop(seq_a, max_protein_length, pair_key[0])
        seq_b = stable_crop(seq_b, max_protein_length, pair_key[1])
        record = BioSeqRecord(
            chains=[
                BioSeqChain(seq_a, "protein_a"),
                BioSeqChain(seq_b, "protein_b"),
            ],
            task_type="ppi",
            source="string_ppi",
            split=split,
            labels={"relation": "binding", "score": row.get("score")},
        )
        yield semantic_row(record, split)
        kept += 1
        if limit is not None and kept >= limit:
            break


def build_source(
    name: str,
    split: str,
    output_dir: Path,
    limit: int | None,
    ppi_max_protein_length: int,
    force: bool,
) -> dict[str, Any]:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Building grammar Arrow data requires the `datasets` package") from exc

    target = output_dir / name / split
    if target.exists():
        if not force:
            dataset = Dataset.load_from_disk(str(target))
            return {"source": name, "split": split, "rows": len(dataset), "path": str(target)}
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if name in {"oas", "ots"}:
        generator = lambda: iter_oas_or_ots(name, split, limit)
    elif name == "tcr":
        generator = lambda: iter_tcr(split, limit)
    elif name == "ppi":
        generator = lambda: iter_ppi(split, limit, ppi_max_protein_length)
    else:
        raise ValueError(f"Unsupported grammar source: {name}")

    dataset = Dataset.from_generator(generator, cache_dir=str(output_dir / ".cache"))
    dataset.save_to_disk(str(target), max_shard_size="512MB")
    return {"source": name, "split": split, "rows": len(dataset), "path": str(target)}


def main() -> None:
    args = parse_args()
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "format": "bioseq_grammar_v1_semantic_arrow",
        "ppi_max_protein_length": args.ppi_max_protein_length,
        "datasets": [],
    }
    for split in splits:
        for source in sources:
            result = build_source(
                source,
                split,
                args.output_dir,
                args.max_records_per_source,
                args.ppi_max_protein_length,
                args.force,
            )
            manifest["datasets"].append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
