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
from dllm.pipelines.qwen3_vl_arch.data.ppi_relations import infer_grammar_relation  # noqa: E402
from dllm.pipelines.qwen3_vl_arch.data.ppi_splits import (  # noqa: E402
    BERNETT_STRING_90_90,
    normalize_split_name,
    splits_allowed_for_grammar_mix,
    validate_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GRAMMAR_DATA_DIR)
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument(
        "--sources",
        default="oas,ots,tcr,ppi",
        help=(
            "Comma-separated sources: oas, ots, tcr, ppi, mint_ppi, neutralization. "
            "mint_ppi requires prebuilt shards (build_mint_grammar_shards.py). "
            "neutralization uses unified CSV or prebuilt shards."
        ),
    )
    parser.add_argument("--ppi-max-protein-length", type=int, default=1024)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--ppi-split-policy",
        default=BERNETT_STRING_90_90.policy_id,
        help=(
            "Canonical PPI split policy (default: bernett_string_90_90_hf). "
            "For MINT-scale pretraining use mint_string_pretrain_v1 via build_mint_string_splits.py."
        ),
    )
    return parser.parse_args()


def assert_ppi_splits_allowed(requested_splits: list[str], policy_id: str) -> None:
    allowed = set(splits_allowed_for_grammar_mix("string_model_org_90_90_split"))
    for split in requested_splits:
        normalized = normalize_split_name(split)
        if normalized not in allowed:
            raise ValueError(
                f"PPI split {split!r} is not allowed for grammar mixing under {policy_id}. "
                f"Allowed: {', '.join(sorted(allowed))}. "
                "Do not merge published test/eval splits into training."
            )
        validate_split("string_model_org_90_90_split", normalized, policy_id=policy_id)


def semantic_row(record: BioSeqRecord, split: str, *, default_relation: str = "unknown") -> dict[str, Any]:
    return {
        "chains": record.sequences,
        "roles": record.chain_roles,
        "task_type": record.task_type,
        "source": record.source,
        "split": split,
        "relation": str(record.labels.get("relation", default_relation)),
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
        record.labels.setdefault("relation", "binding")
        yield semantic_row(record, split, default_relation="binding")


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
        record.labels.setdefault("relation", "binding")
        yield semantic_row(record, split, default_relation="binding")
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
            labels={
                "relation": infer_grammar_relation(
                    source_id="string_model_org_90_90_split",
                    task_family="ppi_binary",
                    string_channel="physical",
                ),
                "score": row.get("score"),
            },
        )
        yield semantic_row(record, split)
        kept += 1
        if limit is not None and kept >= limit:
            break


def verify_prebuilt_shard(name: str, split: str, output_dir: Path, force: bool) -> dict[str, Any]:
    from datasets import Dataset

    target = output_dir / name / split
    if not target.exists():
        raise FileNotFoundError(
            f"Missing prebuilt shard {target}. "
            f"Run scripts/data/build_{name}_grammar_shards.py first."
        )
    if force:
        raise ValueError(
            f"Source {name} is prebuilt externally; rebuild with the dedicated script, not --force here."
        )
    dataset = Dataset.load_from_disk(str(target))
    return {"source": name, "split": split, "rows": len(dataset), "path": str(target), "prebuilt": True}


def build_neutralization_shard(
    split: str,
    output_dir: Path,
    limit: int | None,
    force: bool,
) -> dict[str, Any]:
    unified_csv = PROJECT_ROOT / "data/ppi_task_raw/processed/interaction_records_unified.csv"
    prebuilt = output_dir / "neutralization" / split
    if prebuilt.exists() and not force:
        from datasets import Dataset

        dataset = Dataset.load_from_disk(str(prebuilt))
        return {"source": "neutralization", "split": split, "rows": len(dataset), "path": str(prebuilt)}
    if not unified_csv.exists():
        raise FileNotFoundError(
            f"Missing {unified_csv}. Run scripts/data/build_ppi_unified_csv.py first."
        )

    import csv

    from dllm.pipelines.qwen3_vl_arch.data.grammar_builders import (
        antibody_neutralization_record,
        semantic_row,
    )

    def generator() -> Iterator[dict[str, Any]]:
        kept = 0
        with unified_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("source_id") != "covabdab_neutralization":
                    continue
                heavy = str(row.get("antibody_heavy") or "").strip()
                light = str(row.get("antibody_light") or "").strip()
                ab_type = str(row.get("entity_a_id") or "").lower()
                is_nanobody = "nb" in ab_type
                record = antibody_neutralization_record(
                    heavy,
                    light or None,
                    split=split,
                    is_nanobody=is_nanobody,
                )
                if record is None:
                    continue
                yield semantic_row(record, split)
                kept += 1
                if limit is not None and kept >= limit:
                    break

    from datasets import Dataset

    if prebuilt.exists() and force:
        shutil.rmtree(prebuilt)
    prebuilt.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(generator, cache_dir=str(output_dir / ".cache"))
    dataset.save_to_disk(str(prebuilt), max_shard_size="128MB")
    return {"source": "neutralization", "split": split, "rows": len(dataset), "path": str(prebuilt)}


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

    if name == "mint_ppi":
        return verify_prebuilt_shard("mint_ppi", split, output_dir, force)
    if name == "neutralization":
        return build_neutralization_shard(split, output_dir, limit, force)

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
    if "ppi" in sources:
        assert_ppi_splits_allowed(splits, args.ppi_split_policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "format": "bioseq_grammar_v1_semantic_arrow",
        "ppi_max_protein_length": args.ppi_max_protein_length,
        "ppi_split_policy": args.ppi_split_policy if "ppi" in sources else None,
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
