from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from torch.utils.data import IterableDataset

from .records import (
    BioSeqChain,
    BioSeqRecord,
    REGION_ORDER,
    chain_role_from_processed_type,
    compact_metadata,
    is_valid_protein_sequence,
    normalize_sequence,
    task_type_from_processed_source,
)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DEFAULT_DATA_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/data")
DEFAULT_OAS_DIR = DEFAULT_DATA_ROOT / "oas_previous_clean/splits/compat_for_current_loader_oasrule"
DEFAULT_OTS_DIR = DEFAULT_DATA_ROOT / "ots_paired_clean/final"
DEFAULT_NANOBODY_DIR = DEFAULT_DATA_ROOT / "nanobody_processed/step6_final"
DEFAULT_PROCESSED_V2_DIR = DEFAULT_DATA_ROOT / "processed_v2"
DEFAULT_PPI_DIR = DEFAULT_DATA_ROOT / "ppi/string_model_org_90_90_split"


def _is_heavy(row: dict[str, Any], prefix: str) -> bool:
    anarci_type = str(row.get(f"{prefix}_anarci_type", "")).strip().upper()
    chain_type = str(row.get(f"{prefix}_type", "")).strip().lower()
    return anarci_type == "H" or chain_type == "heavy"


def _is_beta(row: dict[str, Any], prefix: str) -> bool:
    anarci_type = str(row.get(f"{prefix}_anarci_type", "")).strip().upper()
    chain_type = str(row.get(f"{prefix}_type", "")).strip().lower()
    return anarci_type == "B" or chain_type in {"beta", "trb"}


def _regions(row: dict[str, Any], prefix: str) -> dict[str, str]:
    return {
        region: normalize_sequence(row.get(f"{prefix}_{region}"))
        for region in REGION_ORDER
        if normalize_sequence(row.get(f"{prefix}_{region}"))
    }


def _chain_metadata(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return compact_metadata(
        row,
        (
            f"{prefix}_anarci_type",
            f"{prefix}_type",
            f"{prefix}_v",
            f"{prefix}_j",
            f"{prefix}_cluster",
            f"{prefix}_cdr3",
        ),
    )


def _split_path(path: Path, split: str) -> Path:
    return path if path.is_file() else path / f"{split}.csv"


def oas_row_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    chain1 = normalize_sequence(row.get("cleaned_chain1_seq"))
    chain2 = normalize_sequence(row.get("cleaned_chain2_seq"))
    if not is_valid_protein_sequence(chain1) or not is_valid_protein_sequence(chain2):
        return None

    first = BioSeqChain(chain1, "antibody_heavy" if _is_heavy(row, "chain1") else "antibody_light", _regions(row, "chain1"), _chain_metadata(row, "chain1"))
    second = BioSeqChain(chain2, "antibody_heavy" if _is_heavy(row, "chain2") else "antibody_light", _regions(row, "chain2"), _chain_metadata(row, "chain2"))
    chains = [first, second]
    if second.role == "antibody_heavy" and first.role != "antibody_heavy":
        chains = [second, first]

    return BioSeqRecord(
        chains=chains,
        task_type="antibody",
        source="oas_paired",
        split=str(row.get("split") or split or "").strip() or None,
        metadata=compact_metadata(row, ("species", "data_type", "source_file", "pair_cluster", "cluster_id")),
        weight=weight,
    )


def ots_row_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    chain1 = normalize_sequence(row.get("cleaned_chain1_seq"))
    chain2 = normalize_sequence(row.get("cleaned_chain2_seq"))
    if not is_valid_protein_sequence(chain1) or not is_valid_protein_sequence(chain2):
        return None

    first = BioSeqChain(chain1, "tcr_beta" if _is_beta(row, "chain1") else "tcr_alpha", _regions(row, "chain1"), _chain_metadata(row, "chain1"))
    second = BioSeqChain(chain2, "tcr_beta" if _is_beta(row, "chain2") else "tcr_alpha", _regions(row, "chain2"), _chain_metadata(row, "chain2"))
    chains = [first, second]
    if second.role == "tcr_beta" and first.role != "tcr_beta":
        chains = [second, first]

    return BioSeqRecord(
        chains=chains,
        task_type="tcr",
        source="ots_paired",
        split=str(row.get("split") or split or "").strip() or None,
        metadata=compact_metadata(row, ("species", "data_type", "source_file", "pair_cluster", "cluster_id")),
        weight=weight,
    )


def nanobody_row_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    sequence = normalize_sequence(row.get("cleaned_seq") or row.get("vhh_seq"))
    if not is_valid_protein_sequence(sequence):
        return None
    regions = {region: normalize_sequence(row.get(region)) for region in REGION_ORDER if normalize_sequence(row.get(region))}
    chain = BioSeqChain(
        sequence=sequence,
        role="nanobody_vhh",
        regions=regions,
        metadata=compact_metadata(row, ("source", "anarci_chain_type", "cluster_id")),
    )
    return BioSeqRecord(
        chains=[chain],
        task_type="antibody",
        source="nanobody",
        split=str(row.get("split") or split or "").strip() or None,
        metadata=compact_metadata(row, ("source", "cluster_id")),
        weight=weight,
    )


def processed_json_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    sequences = [normalize_sequence(sequence) for sequence in row.get("chains", [])]
    if not sequences or not all(is_valid_protein_sequence(sequence) for sequence in sequences):
        return None
    roles = [chain_role_from_processed_type(value) for value in row.get("types", [])]
    if len(roles) < len(sequences):
        roles.extend(["other"] * (len(sequences) - len(roles)))
    source = str(row.get("source") or "processed")
    if source.lower() == "ppi" and len(roles) >= 2:
        roles[0] = "protein_a"
        roles[1] = "protein_b"
    chains = [BioSeqChain(sequence, roles[index]) for index, sequence in enumerate(sequences)]
    metadata: dict[str, Any] = {}
    if "targets" in row:
        metadata["targets"] = row["targets"]
    return BioSeqRecord(
        chains=chains,
        task_type=task_type_from_processed_source(source),
        source=source,
        split=split,
        metadata=metadata,
        weight=weight,
    )


@dataclass(frozen=True)
class CsvSourceConfig:
    name: str
    path: Path
    row_to_record: Callable[[dict[str, Any], str | None, float], BioSeqRecord | None]
    split: str = "train"
    weight: float = 1.0
    max_records: int | None = None


class CsvBioSeqSource(IterableDataset):
    def __init__(self, config: CsvSourceConfig) -> None:
        super().__init__()
        self.config = config
        self.path = _split_path(config.path, config.split)
        if not self.path.is_file():
            raise FileNotFoundError(f"CSV source not found: {self.path}")

    def iter_records(self, shard_index: int = 0, num_shards: int = 1) -> Iterator[BioSeqRecord]:
        kept = 0
        with self.path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for raw_index, row in enumerate(reader):
                if raw_index % num_shards != shard_index:
                    continue
                record = self.config.row_to_record(row, self.config.split, self.config.weight)
                if record is None:
                    continue
                yield record
                kept += 1
                if self.config.max_records is not None and kept >= self.config.max_records:
                    break

    def __iter__(self) -> Iterator[BioSeqRecord]:
        yield from self.iter_records()


@dataclass(frozen=True)
class JsonlSourceConfig:
    name: str
    path: Path
    split: str = "train"
    weight: float = 1.0
    max_records: int | None = None


class ProcessedJsonlSource(IterableDataset):
    def __init__(self, config: JsonlSourceConfig) -> None:
        super().__init__()
        self.config = config
        self.path = config.path if config.path.is_file() else config.path / f"{config.split}.jsonl"
        if not self.path.is_file():
            raise FileNotFoundError(f"JSONL source not found: {self.path}")

    def iter_records(self, shard_index: int = 0, num_shards: int = 1) -> Iterator[BioSeqRecord]:
        kept = 0
        with self.path.open() as handle:
            for raw_index, line in enumerate(handle):
                if raw_index % num_shards != shard_index or not line.strip():
                    continue
                record = processed_json_to_record(json.loads(line), self.config.split, self.config.weight)
                if record is None:
                    continue
                yield record
                kept += 1
                if self.config.max_records is not None and kept >= self.config.max_records:
                    break

    def __iter__(self) -> Iterator[BioSeqRecord]:
        yield from self.iter_records()


@dataclass(frozen=True)
class PpiArrowSourceConfig:
    name: str
    path: Path = DEFAULT_PPI_DIR
    split: str = "train"
    weight: float = 1.0
    max_records: int | None = None


class PpiArrowSource(IterableDataset):
    def __init__(self, config: PpiArrowSourceConfig) -> None:
        super().__init__()
        self.config = config
        if not config.path.exists():
            raise FileNotFoundError(f"PPI Arrow source not found: {config.path}")

    def iter_records(self, shard_index: int = 0, num_shards: int = 1) -> Iterator[BioSeqRecord]:
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError("PpiArrowSource requires the `datasets` package") from exc

        dataset = load_from_disk(str(self.config.path))[self.config.split]
        kept = 0
        for raw_index, row in enumerate(dataset):
            if raw_index % num_shards != shard_index:
                continue
            seq_a = normalize_sequence(row.get("SeqA"))
            seq_b = normalize_sequence(row.get("SeqB"))
            if not is_valid_protein_sequence(seq_a) or not is_valid_protein_sequence(seq_b):
                continue
            yield BioSeqRecord(
                chains=[BioSeqChain(seq_a, "protein_a"), BioSeqChain(seq_b, "protein_b")],
                task_type="ppi",
                source="ppi",
                split=self.config.split,
                metadata=compact_metadata(row, ("IDs", "OrgA", "OrgB")),
                labels={"score": row.get("score")} if row.get("score") is not None else {},
                weight=self.config.weight,
            )
            kept += 1
            if self.config.max_records is not None and kept >= self.config.max_records:
                break

    def __iter__(self) -> Iterator[BioSeqRecord]:
        yield from self.iter_records()


def default_source_configs(split: str = "train", max_records: int | None = None) -> list[CsvSourceConfig | JsonlSourceConfig]:
    return [
        CsvSourceConfig("oas", DEFAULT_OAS_DIR, oas_row_to_record, split=split, max_records=max_records),
        CsvSourceConfig("ots", DEFAULT_OTS_DIR, ots_row_to_record, split=split, max_records=max_records),
        CsvSourceConfig("nanobody", DEFAULT_NANOBODY_DIR, nanobody_row_to_record, split=split, max_records=max_records),
        JsonlSourceConfig("processed_v2", DEFAULT_PROCESSED_V2_DIR, split="val" if split in {"valid", "validation"} else split, max_records=max_records),
    ]


def source_from_config(config: CsvSourceConfig | JsonlSourceConfig | PpiArrowSourceConfig) -> IterableDataset:
    if isinstance(config, CsvSourceConfig):
        return CsvBioSeqSource(config)
    if isinstance(config, JsonlSourceConfig):
        return ProcessedJsonlSource(config)
    if isinstance(config, PpiArrowSourceConfig):
        return PpiArrowSource(config)
    raise TypeError(f"Unsupported source config: {type(config)!r}")
