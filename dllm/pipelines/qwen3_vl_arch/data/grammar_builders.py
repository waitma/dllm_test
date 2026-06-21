"""Shared helpers for grammar-v1 Arrow shard builders."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterator

_records_path = Path(__file__).resolve().parent / "records.py"
_records_name = "_bioseq_data_records"
if _records_name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_records_name, _records_path)
    assert _spec and _spec.loader
    _records_mod = importlib.util.module_from_spec(_spec)
    sys.modules[_records_name] = _records_mod
    _spec.loader.exec_module(_records_mod)
else:
    _records_mod = sys.modules[_records_name]

BioSeqChain = _records_mod.BioSeqChain
BioSeqRecord = _records_mod.BioSeqRecord
is_valid_protein_sequence = _records_mod.is_valid_protein_sequence
normalize_sequence = _records_mod.normalize_sequence


def stable_crop(sequence: str, max_length: int, key: str) -> str:
    if len(sequence) <= max_length:
        return sequence
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    start = int.from_bytes(digest, "little") % (len(sequence) - max_length + 1)
    return sequence[start : start + max_length]


def semantic_row(record: BioSeqRecord, split: str) -> dict[str, Any]:
    return {
        "chains": record.sequences,
        "roles": record.chain_roles,
        "task_type": record.task_type,
        "source": record.source,
        "split": split,
        "relation": str(record.labels.get("relation", "unknown")),
        "weight": float(record.weight),
    }


def ppi_record(
    seq_a: str,
    seq_b: str,
    *,
    split: str,
    relation: str = "binding",
    source: str = "string_ppi",
    pair_key: tuple[str, str] | None = None,
    max_protein_length: int = 1024,
) -> BioSeqRecord | None:
    seq_a = normalize_sequence(seq_a)
    seq_b = normalize_sequence(seq_b)
    if not is_valid_protein_sequence(seq_a) or not is_valid_protein_sequence(seq_b):
        return None
    key = pair_key or tuple(sorted((seq_a[:32], seq_b[:32])))
    seq_a = stable_crop(seq_a, max_protein_length, key[0])
    seq_b = stable_crop(seq_b, max_protein_length, key[1])
    return BioSeqRecord(
        chains=[
            BioSeqChain(seq_a, "protein_a"),
            BioSeqChain(seq_b, "protein_b"),
        ],
        task_type="ppi",
        source=source,
        split=split,
        labels={"relation": relation},
    )


def antibody_neutralization_record(
    heavy: str,
    light: str | None,
    *,
    split: str = "train",
    source: str = "covabdab_neutralization",
    is_nanobody: bool = False,
) -> BioSeqRecord | None:
    heavy = normalize_sequence(heavy)
    light = normalize_sequence(light) if light else ""
    if not is_valid_protein_sequence(heavy):
        return None
    if is_nanobody or not light or light.upper() == "ND":
        return BioSeqRecord(
            chains=[BioSeqChain(heavy, "nanobody_vhh")],
            task_type="antibody_neutralization",
            source=source,
            split=split,
            labels={"relation": "neutralization"},
        )
    if not is_valid_protein_sequence(light):
        return None
    return BioSeqRecord(
        chains=[
            BioSeqChain(heavy, "antibody_heavy"),
            BioSeqChain(light, "antibody_light"),
        ],
        task_type="antibody_neutralization",
        source=source,
        split=split,
        labels={"relation": "neutralization"},
    )


def iter_semantic_rows(records: Iterator[BioSeqRecord | None], split: str) -> Iterator[dict[str, Any]]:
    for record in records:
        if record is None:
            continue
        yield semantic_row(record, split)
