"""Dataset loaders that feed the exact Ophiuchus-Ab diffusion model.

This module loads the three local immune-sequence corpora used for BioSeq
training and normalizes every row into a minimal ``{"chains", "task_type",
"source", "weight"}`` record that :class:`MultiChainDynamicCollator` understands:

* OAS paired antibody  -> two chains (heavy oriented to slot 0)
* OTS paired TCR        -> two chains (beta oriented to slot 0)
* nanobody (VHH)        -> one chain

All paths must be absolute, following the project rule. The loaders stream the
CSV files and stop after ``max_rows`` valid rows, so the multi-GB ``train.csv``
files never need to be fully materialized for a smoke run.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from torch.utils.data import ConcatDataset, Dataset

from .adapters import is_valid_protein_sequence, normalize_sequence

# csv fields such as full FR/CDR region strings can be long; lift the limit once.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DEFAULT_DATA_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/data")

OAS_DEFAULT_DIR = DEFAULT_DATA_ROOT / "oas_previous_clean/splits/compat_for_current_loader_oasrule"
OTS_DEFAULT_DIR = DEFAULT_DATA_ROOT / "ots_paired_clean/final"
NANOBODY_DEFAULT_DIR = DEFAULT_DATA_ROOT / "nanobody_processed/step6_final"


@dataclass
class ImmuneSourceSpec:
    name: str
    path: Path
    row_to_record: Callable[[dict[str, Any]], dict[str, Any] | None]
    task_type: str
    weight: float = 1.0


def _is_heavy(anarci_type: str, chain_type: str) -> bool:
    return anarci_type.strip().upper() == "H" or chain_type.strip().lower() == "heavy"


def _is_beta(anarci_type: str, chain_type: str) -> bool:
    return anarci_type.strip().upper() == "B" or chain_type.strip().lower() in {"beta", "trb"}


def oas_paired_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    chain1 = normalize_sequence(row.get("cleaned_chain1_seq"))
    chain2 = normalize_sequence(row.get("cleaned_chain2_seq"))
    if not is_valid_protein_sequence(chain1) or not is_valid_protein_sequence(chain2):
        return None
    chain2_heavy = _is_heavy(str(row.get("chain2_anarci_type", "")), str(row.get("chain2_type", "")))
    chain1_heavy = _is_heavy(str(row.get("chain1_anarci_type", "")), str(row.get("chain1_type", "")))
    if chain2_heavy and not chain1_heavy:
        chains = [chain2, chain1]
    else:
        chains = [chain1, chain2]
    return {"chains": chains, "task_type": "antibody", "source": "oas_paired"}


def ots_paired_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    chain1 = normalize_sequence(row.get("cleaned_chain1_seq"))
    chain2 = normalize_sequence(row.get("cleaned_chain2_seq"))
    if not is_valid_protein_sequence(chain1) or not is_valid_protein_sequence(chain2):
        return None
    chain2_beta = _is_beta(str(row.get("chain2_anarci_type", "")), str(row.get("chain2_type", "")))
    chain1_beta = _is_beta(str(row.get("chain1_anarci_type", "")), str(row.get("chain1_type", "")))
    if chain2_beta and not chain1_beta:
        chains = [chain2, chain1]
    else:
        chains = [chain1, chain2]
    return {"chains": chains, "task_type": "tcr", "source": "ots_paired"}


def nanobody_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    chain = normalize_sequence(row.get("cleaned_seq") or row.get("vhh_seq"))
    if not is_valid_protein_sequence(chain):
        return None
    return {"chains": [chain], "task_type": "antibody", "source": "nanobody"}


def _iter_csv_records(
    path: Path,
    row_to_record: Callable[[dict[str, Any]], dict[str, Any] | None],
    max_rows: int | None,
) -> Iterator[dict[str, Any]]:
    kept = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record = row_to_record(row)
            if record is None:
                continue
            yield record
            kept += 1
            if max_rows is not None and kept >= max_rows:
                break


class ImmuneCsvDataset(Dataset):
    """In-memory map-style dataset built from one immune CSV source."""

    def __init__(self, spec: ImmuneSourceSpec, split: str = "train", max_rows: int | None = None) -> None:
        path = spec.path if spec.path.is_file() else spec.path / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Immune CSV not found: {path}")
        self.name = spec.name
        self.weight = spec.weight
        self.records: list[dict[str, Any]] = []
        for record in _iter_csv_records(path, spec.row_to_record, max_rows):
            record["weight"] = spec.weight
            self.records.append(record)
        if not self.records:
            raise ValueError(f"No valid rows loaded from {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def default_immune_specs(
    oas_dir: Path | str = OAS_DEFAULT_DIR,
    ots_dir: Path | str = OTS_DEFAULT_DIR,
    nanobody_dir: Path | str = NANOBODY_DEFAULT_DIR,
    oas_weight: float = 1.0,
    ots_weight: float = 1.0,
    nanobody_weight: float = 1.0,
) -> list[ImmuneSourceSpec]:
    return [
        ImmuneSourceSpec("oas", Path(oas_dir), oas_paired_row_to_record, "antibody", oas_weight),
        ImmuneSourceSpec("ots", Path(ots_dir), ots_paired_row_to_record, "tcr", ots_weight),
        ImmuneSourceSpec("nanobody", Path(nanobody_dir), nanobody_row_to_record, "antibody", nanobody_weight),
    ]


def build_mixed_immune_dataset(
    specs: list[ImmuneSourceSpec] | None = None,
    split: str = "train",
    max_rows_per_source: int | None = None,
) -> tuple[ConcatDataset, dict[str, int]]:
    """Build a concatenated dataset over the three immune sources.

    Returns the ``ConcatDataset`` plus a ``{source_name: num_rows}`` summary.
    """

    specs = specs or default_immune_specs()
    datasets: list[ImmuneCsvDataset] = []
    counts: dict[str, int] = {}
    for spec in specs:
        dataset = ImmuneCsvDataset(spec, split=split, max_rows=max_rows_per_source)
        datasets.append(dataset)
        counts[spec.name] = len(dataset)
    return ConcatDataset(datasets), counts
