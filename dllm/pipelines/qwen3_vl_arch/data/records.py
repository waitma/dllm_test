from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_PROTEIN_CHARS = set("ACDEFGHIKLMNPQRSTVWYXBZUO.-")
REGION_ORDER = ("FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4")

CHAIN_ROLE_TO_ID = {
    "unknown": 0,
    "antibody_heavy": 1,
    "antibody_light": 2,
    "nanobody_vhh": 3,
    "tcr_alpha": 4,
    "tcr_beta": 5,
    "peptide": 6,
    "antigen": 7,
    "mhc": 8,
    "protein_a": 9,
    "protein_b": 10,
    "other": 11,
}

TASK_TYPE_TO_ID = {
    "generic": 0,
    "antibody": 1,
    "tcr": 2,
    "tcr_epitope": 3,
    "tcr_pmhc": 4,
    "ppi": 5,
    "antibody_antigen": 6,
    "nanobody_antigen": 7,
}


def normalize_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper().replace("J", "L")


def is_valid_protein_sequence(sequence: str) -> bool:
    return bool(sequence) and all(residue in VALID_PROTEIN_CHARS for residue in sequence)


def compact_metadata(row: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            metadata[key] = value
    return metadata


@dataclass(frozen=True)
class BioSeqChain:
    sequence: str
    role: str
    regions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = normalize_sequence(self.sequence)
        object.__setattr__(self, "sequence", normalized)
        if not is_valid_protein_sequence(normalized):
            raise ValueError(f"Invalid protein sequence for role {self.role}: {self.sequence!r}")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"sequence": self.sequence, "role": self.role}
        if self.regions:
            data["regions"] = self.regions
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    def region_span(self, region_name: str) -> tuple[int, int] | None:
        region = normalize_sequence(self.regions.get(region_name))
        if not region:
            return None

        cursor = 0
        for name in REGION_ORDER:
            current = normalize_sequence(self.regions.get(name))
            if not current:
                continue
            start = self.sequence.find(current, cursor)
            if start < 0:
                start = self.sequence.find(current)
            if start < 0:
                continue
            end = start + len(current)
            if name == region_name:
                return start, end
            cursor = end
        return None


@dataclass(frozen=True)
class BioSeqRecord:
    chains: list[BioSeqChain]
    task_type: str
    source: str
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.chains:
            raise ValueError("BioSeqRecord requires at least one chain")

    @property
    def chain_roles(self) -> list[str]:
        return [chain.role for chain in self.chains]

    @property
    def sequences(self) -> list[str]:
        return [chain.sequence for chain in self.chains]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "chains": [chain.sequence for chain in self.chains],
            "chain_roles": self.chain_roles,
            "task_type": self.task_type,
            "source": self.source,
            "weight": self.weight,
        }
        regions = {str(i): chain.regions for i, chain in enumerate(self.chains) if chain.regions}
        chain_metadata = {str(i): chain.metadata for i, chain in enumerate(self.chains) if chain.metadata}
        if self.split:
            data["split"] = self.split
        if regions:
            data["regions"] = regions
        if self.metadata:
            data["metadata"] = self.metadata
        if chain_metadata:
            data["chain_metadata"] = chain_metadata
        if self.labels:
            data["labels"] = self.labels
        return data


def chain_role_from_processed_type(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "alpha": "tcr_alpha",
        "a": "tcr_alpha",
        "tra": "tcr_alpha",
        "beta": "tcr_beta",
        "b": "tcr_beta",
        "trb": "tcr_beta",
        "antigen": "antigen",
        "peptide": "peptide",
        "mhc": "mhc",
        "heavy": "antibody_heavy",
        "light": "antibody_light",
        "h": "antibody_heavy",
        "l": "antibody_light",
        "other": "other",
    }.get(normalized, normalized or "other")


def task_type_from_processed_source(source: Any) -> str:
    normalized = str(source or "").strip().lower()
    if normalized == "ppi":
        return "ppi"
    if normalized in {"vdjdb", "mcpas", "mira", "iedb", "atlas", "deepinsight", "tcrdesign"}:
        return "tcr_epitope"
    return "generic"
