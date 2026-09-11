"""Shared immune sequence records.

Use ordinary chain construction when ingesting raw sequences::

    from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
    record = BioSeqRecord([BioSeqChain("ACD", "peptide")], "generic", "example")

Use ``BioSeqRecord.from_prepared_dict(record.to_dict())`` only for trusted,
already-prepared rows to avoid repeating chain normalization and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_PROTEIN_CHARS = set("ACDEFGHIKLMNPQRSTVWYXBZUO.-")
DEFAULT_MAX_PROTEIN_LENGTH = 1024
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
    identifiers: dict[str, str] = field(default_factory=dict)

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
        if self.identifiers:
            data["identifiers"] = self.identifiers
        return data

    @classmethod
    def from_prepared_dict(cls, row: dict[str, Any]) -> "BioSeqRecord":
        """Deserialize a trusted canonical row without sequence normalization.

        Basic shape errors raise immediately. This is for prepared rows only;
        sequence content is neither normalized nor validated a second time::

            row = record.to_dict()
            restored = BioSeqRecord.from_prepared_dict(row)
            assert restored.to_dict() == row
        """

        if not isinstance(row, dict):
            raise TypeError("prepared BioSeqRecord must be a dict")
        sequences = row.get("chains")
        roles = row.get("chain_roles")
        if not isinstance(sequences, list) or not isinstance(roles, list):
            raise TypeError("prepared record requires list fields 'chains' and 'chain_roles'")
        if not sequences:
            raise ValueError("prepared BioSeqRecord requires at least one chain")
        if len(sequences) != len(roles):
            raise ValueError("prepared chains and chain_roles must have equal lengths")
        if not all(isinstance(value, str) for value in sequences):
            raise TypeError("prepared chain sequences must be strings")
        if not all(isinstance(value, str) for value in roles):
            raise TypeError("prepared chain roles must be strings")

        def indexed_mapping(name: str) -> dict[str, dict[str, Any]]:
            value = row.get(name, {})
            if not isinstance(value, dict):
                raise TypeError(f"prepared {name} must be a dict")
            expected = {str(index) for index in range(len(sequences))}
            if not set(value).issubset(expected):
                raise ValueError(f"prepared {name} keys must be string chain indices")
            for key, item in value.items():
                if not isinstance(item, dict):
                    raise TypeError(f"prepared {name}[{key!r}] must be a dict")
                if name == "regions" and not all(
                    isinstance(region, str) and isinstance(sequence, str)
                    for region, sequence in item.items()
                ):
                    raise TypeError(f"prepared {name}[{key!r}] must map strings to strings")
            return value

        regions = indexed_mapping("regions")
        chain_metadata = indexed_mapping("chain_metadata")
        metadata = row.get("metadata", {})
        labels = row.get("labels", {})
        identifiers = row.get("identifiers", {})
        if not isinstance(metadata, dict) or not isinstance(labels, dict):
            raise TypeError("prepared metadata and labels must be dicts")
        if not isinstance(identifiers, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in identifiers.items()
        ):
            raise TypeError("prepared identifiers must be a dict[str, str]")
        task_type = row.get("task_type")
        source = row.get("source")
        split = row.get("split")
        if not isinstance(task_type, str) or not isinstance(source, str):
            raise TypeError("prepared task_type and source must be strings")
        if split is not None and not isinstance(split, str):
            raise TypeError("prepared split must be a string or None")
        weight = row.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("prepared weight must be numeric")

        prepared_chains: list[BioSeqChain] = []
        for index, (sequence, role) in enumerate(zip(sequences, roles)):
            chain = object.__new__(BioSeqChain)
            object.__setattr__(chain, "sequence", sequence)
            object.__setattr__(chain, "role", role)
            object.__setattr__(chain, "regions", dict(regions.get(str(index), {})))
            object.__setattr__(chain, "metadata", dict(chain_metadata.get(str(index), {})))
            prepared_chains.append(chain)
        return cls(
            chains=prepared_chains,
            task_type=task_type,
            source=source,
            split=split,
            metadata=dict(metadata),
            labels=dict(labels),
            weight=weight,
            identifiers=dict(identifiers),
        )


def record_within_max_protein_length(
    record: BioSeqRecord,
    max_protein_length: int = DEFAULT_MAX_PROTEIN_LENGTH,
) -> bool:
    """Return True when every chain in the record is at most ``max_protein_length`` aa."""

    if max_protein_length <= 0:
        return True
    return all(len(chain.sequence) <= max_protein_length for chain in record.chains)


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
