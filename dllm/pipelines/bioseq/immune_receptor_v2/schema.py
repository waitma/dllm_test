"""Define and validate the ``bioseq.v2`` immune-receptor record schema.

This module has no third-party dependencies. Smoke test it with::

    conda run -n pllm python -m pytest \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_immune_receptor_v2.py -q
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


BIOSEQ_V2_SCHEMA_VERSION = "bioseq.v2"
VALID_PROTEIN_CHARS = frozenset("ACDEFGHIKLMNPQRSTVWYXBZJUO")
VALID_ROLES = frozenset(
    {
        "antibody_heavy",
        "antibody_light",
        "tcr_alpha",
        "tcr_beta",
        "peptide",
        "mhc_alpha",
        "mhc_beta2m",
        "antigen",
    }
)
VALID_SEQUENCE_SCOPES = frozenset(
    {
        "full_chain",
        "variable_domain",
        "cdr3",
        "peptide",
        "mhc_full_chain",
        "mhc_pseudosequence",
        "antigen_construct",
        "unknown",
    }
)
VALID_EVIDENCE_TIERS = frozenset({"A", "B", "C", "D"})
VALID_MAPPING_CONFIDENCE = frozenset(
    {"exact_sequence", "exact_accession", "construct_inferred", "name_only", "unmapped"}
)
VALID_CENSORS = frozenset({"=", "<", "<=", ">", ">=", "range", "unknown"})
VALID_DIRECTIONS = frozenset({"higher", "lower", "none", "unknown"})
RECEPTOR_ROLES = frozenset(
    {"antibody_heavy", "antibody_light", "tcr_alpha", "tcr_beta"}
)
MISSING_TEXT = frozenset({"", "na", "n/a", "nan", "none", "null", "nd", "unknown", "-"})


def clean_text(value: Any) -> str:
    """Return stripped text while treating common table sentinels as missing."""

    text = str(value or "").strip()
    return "" if text.lower() in MISSING_TEXT else text


def normalize_sequence(value: Any) -> str:
    """Normalize whitespace/case without changing ambiguous amino-acid identity."""

    return re.sub(r"\s+", "", clean_text(value)).upper()


def is_valid_sequence(sequence: str) -> bool:
    return bool(sequence) and all(residue in VALID_PROTEIN_CHARS for residue in sequence)


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for identifiers and manifests."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def stable_digest(value: Any, *, prefix: str = "", length: int = 24) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}"


def _drop_empty(value: Any) -> Any:
    """Recursively remove empty optional values but retain false and zero."""

    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _drop_empty(item)) not in (None, "", [], {})
        }
    if isinstance(value, (list, tuple)):
        return [
            cleaned
            for item in value
            if (cleaned := _drop_empty(item)) not in (None, "", [], {})
        ]
    return value


def parse_float(value: Any) -> float | None:
    """Parse a finite floating-point table value, returning ``None`` on sentinels."""

    text = clean_text(value)
    if not text:
        return None
    try:
        parsed = float(text.replace(",", ""))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class SequenceEntity:
    """One explicitly typed biological sequence."""

    role: str
    sequence: str
    sequence_scope: str
    species: str = ""
    observed: bool = True
    reconstructed: bool = False
    genes: Mapping[str, str] = field(default_factory=dict)
    regions: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    entity_id: str = ""

    def __post_init__(self) -> None:
        role = clean_text(self.role).lower()
        scope = clean_text(self.sequence_scope).lower() or "unknown"
        sequence = normalize_sequence(self.sequence)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "sequence_scope", scope)
        object.__setattr__(self, "sequence", sequence)
        if role not in VALID_ROLES:
            raise ValueError(f"Unsupported bioseq.v2 role: {role!r}")
        if scope not in VALID_SEQUENCE_SCOPES:
            raise ValueError(f"Unsupported sequence scope: {scope!r}")
        if not is_valid_sequence(sequence):
            raise ValueError(f"Invalid amino-acid sequence for {role}: {sequence[:80]!r}")
        if self.observed and self.reconstructed:
            raise ValueError("A sequence cannot be both observed and reconstructed")
        if not self.entity_id:
            object.__setattr__(
                self,
                "entity_id",
                stable_digest(
                    {
                        "role": role,
                        "sequence": sequence,
                        "sequence_scope": scope,
                        "species": clean_text(self.species),
                    },
                    prefix="ent_",
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "SequenceEntity":
        return cls(**dict(row))


@dataclass(frozen=True)
class SourceReference:
    """Provenance for one original row or released artifact."""

    source_name: str
    source_record_id: str
    source_version: str = ""
    raw_path: str = ""
    raw_sha256: str = ""
    original_split: str = ""
    study_id: str = ""
    donor_id: str = ""
    assay_id: str = ""
    pmid: str = ""
    doi: str = ""
    url: str = ""
    license: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not clean_text(self.source_name):
            raise ValueError("source_name is required")
        if not clean_text(self.source_record_id):
            raise ValueError("source_record_id is required")

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "SourceReference":
        return cls(**dict(row))


@dataclass(frozen=True)
class Measurement:
    """Assay label/value with raw and normalized representations."""

    relation: str
    label: Any = None
    raw_value: Any = None
    raw_unit: str = ""
    value: float | None = None
    unit: str = ""
    censor: str = "="
    direction: str = "unknown"
    assay_type: str = ""
    assay_readout: str = ""
    negative_type: str = ""
    confidence: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not clean_text(self.relation):
            raise ValueError("measurement relation is required")
        if self.censor not in VALID_CENSORS:
            raise ValueError(f"Unsupported censor: {self.censor!r}")
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(f"Unsupported measurement direction: {self.direction!r}")
        if self.value is not None and not math.isfinite(float(self.value)):
            raise ValueError("measurement value must be finite")

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Measurement":
        return cls(**dict(row))


@dataclass(frozen=True)
class Evidence:
    tier: str
    basis: str
    direct_measurement: bool
    derived_from_aggregate: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.tier not in VALID_EVIDENCE_TIERS:
            raise ValueError(f"Unsupported evidence tier: {self.tier!r}")
        if not clean_text(self.basis):
            raise ValueError("evidence basis is required")

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Evidence":
        return cls(**dict(row))


@dataclass(frozen=True)
class TargetMapping:
    target_name_raw: str = ""
    canonical_target_id: str = ""
    accession: str = ""
    construct: str = ""
    variant: str = ""
    mapping_confidence: str = "unmapped"
    mapping_source: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.mapping_confidence not in VALID_MAPPING_CONFIDENCE:
            raise ValueError(
                f"Unsupported target mapping confidence: {self.mapping_confidence!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "TargetMapping":
        return cls(**dict(row))


@dataclass(frozen=True)
class Eligibility:
    training: bool
    evaluation: bool
    core: bool
    pack: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not clean_text(self.pack):
            raise ValueError("eligibility pack is required")
        if self.core and not self.training:
            raise ValueError("A core record must be training-eligible")

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Eligibility":
        data = dict(row)
        data["reasons"] = tuple(data.get("reasons", ()))
        return cls(**data)


def _default_groups(
    entities: Iterable[SequenceEntity],
    context: Mapping[str, Any],
    sources: Iterable[SourceReference],
    target_mapping: TargetMapping | None,
) -> dict[str, str]:
    by_role = {entity.role: entity for entity in entities}
    receptor = [
        (role, by_role[role].sequence)
        for role in sorted(RECEPTOR_ROLES)
        if role in by_role
    ]
    groups: dict[str, str] = {}
    if receptor:
        groups["receptor"] = stable_digest(receptor, prefix="rec_")
    heavy_light = [
        (role, by_role[role].sequence)
        for role in ("antibody_heavy", "antibody_light")
        if role in by_role
    ]
    if heavy_light:
        groups["antibody"] = stable_digest(heavy_light, prefix="ab_")
    peptide = by_role.get("peptide")
    if peptide:
        groups["peptide"] = stable_digest(peptide.sequence, prefix="pep_")
        mhc_identity = clean_text(context.get("mhc_allele"))
        if not mhc_identity and "mhc_alpha" in by_role:
            mhc_identity = by_role["mhc_alpha"].sequence
        if mhc_identity:
            groups["pmhc"] = stable_digest(
                [peptide.sequence, mhc_identity], prefix="pmhc_"
            )
    antigen = by_role.get("antigen")
    if antigen:
        groups["antigen"] = stable_digest(antigen.sequence, prefix="ag_")
    elif target_mapping and (
        target_mapping.canonical_target_id or target_mapping.target_name_raw
    ):
        groups["antigen"] = stable_digest(
            target_mapping.canonical_target_id or target_mapping.target_name_raw.lower(),
            prefix="agname_",
        )
    study_ids = sorted({clean_text(source.study_id) for source in sources if source.study_id})
    if study_ids:
        groups["study"] = stable_digest(study_ids, prefix="study_")
    parent_id = clean_text(context.get("parent_id"))
    if parent_id:
        groups["parent"] = stable_digest(parent_id, prefix="parent_")
    return groups


@dataclass(frozen=True)
class CanonicalRecord:
    """One canonical evidence row; duplicate evidence is merged separately."""

    family: str
    record_type: str
    entities: tuple[SequenceEntity, ...]
    measurement: Measurement
    source_records: tuple[SourceReference, ...]
    evidence: Evidence
    eligibility: Eligibility
    context: Mapping[str, Any] = field(default_factory=dict)
    groups: Mapping[str, str] = field(default_factory=dict)
    target_mapping: TargetMapping | None = None
    derived_from: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    record_id: str = ""
    schema_version: str = BIOSEQ_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BIOSEQ_V2_SCHEMA_VERSION:
            raise ValueError(f"Expected {BIOSEQ_V2_SCHEMA_VERSION}, got {self.schema_version}")
        if not clean_text(self.family) or not clean_text(self.record_type):
            raise ValueError("family and record_type are required")
        if not self.entities:
            raise ValueError("A canonical record requires at least one sequence entity")
        if len({entity.role for entity in self.entities}) != len(self.entities):
            raise ValueError("A canonical record may contain at most one entity per role")
        if not self.source_records:
            raise ValueError("A canonical record requires source provenance")
        groups = _default_groups(
            self.entities, self.context, self.source_records, self.target_mapping
        )
        groups.update({key: value for key, value in self.groups.items() if value})
        object.__setattr__(self, "groups", groups)
        if not self.record_id:
            source_identity = [
                (
                    source.source_name,
                    source.source_version,
                    source.source_record_id,
                    source.original_split,
                    source.study_id,
                    source.donor_id,
                    source.assay_id,
                    source.pmid,
                    source.doi,
                )
                for source in self.source_records
            ]
            object.__setattr__(
                self,
                "record_id",
                stable_digest(
                    {
                        "biological_key": self.biological_key,
                        "measurement": self.measurement.to_dict(),
                        "sources": source_identity,
                        "record_type": self.record_type,
                    },
                    prefix="ir2_",
                ),
            )

    @property
    def biological_key(self) -> str:
        payload = {
            "family": self.family,
            "entities": [
                {
                    "role": entity.role,
                    "sequence": entity.sequence,
                    "sequence_scope": entity.sequence_scope,
                }
                for entity in sorted(self.entities, key=lambda item: item.role)
            ],
            "context": dict(self.context),
            "target": self.target_mapping.to_dict() if self.target_mapping else {},
        }
        return stable_digest(payload, prefix="bio_", length=32)

    @property
    def measurement_key(self) -> str:
        studies = sorted(
            {
                source.study_id or source.doi or source.pmid
                for source in self.source_records
                if source.study_id or source.doi or source.pmid
            }
        )
        return stable_digest(
            {
                "biological_key": self.biological_key,
                "measurement": self.measurement.to_dict(),
                "studies": studies,
            },
            prefix="meas_",
            length=32,
        )

    def entity(self, role: str) -> SequenceEntity | None:
        return next((item for item in self.entities if item.role == role), None)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entities"] = [entity.to_dict() for entity in self.entities]
        data["measurement"] = self.measurement.to_dict()
        data["source_records"] = [source.to_dict() for source in self.source_records]
        data["evidence"] = self.evidence.to_dict()
        data["eligibility"] = self.eligibility.to_dict()
        data["target_mapping"] = (
            self.target_mapping.to_dict() if self.target_mapping else None
        )
        data["biological_key"] = self.biological_key
        data["measurement_key"] = self.measurement_key
        return _drop_empty(data)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "CanonicalRecord":
        data = dict(row)
        data.pop("biological_key", None)
        data.pop("measurement_key", None)
        data["entities"] = tuple(
            SequenceEntity.from_dict(item) for item in data.get("entities", [])
        )
        data["measurement"] = Measurement.from_dict(data["measurement"])
        data["source_records"] = tuple(
            SourceReference.from_dict(item) for item in data.get("source_records", [])
        )
        data["evidence"] = Evidence.from_dict(data["evidence"])
        data["eligibility"] = Eligibility.from_dict(data["eligibility"])
        target = data.get("target_mapping")
        data["target_mapping"] = TargetMapping.from_dict(target) if target else None
        data["derived_from"] = tuple(data.get("derived_from", ()))
        data["flags"] = tuple(data.get("flags", ()))
        return cls(**data)


def validate_domain_record(record: CanonicalRecord) -> list[str]:
    """Return semantic validation errors beyond dataclass shape checks."""

    errors: list[str] = []
    roles = {entity.role for entity in record.entities}
    if any("nanobody" in role or "vhh" in role for role in roles):
        errors.append("nanobody/VHH records are outside the active AB/TCR scope")
    if record.family.startswith("tcr"):
        if not roles.intersection({"tcr_alpha", "tcr_beta"}):
            errors.append("TCR record lacks an alpha or beta receptor sequence")
        if record.eligibility.core and "peptide" not in roles:
            errors.append("core TCR specificity record lacks an exact peptide")
        if record.family == "tcr_pmhc" and record.eligibility.core:
            has_mhc = "mhc_alpha" in roles or clean_text(record.context.get("mhc_allele"))
            if not has_mhc:
                errors.append("core TCR-pMHC record lacks MHC sequence/allele context")
    if record.family == "antibody_antigen":
        if record.eligibility.core:
            required = {"antibody_heavy", "antibody_light", "antigen"}
            if not required.issubset(roles):
                errors.append("core antibody-antigen record must contain H, L, and antigen")
            if not record.target_mapping or record.target_mapping.mapping_confidence not in {
                "exact_sequence",
                "exact_accession",
            }:
                errors.append("core antibody-antigen record lacks exact target mapping")
            if not record.evidence.direct_measurement:
                errors.append("core antibody-antigen record must be a direct measurement")
    if record.family == "antibody_property":
        if not {"antibody_heavy", "antibody_light"}.issubset(roles):
            errors.append("antibody property record must contain paired H/L sequences")
        if "antigen" in roles:
            errors.append("intrinsic antibody property records cannot contain antigen entities")
    if record.record_type == "derived_view":
        if not record.derived_from:
            errors.append("derived view lacks derived_from linkage")
        if not any(entity.reconstructed for entity in record.entities):
            errors.append("derived view has no reconstructed entity")
        if not record.evidence.derived_from_aggregate:
            errors.append("derived view must be marked as derived evidence")
    return errors
