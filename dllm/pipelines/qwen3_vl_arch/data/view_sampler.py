from __future__ import annotations

import json
import random
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import Sequence

from .records import BioSeqChain, BioSeqRecord, REGION_ORDER


FIXED_CONTEXT_ROLES = frozenset(
    {
        "antigen",
        "peptide",
        "epitope",
        "mhc",
        "pmhc",
        "hla",
    }
)

MHC_CONTEXT_ROLES = frozenset({"mhc", "pmhc", "hla"})
MHC_CONDITIONED_TARGET_ROLES = frozenset(
    {
        "tcr_alpha",
        "tcr_beta",
        "peptide",
        "epitope",
    }
)
PEPTIDE_TARGET_ROLES = frozenset({"peptide", "epitope"})
TCR_TARGET_ROLES = frozenset({"tcr_alpha", "tcr_beta"})
ANTIGEN_CONTEXT_ROLES = frozenset({"antigen"})
ANTIBODY_TARGET_ROLES = frozenset({"antibody_heavy", "antibody_light"})
NANOBODY_TARGET_ROLES = frozenset({"nanobody_vhh"})
ANTIBODY_RECEPTOR_ROLES = ANTIBODY_TARGET_ROLES | NANOBODY_TARGET_ROLES

DEFAULT_VIEWS_BY_TASK_TYPE = {
    "antibody": (
        "full_denoise",
        "heavy_to_light",
        "light_to_heavy",
        "fr_to_cdr",
        "single_cdr",
    ),
    "antibody_antigen": (
        "antigen_to_antibody",
        "heavy_antigen_to_light",
        "light_antigen_to_heavy",
        "antigen_fr_to_cdr",
        "antigen_single_cdr",
        "full_denoise",
    ),
    "nanobody_antigen": (
        "antigen_to_nanobody",
        "antigen_fr_to_cdr",
        "antigen_single_cdr",
        "full_denoise",
    ),
    "tcr": (
        "full_denoise",
        "beta_epitope_to_alpha",
        "alpha_epitope_to_beta",
        "fr_to_cdr",
        "single_cdr",
    ),
    "tcr_epitope": (
        "pmhc_to_tcr",
        "beta_epitope_to_alpha",
        "alpha_epitope_to_beta",
        "fr_to_cdr",
        "single_cdr",
        "full_denoise",
    ),
    "tcr_pmhc": (
        "pmhc_to_tcr",
        "pmhc_fr_to_cdr",
        "pmhc_single_cdr",
        "beta_epitope_to_alpha",
        "alpha_epitope_to_beta",
        "full_denoise",
    ),
    "ppi": ("full_denoise",),
    "generic": ("full_denoise",),
}


@dataclass(frozen=True)
class ResidueSpan:
    chain_index: int
    start: int
    end: int
    name: str


@dataclass(frozen=True)
class GenerationView:
    name: str
    target_spans: list[ResidueSpan]
    metadata: dict[str, str] = field(default_factory=dict)


def _full_chain_span(chain_index: int, chain: BioSeqChain, name: str = "full_chain") -> ResidueSpan:
    return ResidueSpan(chain_index=chain_index, start=0, end=len(chain.sequence), name=name)


def _role_indices(record: BioSeqRecord, role: str) -> list[int]:
    return [index for index, chain in enumerate(record.chains) if chain.role == role]


def _role_in(role: str, role_set: frozenset[str]) -> bool:
    normalized = role.strip().lower()
    return normalized in role_set


def _is_fixed_context_role(role: str) -> bool:
    normalized = role.strip().lower()
    return normalized in FIXED_CONTEXT_ROLES or normalized.startswith("mhc") or normalized.startswith("hla")


def _is_mhc_context_role(role: str) -> bool:
    normalized = role.strip().lower()
    return normalized in MHC_CONTEXT_ROLES or normalized.startswith("mhc") or normalized.startswith("hla")


def _is_antigen_role(role: str) -> bool:
    return role.strip().lower() in ANTIGEN_CONTEXT_ROLES


def _metadata_target_indices(record: BioSeqRecord) -> list[int]:
    raw_targets = record.metadata.get("targets")
    if raw_targets is None:
        return []
    if isinstance(raw_targets, str):
        try:
            parsed = json.loads(raw_targets)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in raw_targets.split(",")]
        raw_targets = parsed
    if not isinstance(raw_targets, SequenceABC):
        return []

    indices: list[int] = []
    seen: set[int] = set()
    for raw_index in raw_targets:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(record.chains) and index not in seen:
            indices.append(index)
            seen.add(index)
    return indices


def _eligible_target_indices(record: BioSeqRecord) -> list[int]:
    metadata_indices = _metadata_target_indices(record)
    candidate_indices = metadata_indices or list(range(len(record.chains)))
    target_indices = [
        index
        for index in candidate_indices
        if not _is_fixed_context_role(record.chains[index].role)
    ]
    if target_indices:
        return target_indices

    fallback_indices = [
        index
        for index, chain in enumerate(record.chains)
        if not _is_fixed_context_role(chain.role)
    ]
    if fallback_indices:
        return fallback_indices
    return list(range(len(record.chains)))


def _region_spans(
    record: BioSeqRecord,
    region_prefix: str,
    target_roles: frozenset[str] | None = None,
) -> list[ResidueSpan]:
    spans: list[ResidueSpan] = []
    for chain_index, chain in enumerate(record.chains):
        if target_roles is not None and not _role_in(chain.role, target_roles):
            continue
        for region_name in REGION_ORDER:
            if not region_name.startswith(region_prefix):
                continue
            span = chain.region_span(region_name)
            if span is not None:
                spans.append(ResidueSpan(chain_index, span[0], span[1], region_name))
    return spans


def _has_role_in(record: BioSeqRecord, role_set: frozenset[str]) -> bool:
    return any(_role_in(chain.role, role_set) for chain in record.chains)


def _has_antigen_context(record: BioSeqRecord) -> bool:
    return any(_is_antigen_role(chain.role) for chain in record.chains)


def _has_peptide_context(record: BioSeqRecord) -> bool:
    return any(
        _role_in(chain.role, PEPTIDE_TARGET_ROLES)
        or (record.task_type in {"tcr_epitope", "tcr_pmhc"} and chain.role == "antigen")
        for chain in record.chains
    )


class BioSeqViewSampler:
    """Create token-generation views from a full biological record."""

    def __init__(
        self,
        allowed_views: Sequence[str] | None = None,
        seed: int = 0,
        full_denoise_probability: float = 1.0,
    ) -> None:
        if not 0.0 <= full_denoise_probability <= 1.0:
            raise ValueError("full_denoise_probability must be between 0 and 1")
        self.allowed_views = list(allowed_views) if allowed_views is not None else None
        self.rng = random.Random(seed)
        self.full_denoise_probability = full_denoise_probability

    def sample(self, record: BioSeqRecord) -> GenerationView:
        compatible = self.compatible_views(record)
        if not compatible:
            return self.full_denoise(record)
        view_name = self._select_view_name(compatible)
        view = self.build(record, view_name)
        if view is None:
            return self.full_denoise(record)
        return view

    def sample_batch(self, records: Sequence[BioSeqRecord]) -> list[GenerationView]:
        if not records:
            return []
        common_views: set[str] | None = None
        ordered_views = self.allowed_views or self.default_views_for_record(records[0])
        for record in records:
            compatible = set(self.compatible_views(record))
            common_views = compatible if common_views is None else common_views & compatible
        if common_views:
            candidates = [view_name for view_name in ordered_views if view_name in common_views]
            view_name = self._select_view_name(candidates)
            return [self.build(record, view_name) or self.full_denoise(record) for record in records]
        return [self.full_denoise(record) for record in records]

    def compatible_views(self, record: BioSeqRecord, allowed_views: Sequence[str] | None = None) -> list[str]:
        view_names = list(allowed_views) if allowed_views is not None else (self.allowed_views or self.default_views_for_record(record))
        return [view_name for view_name in view_names if self.build(record, view_name) is not None]

    def _select_view_name(self, compatible_views: Sequence[str]) -> str:
        if not compatible_views:
            return "full_denoise"
        if "full_denoise" not in compatible_views:
            return compatible_views[0] if len(compatible_views) == 1 else self.rng.choice(list(compatible_views))

        condition_views = [view_name for view_name in compatible_views if view_name != "full_denoise"]
        if not condition_views:
            return "full_denoise"
        if self.rng.random() < self.full_denoise_probability:
            return "full_denoise"
        return condition_views[0] if len(condition_views) == 1 else self.rng.choice(condition_views)

    def default_views_for_record(self, record: BioSeqRecord) -> list[str]:
        if _has_antigen_context(record) and _has_role_in(record, ANTIBODY_TARGET_ROLES):
            return list(DEFAULT_VIEWS_BY_TASK_TYPE["antibody_antigen"])
        if _has_antigen_context(record) and _has_role_in(record, NANOBODY_TARGET_ROLES):
            return list(DEFAULT_VIEWS_BY_TASK_TYPE["nanobody_antigen"])
        return list(DEFAULT_VIEWS_BY_TASK_TYPE.get(record.task_type, DEFAULT_VIEWS_BY_TASK_TYPE["generic"]))

    def build(self, record: BioSeqRecord, view_name: str) -> GenerationView | None:
        if view_name == "full_denoise":
            return self.full_denoise(record)
        if view_name == "heavy_to_light":
            return self.role_completion(record, target_role="antibody_light", view_name=view_name)
        if view_name == "light_to_heavy":
            return self.role_completion(record, target_role="antibody_heavy", view_name=view_name)
        if view_name == "antigen_to_antibody":
            return self.antigen_to_roles(record, ANTIBODY_TARGET_ROLES, view_name=view_name)
        if view_name == "antigen_to_nanobody":
            return self.antigen_to_roles(record, NANOBODY_TARGET_ROLES, view_name=view_name)
        if view_name == "heavy_antigen_to_light":
            if not _role_indices(record, "antibody_heavy"):
                return None
            return self.antigen_context_role_completion(record, target_role="antibody_light", view_name=view_name)
        if view_name == "light_antigen_to_heavy":
            if not _role_indices(record, "antibody_light"):
                return None
            return self.antigen_context_role_completion(record, target_role="antibody_heavy", view_name=view_name)
        if view_name == "antigen_fr_to_cdr":
            return self.antigen_context_region_infilling(
                record,
                target_roles=ANTIBODY_RECEPTOR_ROLES,
                region_prefix="CDR",
                view_name=view_name,
            )
        if view_name == "antigen_single_cdr":
            return self.antigen_context_single_region(
                record,
                target_roles=ANTIBODY_RECEPTOR_ROLES,
                region_prefix="CDR",
                view_name=view_name,
            )
        if view_name == "beta_epitope_to_alpha":
            if not _role_indices(record, "tcr_beta"):
                return None
            return self.role_completion(record, target_role="tcr_alpha", view_name=view_name)
        if view_name == "alpha_epitope_to_beta":
            if not _role_indices(record, "tcr_alpha"):
                return None
            return self.role_completion(record, target_role="tcr_beta", view_name=view_name)
        if view_name == "mhc_to_peptide_tcr":
            return self.mhc_conditioned_peptide_tcr(record, view_name=view_name)
        if view_name == "tcr_mhc_to_peptide":
            return self.tcr_mhc_to_peptide(record, view_name=view_name)
        if view_name == "pmhc_to_tcr":
            return self.pmhc_to_tcr(record, view_name=view_name)
        if view_name == "pmhc_fr_to_cdr":
            return self.pmhc_context_region_infilling(record, region_prefix="CDR", view_name=view_name)
        if view_name == "pmhc_single_cdr":
            return self.pmhc_context_single_region(record, region_prefix="CDR", view_name=view_name)
        if view_name == "fr_to_cdr":
            spans = _region_spans(record, "CDR")
            return GenerationView(view_name, spans) if spans else None
        if view_name == "single_cdr":
            spans = _region_spans(record, "CDR")
            return GenerationView(view_name, [self.rng.choice(spans)]) if spans else None
        if view_name == "cdr_to_fr":
            spans = _region_spans(record, "FR")
            return GenerationView(view_name, spans) if spans else None
        raise ValueError(f"Unknown BioSeq generation view: {view_name}")

    def full_denoise(self, record: BioSeqRecord) -> GenerationView:
        return GenerationView(
            name="full_denoise",
            target_spans=[
                _full_chain_span(index, record.chains[index])
                for index in _eligible_target_indices(record)
            ],
        )

    def role_completion(self, record: BioSeqRecord, target_role: str, view_name: str) -> GenerationView | None:
        indices = _role_indices(record, target_role)
        if not indices:
            return None
        return GenerationView(
            name=view_name,
            target_spans=[_full_chain_span(index, record.chains[index], target_role) for index in indices],
        )

    def antigen_to_roles(self, record: BioSeqRecord, target_roles: frozenset[str], view_name: str) -> GenerationView | None:
        if not any(_is_antigen_role(chain.role) for chain in record.chains):
            return None
        target_indices = [
            index
            for index, chain in enumerate(record.chains)
            if _role_in(chain.role, target_roles)
        ]
        if not target_indices:
            return None
        return GenerationView(
            name=view_name,
            target_spans=[
                _full_chain_span(index, record.chains[index], record.chains[index].role)
                for index in target_indices
            ],
        )

    def antigen_context_role_completion(self, record: BioSeqRecord, target_role: str, view_name: str) -> GenerationView | None:
        if not any(_is_antigen_role(chain.role) for chain in record.chains):
            return None
        return self.role_completion(record, target_role=target_role, view_name=view_name)

    def antigen_context_region_infilling(
        self,
        record: BioSeqRecord,
        target_roles: frozenset[str],
        region_prefix: str,
        view_name: str,
    ) -> GenerationView | None:
        if not _has_antigen_context(record):
            return None
        spans = _region_spans(record, region_prefix, target_roles=target_roles)
        return GenerationView(view_name, spans) if spans else None

    def antigen_context_single_region(
        self,
        record: BioSeqRecord,
        target_roles: frozenset[str],
        region_prefix: str,
        view_name: str,
    ) -> GenerationView | None:
        view = self.antigen_context_region_infilling(record, target_roles, region_prefix, view_name)
        if view is None:
            return None
        return GenerationView(view_name, [self.rng.choice(view.target_spans)])

    def pmhc_context_region_infilling(
        self,
        record: BioSeqRecord,
        region_prefix: str,
        view_name: str,
    ) -> GenerationView | None:
        if not any(_is_mhc_context_role(chain.role) for chain in record.chains):
            return None
        if not _has_peptide_context(record):
            return None
        spans = _region_spans(record, region_prefix, target_roles=TCR_TARGET_ROLES)
        return GenerationView(view_name, spans) if spans else None

    def pmhc_context_single_region(
        self,
        record: BioSeqRecord,
        region_prefix: str,
        view_name: str,
    ) -> GenerationView | None:
        view = self.pmhc_context_region_infilling(record, region_prefix, view_name)
        if view is None:
            return None
        return GenerationView(view_name, [self.rng.choice(view.target_spans)])

    def mhc_conditioned_peptide_tcr(self, record: BioSeqRecord, view_name: str) -> GenerationView | None:
        if not any(_is_mhc_context_role(chain.role) for chain in record.chains):
            return None

        target_indices: list[int] = []
        for index, chain in enumerate(record.chains):
            if _role_in(chain.role, MHC_CONDITIONED_TARGET_ROLES):
                target_indices.append(index)
            elif record.task_type in {"tcr_epitope", "tcr_pmhc"} and chain.role == "antigen":
                target_indices.append(index)

        if not target_indices:
            return None
        return GenerationView(
            name=view_name,
            target_spans=[
                _full_chain_span(index, record.chains[index], record.chains[index].role)
                for index in target_indices
            ],
        )

    def tcr_mhc_to_peptide(self, record: BioSeqRecord, view_name: str) -> GenerationView | None:
        if not any(_is_mhc_context_role(chain.role) for chain in record.chains):
            return None
        if not any(_role_in(chain.role, TCR_TARGET_ROLES) for chain in record.chains):
            return None

        target_indices = [
            index
            for index, chain in enumerate(record.chains)
            if _role_in(chain.role, PEPTIDE_TARGET_ROLES)
            or (record.task_type in {"tcr_epitope", "tcr_pmhc"} and chain.role == "antigen")
        ]
        if not target_indices:
            return None
        return GenerationView(
            name=view_name,
            target_spans=[
                _full_chain_span(index, record.chains[index], record.chains[index].role)
                for index in target_indices
            ],
        )

    def pmhc_to_tcr(self, record: BioSeqRecord, view_name: str) -> GenerationView | None:
        if not any(_is_mhc_context_role(chain.role) for chain in record.chains):
            return None
        if not _has_peptide_context(record):
            return None

        target_indices = [
            index
            for index, chain in enumerate(record.chains)
            if _role_in(chain.role, TCR_TARGET_ROLES)
        ]
        if not target_indices:
            return None
        return GenerationView(
            name=view_name,
            target_spans=[
                _full_chain_span(index, record.chains[index], record.chains[index].role)
                for index in target_indices
            ],
        )
