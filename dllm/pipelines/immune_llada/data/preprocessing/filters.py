"""Explicit record filters for immune preprocessing.

Call ``load_blocklists`` on configuration paths, then build a source-specific
list with ``build_filters``.  ``filter_reason`` returns the first matching,
group-prefixed filter name.

Manifest ``filter_names`` is the **union**, across processed sources, of the
``RecordFilter`` names that ``build_filters`` actually constructed. Filters are
per-source (``quality.blank_epitope`` exists only for ``trait`` / ``tcr_native``
/ ``tcr_papers``), so a dataset-level list cannot be read as "every source ran
every name". A filter that was evaluated and dropped zero rows is still listed;
a disabled or empty blocklist never becomes a ``RecordFilter`` and is omitted.
The field is those constructed names, not the ``BLOCKLIST_NAMES`` config keys.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..records import BioSeqRecord

BLOCKLIST_NAMES = (
    "replaces_trait", "trait_benchmark", "ots_benchmark", "oas_benchmark",
    "asd_antibody_benchmark", "asd_nanobody_benchmark", "t4_refbinder", "t2t3_eval",
)
DISABLED_BLOCKLIST_SENTINELS = frozenset({"none", "off", "disabled", "-"})


@dataclass(frozen=True)
class RecordFilter:
    """A small callable predicate; ``True`` means that the record is rejected."""

    name: str
    predicate: Callable[[BioSeqRecord], bool]

    def evaluate(self, record: BioSeqRecord) -> bool:
        return bool(self.predicate(record))

    def __call__(self, record: BioSeqRecord) -> bool:
        return self.evaluate(record)


def load_blocklists(paths: Mapping[str, str | None]) -> dict[str, set[str]]:
    """Load all eight configured blocklists and fail on accidental no-ops.

    ``None`` and the four documented sentinel values explicitly disable a
    filter.  An empty string, missing file, or empty file is an error.
    """
    missing = [name for name in BLOCKLIST_NAMES if name not in paths]
    if missing:
        raise KeyError(f"Missing blocklist configuration keys: {missing}")
    result: dict[str, set[str]] = {}
    for name in BLOCKLIST_NAMES:
        value = paths[name]
        if value is None:
            result[name] = set()
            continue
        raw = str(value).strip()
        if not raw:
            raise ValueError(f"Empty blocklist path for {name!r}; use an explicit disabled sentinel")
        if raw.lower() in DISABLED_BLOCKLIST_SENTINELS:
            result[name] = set()
            continue
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"Blocklist not found for {name}: {path}")
        keys = {line.strip() for line in path.read_text().splitlines() if line.strip()}
        if not keys:
            raise ValueError(f"Blocklist {path} for {name!r} exists but contains no keys")
        result[name] = keys
    return result


def _keys(record: BioSeqRecord, *names: str) -> set[str]:
    return {record.identifiers[name] for name in names if record.identifiers.get(name)}


def _key_filter(name: str, blocked: set[str], *identifier_names: str) -> RecordFilter | None:
    if not blocked:
        return None
    return RecordFilter(name, lambda record: bool(_keys(record, *identifier_names) & blocked))


def _pair_key_filter(name: str, blocked: set[str]) -> RecordFilter | None:
    return _key_filter(name, blocked, "pair_key", "cdr3b_core|epitope")


def _project_cores(keys: set[str]) -> set[str]:
    """Project pair keys once, at filter construction, for repertoire rows."""
    return {key.split("|", 1)[0] for key in keys if "|" in key and key.split("|", 1)[0]} | {key for key in keys if "|" not in key}


def build_filters(
    source: str,
    mix: Sequence[str],
    blocklists: Mapping[str, set[str]],
    max_protein_length: int,
    max_length: int,
) -> list[RecordFilter]:
    """Build legacy decontamination/budgets plus approved offline quality checks."""
    missing = [name for name in BLOCKLIST_NAMES if name not in blocklists]
    if missing:
        raise KeyError(f"Missing blocklist keys: {missing}")
    source = str(source).strip()
    mix_set = set(mix)
    filters: list[RecordFilter] = []

    def add(filter_: RecordFilter | None) -> None:
        if filter_ is not None:
            filters.append(filter_)

    if source == "trait":
        if "trait" in mix_set and "tcr_native" in mix_set:
            add(_pair_key_filter("dedup.replaces_trait", set(blocklists.get("replaces_trait", set()))))
        add(_key_filter("decontam.trait_benchmark", set(blocklists.get("trait_benchmark", set())), "cdr3b_core"))
        add(_pair_key_filter("decontam.t4_refbinder", set(blocklists.get("t4_refbinder", set()))))
        add(_pair_key_filter("decontam.t2t3_eval", set(blocklists.get("t2t3_eval", set()))))
    elif source in {"tcr_native", "tcr_papers"}:
        add(_pair_key_filter("decontam.t4_refbinder", set(blocklists.get("t4_refbinder", set()))))
        add(_pair_key_filter("decontam.t2t3_eval", set(blocklists.get("t2t3_eval", set()))))
    elif source == "ots":
        add(_key_filter("decontam.ots_benchmark", set(blocklists.get("ots_benchmark", set())), "benchmark_key", "cdr3b_core"))
    elif source == "oas":
        add(_key_filter("decontam.oas_benchmark", set(blocklists.get("oas_benchmark", set())), "benchmark_h3_key", "benchmark_heavy_key"))
    elif source == "asd_antibody":
        add(_key_filter("decontam.asd_antibody_benchmark", set(blocklists.get("asd_antibody_benchmark", set())), "benchmark_h3_key", "benchmark_heavy_key"))
    elif source == "asd_nanobody":
        add(_key_filter("decontam.asd_nanobody_benchmark", set(blocklists.get("asd_nanobody_benchmark", set())), "benchmark_h3_key", "benchmark_heavy_key"))
    elif source == "tcr_repertoire":
        projected = _project_cores(set(blocklists.get("t4_refbinder", set())) | set(blocklists.get("t2t3_eval", set())))
        projected |= set(blocklists.get("ots_benchmark", set()))
        add(_key_filter("decontam.repertoire_core_projection", projected, "cdr3b_core", "benchmark_key"))
    elif source not in {"", "all"}:
        raise ValueError(f"Unknown immune source {source!r}")

    if max_protein_length > 0:
        add(RecordFilter("budget.max_protein_length", lambda record: any(len(chain.sequence) > max_protein_length for chain in record.chains)))
    if max_length > 0:
        add(RecordFilter("budget.max_length", lambda record: sum(len(chain.sequence) for chain in record.chains) + 3 * len(record.chains) + 8 > max_length))
    # Keep existing drop attribution; reject same-role pairs before grammar can
    # silently reinterpret them as heterotypic pairs or single-chain examples.
    if source in {"oas", "ots"}:
        add(RecordFilter("quality.homotypic_pair", lambda record: len(record.chains) == 2 and record.chains[0].role == record.chains[1].role))
    # An all-X epitope is the upstream "unknown" placeholder, so the row claims a
    # binding relation with nothing to condition on. It is also invisible to the
    # pair-keyed decontamination: '{cdr3b_core}|{epitope}' cannot be formed, so
    # these rows never get checked against the T4 ref-binder blocklist at all.
    # The epitope is not recoverable from the MHC (70 pseudo-sequences vs
    # thousands of epitopes), hence dropped rather than repaired.
    if source in {"trait", "tcr_native", "tcr_papers"}:
        add(RecordFilter("quality.blank_epitope", _has_blank_epitope))
    return filters


def _has_blank_epitope(record: BioSeqRecord) -> bool:
    return any(
        chain.role == "peptide" and bool(chain.sequence) and set(chain.sequence) == {"X"}
        for chain in record.chains
    )


def filter_reason(record: BioSeqRecord, filters: Sequence[RecordFilter]) -> str | None:
    """Return the first rejection reason, or ``None`` when all predicates pass."""
    for filter_ in filters:
        if filter_(record):
            return filter_.name
    return None


def constructed_filter_names(filters: Sequence[RecordFilter]) -> list[str]:
    """Names of filters that will be evaluated, in evaluation order."""
    return [filter_.name for filter_ in filters]


def union_filter_names(per_source: Iterable[Sequence[str]]) -> list[str]:
    """Dataset-level union of per-source constructed filter names.

    First-seen order follows ``per_source`` (typically the processed-source
    order). See the module docstring for why this is a union rather than a
    per-source map.
    """
    names: list[str] = []
    seen: set[str] = set()
    for group in per_source:
        for name in group:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


__all__ = [
    "BLOCKLIST_NAMES", "DISABLED_BLOCKLIST_SENTINELS", "RecordFilter",
    "build_filters", "constructed_filter_names", "filter_reason", "load_blocklists",
    "union_filter_names",
]
