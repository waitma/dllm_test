"""Deduplication, conflict, and eligibility audits for ``bioseq.v2`` records.

Use through the main data builder or run its focused tests with::

    conda run -n pllm python -m pytest \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_immune_receptor_v2.py -q
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable

from .schema import CanonicalRecord, validate_domain_record


def merge_exact_measurements(
    records: Iterable[CanonicalRecord],
) -> tuple[list[CanonicalRecord], dict]:
    """Merge only identical measurement identities and retain all provenance."""

    grouped: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for record in records:
        grouped[record.measurement_key].append(record)
    merged: list[CanonicalRecord] = []
    duplicate_groups = 0
    for key in sorted(grouped):
        members = grouped[key]
        first = members[0]
        if len(members) == 1:
            merged.append(first)
            continue
        duplicate_groups += 1
        sources = {
            (
                source.source_name,
                source.source_version,
                source.source_record_id,
                source.study_id,
            ): source
            for member in members
            for source in member.source_records
        }
        flags = tuple(sorted({flag for member in members for flag in member.flags}))
        derived_from = tuple(
            sorted({item for member in members for item in member.derived_from})
        )
        merged.append(
            replace(
                first,
                source_records=tuple(sources[key] for key in sorted(sources)),
                flags=flags,
                derived_from=derived_from,
                record_id="",
            )
        )
    return merged, {
        "input_records": sum(len(members) for members in grouped.values()),
        "output_records": len(merged),
        "exact_duplicate_groups": duplicate_groups,
    }


def audit_records(records: Iterable[CanonicalRecord]) -> dict:
    rows = list(records)
    families = Counter(record.family for record in rows)
    sources = Counter(
        source.source_name for record in rows for source in record.source_records
    )
    packs = Counter(record.eligibility.pack for record in rows)
    tiers = Counter(record.evidence.tier for record in rows)
    flags = Counter(flag for record in rows for flag in record.flags)
    domain_errors: list[dict] = []
    for record in rows:
        errors = validate_domain_record(record)
        if errors:
            domain_errors.append({"record_id": record.record_id, "errors": errors})

    biological_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    biological_measurements: Counter[str] = Counter()
    receptor_peptide_labels: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for record in rows:
        biological_measurements[record.biological_key] += 1
        if record.measurement.label is not None:
            relation_key = (record.biological_key, record.measurement.relation)
            biological_labels[relation_key].add(str(record.measurement.label))
            receptor = record.groups.get("receptor", "")
            peptide = record.groups.get("peptide", "")
            if receptor and peptide:
                receptor_peptide_labels[
                    (receptor, peptide, record.measurement.relation)
                ].add(str(record.measurement.label))
    conflicts = {
        f"{key[0]}:{key[1]}": sorted(values)
        for key, values in biological_labels.items()
        if len(values) > 1
    }
    receptor_peptide_conflicts = {
        "|".join(key): sorted(values)
        for key, values in receptor_peptide_labels.items()
        if len(values) > 1
    }

    return {
        "schema_version": "immune_receptor_audit.v1",
        "record_count": len(rows),
        "unique_record_ids": len({record.record_id for record in rows}),
        "unique_biological_keys": len(biological_measurements),
        "family_counts": dict(sorted(families.items())),
        "source_counts": dict(sorted(sources.items())),
        "pack_counts": dict(sorted(packs.items())),
        "evidence_tier_counts": dict(sorted(tiers.items())),
        "flag_counts": dict(sorted(flags.items())),
        "training_eligible": sum(record.eligibility.training for record in rows),
        "core_eligible": sum(record.eligibility.core for record in rows),
        "domain_error_count": len(domain_errors),
        "domain_errors_sample": domain_errors[:100],
        "conflicting_biological_keys": len(conflicts),
        "conflicts_sample": dict(list(sorted(conflicts.items()))[:100]),
        "receptor_peptide_label_conflicts_ignoring_mhc": len(
            receptor_peptide_conflicts
        ),
        "receptor_peptide_conflicts_sample": dict(
            list(sorted(receptor_peptide_conflicts.items()))[:100]
        ),
        "biological_keys_with_multiple_measurements": sum(
            count > 1 for count in biological_measurements.values()
        ),
    }
