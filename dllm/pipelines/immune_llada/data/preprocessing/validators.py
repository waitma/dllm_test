"""Validation helpers for the trusted semantic dataset boundary.

Raw rows are validated by source adapters. Prepared rows are validated once at
write time and again with a cheap shape check when loaded; neither path performs
training-time filtering or sample replacement.
"""

from __future__ import annotations

from typing import Any

from ..records import BioSeqRecord

SCHEMA_VERSION = "immune_llada.semantic.v1"


def validate_prepared_row(row: dict[str, Any]) -> BioSeqRecord:
    """Raise immediately if a prepared row cannot be restored losslessly."""
    record = BioSeqRecord.from_prepared_dict(row)
    if row.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported prepared schema: {row.get('schema_version')!r}")
    return record


def add_schema_version(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with the dataset schema marker used by manifests/loaders."""
    result = dict(row)
    result["schema_version"] = SCHEMA_VERSION
    return result


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset manifest schema: {manifest.get('schema_version')!r}")
    if manifest.get("format") != "jsonl":
        raise ValueError(f"Unsupported prepared dataset format: {manifest.get('format')!r}")
    if not isinstance(manifest.get("splits"), dict):
        raise ValueError("Dataset manifest requires a 'splits' mapping")


__all__ = ["SCHEMA_VERSION", "add_schema_version", "validate_manifest", "validate_prepared_row"]
