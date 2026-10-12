"""Shared provenance labels for downstream baseline results.

Run the repository-wide audit after changing these labels::

    python downstream/benchmark/scripts/audit_baseline_provenance.py

Baseline numbers must state both *where the number came from* and whether the
local protocol is comparable to the anchor paper.  These fields deliberately do
not infer provenance from a model name such as ``official``.
"""

from __future__ import annotations

from typing import Any


PAPER_REPORTED = "paper_reported"
OFFICIAL_ARTIFACT_RESCORED = "official_artifact_rescored"
OFFICIAL_CODE_RERUN = "official_code_rerun"
LOCAL_REIMPLEMENTATION = "local_reimplementation"
LOCAL_CONTROL = "local_control"

EVIDENCE_TYPES = {
    PAPER_REPORTED,
    OFFICIAL_ARTIFACT_RESCORED,
    OFFICIAL_CODE_RERUN,
    LOCAL_REIMPLEMENTATION,
    LOCAL_CONTROL,
}

EXACT = "exact"
COMPATIBLE = "compatible"
MISMATCHED = "mismatched"
NOT_COMPARABLE = "not_comparable"

ALIGNMENTS = {EXACT, COMPATIBLE, MISMATCHED, NOT_COMPARABLE, "unknown"}


def baseline_provenance(
    *,
    evidence_type: str,
    protocol_alignment: str,
    citation: str,
    source_location: str,
    notes: str = "",
) -> dict[str, Any]:
    """Build and validate the provenance block stored in metrics/summary JSON."""

    if evidence_type not in EVIDENCE_TYPES:
        raise ValueError(f"unknown evidence_type={evidence_type!r}")
    if protocol_alignment not in ALIGNMENTS:
        raise ValueError(f"unknown protocol_alignment={protocol_alignment!r}")
    if not citation:
        raise ValueError("baseline citation must be non-empty")
    if not source_location:
        raise ValueError("baseline source_location must be non-empty")
    return {
        "evidence_type": evidence_type,
        "protocol_alignment": protocol_alignment,
        "paper_comparable": protocol_alignment in {EXACT, COMPATIBLE},
        "citation": citation,
        "source_location": source_location,
        "notes": notes,
    }


def validate_provenance(payload: dict[str, Any]) -> list[str]:
    """Return validation errors for a metrics payload instead of raising."""

    prov = payload.get("baseline_provenance")
    if not isinstance(prov, dict):
        return ["missing baseline_provenance object"]
    errors = []
    if prov.get("evidence_type") not in EVIDENCE_TYPES:
        errors.append(f"invalid evidence_type={prov.get('evidence_type')!r}")
    if prov.get("protocol_alignment") not in ALIGNMENTS:
        errors.append(f"invalid protocol_alignment={prov.get('protocol_alignment')!r}")
    if not prov.get("citation"):
        errors.append("missing citation")
    if not prov.get("source_location"):
        errors.append("missing source_location")
    return errors
