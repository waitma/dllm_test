"""Conservative result contracts. Run: python -m pytest scripts/tests/test_downstream_reporting.py.

Reports are derived artifacts: never write into a source result directory or
infer comparison eligibility from a model name or an evidence-layer label.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_FIELDS = ("dataset", "split", "input", "metric", "aggregation", "selection")
REPO = Path(__file__).resolve().parents[3]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_identity(path: str | Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def new_report_dir(path: str | Path | None, name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = Path(path) if path else REPO / "output" / "downstream_reports" / f"{stamp}_{name}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination.resolve()


def compare_protocols(local: dict | None, reference: dict | None) -> dict:
    """Require explicit, equal contracts on every axis before computing deltas."""
    local, reference = local or {}, reference or {}
    unknown, mismatched = [], []
    for field in PROTOCOL_FIELDS:
        left, right = local.get(field), reference.get(field)
        if left in (None, "", "unknown", {}, []) or right in (None, "", "unknown", {}, []):
            unknown.append(field)
        elif left != right:
            mismatched.append(field)
    allowed = not unknown and not mismatched
    return {
        "protocol_alignment": "mismatched" if mismatched else "unknown" if unknown else "exact",
        "comparison_allowed": allowed,
        "unknown_fields": unknown,
        "mismatched_fields": mismatched,
        "rank_eligible": allowed,
    }


def write_report_manifest(destination: Path, inputs: list[Path], *, policy: str) -> None:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": policy,
        "inputs": [file_identity(p) for p in sorted(set(inputs)) if p.is_file()],
        "outputs": [file_identity(p) for p in sorted(destination.iterdir()) if p.is_file()],
    }
    (destination / "report_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
