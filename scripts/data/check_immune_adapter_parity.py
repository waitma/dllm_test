"""Compare legacy and canonical immune source adapters on sampled raw CSV rows.

This is an audit tool, not a training dependency. It deliberately reads raw CSV
only during an offline migration check and never creates model-ready token data.

Example::

    python scripts/data/check_immune_adapter_parity.py \
        --split train --max-rows 5000 \
        --output refactor_baseline/immune_adapter_parity_train.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dllm.pipelines.immune_llada.data.records import BioSeqRecord
from scripts.data.run_immune_full_parity import BASELINE_COMMIT, load_legacy
from dllm.pipelines.immune_llada.data.registry import (
    source_spec,
    source_split_path,
)
from dllm.pipelines.immune_llada.data.sources import row_to_record


def legacy_adapters(legacy: Any) -> dict[str, Callable[[dict[str, Any]], dict[str, Any] | None]]:
    """Return adapters imported exclusively from the pinned historical blob."""
    datasets = legacy.datasets
    return {
        "oas": datasets.oas_paired_row_to_record,
        "ots": datasets.ots_paired_row_to_record,
        "asd_antibody": datasets.asd_antibody_row_to_record,
        "asd_nanobody": datasets.asd_nanobody_row_to_record,
        "trait": datasets.trait_row_to_record,
        "tcr_native": datasets.tcr_native_row_to_record,
        "tcr_papers": datasets.tcr_native_row_to_record,
        "tcr_repertoire": datasets.tcr_repertoire_row_to_record,
    }


def _canonical_signature(record: BioSeqRecord) -> dict[str, Any]:
    return {
        "chains": record.sequences,
        "task_type": record.task_type,
        "source": record.source,
        "roles": record.chain_roles,
        "relation": record.labels.get("relation"),
    }


def _legacy_signature(source: str, record: dict[str, Any]) -> dict[str, Any]:
    # Legacy OAS/OTS adapters did not emit roles or relation labels. For those
    # sources, compare exactly the information the old loader exposed.
    return {
        "chains": record.get("chains"),
        "task_type": record.get("task_type"),
        "source": record.get("source"),
        "roles": record.get("roles"),
        "relation": record.get("relation"),
    }


def _compare_signatures(source: str, legacy: dict[str, Any], canonical: BioSeqRecord) -> list[str]:
    old = _legacy_signature(source, legacy)
    new = _canonical_signature(canonical)
    differences: list[str] = []
    for field in ("chains", "task_type", "source"):
        if old[field] != new[field]:
            differences.append(field)
    if old["roles"] is not None and old["roles"] != new["roles"]:
        differences.append("roles")
    if old["relation"] is not None and old["relation"] != new["relation"]:
        differences.append("relation")
    return differences


def _sample_source(
    source: str,
    split: str,
    max_rows: int,
    mismatch_limit: int,
    legacy: Any,
) -> dict[str, Any]:
    path = source_split_path(source_spec(source), split)
    if not path.is_file():
        raise FileNotFoundError(f"raw source file not found: {path}")

    stats: Counter[str] = Counter()
    mismatch_fields: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    legacy_adapter = legacy_adapters(legacy)[source]

    truncated = False
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            if stats["rows_read"] >= max_rows:
                truncated = True
                break
            stats["rows_read"] += 1
            try:
                old_record = legacy_adapter(row)
            except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError, csv.Error) as exc:  # pragma: no cover - audit diagnostics
                stats["legacy_errors"] += 1
                if len(examples) < mismatch_limit:
                    examples.append({"row": row_number, "kind": "legacy_error", "error": repr(exc)})
                continue
            try:
                new_record = row_to_record(source, row, split=split)
            except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError, csv.Error) as exc:  # pragma: no cover - audit diagnostics
                stats["canonical_errors"] += 1
                if len(examples) < mismatch_limit:
                    examples.append({"row": row_number, "kind": "canonical_error", "error": repr(exc)})
                continue

            if old_record is None and new_record is None:
                stats["both_dropped"] += 1
                continue
            if old_record is None:
                stats["legacy_only_drop"] += 1
                if len(examples) < mismatch_limit:
                    examples.append({"row": row_number, "kind": "legacy_only_drop"})
                continue
            if new_record is None:
                stats["canonical_only_drop"] += 1
                if len(examples) < mismatch_limit:
                    examples.append({"row": row_number, "kind": "canonical_only_drop"})
                continue

            stats["both_kept"] += 1
            fields = _compare_signatures(source, old_record, new_record)
            if not fields:
                stats["matching_kept"] += 1
                continue
            stats["mismatches"] += 1
            mismatch_fields.update(fields)
            if len(examples) < mismatch_limit:
                examples.append(
                    {
                        "row": row_number,
                        "kind": "kept_mismatch",
                        "fields": fields,
                        "legacy": _legacy_signature(source, old_record),
                        "canonical": _canonical_signature(new_record),
                    }
                )

    return {
        "source": source,
        "split": split,
        "path": str(path),
        "stats": dict(stats),
        "truncated": truncated,
        "mismatch_fields": dict(mismatch_fields),
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=5000, help="Raw rows per source; use a positive value")
    parser.add_argument("--mismatch-limit", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.mismatch_limit < 0:
        raise ValueError("--mismatch-limit must not be negative")
    sources = [item.strip() for item in args.source.split("+") if item.strip()]
    if not sources:
        raise ValueError("--source must contain at least one source")
    supported_sources = {
        "oas", "ots", "asd_antibody", "asd_nanobody", "trait",
        "tcr_native", "tcr_papers", "tcr_repertoire",
    }
    unknown = sorted(set(sources) - supported_sources)
    if unknown:
        raise ValueError(f"Unsupported parity sources: {unknown}")

    with tempfile.TemporaryDirectory(prefix="immune-parity-legacy-") as parent:
        snapshot = Path(parent) / "snapshot"
        legacy = load_legacy(snapshot, BASELINE_COMMIT)
        result = {
            "tool": "check_immune_adapter_parity",
            "split": args.split,
            "max_rows_per_source": args.max_rows,
            "baseline": {
                "commit": BASELINE_COMMIT,
                "legacy_blob_sha256": legacy.hashes,
                "snapshot": "temporary (not retained)",
            },
            "sources": [
                _sample_source(source, args.split, args.max_rows, args.mismatch_limit, legacy)
                for source in sources
            ],
        }
    result["summary"] = dict(
        Counter(
            key
            for source_result in result["sources"]
            for key, value in source_result["stats"].items()
            for _ in range(value)
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"parity report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
