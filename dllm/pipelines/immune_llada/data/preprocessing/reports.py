"""Serializable statistics emitted by immune preprocessing."""

from __future__ import annotations

from collections import Counter
from typing import Any

# Drop / schema counts stay in ``totals()``. These three are kept-record
# transformations, not first-failure drops: they live on each source entry
# and are summed separately so ``count_immune_drops`` keeps comparing the
# original six keys.
AUDIT_COUNT_KEYS = ("downgraded_all_x_mhc", "beta_only_completed", "alpha_only_completed")


def new_source_stats(source: str, split: str, path: str) -> dict[str, Any]:
    return {
        "source": source,
        "split": split,
        "path": path,
        "raw_rows": 0,
        "converted_rows": 0,
        "kept_rows": 0,
        "dropped_schema": 0,
        "dropped_filters": 0,
        "filter_reasons": {},
        "filter_names": [],
        "downgraded_all_x_mhc": 0,
        "beta_only_completed": 0,
        "alpha_only_completed": 0,
        "errors": 0,
        "shards": [],
    }


def add_filter_drop(stats: dict[str, Any], reason: str) -> None:
    stats["dropped_filters"] += 1
    reasons = Counter(stats["filter_reasons"])
    reasons[reason] += 1
    stats["filter_reasons"] = dict(sorted(reasons.items()))


def totals(source_stats: list[dict[str, Any]]) -> dict[str, int]:
    keys = ("raw_rows", "converted_rows", "kept_rows", "dropped_schema", "dropped_filters", "errors")
    return {key: sum(int(item.get(key, 0)) for item in source_stats) for key in keys}


def audit_totals(source_stats: list[dict[str, Any]]) -> dict[str, int]:
    """Sum the kept-record MHC-strip and partner-completion counters."""
    return {key: sum(int(item.get(key, 0)) for item in source_stats) for key in AUDIT_COUNT_KEYS}


__all__ = ["AUDIT_COUNT_KEYS", "add_filter_drop", "audit_totals", "new_source_stats", "totals"]
