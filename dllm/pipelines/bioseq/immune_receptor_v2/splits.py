"""Deterministic group-disjoint split manifests for immune-receptor records.

Run unit tests with::

    conda run -n pllm python -m pytest \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_immune_receptor_v2.py -q
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from .schema import CanonicalRecord, stable_digest


@dataclass(frozen=True)
class SplitProtocol:
    protocol_id: str
    group_fields: tuple[str, ...]
    ratios: tuple[float, float, float] = (0.9, 0.05, 0.05)
    split_names: tuple[str, str, str] = ("train", "valid", "test")
    seed: int = 42
    description: str = ""

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("protocol_id is required")
        if not self.group_fields:
            raise ValueError("At least one group field is required")
        if len(self.ratios) != len(self.split_names):
            raise ValueError("ratios and split_names must have the same length")
        if any(ratio < 0 for ratio in self.ratios):
            raise ValueError("split ratios cannot be negative")
        if abs(sum(self.ratios) - 1.0) > 1e-8:
            raise ValueError("split ratios must sum to one")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def _components(
    records: Sequence[CanonicalRecord], group_fields: Sequence[str]
) -> list[list[int]]:
    union_find = _UnionFind(len(records))
    first_seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for field_name in group_fields:
            group_value = record.groups.get(field_name, "")
            if not group_value:
                group_value = stable_digest(
                    [record.record_id, field_name], prefix="singleton_"
                )
            key = (field_name, group_value)
            if key in first_seen:
                union_find.union(index, first_seen[key])
            else:
                first_seen[key] = index
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[union_find.find(index)].append(index)
    return list(grouped.values())


def build_split_manifest(
    records: Iterable[CanonicalRecord], protocol: SplitProtocol
) -> dict:
    """Assign connected group components without using row order or Python hash."""

    rows = list(records)
    if not rows:
        raise ValueError("Cannot split an empty record collection")
    components = _components(rows, protocol.group_fields)
    components.sort(
        key=lambda component: (
            -len(component),
            stable_digest(
                [protocol.seed, sorted(rows[index].record_id for index in component)],
                length=64,
            ),
        )
    )
    target_counts = {
        split: len(rows) * ratio
        for split, ratio in zip(protocol.split_names, protocol.ratios)
    }
    current_counts = Counter({split: 0 for split in protocol.split_names})
    assignments: dict[str, str] = {}
    component_rows: list[dict] = []
    for component in components:
        component_record_ids = sorted(rows[index].record_id for index in component)
        tie_order = list(protocol.split_names)
        split = max(
            tie_order,
            key=lambda name: (
                target_counts[name] - current_counts[name],
                -current_counts[name],
                -tie_order.index(name),
            ),
        )
        component_id = stable_digest(
            component_record_ids, prefix="cmp_"
        )
        for index in component:
            assignments[rows[index].record_id] = split
        current_counts[split] += len(component)
        component_rows.append(
            {"component_id": component_id, "split": split, "size": len(component)}
        )
    assignment_rows = [
        {
            "record_id": record.record_id,
            "biological_key": record.biological_key,
            "split": assignments[record.record_id],
            "groups": {
                field_name: record.groups.get(field_name, "")
                for field_name in protocol.group_fields
            },
        }
        for record in sorted(rows, key=lambda item: item.record_id)
    ]
    result = {
        "schema_version": "immune_receptor_split.v1",
        "protocol": asdict(protocol),
        "record_count": len(rows),
        "component_count": len(components),
        "split_counts": dict(sorted(current_counts.items())),
        "assignments": assignment_rows,
        "components": component_rows,
    }
    result["audit"] = verify_split_manifest(rows, result)
    actual_ratios = {
        split: current_counts[split] / len(rows) for split in protocol.split_names
    }
    target_ratios = dict(zip(protocol.split_names, protocol.ratios))
    result["balance"] = {
        "target_ratios": target_ratios,
        "actual_ratios": actual_ratios,
        "max_absolute_ratio_deviation": max(
            abs(actual_ratios[split] - target_ratios[split])
            for split in protocol.split_names
        ),
        "largest_component_records": max(len(component) for component in components),
        "largest_component_fraction": max(len(component) for component in components)
        / len(rows),
        "warning": (
            "Large connected components can make the requested ratio mathematically "
            "unattainable; group disjointness takes precedence over ratio balance."
        ),
    }
    return result


def verify_split_manifest(
    records: Iterable[CanonicalRecord], manifest: Mapping
) -> dict:
    rows = list(records)
    protocol = manifest["protocol"]
    group_fields = tuple(protocol["group_fields"])
    assignment = {
        row["record_id"]: row["split"] for row in manifest.get("assignments", [])
    }
    missing = sorted(record.record_id for record in rows if record.record_id not in assignment)
    unknown = sorted(set(assignment) - {record.record_id for record in rows})
    leakage: dict[str, dict[str, list[str]]] = {}
    for field_name in group_fields:
        seen: dict[str, set[str]] = defaultdict(set)
        for record in rows:
            value = record.groups.get(field_name, "")
            if value and record.record_id in assignment:
                seen[value].add(assignment[record.record_id])
        leaked = {
            value: sorted(splits)
            for value, splits in seen.items()
            if len(splits) > 1
        }
        if leaked:
            leakage[field_name] = leaked
    return {
        "passed": not missing and not unknown and not leakage,
        "missing_record_ids": missing,
        "unknown_record_ids": unknown,
        "group_leakage": leakage,
    }
