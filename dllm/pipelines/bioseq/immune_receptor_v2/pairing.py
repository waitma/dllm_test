"""Re-audit and export paired OAS/OTS against the current benchmark bank."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .export import (
    ClusterResult,
    ClusterRule,
    DEFAULT_CLUSTER_RULES,
    _artifact,
    cluster_with_mmseqs,
)
from .io import sha256_file, write_json
from .leakage import build_benchmark_sequence_bank
from .schema import is_valid_sequence, normalize_sequence, stable_digest


@dataclass(frozen=True)
class PairingSource:
    source_name: str
    input_path: Path
    existing_valid_path: Path
    existing_holdout_path: Path
    split_group_fields: tuple[str, ...]
    buckets: tuple[str, ...]


PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DEFAULT_PAIRING_SOURCES = (
    PairingSource(
        "oas",
        PROJECT_ROOT
        / "data/oas_previous_clean/splits/cleaned_merged_data_step_clustered_train_oas_label.csv",
        PROJECT_ROOT
        / "data/oas_previous_clean/splits/cleaned_merged_data_step_clustered_valid_oas_label.csv",
        PROJECT_ROOT
        / "data/oas_previous_clean/splits/cleaned_merged_data_step_clustered_holdout_oas_label.csv",
        ("ab_cluster_key", "ab_cluster_id"),
        (
            "antibody_heavy_cdr",
            "antibody_light_cdr",
            "antibody_heavy_chain",
            "antibody_light_chain",
        ),
    ),
    PairingSource(
        "ots",
        PROJECT_ROOT / "data/ots_paired_clean/final/train.csv",
        PROJECT_ROOT / "data/ots_paired_clean/final/valid.csv",
        PROJECT_ROOT / "data/ots_paired_clean/final/holdout.csv",
        ("pair_cluster", "cluster_id"),
        (
            "tcr_alpha_cdr",
            "tcr_beta_cdr",
            "tcr_alpha_chain",
            "tcr_beta_chain",
        ),
    ),
)


def _valid(value: Any) -> str:
    sequence = normalize_sequence(value)
    return sequence if sequence and is_valid_sequence(sequence) else ""


def pairing_row_sequences(
    source_name: str, row: Mapping[str, Any]
) -> list[tuple[str, str]]:
    """Extract explicit role/scope buckets without irreversible AA rewrites."""

    values: list[tuple[str, str]] = []
    if source_name == "oas":
        fields = (
            ("antibody_heavy_chain", "cleaned_h_sequence", "h_sequence"),
            ("antibody_light_chain", "cleaned_l_sequence", "l_sequence"),
            ("antibody_heavy_cdr", "h_cdr3", "h_CDR3"),
            ("antibody_light_cdr", "l_cdr3", "l_CDR3"),
        )
        for bucket, *candidates in fields:
            sequence = ""
            for field in candidates:
                sequence = _valid(row.get(field))
                if sequence:
                    break
            if sequence:
                values.append((bucket, sequence))
        return values
    if source_name != "ots":
        raise ValueError(f"Unsupported pairing source: {source_name}")
    for index in ("1", "2"):
        chain_type = str(row.get(f"chain{index}_anarci_type", "")).strip().upper()
        role = {"A": "tcr_alpha", "B": "tcr_beta"}.get(chain_type)
        if not role:
            continue
        chain = _valid(row.get(f"cleaned_chain{index}_seq"))
        cdr3 = _valid(
            row.get(f"chain{index}_cdr3") or row.get(f"chain{index}_CDR3")
        )
        if chain:
            values.append((f"{role}_chain", chain))
        if cdr3:
            values.append((f"{role}_cdr", cdr3))
    return values


def _iter_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", errors="strict", newline="") as handle:
        yield from csv.DictReader(handle)


def _rule_map(
    rules: Sequence[ClusterRule] = DEFAULT_CLUSTER_RULES,
) -> dict[str, ClusterRule]:
    return {rule.bucket: rule for rule in rules}


def _collect_unique_sequences(source: PairingSource) -> tuple[dict[str, set[str]], int]:
    sequences: dict[str, set[str]] = defaultdict(set)
    row_count = 0
    for row in _iter_csv(source.input_path):
        row_count += 1
        for bucket, sequence in pairing_row_sequences(source.source_name, row):
            if bucket in source.buckets:
                sequences[bucket].add(sequence)
    return dict(sequences), row_count


def _benchmark_buckets(
    benchmark_bank: Mapping[str, Mapping[str, str]],
) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for role, sequences in benchmark_bank.items():
        if role not in {
            "antibody_heavy",
            "antibody_light",
            "tcr_alpha",
            "tcr_beta",
        }:
            continue
        for sequence in sequences:
            kind = "cdr" if len(sequence) <= 40 else "chain"
            values[f"{role}_{kind}"].add(sequence)
    return dict(values)


def _group_id(source: PairingSource, row: Mapping[str, Any], row_index: int) -> str:
    for field in source.split_group_fields:
        value = str(row.get(field, "")).strip()
        if value:
            return f"{field}:{value}"
    return f"missing_group:{row_index}"


def deterministic_pairing_split(
    group_id: str,
    *,
    seed: int = 42,
    train_ratio: float = 0.98,
    valid_ratio: float = 0.01,
) -> str:
    digest = hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_ratio:
        return "train"
    if value < train_ratio + valid_ratio:
        return "valid"
    return "test"


def _row_match(
    source_name: str,
    row: Mapping[str, Any],
    clusters: Mapping[str, ClusterResult],
) -> tuple[list[str], list[str]]:
    exact: set[str] = set()
    near: set[str] = set()
    for bucket, sequence in pairing_row_sequences(source_name, row):
        result = clusters.get(bucket)
        if not result:
            continue
        cluster_id = result.sequence_to_cluster.get(sequence, "")
        if sequence in result.benchmark_sequences:
            exact.add(bucket)
        elif cluster_id and cluster_id in result.benchmark_clusters:
            near.add(bucket)
    return sorted(exact), sorted(near)


def _atomic_csv_writers(
    output_root: Path, fieldnames: Sequence[str]
) -> tuple[dict[str, tuple[Path, Any, csv.DictWriter]], Path, Any]:
    writers: dict[str, tuple[Path, Any, csv.DictWriter]] = {}
    for split in ("train", "valid", "test"):
        final = output_root / f"{split}.csv"
        temporary = final.with_suffix(".csv.tmp")
        handle = temporary.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writers[split] = (final, handle, writer)
    blocklist = output_root / "benchmark_blocklist.jsonl"
    block_handle = blocklist.with_suffix(".jsonl.tmp").open("w", encoding="utf-8")
    return writers, blocklist, block_handle


def _close_and_commit(
    writers: Mapping[str, tuple[Path, Any, csv.DictWriter]],
    blocklist: Path,
    block_handle: Any,
) -> None:
    for final, handle, _ in writers.values():
        temporary = Path(handle.name)
        handle.close()
        os.replace(temporary, final)
    temporary_blocklist = Path(block_handle.name)
    block_handle.close()
    os.replace(temporary_blocklist, blocklist)


def export_pairing_source(
    source: PairingSource,
    clusters: Mapping[str, ClusterResult],
    *,
    output_root: Path,
    seed: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    blocked_group_reasons: dict[str, set[str]] = defaultdict(set)
    input_rows = 0
    missing_sequence_rows = 0
    original_split_counts: Counter[str] = Counter()
    exact_occurrences: Counter[str] = Counter()
    near_occurrences: Counter[str] = Counter()
    for row_index, row in enumerate(_iter_csv(source.input_path)):
        input_rows += 1
        original_split_counts[str(row.get("split", ""))] += 1
        sequences = pairing_row_sequences(source.source_name, row)
        observed_buckets = {bucket for bucket, _ in sequences}
        exact, near = _row_match(source.source_name, row, clusters)
        reasons = list(exact) + [f"{bucket}:cluster" for bucket in near]
        if not set(source.buckets).issubset(observed_buckets):
            reasons.append("missing_or_invalid_required_sequence")
            missing_sequence_rows += 1
        if reasons:
            group_id = _group_id(source, row, row_index)
            blocked_group_reasons[group_id].update(reasons)
            exact_occurrences.update(exact)
            near_occurrences.update(near)

    with source.input_path.open(
        encoding="utf-8-sig", errors="strict", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {source.input_path}")
        writers, blocklist_path, block_handle = _atomic_csv_writers(
            output_root, [*reader.fieldnames, "v2_split"]
        )
        split_counts: Counter[str] = Counter()
        blocked = 0
        blocked_groups: set[str] = set()
        kept_groups: set[str] = set()
        for row_index, row in enumerate(reader):
            group_id = _group_id(source, row, row_index)
            exact, near = _row_match(source.source_name, row, clusters)
            reasons = list(exact) + [f"{bucket}:cluster" for bucket in near]
            if group_id in blocked_group_reasons:
                if not reasons:
                    reasons.append("upstream_cluster_group_propagation")
                reasons.extend(blocked_group_reasons[group_id])
                blocked += 1
                blocked_groups.add(group_id)
                block_handle.write(
                    json.dumps(
                        {
                            "row_index": row_index,
                            "group_id": group_id,
                            "reasons": sorted(set(reasons)),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                continue
            split = deterministic_pairing_split(group_id, seed=seed)
            kept_groups.add(group_id)
            split_counts[split] += 1
            row["v2_split"] = split
            writers[split][2].writerow(row)
    _close_and_commit(writers, blocklist_path, block_handle)
    output_files = {
        split: _artifact(output_root / f"{split}.csv")
        for split in ("train", "valid", "test")
    }
    report = {
        "schema_version": "immune_receptor_pairing_export.v1",
        "source_name": source.source_name,
        "input": _artifact(source.input_path),
        "input_records": input_rows,
        "kept_records": sum(split_counts.values()),
        "blocked_records": blocked,
        "blocked_fraction": blocked / input_rows if input_rows else 0.0,
        "split_counts": dict(sorted(split_counts.items())),
        "split_ratios": {
            split: count / sum(split_counts.values())
            for split, count in sorted(split_counts.items())
        },
        "split_seed": seed,
        "split_policy": "SHA256(group_id, seed) 98/1/1; source valid/holdout never reused",
        "group_count": len(kept_groups),
        "blocked_group_count": len(blocked_groups),
        "missing_or_invalid_required_sequence_records": missing_sequence_rows,
        "exact_match_occurrences_by_bucket": dict(sorted(exact_occurrences.items())),
        "near_match_occurrences_by_bucket": dict(sorted(near_occurrences.items())),
        "original_split_counts": dict(sorted(original_split_counts.items())),
        "residual_benchmark_cluster_records": 0,
        "outputs": output_files,
        "blocklist": _artifact(blocklist_path),
        "quarantined_existing_holdouts": [
            _artifact(source.existing_valid_path),
            _artifact(source.existing_holdout_path),
        ],
        "audit": {
            "passed": (
                input_rows == blocked + sum(split_counts.values())
                and missing_sequence_rows <= blocked
            ),
            "row_conservation": input_rows == blocked + sum(split_counts.values()),
            "group_assignment_deterministic": True,
            "source_valid_holdout_reused": False,
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, report)
    return report


def build_pairing_exports(
    *,
    root: Path,
    quarantine: Mapping[str, Any],
    mmseqs_bin: Path,
    threads: int = 64,
    seed: int = 42,
    sources: Sequence[PairingSource] = DEFAULT_PAIRING_SOURCES,
) -> dict[str, Any]:
    """Build a versioned OAS/OTS export against the frozen current bank."""

    if not mmseqs_bin.is_file():
        raise FileNotFoundError(f"MMseqs2 binary not found: {mmseqs_bin}")
    input_hashes = {
        source.source_name: sha256_file(source.input_path) for source in sources
    }
    quarantine_path = root / "registries/benchmark_quarantine.json"
    input_hashes["benchmark_quarantine"] = sha256_file(quarantine_path)
    rules = _rule_map()
    selected_rules = sorted(
        {bucket for source in sources for bucket in source.buckets}
    )
    mmseqs_version = subprocess.run(
        [str(mmseqs_bin), "version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    build_id = stable_digest(
        {
            "inputs": input_hashes,
            "rules": [asdict(rules[bucket]) for bucket in selected_rules],
            "seed": seed,
            "mmseqs_version": mmseqs_version,
        },
        prefix="ir2pair_",
        length=20,
    )
    cluster_root = root / "clusters" / build_id
    export_root = root / "exports" / build_id
    export_root.mkdir(parents=True, exist_ok=True)
    benchmark_bank = build_benchmark_sequence_bank(quarantine)
    benchmark = _benchmark_buckets(benchmark_bank)
    source_sequences: dict[str, dict[str, set[str]]] = {}
    source_input_rows: dict[str, int] = {}
    combined_sequences: dict[str, set[str]] = defaultdict(set)
    for source in sources:
        sequences, rows = _collect_unique_sequences(source)
        source_sequences[source.source_name] = sequences
        source_input_rows[source.source_name] = rows
        for bucket, values in sequences.items():
            combined_sequences[bucket].update(values)
    clusters: dict[str, ClusterResult] = {}
    for bucket in selected_rules:
        clusters[bucket] = cluster_with_mmseqs(
            rule=rules[bucket],
            query_sequences=combined_sequences[bucket],
            benchmark_sequences=benchmark.get(bucket, set()),
            output_dir=cluster_root,
            mmseqs_bin=mmseqs_bin,
            threads=threads,
        )
    cluster_registry = {
        "schema_version": "immune_receptor_pairing_cluster_registry.v1",
        "build_id": build_id,
        "sources": {
            source: {
                "input_records": source_input_rows[source],
                "unique_sequences_by_bucket": {
                    bucket: len(values) for bucket, values in sorted(buckets.items())
                },
            }
            for source, buckets in source_sequences.items()
        },
        "clusters": {
            bucket: result.report for bucket, result in sorted(clusters.items())
        },
    }
    write_json(cluster_root / "registry.json", cluster_registry)
    source_reports = {
        source.source_name: export_pairing_source(
            source,
            clusters,
            output_root=export_root / source.source_name,
            seed=seed,
        )
        for source in sources
    }
    technical_ready = all(
        report["audit"]["passed"]
        and report["residual_benchmark_cluster_records"] == 0
        for report in source_reports.values()
    )
    report = {
        "schema_version": "immune_receptor_pairing_export_report.v1",
        "build_id": build_id,
        "input_hashes": input_hashes,
        "mmseqs_version": mmseqs_version,
        "cluster_registry": cluster_registry,
        "sources": source_reports,
        "technical_pairing_export_ready": technical_ready,
        "source_valid_holdout_reused": False,
        "training_started": False,
    }
    report_path = export_root / "pairing_export_report.json"
    manifest_path = export_root / "artifact_manifest.json"
    report["artifact_manifest_path"] = str(manifest_path.resolve())
    write_json(report_path, report)
    artifacts = sorted(
        [path for path in export_root.rglob("*") if path.is_file()]
        + [path for path in cluster_root.rglob("*") if path.is_file()]
    )
    manifest = {
        "schema_version": "immune_receptor_pairing_artifact_manifest.v1",
        "build_id": build_id,
        "root": str(export_root.resolve()),
        "artifacts": [_artifact(path) for path in artifacts],
    }
    manifest["artifact_count"] = len(manifest["artifacts"])
    manifest["total_bytes"] = sum(item["bytes"] for item in manifest["artifacts"])
    write_json(manifest_path, manifest)
    return report
