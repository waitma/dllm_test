"""Cluster-aware benchmark decontamination and immutable core exports.

This module deliberately exports only high-confidence ``eligibility.core``
records. Auxiliary, proxy, pool, reconstructed, and source-released synthetic
negative packs remain canonical candidates until a training recipe explicitly
opts into them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import sha256_file, write_json, write_jsonl
from .leakage import audit_benchmark_leakage, build_benchmark_sequence_bank
from .schema import (
    CanonicalRecord,
    Eligibility,
    Evidence,
    Measurement,
    SourceReference,
    clean_text,
    is_valid_sequence,
    normalize_sequence,
    stable_digest,
    validate_domain_record,
)
from .splits import SplitProtocol, build_split_manifest


RECEPTOR_ROLES = {
    "tcr_alpha",
    "tcr_beta",
    "antibody_heavy",
    "antibody_light",
}


@dataclass(frozen=True)
class ClusterRule:
    bucket: str
    min_sequence_identity: float
    coverage: float
    coverage_mode: int = 0
    cluster_mode: int = 1
    rationale: str = ""


DEFAULT_CLUSTER_RULES = (
    ClusterRule(
        "antibody_heavy_cdr",
        0.70,
        0.80,
        rationale="CDR-H3 family-level quarantine used by the project data policy.",
    ),
    ClusterRule("antibody_light_cdr", 0.70, 0.80),
    ClusterRule(
        "antibody_heavy_chain",
        0.95,
        0.80,
        rationale="Near-identical antibody variable/full-chain quarantine.",
    ),
    ClusterRule("antibody_light_chain", 0.95, 0.80),
    ClusterRule(
        "tcr_alpha_cdr",
        0.80,
        0.80,
        rationale="Short CDR3 near-neighbor quarantine; stricter than exact clonotype.",
    ),
    ClusterRule("tcr_beta_cdr", 0.80, 0.80),
    ClusterRule("tcr_alpha_chain", 0.95, 0.80),
    ClusterRule("tcr_beta_chain", 0.95, 0.80),
    ClusterRule(
        "peptide",
        0.80,
        0.90,
        rationale="Epitope-family split/quarantine for unseen-peptide protocols.",
    ),
    ClusterRule(
        "antigen",
        0.80,
        0.80,
        rationale="Antigen-family split; benchmark quarantine applies when sequences exist.",
    ),
)


@dataclass
class ClusterResult:
    rule: ClusterRule
    sequence_to_cluster: dict[str, str]
    benchmark_sequences: set[str]
    benchmark_clusters: set[str]
    mapping_path: Path
    report_path: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class ExportProtocol:
    domain: str
    protocol_id: str
    group_fields: tuple[str, ...]
    description: str
    include_benchmark_peptide: bool = False
    include_benchmark_antigen: bool = False
    required_original_groups: tuple[str, ...] = ()
    record_family: str = ""


TCR_RECEPTOR_CLUSTER_FIELDS = (
    "tcr_alpha_cdr_cluster",
    "tcr_alpha_chain_cluster",
    "tcr_beta_cdr_cluster",
    "tcr_beta_chain_cluster",
)
ANTIBODY_RECEPTOR_CLUSTER_FIELDS = (
    "antibody_heavy_cdr_cluster",
    "antibody_heavy_chain_cluster",
    "antibody_light_cdr_cluster",
    "antibody_light_chain_cluster",
)


DEFAULT_EXPORT_PROTOCOLS = (
    ExportProtocol(
        "tcr",
        "tcr_receptor_cluster_disjoint_seen_peptide_v2",
        TCR_RECEPTOR_CLUSTER_FIELDS,
        "Receptor-cluster-disjoint split; peptide families may be shared.",
    ),
    ExportProtocol(
        "tcr",
        "tcr_unseen_peptide_cluster_v2",
        ("peptide_cluster",),
        "Peptide-family-disjoint split and peptide benchmark quarantine.",
        include_benchmark_peptide=True,
    ),
    ExportProtocol(
        "tcr",
        "tcr_unseen_pmhc_cluster_v2",
        ("pmhc_cluster",),
        "Peptide-family plus exact-MHC identity is disjoint across splits.",
        include_benchmark_peptide=True,
    ),
    ExportProtocol(
        "tcr",
        "tcr_study_holdout_decontaminated_v2",
        ("study",),
        "Study-disjoint diagnostic after receptor benchmark decontamination.",
        required_original_groups=("study",),
    ),
    ExportProtocol(
        "antibody",
        "antibody_cluster_disjoint_v2",
        ANTIBODY_RECEPTOR_CLUSTER_FIELDS,
        "Heavy/light sequence-cluster-disjoint antibody-antigen split.",
        record_family="antibody_antigen",
    ),
    ExportProtocol(
        "antibody",
        "antigen_cluster_disjoint_v2",
        ("antigen_cluster",),
        "Antigen-family-disjoint antibody-antigen split.",
        include_benchmark_antigen=True,
        record_family="antibody_antigen",
    ),
    ExportProtocol(
        "antibody",
        "antibody_antigen_cluster_joint_hard_v2",
        ANTIBODY_RECEPTOR_CLUSTER_FIELDS + ("antigen_cluster",),
        "Connected components across antibody-chain and antigen clusters.",
        include_benchmark_antigen=True,
        record_family="antibody_antigen",
    ),
    ExportProtocol(
        "antibody",
        "antibody_property_parent_decontaminated_v2",
        ("parent",),
        "Parent-disjoint intrinsic-property split after receptor quarantine.",
        required_original_groups=("parent",),
        record_family="antibody_property",
    ),
    ExportProtocol(
        "antibody",
        "antibody_property_study_decontaminated_v2",
        ("study",),
        "Study-disjoint intrinsic-property split after receptor quarantine.",
        required_original_groups=("study",),
        record_family="antibody_property",
    ),
)


def _scope_bucket(role: str, sequence: str, sequence_scope: str = "") -> str | None:
    if role in RECEPTOR_ROLES:
        scope = sequence_scope.lower()
        kind = "cdr" if "cdr3" in scope or len(sequence) <= 40 else "chain"
        return f"{role}_{kind}"
    if role in {"peptide", "antigen"}:
        return role
    return None


def record_cluster_sequences(record: CanonicalRecord) -> list[tuple[str, str]]:
    """Return unique ``(bucket, sequence)`` candidates, including annotated CDR3."""

    values: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entity in record.entities:
        bucket = _scope_bucket(entity.role, entity.sequence, entity.sequence_scope)
        if bucket and (bucket, entity.sequence) not in seen:
            seen.add((bucket, entity.sequence))
            values.append((bucket, entity.sequence))
        if entity.role not in RECEPTOR_ROLES:
            continue
        cdr3 = ""
        for key, value in entity.regions.items():
            if str(key).upper().replace("_", "") == "CDR3":
                cdr3 = normalize_sequence(value)
                break
        if cdr3 and is_valid_sequence(cdr3):
            cdr_bucket = f"{entity.role}_cdr"
            if (cdr_bucket, cdr3) not in seen:
                seen.add((cdr_bucket, cdr3))
                values.append((cdr_bucket, cdr3))
    return values


def benchmark_cluster_sequences(
    benchmark_bank: Mapping[str, Mapping[str, str]],
) -> dict[str, set[str]]:
    buckets: dict[str, set[str]] = defaultdict(set)
    for role, values in benchmark_bank.items():
        for sequence in values:
            bucket = _scope_bucket(role, sequence)
            if bucket:
                buckets[bucket].add(sequence)
    return dict(buckets)


def _cluster_input_digest(sequences: Sequence[str], benchmark: set[str]) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(b"B" if sequence in benchmark else b"Q")
        digest.update(b"\0")
        digest.update(sequence.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_cluster_id(bucket: str, sequences: Iterable[str]) -> str:
    digest = hashlib.sha256()
    digest.update(bucket.encode("utf-8"))
    digest.update(b"\0")
    for sequence in sorted(sequences):
        digest.update(sequence.encode("ascii"))
        digest.update(b"\n")
    return f"clu_{bucket}_{digest.hexdigest()[:24]}"


def _load_cluster_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            mapping[row["sequence"]] = row["cluster_id"]
    return mapping


def cluster_with_mmseqs(
    *,
    rule: ClusterRule,
    query_sequences: Iterable[str],
    benchmark_sequences: Iterable[str],
    output_dir: Path,
    mmseqs_bin: Path,
    threads: int,
) -> ClusterResult:
    """Cluster one role/scope bucket and cache it by content digest."""

    benchmark = set(benchmark_sequences)
    sequences = sorted(set(query_sequences) | benchmark)
    if not sequences:
        raise ValueError(f"No sequences for cluster bucket {rule.bucket}")
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / f"{rule.bucket}.tsv"
    report_path = output_dir / f"{rule.bucket}.json"
    input_digest = _cluster_input_digest(sequences, benchmark)
    rule_payload = asdict(rule)
    if mapping_path.exists() and report_path.exists():
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            prior.get("input_sha256") == input_digest
            and prior.get("rule") == rule_payload
            and prior.get("mapping_sha256") == sha256_file(mapping_path)
        ):
            mapping = _load_cluster_mapping(mapping_path)
            benchmark_clusters = {mapping[item] for item in benchmark if item in mapping}
            return ClusterResult(
                rule,
                mapping,
                benchmark,
                benchmark_clusters,
                mapping_path,
                report_path,
                prior,
            )

    temporary_root = Path(tempfile.mkdtemp(prefix=f"{rule.bucket}_", dir=output_dir))
    try:
        fasta = temporary_root / "input.fasta"
        identifiers = {f"s{index}": sequence for index, sequence in enumerate(sequences)}
        with fasta.open("w", encoding="ascii") as handle:
            for identifier, sequence in identifiers.items():
                handle.write(f">{identifier}\n{sequence}\n")
        prefix = temporary_root / "result"
        command = [
            str(mmseqs_bin),
            "easy-linclust",
            str(fasta),
            str(prefix),
            str(temporary_root / "scratch"),
            "--min-seq-id",
            str(rule.min_sequence_identity),
            "-c",
            str(rule.coverage),
            "--cov-mode",
            str(rule.coverage_mode),
            "--cluster-mode",
            str(rule.cluster_mode),
            "--threads",
            str(max(1, threads)),
            "--remove-tmp-files",
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        cluster_members: dict[str, set[str]] = {
            identifier: {identifier} for identifier in identifiers
        }
        member_to_representative: dict[str, str] = {
            identifier: identifier for identifier in identifiers
        }
        cluster_tsv = Path(str(prefix) + "_cluster.tsv")
        with cluster_tsv.open(encoding="utf-8") as handle:
            for line in handle:
                representative, member = line.rstrip("\n").split("\t")[:2]
                cluster_members.setdefault(representative, set()).add(member)
                member_to_representative[member] = representative
        sequence_to_cluster: dict[str, str] = {}
        stable_ids: dict[str, str] = {}
        for identifier, sequence in identifiers.items():
            representative = member_to_representative.get(identifier, identifier)
            members = cluster_members.get(representative, {identifier})
            if representative not in stable_ids:
                stable_ids[representative] = _stable_cluster_id(
                    rule.bucket,
                    (identifiers[member] for member in members),
                )
            sequence_to_cluster[sequence] = stable_ids[representative]
        temporary_mapping = mapping_path.with_suffix(".tsv.tmp")
        with temporary_mapping.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ("sequence", "cluster_id", "is_benchmark", "sequence_sha256")
            )
            for sequence in sequences:
                writer.writerow(
                    (
                        sequence,
                        sequence_to_cluster[sequence],
                        int(sequence in benchmark),
                        hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                    )
                )
        os.replace(temporary_mapping, mapping_path)
        benchmark_clusters = {
            sequence_to_cluster[sequence] for sequence in benchmark
        }
        report = {
            "schema_version": "immune_receptor_sequence_clusters.v1",
            "rule": rule_payload,
            "input_sha256": input_digest,
            "sequence_count": len(sequences),
            "query_sequence_count": len(set(query_sequences)),
            "benchmark_sequence_count": len(benchmark),
            "cluster_count": len(set(sequence_to_cluster.values())),
            "benchmark_connected_cluster_count": len(benchmark_clusters),
            "benchmark_connected_sequence_count": sum(
                cluster_id in benchmark_clusters
                for cluster_id in sequence_to_cluster.values()
            ),
            "mmseqs_binary": str(mmseqs_bin.resolve()),
            "mmseqs_version": subprocess.run(
                [str(mmseqs_bin), "version"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "command": command,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "mapping_path": str(mapping_path.resolve()),
            "mapping_sha256": sha256_file(mapping_path),
        }
        write_json(report_path, report)
        return ClusterResult(
            rule,
            sequence_to_cluster,
            benchmark,
            benchmark_clusters,
            mapping_path,
            report_path,
            report,
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def build_cluster_registry(
    records: Iterable[CanonicalRecord],
    benchmark_bank: Mapping[str, Mapping[str, str]],
    *,
    output_dir: Path,
    mmseqs_bin: Path,
    threads: int,
    rules: Sequence[ClusterRule] = DEFAULT_CLUSTER_RULES,
) -> tuple[dict[str, ClusterResult], dict[str, Any]]:
    query: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for bucket, sequence in record_cluster_sequences(record):
            query[bucket].add(sequence)
    benchmark = benchmark_cluster_sequences(benchmark_bank)
    results: dict[str, ClusterResult] = {}
    for rule in rules:
        if not query.get(rule.bucket):
            continue
        results[rule.bucket] = cluster_with_mmseqs(
            rule=rule,
            query_sequences=query[rule.bucket],
            benchmark_sequences=benchmark.get(rule.bucket, set()),
            output_dir=output_dir,
            mmseqs_bin=mmseqs_bin,
            threads=threads,
        )
    registry = {
        "schema_version": "immune_receptor_cluster_registry.v1",
        "rules": [result.report for result in results.values()],
        "bucket_count": len(results),
        "total_unique_bucket_sequences": sum(
            result.report["sequence_count"] for result in results.values()
        ),
    }
    write_json(output_dir / "registry.json", registry)
    return results, registry


def cluster_groups_for_record(
    record: CanonicalRecord, clusters: Mapping[str, ClusterResult]
) -> dict[str, str]:
    groups: dict[str, str] = {}
    for bucket, sequence in record_cluster_sequences(record):
        result = clusters.get(bucket)
        if result and sequence in result.sequence_to_cluster:
            groups[f"{bucket}_cluster"] = result.sequence_to_cluster[sequence]
    peptide_cluster = groups.get("peptide_cluster", "")
    if peptide_cluster:
        mhc = clean_text(record.context.get("mhc_allele"))
        if not mhc:
            entity = record.entity("mhc_alpha")
            mhc = entity.sequence if entity else ""
        if mhc:
            groups["pmhc_cluster"] = stable_digest(
                [peptide_cluster, mhc.upper().replace(" ", "")],
                prefix="pmhcc_",
            )
    return groups


def with_cluster_groups(
    record: CanonicalRecord, clusters: Mapping[str, ClusterResult]
) -> CanonicalRecord:
    groups = dict(record.groups)
    groups.update(cluster_groups_for_record(record, clusters))
    return replace(record, groups=groups, record_id="")


def benchmark_match_for_record(
    record: CanonicalRecord,
    clusters: Mapping[str, ClusterResult],
    *,
    include_peptide: bool,
    include_antigen: bool,
) -> dict[str, list[str]]:
    exact: set[str] = set()
    near: set[str] = set()
    cluster_ids: set[str] = set()
    for bucket, sequence in record_cluster_sequences(record):
        role = bucket.rsplit("_", 1)[0] if bucket.endswith(("_cdr", "_chain")) else bucket
        relevant = (
            role in RECEPTOR_ROLES
            or (bucket == "peptide" and include_peptide)
            or (bucket == "antigen" and include_antigen)
        )
        if not relevant:
            continue
        result = clusters.get(bucket)
        if not result:
            continue
        cluster_id = result.sequence_to_cluster.get(sequence, "")
        if sequence in result.benchmark_sequences:
            exact.add(bucket)
        elif cluster_id and cluster_id in result.benchmark_clusters:
            near.add(bucket)
        if cluster_id and cluster_id in result.benchmark_clusters:
            cluster_ids.add(cluster_id)
    return {
        "exact_buckets": sorted(exact),
        "near_buckets": sorted(near),
        "benchmark_cluster_ids": sorted(cluster_ids),
    }


def filter_benchmark_contamination(
    records: Iterable[CanonicalRecord],
    clusters: Mapping[str, ClusterResult],
    *,
    include_peptide: bool = False,
    include_antigen: bool = False,
    forced_exact_record_ids: Iterable[str] = (),
) -> tuple[list[CanonicalRecord], list[dict[str, Any]], dict[str, Any]]:
    forced = set(forced_exact_record_ids)
    kept: list[CanonicalRecord] = []
    blocked: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    pack_counts: Counter[str] = Counter()
    for record in records:
        match = benchmark_match_for_record(
            record,
            clusters,
            include_peptide=include_peptide,
            include_antigen=include_antigen,
        )
        reasons = list(match["exact_buckets"]) + [
            f"{bucket}:cluster" for bucket in match["near_buckets"]
        ]
        if record.record_id in forced:
            reasons.append("forced_exact_interaction_or_entity")
        if reasons:
            reason_counts.update(reasons)
            pack_counts[record.eligibility.pack] += 1
            blocked.append(
                {
                    "record_id": record.record_id,
                    "pack": record.eligibility.pack,
                    "reasons": sorted(set(reasons)),
                    **match,
                }
            )
        else:
            kept.append(with_cluster_groups(record, clusters))
    report = {
        "schema_version": "immune_receptor_decontamination.v1",
        "input_records": len(kept) + len(blocked),
        "kept_records": len(kept),
        "blocked_records": len(blocked),
        "blocked_fraction": (
            len(blocked) / (len(kept) + len(blocked))
            if kept or blocked
            else 0.0
        ),
        "blocked_reason_occurrences": dict(sorted(reason_counts.items())),
        "blocked_records_by_pack": dict(sorted(pack_counts.items())),
        "include_benchmark_peptide": include_peptide,
        "include_benchmark_antigen": include_antigen,
    }
    return kept, blocked, report


def _is_positive(record: CanonicalRecord) -> bool:
    label = record.measurement.label
    if isinstance(label, bool):
        return label
    if isinstance(label, (int, float)):
        return float(label) > 0
    return str(label).strip().lower() in {"1", "true", "positive", "pos", "yes"}


def _tcr_interaction_key(record: CanonicalRecord) -> tuple[str, str, str, str] | None:
    alpha = record.entity("tcr_alpha")
    beta = record.entity("tcr_beta")
    peptide = record.entity("peptide")
    if not alpha or not beta or not peptide:
        return None
    mhc = clean_text(record.context.get("mhc_allele"))
    if not mhc:
        mhc_entity = record.entity("mhc_alpha")
        mhc = mhc_entity.sequence if mhc_entity else ""
    return alpha.sequence, beta.sequence, peptide.sequence, mhc.upper().replace(" ", "")


def _synthetic_negative(
    receptor: CanonicalRecord,
    ligand: CanonicalRecord,
    *,
    protocol_id: str,
) -> CanonicalRecord:
    receptor_roles = {"tcr_alpha", "tcr_beta"}
    ligand_roles = {"peptide", "mhc_alpha", "mhc_beta2m"}
    entities = tuple(
        [entity for entity in receptor.entities if entity.role in receptor_roles]
        + [entity for entity in ligand.entities if entity.role in ligand_roles]
    )
    context = dict(ligand.context)
    context.update(
        {
            "negative_generation_protocol": protocol_id,
            "negative_receptor_parent": receptor.record_id,
            "negative_ligand_parent": ligand.record_id,
        }
    )
    source_id = stable_digest(
        [protocol_id, receptor.record_id, ligand.record_id],
        prefix="negsrc_",
        length=32,
    )
    return CanonicalRecord(
        family="tcr_pmhc",
        record_type="synthetic_negative_train_only",
        entities=entities,
        measurement=Measurement(
            "specificity",
            label=0,
            direction="higher",
            assay_type="in-silico train-only derangement",
            negative_type="train_only_receptor_pmhc_derangement",
            confidence="synthetic_low",
        ),
        source_records=(
            SourceReference(
                "immune_receptor_v2_export",
                source_id,
                source_version="decontaminated_v1",
                original_split="train",
                metadata={
                    "receptor_parent": receptor.record_id,
                    "ligand_parent": ligand.record_id,
                    "protocol_id": protocol_id,
                },
            ),
        ),
        evidence=Evidence(
            "D",
            "deterministic train-only receptor-pMHC derangement",
            False,
            notes="Potential false negatives remain isolated from measured negatives.",
        ),
        eligibility=Eligibility(
            True,
            False,
            False,
            "tcr_train_only_synthetic_negative",
            ("synthetic negative", "train split only"),
        ),
        context=context,
        derived_from=(receptor.record_id, ligand.record_id),
        flags=("synthetic_negative", "train_only", "do_not_use_for_evaluation"),
    )


def generate_train_only_tcr_negatives(
    train_records: Iterable[CanonicalRecord],
    all_known_records: Iterable[CanonicalRecord],
    *,
    protocol_id: str,
    ratio: float = 1.0,
    seed: int = 42,
) -> tuple[list[CanonicalRecord], dict[str, Any]]:
    """Derange pMHC only within train and reject every known positive key."""

    train_positive = sorted(
        (
            record
            for record in train_records
            if _is_positive(record) and _tcr_interaction_key(record)
        ),
        key=lambda record: record.record_id,
    )
    desired = min(
        math.ceil(len(train_positive) * max(0.0, ratio)),
        len(train_positive),
    )
    known_positive = {
        key
        for record in all_known_records
        if _is_positive(record)
        if (key := _tcr_interaction_key(record)) is not None
    }
    generated_keys: set[tuple[str, str, str, str]] = set()
    generated: list[CanonicalRecord] = []
    attempts = 0
    if train_positive:
        donors = sorted(
            train_positive,
            key=lambda record: stable_digest([seed, "ligand", record.record_id], length=64),
        )
        for receptor in train_positive:
            if len(generated) >= desired:
                break
            start = int(
                stable_digest([seed, "offset", receptor.record_id], length=16), 16
            ) % len(donors)
            for shift in range(1, len(donors) + 1):
                attempts += 1
                ligand = donors[(start + shift) % len(donors)]
                if ligand.record_id == receptor.record_id:
                    continue
                candidate = _synthetic_negative(
                    receptor,
                    ligand,
                    protocol_id=protocol_id,
                )
                key = _tcr_interaction_key(candidate)
                if key is None or key in known_positive or key in generated_keys:
                    continue
                generated_keys.add(key)
                generated.append(candidate)
                break
    errors = [
        error
        for record in generated
        for error in validate_domain_record(record)
    ]
    report = {
        "schema_version": "immune_receptor_train_negative_generation.v1",
        "protocol_id": protocol_id,
        "seed": seed,
        "requested_ratio": ratio,
        "train_positive_records": len(train_positive),
        "desired_negative_records": desired,
        "generated_negative_records": len(generated),
        "attempts": attempts,
        "known_positive_key_count": len(known_positive),
        "generated_unique_key_count": len(generated_keys),
        "known_positive_collisions": 0,
        "domain_error_count": len(errors),
        "evaluation_eligible_negative_records": sum(
            record.eligibility.evaluation for record in generated
        ),
    }
    return generated, report


def _forced_tcr_ids(report: Mapping[str, Any], include_peptide: bool) -> set[str]:
    keys = (
        "exact_receptor_record_ids",
        "exact_receptor_peptide_pair_record_ids",
        "exact_paired_tcr_peptide_record_ids",
        "exact_receptor_pmhc_record_ids",
    )
    values = {
        record_id
        for key in keys
        for record_id in report.get(key, [])
    }
    if include_peptide:
        values.update(report.get("exact_peptide_record_ids", []))
    return values


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_decontaminated_core_exports(
    *,
    root: Path,
    tcr_records: Sequence[CanonicalRecord],
    antibody_records: Sequence[CanonicalRecord],
    quarantine: Mapping[str, Any],
    mmseqs_bin: Path,
    threads: int = 32,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Build versioned, cluster-aware core exports and their audit manifest."""

    if not mmseqs_bin.is_file():
        raise FileNotFoundError(f"MMseqs2 binary not found: {mmseqs_bin}")
    tcr_core = [
        record
        for record in tcr_records
        if record.eligibility.core and record.record_type != "derived_view"
    ]
    antibody_core = [record for record in antibody_records if record.eligibility.core]
    input_paths = {
        "tcr": root / "canonical/tcr/tcr_union.jsonl",
        "antibody": root / "canonical/antibody/antibody_union.jsonl",
        "quarantine": root / "registries/benchmark_quarantine.json",
    }
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    mmseqs_version = subprocess.run(
        [str(mmseqs_bin), "version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    build_id = stable_digest(
        {
            "inputs": input_hashes,
            "rules": [asdict(rule) for rule in DEFAULT_CLUSTER_RULES],
            "protocols": [asdict(protocol) for protocol in DEFAULT_EXPORT_PROTOCOLS],
            "negative_ratio": negative_ratio,
            "seed": seed,
            "mmseqs_version": mmseqs_version,
        },
        prefix="ir2exp_",
        length=20,
    )
    cluster_root = root / "clusters" / build_id
    export_root = root / "exports" / build_id
    export_root.mkdir(parents=True, exist_ok=True)
    benchmark_bank = build_benchmark_sequence_bank(quarantine)
    clusters, cluster_registry = build_cluster_registry(
        [*tcr_core, *antibody_core],
        benchmark_bank,
        output_dir=cluster_root,
        mmseqs_bin=mmseqs_bin,
        threads=threads,
    )
    tcr_leakage = audit_benchmark_leakage(
        tcr_core,
        quarantine,
        benchmark_bank=benchmark_bank,
    )
    antibody_leakage = audit_benchmark_leakage(
        antibody_core,
        quarantine,
        benchmark_bank=benchmark_bank,
    )
    protocol_reports: dict[str, Any] = {}
    export_artifacts: list[Path] = []
    for protocol in DEFAULT_EXPORT_PROTOCOLS:
        domain_rows = tcr_core if protocol.domain == "tcr" else antibody_core
        if protocol.record_family:
            domain_rows = [
                record for record in domain_rows if record.family == protocol.record_family
            ]
        leakage = tcr_leakage if protocol.domain == "tcr" else antibody_leakage
        forced = (
            _forced_tcr_ids(leakage, protocol.include_benchmark_peptide)
            if protocol.domain == "tcr"
            else set(leakage.get("exact_receptor_record_ids", []))
        )
        kept, blocklist, decontamination = filter_benchmark_contamination(
            domain_rows,
            clusters,
            include_peptide=protocol.include_benchmark_peptide,
            include_antigen=protocol.include_benchmark_antigen,
            forced_exact_record_ids=forced,
        )
        if protocol.required_original_groups:
            missing_group = [
                record
                for record in kept
                if not all(
                    record.groups.get(field)
                    for field in protocol.required_original_groups
                )
            ]
            missing_ids = {record.record_id for record in missing_group}
            kept = [record for record in kept if record.record_id not in missing_ids]
            decontamination["missing_required_group_records"] = len(missing_group)
        if not kept:
            raise RuntimeError(f"No records remain for {protocol.protocol_id}")
        split_protocol = SplitProtocol(
            protocol.protocol_id,
            protocol.group_fields,
            seed=seed,
            description=protocol.description,
        )
        split_manifest = build_split_manifest(kept, split_protocol)
        assignment = {
            row["record_id"]: row["split"]
            for row in split_manifest["assignments"]
        }
        protocol_root = export_root / protocol.domain / protocol.protocol_id
        protocol_root.mkdir(parents=True, exist_ok=True)
        split_paths: dict[str, Path] = {}
        split_records: dict[str, list[CanonicalRecord]] = {}
        for split in ("train", "valid", "test"):
            rows = sorted(
                (
                    record
                    for record in kept
                    if assignment[record.record_id] == split
                ),
                key=lambda record: record.record_id,
            )
            path = protocol_root / f"{split}.jsonl"
            write_jsonl(path, rows)
            split_paths[split] = path
            split_records[split] = rows
            export_artifacts.append(path)
        negative_report: dict[str, Any] | None = None
        negative_path: Path | None = None
        if protocol.domain == "tcr":
            negatives, negative_report = generate_train_only_tcr_negatives(
                split_records["train"],
                tcr_records,
                protocol_id=protocol.protocol_id,
                ratio=negative_ratio,
                seed=seed,
            )
            negatives = [with_cluster_groups(record, clusters) for record in negatives]
            negative_path = protocol_root / "train_synthetic_negatives.jsonl"
            write_jsonl(negative_path, negatives)
            export_artifacts.append(negative_path)
        blocklist_path = protocol_root / "benchmark_blocklist.jsonl"
        write_jsonl(blocklist_path, blocklist)
        split_manifest_path = protocol_root / "split_manifest.json"
        write_json(split_manifest_path, split_manifest)
        export_artifacts.extend((blocklist_path, split_manifest_path))
        residual = [
            record.record_id
            for record in kept
            if any(
                benchmark_match_for_record(
                    record,
                    clusters,
                    include_peptide=protocol.include_benchmark_peptide,
                    include_antigen=protocol.include_benchmark_antigen,
                ).values()
            )
            or record.record_id in forced
        ]
        protocol_report = {
            "domain": protocol.domain,
            "protocol": asdict(protocol),
            "decontamination": decontamination,
            "split_counts": split_manifest["split_counts"],
            "split_audit": split_manifest["audit"],
            "balance": split_manifest["balance"],
            "residual_benchmark_match_records": len(residual),
            "negative_generation": negative_report,
            "files": {
                split: _artifact(path) for split, path in split_paths.items()
            },
            "blocklist": _artifact(blocklist_path),
            "split_manifest": _artifact(split_manifest_path),
        }
        if negative_path:
            protocol_report["files"]["train_synthetic_negatives"] = _artifact(
                negative_path
            )
        protocol_report_path = protocol_root / "report.json"
        write_json(protocol_report_path, protocol_report)
        export_artifacts.append(protocol_report_path)
        protocol_reports[protocol.protocol_id] = protocol_report

    cluster_artifacts = sorted(path for path in cluster_root.rglob("*") if path.is_file())
    all_artifacts = sorted(set(export_artifacts + cluster_artifacts))
    technical_ready = all(
        report["split_audit"]["passed"]
        and report["residual_benchmark_match_records"] == 0
        and (
                report["negative_generation"] is None
                or (
                    report["negative_generation"]["generated_negative_records"]
                    == report["negative_generation"]["desired_negative_records"]
                    and
                    report["negative_generation"]["known_positive_collisions"] == 0
                and report["negative_generation"]["domain_error_count"] == 0
                and report["negative_generation"][
                    "evaluation_eligible_negative_records"
                ]
                == 0
            )
        )
        for report in protocol_reports.values()
    )
    report = {
        "schema_version": "immune_receptor_core_export_report.v1",
        "build_id": build_id,
        "input_artifacts": {
            name: {"path": str(input_paths[name].resolve()), "sha256": digest}
            for name, digest in input_hashes.items()
        },
        "scope": {
            "included": "high-confidence TCR specificity, antibody-antigen, and antibody-property core",
            "excluded": "proxy, pool, reconstructed, low-confidence, and source-released synthetic-negative auxiliary packs",
            "oas_ots_pairing": "referenced separately; not duplicated in this core recognition export",
        },
        "cluster_registry": cluster_registry,
        "mmseqs_version": mmseqs_version,
        "protocols": protocol_reports,
        "technical_core_export_ready": technical_ready,
        "rights_review_complete": False,
        "redistribution_ready": False,
        "training_started": False,
        "remaining_release_blockers": [
            "SAbDab2 and CATNAP training-use/redistribution rights require project-level review.",
            "OAS/OTS pairing assets require a current benchmark-bank re-audit before a combined recipe manifest is released.",
        ],
    }
    report_path = export_root / "export_report.json"
    manifest_path = export_root / "artifact_manifest.json"
    report["artifact_manifest_path"] = str(manifest_path.resolve())
    write_json(report_path, report)
    all_artifacts.append(report_path)
    artifact_manifest = {
        "schema_version": "immune_receptor_export_artifact_manifest.v1",
        "build_id": build_id,
        "root": str(export_root.resolve()),
        "artifacts": [_artifact(path) for path in sorted(set(all_artifacts))],
    }
    artifact_manifest["artifact_count"] = len(artifact_manifest["artifacts"])
    artifact_manifest["total_bytes"] = sum(
        item["bytes"] for item in artifact_manifest["artifacts"]
    )
    write_json(manifest_path, artifact_manifest)
    return report
