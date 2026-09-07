#!/usr/bin/env python3
"""Build canonical AB/TCR data, registries, splits, and audits without training.

Examples::

    conda run -n pllm python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py inventory
    conda run -n pllm python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py all
    conda run -n pllm python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py all \
      --limit-per-source 100
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator


PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/immune_receptor_v2"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.bioseq.immune_receptor_v2.adapters.antibody import (  # noqa: E402
    build_target_registry,
    iter_abrank,
    iter_catnap,
    iter_cov_abdab,
    iter_flab_properties,
    iter_kothiwal,
    iter_sabdab2,
)
from dllm.pipelines.bioseq.immune_receptor_v2.adapters.tcr import (  # noqa: E402
    iter_fullchain_derived,
    iter_iedb,
    iter_mcpas,
    iter_mira,
    iter_piste,
    iter_vdjdb,
)
from dllm.pipelines.bioseq.immune_receptor_v2.audit import (  # noqa: E402
    audit_records,
    merge_exact_measurements,
)
from dllm.pipelines.bioseq.immune_receptor_v2.export import (  # noqa: E402
    build_decontaminated_core_exports,
)
from dllm.pipelines.bioseq.immune_receptor_v2.io import (  # noqa: E402
    iter_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from dllm.pipelines.bioseq.immune_receptor_v2.pairing import (  # noqa: E402
    build_pairing_exports,
)
from dllm.pipelines.bioseq.immune_receptor_v2.leakage import (  # noqa: E402
    audit_benchmark_leakage,
    build_benchmark_interaction_bank,
    build_benchmark_sequence_bank,
)
from dllm.pipelines.bioseq.immune_receptor_v2.registry import (  # noqa: E402
    build_source_registry,
    freeze_benchmark_inputs,
)
from dllm.pipelines.bioseq.immune_receptor_v2.schema import (  # noqa: E402
    CanonicalRecord,
    stable_digest,
)
from dllm.pipelines.bioseq.immune_receptor_v2.splits import (  # noqa: E402
    SplitProtocol,
    build_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "inventory",
            "build-tcr",
            "build-antibody",
            "split",
            "audit",
            "finalize",
            "export",
            "pairing-export",
            "recipe",
            "all",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument(
        "--no-checksums",
        action="store_true",
        help="Skip SHA256 only for quick development runs; production manifests require it.",
    )
    parser.add_argument(
        "--exclude-fullchain-derived", action="store_true", help="Skip legacy derived view."
    )
    parser.add_argument(
        "--exclude-cov-proxy", action="store_true", help="Skip name-only CoV-AbDab auxiliary rows."
    )
    parser.add_argument(
        "--mmseqs-bin",
        type=Path,
        default=Path(
            "/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/mmseqs"
        ),
        help="Absolute MMseqs2 executable used by the export command.",
    )
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Train-only synthetic negatives per positive, capped at one per positive.",
    )
    return parser.parse_args()


def _layout(root: Path) -> None:
    for relative in (
        "raw",
        "canonical/tcr",
        "canonical/antibody",
        "registries",
        "splits/tcr",
        "splits/antibody",
        "exports",
        "clusters",
        "reports",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _checksum_map(registry: dict) -> dict[str, str]:
    return {
        item["path"]: item.get("sha256", "")
        for source in registry.get("sources", [])
        for item in source.get("files", [])
    }


def _source_status(registry: dict, name: str) -> str:
    for source in registry.get("sources", []):
        if source["source_name"] == name:
            return source["status"]
    return "missing"


def run_inventory(root: Path, *, checksums: bool) -> tuple[dict, dict]:
    _layout(root)
    registry = build_source_registry(checksums=checksums)
    quarantine = freeze_benchmark_inputs(checksums=checksums)
    write_json(root / "registries/source_registry.json", registry)
    write_json(root / "registries/benchmark_quarantine.json", quarantine)
    print(
        json.dumps(
            {
                "source_status": Counter(
                    source["status"] for source in registry["sources"]
                ),
                "benchmark_files": quarantine["file_count"],
            },
            default=dict,
            sort_keys=True,
        ),
        flush=True,
    )
    return registry, quarantine


def _write_source(
    output: Path, records: Iterable[CanonicalRecord]
) -> tuple[list[CanonicalRecord], dict]:
    rows = list(records)
    count = write_jsonl(output, rows)
    report = audit_records(rows)
    report["output_path"] = str(output.resolve())
    report["output_sha256"] = sha256_file(output)
    if count != report["record_count"]:
        raise RuntimeError("write count differs from audit count")
    return rows, report


def build_tcr(
    root: Path,
    registry: dict,
    *,
    limit: int | None,
    include_fullchain: bool,
) -> list[CanonicalRecord]:
    checksums = _checksum_map(registry)
    source_builders: list[tuple[str, Callable[[], Iterator[CanonicalRecord]]]] = []
    piste_root = (
        PROJECT_ROOT
        / "data/ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/random"
    )
    if _source_status(registry, "piste") == "raw_verified":
        source_builders.append(
            (
                "piste",
                lambda: iter_piste(
                    piste_root, raw_sha256=checksums, limit=limit
                ),
            )
        )
    vdjdb = PROJECT_ROOT / "data/tcr/vdjdb_full.txt"
    if _source_status(registry, "vdjdb") == "raw_verified":
        source_builders.append(
            (
                "vdjdb",
                lambda: iter_vdjdb(
                    vdjdb,
                    raw_sha256=checksums.get(str(vdjdb.resolve()), ""),
                    limit=limit,
                ),
            )
        )
    iedb_tcell = PROJECT_ROOT / "data/tcr/tcell_full_v3.zip"
    iedb_tcr = PROJECT_ROOT / "data/tcr/tcr_full_v3.zip"
    if _source_status(registry, "iedb_tcell") == "raw_verified":
        source_builders.append(
            (
                "iedb_tcell",
                lambda: iter_iedb(
                    iedb_tcr,
                    iedb_tcell,
                    raw_sha256=checksums,
                    limit=limit,
                ),
            )
        )
    mcpas = PROJECT_ROOT / "data/tcr/McPAS-TCR.csv"
    if _source_status(registry, "mcpas") == "raw_verified":
        source_builders.append(
            (
                "mcpas",
                lambda: iter_mcpas(
                    mcpas,
                    raw_sha256=checksums.get(str(mcpas.resolve()), ""),
                    limit=limit,
                ),
            )
        )
    mira_root = PROJECT_ROOT / "data/tcr/MIRA/ImmuneCODE-MIRA-Release002.1"
    if _source_status(registry, "mira") == "raw_verified":
        source_builders.append(
            (
                "mira",
                lambda: iter_mira(mira_root, raw_sha256=checksums, limit=limit),
            )
        )
    fullchain_root = PROJECT_ROOT / "data/tcr_pmhc_fulllength"
    if (
        include_fullchain
        and _source_status(registry, "tcr_pmhc_fulllength_derived")
        == "raw_verified"
    ):
        source_builders.append(
            (
                "fullchain_derived",
                lambda: iter_fullchain_derived(
                    fullchain_root, raw_sha256=checksums, limit=limit
                ),
            )
        )

    all_rows: list[CanonicalRecord] = []
    reports: dict[str, dict] = {}
    for source_name, builder in source_builders:
        print(f"building TCR source {source_name}", flush=True)
        rows, report = _write_source(
            root / f"canonical/tcr/{source_name}.jsonl", builder()
        )
        all_rows.extend(rows)
        reports[source_name] = report
        print(f"  {source_name}: {len(rows)} canonical records", flush=True)
    merged, merge_report = merge_exact_measurements(all_rows)
    union_path = root / "canonical/tcr/tcr_union.jsonl"
    write_jsonl(union_path, merged)
    union_report = audit_records(merged)
    union_report["output_path"] = str(union_path.resolve())
    union_report["output_sha256"] = sha256_file(union_path)
    union_report["merge"] = merge_report
    union_report["sources"] = reports
    union_report["blocked_sources"] = {
        name: _source_status(registry, name)
        for name in ("iedb_tcell",)
        if _source_status(registry, name) != "raw_verified"
    }
    write_json(root / "reports/tcr_build_report.json", union_report)
    return merged


def build_antibody(
    root: Path,
    registry: dict,
    *,
    limit: int | None,
    include_cov_proxy: bool,
) -> list[CanonicalRecord]:
    checksums = _checksum_map(registry)
    source_builders: list[tuple[str, Callable[[], Iterator[CanonicalRecord]]]] = []
    binding_root = PROJECT_ROOT / "data/ppi_task_raw/raw/flab/FLAb/data/binding"
    abrank = binding_root / "AbRank_dataset.csv.zip"
    if _source_status(registry, "abrank") == "raw_verified":
        source_builders.append(
            (
                "abrank",
                lambda: iter_abrank(
                    abrank,
                    raw_sha256=checksums.get(str(abrank.resolve()), ""),
                    limit=limit,
                ),
            )
        )
    if _source_status(registry, "kothiwal2025") == "raw_verified":
        source_builders.append(
            (
                "kothiwal2025",
                lambda: iter_kothiwal(
                    binding_root, raw_sha256=checksums, limit=limit
                ),
            )
        )
    catnap_roots = sorted(
        path
        for path in (PROJECT_ROOT / "data/antibody_raw/catnap").glob("*")
        if path.is_dir()
    )
    if catnap_roots and _source_status(registry, "catnap") == "raw_verified":
        catnap_root = catnap_roots[-1]
        source_builders.append(
            (
                "catnap",
                lambda: iter_catnap(
                    catnap_root,
                    raw_sha256=checksums,
                    limit=limit,
                ),
            )
        )
    sabdab2 = PROJECT_ROOT / "data/sabdab2_ml/raw/splits.tar.gz"
    if _source_status(registry, "sabdab2_ml") == "raw_verified":
        source_builders.append(
            (
                "sabdab2_ml",
                lambda: iter_sabdab2(
                    sabdab2,
                    raw_sha256=checksums.get(str(sabdab2.resolve()), ""),
                    limit=limit,
                ),
            )
        )
    flab_root = PROJECT_ROOT / "data/ppi_task_raw/raw/flab/FLAb/data"
    if _source_status(registry, "flab_properties") == "raw_verified":
        source_builders.append(
            (
                "flab_properties",
                lambda: iter_flab_properties(
                    flab_root, raw_sha256=checksums, limit=limit
                ),
            )
        )
    cov = (
        PROJECT_ROOT
        / "data/ppi_task_raw/raw/covabdab_neutralization/CoV-AbDab_080224.csv"
    )
    if include_cov_proxy and _source_status(registry, "cov_abdab") == "raw_verified":
        source_builders.append(
            (
                "cov_abdab",
                lambda: iter_cov_abdab(
                    cov,
                    raw_sha256=checksums.get(str(cov.resolve()), ""),
                    limit=limit,
                ),
            )
        )

    all_rows: list[CanonicalRecord] = []
    reports: dict[str, dict] = {}
    for source_name, builder in source_builders:
        print(f"building antibody source {source_name}", flush=True)
        rows, report = _write_source(
            root / f"canonical/antibody/{source_name}.jsonl", builder()
        )
        all_rows.extend(rows)
        reports[source_name] = report
        print(f"  {source_name}: {len(rows)} canonical records", flush=True)
    merged, merge_report = merge_exact_measurements(all_rows)
    union_path = root / "canonical/antibody/antibody_union.jsonl"
    write_jsonl(union_path, merged)
    target_registry = build_target_registry(merged)
    write_json(root / "registries/antibody_targets.json", target_registry)
    union_report = audit_records(merged)
    union_report["output_path"] = str(union_path.resolve())
    union_report["output_sha256"] = sha256_file(union_path)
    union_report["merge"] = merge_report
    union_report["sources"] = reports
    union_report["blocked_sources"] = {
        name: _source_status(registry, name)
        for name in ("sabdab2_ml", "catnap")
        if _source_status(registry, name) != "raw_verified"
    }
    write_json(root / "reports/antibody_build_report.json", union_report)
    return merged


def _load_union(path: Path) -> list[CanonicalRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical union: {path}")
    return list(iter_jsonl(path, as_records=True))  # type: ignore[arg-type]


def build_splits(root: Path, tcr: list[CanonicalRecord], antibody: list[CanonicalRecord]) -> dict:
    split_reports: dict[str, dict] = {}
    tcr_core = [
        record
        for record in tcr
        if record.eligibility.training
        and record.eligibility.core
        and record.record_type != "derived_view"
    ]
    tcr_protocols = (
        SplitProtocol(
            "tcr_receptor_disjoint_seen_peptide_v1",
            ("receptor",),
            description=(
                "Receptor-disjoint 90/5/5 split; peptide overlap is reported, not forced."
            ),
        ),
        SplitProtocol(
            "tcr_unseen_peptide_v1",
            ("peptide",),
            description="Every exact peptide is assigned to only one split.",
        ),
        SplitProtocol(
            "tcr_unseen_pmhc_v1",
            ("pmhc",),
            description="Every peptide-MHC identity is assigned to only one split.",
        ),
        SplitProtocol(
            "tcr_study_holdout_v1",
            ("study",),
            description="Study/provenance-disjoint split.",
        ),
    )
    for protocol in tcr_protocols:
        eligible = [
            record
            for record in tcr_core
            if all(record.groups.get(field) for field in protocol.group_fields)
        ]
        if not eligible:
            continue
        manifest = build_split_manifest(eligible, protocol)
        write_json(root / f"splits/tcr/{protocol.protocol_id}.json", manifest)
        split_reports[protocol.protocol_id] = {
            "records": len(eligible),
            "counts": manifest["split_counts"],
            "audit": manifest["audit"],
            "balance": manifest["balance"],
        }

    antibody_interaction = [
        record
        for record in antibody
        if record.family == "antibody_antigen"
        and record.eligibility.training
        and record.eligibility.core
    ]
    antibody_protocols = (
        SplitProtocol("antibody_disjoint_v1", ("antibody",)),
        SplitProtocol("antigen_disjoint_v1", ("antigen",)),
        SplitProtocol("antibody_antigen_joint_hard_v1", ("antibody", "antigen")),
    )
    for protocol in antibody_protocols:
        eligible = [
            record
            for record in antibody_interaction
            if all(record.groups.get(field) for field in protocol.group_fields)
        ]
        if not eligible:
            continue
        manifest = build_split_manifest(eligible, protocol)
        write_json(root / f"splits/antibody/{protocol.protocol_id}.json", manifest)
        split_reports[protocol.protocol_id] = {
            "records": len(eligible),
            "counts": manifest["split_counts"],
            "audit": manifest["audit"],
            "balance": manifest["balance"],
        }

    properties = [
        record
        for record in antibody
        if record.family == "antibody_property" and record.eligibility.training
    ]
    for protocol in (
        SplitProtocol("antibody_property_parent_v1", ("parent",)),
        SplitProtocol("antibody_property_study_v1", ("study",)),
    ):
        eligible = [
            record
            for record in properties
            if all(record.groups.get(field) for field in protocol.group_fields)
        ]
        if not eligible:
            continue
        manifest = build_split_manifest(eligible, protocol)
        write_json(root / f"splits/antibody/{protocol.protocol_id}.json", manifest)
        split_reports[protocol.protocol_id] = {
            "records": len(eligible),
            "counts": manifest["split_counts"],
            "audit": manifest["audit"],
            "balance": manifest["balance"],
        }
    write_json(root / "reports/split_build_report.json", split_reports)
    return split_reports


def run_audit(
    root: Path,
    tcr: list[CanonicalRecord],
    antibody: list[CanonicalRecord],
    quarantine: dict,
) -> dict:
    tcr_report = audit_records(tcr)
    antibody_report = audit_records(antibody)
    write_json(
        root / "registries/antibody_targets.json",
        build_target_registry(antibody),
    )
    benchmark_bank = build_benchmark_sequence_bank(quarantine)
    interaction_bank = build_benchmark_interaction_bank(quarantine)
    tcr_leakage = audit_benchmark_leakage(
        tcr,
        quarantine,
        benchmark_bank=benchmark_bank,
        interaction_bank=interaction_bank,
    )
    antibody_leakage = audit_benchmark_leakage(
        antibody,
        quarantine,
        benchmark_bank=benchmark_bank,
        interaction_bank=interaction_bank,
    )
    write_json(root / "reports/tcr_audit.json", tcr_report)
    write_json(root / "reports/antibody_audit.json", antibody_report)
    write_json(root / "reports/tcr_benchmark_leakage.json", tcr_leakage)
    write_json(root / "reports/antibody_benchmark_leakage.json", antibody_leakage)
    split_report_path = root / "reports/split_build_report.json"
    split_report = (
        json.loads(split_report_path.read_text(encoding="utf-8"))
        if split_report_path.exists()
        else {}
    )
    split_audits_pass = bool(split_report) and all(
        report.get("audit", {}).get("passed") for report in split_report.values()
    )
    tcr_build_path = root / "reports/tcr_build_report.json"
    antibody_build_path = root / "reports/antibody_build_report.json"
    tcr_build = (
        json.loads(tcr_build_path.read_text(encoding="utf-8"))
        if tcr_build_path.exists()
        else {}
    )
    antibody_build = (
        json.loads(antibody_build_path.read_text(encoding="utf-8"))
        if antibody_build_path.exists()
        else {}
    )
    required_tcr_sources = {"piste", "vdjdb", "mcpas", "iedb_tcell"}
    required_antibody_sources = {
        "abrank",
        "kothiwal2025",
        "flab_properties",
        "sabdab2_ml",
        "catnap",
    }
    built_tcr_sources = set(tcr_build.get("sources", {}))
    built_antibody_sources = set(antibody_build.get("sources", {}))
    missing_canonical_sources = sorted(
        (required_tcr_sources - built_tcr_sources)
        | (required_antibody_sources - built_antibody_sources)
    )
    schema_valid = (
        tcr_report["domain_error_count"] == 0
        and antibody_report["domain_error_count"] == 0
        and tcr_report["record_count"] == tcr_report["unique_record_ids"]
        and antibody_report["record_count"]
        == antibody_report["unique_record_ids"]
    )
    export_blockers = [
        (
            "benchmark exact-entity, exact-interaction, and near-neighbor "
            "quarantine is audited but not yet applied to immutable exports"
        ),
        (
            "sequence-cluster decontamination and final train-only negative "
            "generation have not been materialized"
        ),
        (
            "SAbDab2 and CATNAP redistribution/training-use rights require "
            "project-level review before any redistributed export"
        ),
    ]
    if missing_canonical_sources:
        export_blockers.insert(
            0,
            "required sources not canonicalized: "
            + ", ".join(missing_canonical_sources),
        )
    technical_export: dict | None = None
    export_reports = sorted((root / "exports").glob("*/export_report.json"))
    for candidate in reversed(export_reports):
        value = json.loads(candidate.read_text(encoding="utf-8"))
        if value.get("technical_core_export_ready"):
            technical_export = value
            export_blockers = list(value.get("remaining_release_blockers", []))
            break
    pairing_export: dict | None = None
    pairing_reports = sorted(
        (root / "exports").glob("*/pairing_export_report.json")
    )
    for candidate in reversed(pairing_reports):
        value = json.loads(candidate.read_text(encoding="utf-8"))
        if value.get("technical_pairing_export_ready"):
            pairing_export = value
            break
    if technical_export and pairing_export:
        export_blockers = [
            "SAbDab2 and CATNAP training-use/redistribution rights require project-level review."
        ]
    summary = {
        "tcr": {
            "records": len(tcr),
            "domain_errors": tcr_report["domain_error_count"],
            "benchmark_records_with_exact_receptor_entity": tcr_leakage[
                "records_with_exact_receptor_entity"
            ],
            "benchmark_records_with_exact_peptide_entity": tcr_leakage[
                "records_with_exact_peptide_entity"
            ],
            "benchmark_exact_receptor_peptide_pair_records": tcr_leakage[
                "exact_receptor_peptide_pair_records"
            ],
            "benchmark_exact_paired_tcr_peptide_records": tcr_leakage[
                "exact_paired_tcr_peptide_records"
            ],
            "benchmark_exact_receptor_pmhc_records": tcr_leakage[
                "exact_receptor_pmhc_records"
            ],
            "benchmark_records_with_any_exact_entity": tcr_leakage[
                "exact_match_record_count"
            ],
            "benchmark_near_match_records": tcr_leakage[
                "near_match_record_count"
            ],
        },
        "antibody": {
            "records": len(antibody),
            "domain_errors": antibody_report["domain_error_count"],
            "benchmark_records_with_exact_receptor_entity": antibody_leakage[
                "records_with_exact_receptor_entity"
            ],
            "benchmark_records_with_any_exact_entity": antibody_leakage[
                "exact_match_record_count"
            ],
            "benchmark_near_match_records": antibody_leakage[
                "near_match_record_count"
            ],
        },
        "pipeline_status": {
            "schema_valid": schema_valid,
            "canonicalized": schema_valid and not missing_canonical_sources,
            "required_sources": {
                "tcr": sorted(required_tcr_sources),
                "antibody": sorted(required_antibody_sources),
            },
            "built_sources": {
                "tcr": sorted(built_tcr_sources),
                "antibody": sorted(built_antibody_sources),
            },
            "missing_canonical_sources": missing_canonical_sources,
            "split_ready": split_audits_pass,
            "technical_core_export_ready": bool(technical_export),
            "technical_pairing_export_ready": bool(pairing_export),
            "technical_data_export_ready": bool(technical_export and pairing_export),
            "export_ready": False,
            "export_blockers": export_blockers,
        },
        "training_started": False,
    }
    if technical_export:
        summary["pipeline_status"].update(
            {
                "core_export_build_id": technical_export["build_id"],
                "core_export_report": str(
                    (
                        root
                        / "exports"
                        / technical_export["build_id"]
                        / "export_report.json"
                    ).resolve()
                ),
                "rights_review_complete": technical_export[
                    "rights_review_complete"
                ],
                "redistribution_ready": technical_export[
                    "redistribution_ready"
                ],
            }
        )
    if pairing_export:
        summary["pipeline_status"].update(
            {
                "pairing_export_build_id": pairing_export["build_id"],
                "pairing_export_report": str(
                    (
                        root
                        / "exports"
                        / pairing_export["build_id"]
                        / "pairing_export_report.json"
                    ).resolve()
                ),
                "training_start_blockers": [
                    "SAbDab2 and CATNAP training-use rights require project-level review",
                    "runtime view sampler/per-chain target masks and relation-target rendering remain outside this data-only execution",
                    "final four-plane sampling weights have not been frozen",
                ],
            }
        )
    write_json(root / "reports/summary.json", summary)
    artifact_paths = sorted(
        [
            path
            for base in (root / "canonical", root / "splits")
            for path in base.rglob("*")
            if path.is_file() and not path.name.endswith((".tmp", ".partial"))
        ]
    )
    artifact_manifest = {
        "schema_version": "immune_receptor_artifact_manifest.v1",
        "root": str(root.resolve()),
        "artifacts": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    artifact_manifest["artifact_count"] = len(artifact_manifest["artifacts"])
    artifact_manifest["total_bytes"] = sum(
        item["bytes"] for item in artifact_manifest["artifacts"]
    )
    write_json(root / "registries/artifact_manifest.json", artifact_manifest)
    return summary


def _record_export_status(root: Path, export_report: dict) -> None:
    """Attach the technical core-export gate without claiming rights approval."""

    summary_path = root / "reports/summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {"pipeline_status": {}, "training_started": False}
    )
    pipeline = summary.setdefault("pipeline_status", {})
    pipeline["technical_core_export_ready"] = export_report[
        "technical_core_export_ready"
    ]
    pipeline["core_export_build_id"] = export_report["build_id"]
    pipeline["core_export_report"] = str(
        (
            root
            / "exports"
            / export_report["build_id"]
            / "export_report.json"
        ).resolve()
    )
    pipeline["rights_review_complete"] = export_report["rights_review_complete"]
    pipeline["redistribution_ready"] = export_report["redistribution_ready"]
    pipeline["export_ready"] = False
    pipeline["export_blockers"] = export_report["remaining_release_blockers"]
    summary["training_started"] = False
    write_json(summary_path, summary)


def _record_pairing_export_status(root: Path, pairing_report: dict) -> None:
    summary_path = root / "reports/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pipeline = summary.setdefault("pipeline_status", {})
    pipeline["technical_pairing_export_ready"] = pairing_report[
        "technical_pairing_export_ready"
    ]
    pipeline["pairing_export_build_id"] = pairing_report["build_id"]
    pipeline["pairing_export_report"] = str(
        (
            root
            / "exports"
            / pairing_report["build_id"]
            / "pairing_export_report.json"
        ).resolve()
    )
    pipeline["technical_data_export_ready"] = bool(
        pipeline.get("technical_core_export_ready")
        and pairing_report["technical_pairing_export_ready"]
    )
    pipeline["export_ready"] = False
    pipeline["export_blockers"] = [
        "SAbDab2 and CATNAP training-use/redistribution rights require project-level review."
    ]
    pipeline["training_start_blockers"] = [
        "SAbDab2 and CATNAP training-use rights require project-level review",
        "runtime view sampler/per-chain target masks and relation-target rendering remain outside this data-only execution",
        "final four-plane sampling weights have not been frozen",
    ]
    summary["training_started"] = False
    write_json(summary_path, summary)


def build_candidate_recipe(root: Path) -> dict:
    """Combine immutable pairing and recognition exports without starting training."""

    summary_path = root / "reports/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pipeline = summary["pipeline_status"]
    if not pipeline.get("technical_data_export_ready"):
        raise RuntimeError("Core and pairing technical exports must both be ready")
    core_report_path = Path(pipeline["core_export_report"])
    pairing_report_path = Path(pipeline["pairing_export_report"])
    core = json.loads(core_report_path.read_text(encoding="utf-8"))
    pairing = json.loads(pairing_report_path.read_text(encoding="utf-8"))
    primary_protocols = {
        "tcr_recognition": "tcr_receptor_cluster_disjoint_seen_peptide_v2",
        "antibody_recognition": "antibody_cluster_disjoint_v2",
        "antibody_properties": "antibody_property_parent_decontaminated_v2",
    }
    diagnostic_protocols = {
        protocol_id: report["files"]
        for protocol_id, report in core["protocols"].items()
        if protocol_id not in set(primary_protocols.values())
    }
    code_paths = [
        PROJECT_ROOT / "dllm/pipelines/bioseq/immune_receptor_v2/schema.py",
        PROJECT_ROOT / "dllm/pipelines/bioseq/immune_receptor_v2/leakage.py",
        PROJECT_ROOT / "dllm/pipelines/bioseq/immune_receptor_v2/splits.py",
        PROJECT_ROOT / "dllm/pipelines/bioseq/immune_receptor_v2/export.py",
        PROJECT_ROOT / "dllm/pipelines/bioseq/immune_receptor_v2/pairing.py",
        PROJECT_ROOT / "scripts/data/build_immune_receptor_v2.py",
    ]
    recipe_id = stable_digest(
        {
            "core_export": core["build_id"],
            "pairing_export": pairing["build_id"],
            "primary_protocols": primary_protocols,
            "recipe_schema": "immune_receptor_training_recipe_candidate.v1",
        },
        prefix="ir2recipe_",
        length=20,
    )
    recipe_root = root / "exports" / recipe_id
    recipe_path = recipe_root / "recipe_manifest.json"
    recipe = {
        "schema_version": "immune_receptor_training_recipe_candidate.v1",
        "recipe_id": recipe_id,
        "scope": {
            "included": [
                "paired antibody H/L",
                "paired TCR alpha/beta",
                "antibody-antigen core",
                "TCR-peptide/pMHC core",
                "paired antibody intrinsic-property core",
            ],
            "excluded": [
                "MINT/general-PPI",
                "nanobody/VHH/VNAR",
                "specificity-free bulk TCR",
                "proxy/name-only antibody targets",
                "pool-level TCR responses",
                "source-released synthetic negatives",
                "reconstructed full-chain evidence as independent supervision",
            ],
        },
        "data_planes": {
            "antibody_pairing": pairing["sources"]["oas"]["outputs"],
            "tcr_pairing": pairing["sources"]["ots"]["outputs"],
            "antibody_recognition": core["protocols"][
                primary_protocols["antibody_recognition"]
            ]["files"],
            "tcr_recognition": core["protocols"][
                primary_protocols["tcr_recognition"]
            ]["files"],
        },
        "supplemental_core": {
            "antibody_properties": core["protocols"][
                primary_protocols["antibody_properties"]
            ]["files"]
        },
        "primary_protocols": primary_protocols,
        "diagnostic_protocols": diagnostic_protocols,
        "negative_policy": {
            "tcr_train_synthetic_negatives": core["protocols"][
                primary_protocols["tcr_recognition"]
            ]["files"]["train_synthetic_negatives"],
            "validation_test_synthetic_negatives": 0,
            "known_positive_collision_count": core["protocols"][
                primary_protocols["tcr_recognition"]
            ]["negative_generation"]["known_positive_collisions"],
            "mixing_weight": "not frozen; keep as a separate low-confidence pack",
        },
        "provenance": {
            "core_export_report": {
                "path": str(core_report_path.resolve()),
                "sha256": sha256_file(core_report_path),
            },
            "pairing_export_report": {
                "path": str(pairing_report_path.resolve()),
                "sha256": sha256_file(pairing_report_path),
            },
            "pipeline_code": [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for path in code_paths
            ],
        },
        "gates": {
            "technical_data_ready": True,
            "rights_review_complete": False,
            "runtime_views_ready": False,
            "sampling_weights_frozen": False,
            "training_ready": False,
            "training_started": False,
        },
        "remaining_blockers": [
            "SAbDab2 and CATNAP training-use/redistribution rights require project-level review.",
            "Implement and test per-chain/per-region fixed masks and relation-target rendering.",
            "Freeze four-plane sampling weights and token-budget batching in a separate model/training change.",
        ],
    }
    write_json(recipe_path, recipe)
    pipeline["candidate_recipe_id"] = recipe_id
    pipeline["candidate_recipe_manifest"] = str(recipe_path.resolve())
    pipeline["candidate_recipe_sha256"] = sha256_file(recipe_path)
    pipeline["training_start_blockers"] = list(recipe["remaining_blockers"])
    summary["training_started"] = False
    write_json(summary_path, summary)
    return recipe


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    if not root.is_absolute():
        raise ValueError("--output-root must resolve to an absolute path")
    _layout(root)
    registry_path = root / "registries/source_registry.json"
    quarantine_path = root / "registries/benchmark_quarantine.json"
    if args.command in {"inventory", "all"} or not registry_path.exists():
        registry, quarantine = run_inventory(
            root, checksums=not args.no_checksums
        )
    else:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
    if args.command == "inventory":
        return

    tcr_path = root / "canonical/tcr/tcr_union.jsonl"
    antibody_path = root / "canonical/antibody/antibody_union.jsonl"
    tcr: list[CanonicalRecord] = []
    antibody: list[CanonicalRecord] = []
    if args.command in {"build-tcr", "all"}:
        tcr = build_tcr(
            root,
            registry,
            limit=args.limit_per_source,
            include_fullchain=not args.exclude_fullchain_derived,
        )
    elif args.command in {"split", "audit", "finalize", "export"}:
        tcr = _load_union(tcr_path) if tcr_path.exists() else []
    if args.command == "build-tcr":
        return

    if args.command in {"build-antibody", "all"}:
        antibody = build_antibody(
            root,
            registry,
            limit=args.limit_per_source,
            include_cov_proxy=not args.exclude_cov_proxy,
        )
    elif args.command in {"split", "audit", "finalize", "export"}:
        antibody = _load_union(antibody_path) if antibody_path.exists() else []
    if args.command == "build-antibody":
        return

    if args.command in {"split", "finalize", "all"}:
        if not tcr or not antibody:
            raise RuntimeError("Both canonical unions are required before split")
        split_report = build_splits(root, tcr, antibody)
        print(json.dumps({"splits": split_report}, sort_keys=True), flush=True)
    if args.command in {"audit", "finalize", "all"}:
        if not tcr or not antibody:
            raise RuntimeError("Both canonical unions are required before audit")
        summary = run_audit(root, tcr, antibody, quarantine)
        print(json.dumps(summary, sort_keys=True), flush=True)
    if args.command == "export":
        if not tcr or not antibody:
            raise RuntimeError("Both canonical unions are required before export")
        export_report = build_decontaminated_core_exports(
            root=root,
            tcr_records=tcr,
            antibody_records=antibody,
            quarantine=quarantine,
            mmseqs_bin=args.mmseqs_bin.expanduser().resolve(),
            threads=args.threads,
            negative_ratio=args.negative_ratio,
        )
        _record_export_status(root, export_report)
        print(
            json.dumps(
                {
                    "build_id": export_report["build_id"],
                    "technical_core_export_ready": export_report[
                        "technical_core_export_ready"
                    ],
                    "rights_review_complete": export_report[
                        "rights_review_complete"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args.command == "pairing-export":
        pairing_report = build_pairing_exports(
            root=root,
            quarantine=quarantine,
            mmseqs_bin=args.mmseqs_bin.expanduser().resolve(),
            threads=args.threads,
        )
        _record_pairing_export_status(root, pairing_report)
        print(
            json.dumps(
                {
                    "build_id": pairing_report["build_id"],
                    "technical_pairing_export_ready": pairing_report[
                        "technical_pairing_export_ready"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args.command == "recipe":
        recipe = build_candidate_recipe(root)
        print(
            json.dumps(
                {
                    "recipe_id": recipe["recipe_id"],
                    "technical_data_ready": recipe["gates"][
                        "technical_data_ready"
                    ],
                    "training_ready": recipe["gates"]["training_ready"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
