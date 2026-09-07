"""Source registry and raw-artifact integrity checks for immune-receptor v2.

Generate the local inventory with::

    conda run -n pllm python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py inventory
"""

from __future__ import annotations

import csv
import glob
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .io import sha256_file


PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")


@dataclass(frozen=True)
class SourceSpec:
    source_name: str
    family: str
    raw_patterns: tuple[str, ...]
    expected_format: str
    intended_pack: str
    required: bool
    notes: str = ""
    source_version: str = ""
    source_url: str = ""
    license: str = ""


DEFAULT_SOURCE_SPECS = (
    SourceSpec(
        "oas_paired",
        "antibody_pairing",
        (
            "data/oas_previous_clean/splits/cleaned_merged_data_step_clustered_*_oas_label.csv",
        ),
        "csv",
        "antibody_pair_core",
        True,
        "Existing project-owned paired H/L splits; inventory only in this phase.",
    ),
    SourceSpec(
        "ots_paired",
        "tcr_pairing",
        ("data/ots_paired_clean/final/*.csv",),
        "csv",
        "tcr_pair_core",
        True,
        "Existing project-owned paired alpha/beta splits; inventory only in this phase.",
    ),
    SourceSpec(
        "piste",
        "tcr_pmhc",
        (
            "data/ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/random/train_data.csv",
            "data/ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/random/val_data.csv",
            "data/ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/random/test_data.csv",
        ),
        "csv",
        "tcr_aux_beta_and_synthetic_negative",
        True,
        "Original split is retained as provenance, never inherited as the v2 training split.",
    ),
    SourceSpec(
        "vdjdb",
        "tcr_specificity",
        ("data/tcr/vdjdb_full.txt",),
        "tsv",
        "tcr_core_or_aux_beta",
        True,
    ),
    SourceSpec(
        "mcpas",
        "tcr_specificity",
        ("data/tcr/McPAS-TCR.csv",),
        "csv",
        "tcr_core_or_aux_beta",
        True,
    ),
    SourceSpec(
        "iedb_tcell",
        "tcr_specificity",
        (
            "data/tcr/tcell_full_v3.zip",
            "data/tcr/tcr_full_v3.zip",
        ),
        "zip",
        "tcr_core_or_aux_beta",
        True,
        "Official IEDB CSV metric exports; TCR associations are joined to assay outcomes.",
        "2026-07-28",
        "https://www.iedb.org/database_export_v3.php",
        "CC BY 4.0",
    ),
    SourceSpec(
        "mira",
        "tcr_specificity_pool",
        ("data/tcr/MIRA/ImmuneCODE-MIRA-Release002.1/peptide-detail-c*.csv",),
        "csv",
        "tcr_aux_pool",
        False,
    ),
    SourceSpec(
        "tcr_pmhc_fulllength_derived",
        "tcr_pmhc",
        (
            "data/tcr_pmhc_fulllength/records_*.jsonl",
            "data/tcr_pmhc_fulllength/thimble_out.tsv",
        ),
        "jsonl_tsv",
        "tcr_fullchain_derived",
        False,
        "Reconstructed view of VDJdb/McPAS evidence, not an additive evidence source.",
    ),
    SourceSpec(
        "abrank",
        "antibody_antigen",
        ("data/ppi_task_raw/raw/flab/FLAb/data/binding/AbRank_dataset.csv.zip",),
        "zip",
        "antibody_antigen_exact_or_aux",
        True,
    ),
    SourceSpec(
        "kothiwal2025",
        "antibody_antigen",
        ("data/ppi_task_raw/raw/flab/FLAb/data/binding/kothiwal2025htp_*.csv",),
        "csv",
        "antibody_antigen_exact",
        True,
    ),
    SourceSpec(
        "flab_properties",
        "antibody_property",
        (
            "data/ppi_task_raw/raw/flab/FLAb/data/aggregation/*.csv",
            "data/ppi_task_raw/raw/flab/FLAb/data/expression/*.csv",
            "data/ppi_task_raw/raw/flab/FLAb/data/immunogenicity/*.csv",
            "data/ppi_task_raw/raw/flab/FLAb/data/pharmacokinetics/*.csv",
            "data/ppi_task_raw/raw/flab/FLAb/data/polyreactivity/*.csv",
            "data/ppi_task_raw/raw/flab/FLAb/data/thermostability/*.csv",
            "data/ppi_task_raw/raw/flab/FLAb/data/flab_metadata.csv",
        ),
        "csv",
        "antibody_property",
        True,
    ),
    SourceSpec(
        "cov_abdab",
        "antibody_antigen_proxy",
        ("data/ppi_task_raw/raw/covabdab_neutralization/CoV-AbDab_080224.csv",),
        "csv",
        "antibody_antigen_proxy",
        False,
        "Targets are names/variants rather than exact antigen sequences.",
    ),
    SourceSpec(
        "sabdab2_ml",
        "antibody_antigen",
        ("data/sabdab2_ml/raw/splits.tar.gz",),
        "tar.gz",
        "antibody_antigen_exact",
        True,
        (
            "Pinned official release. Only antigen-aware rows with paired H/L and "
            "resolved protein/peptide antigen sequences enter the canonical build; "
            "the released split is provenance only."
        ),
        "0.1.0 / Zenodo record 20083995",
        "https://doi.org/10.5281/zenodo.20083995",
        "not declared in Zenodo metadata; copyright notice present",
    ),
    SourceSpec(
        "catnap",
        "antibody_antigen",
        ("data/antibody_raw/catnap/**/*",),
        "mixed",
        "antibody_antigen_exact_accession",
        False,
        "Dated LANL CATNAP snapshot; neutralization values are never thresholded into labels.",
        "2026-08-01",
        "https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/download_db.comp",
        "not declared on download page",
    ),
)


def _resolve_patterns(patterns: Iterable[str], project_root: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        absolute_pattern = str(project_root / pattern)
        for value in glob.glob(absolute_pattern, recursive=True):
            path = Path(value)
            if path.is_file() and not path.name.endswith(".aria2"):
                paths.add(path.resolve())
    return sorted(paths)


def _integrity(path: Path, expected_format: str) -> tuple[str, str]:
    partial_marker = Path(str(path) + ".aria2")
    if partial_marker.exists():
        return "incomplete", f"partial-download marker exists: {partial_marker}"
    try:
        if expected_format == "zip" or path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                if not archive.infolist():
                    return "invalid", "ZIP contains no members"
        elif expected_format == "tar.gz" or path.name.endswith(".tar.gz"):
            with tarfile.open(path, "r:gz") as archive:
                next(iter(archive), None)
        elif expected_format in {"csv", "tsv"} or path.suffix.lower() in {
            ".csv",
            ".tsv",
            ".txt",
        }:
            delimiter = "\t" if expected_format == "tsv" or path.suffix.lower() in {
                ".tsv",
                ".txt",
            } else ","
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
                header = next(csv.reader(handle, delimiter=delimiter), [])
            if len(header) < 2:
                return "invalid", "tabular file has fewer than two header fields"
    except (OSError, EOFError, tarfile.TarError, zipfile.BadZipFile) as exc:
        return "invalid", f"{type(exc).__name__}: {exc}"
    return "verified", ""


def inspect_source(
    spec: SourceSpec,
    *,
    project_root: Path = PROJECT_ROOT,
    checksums: bool = True,
) -> dict:
    paths = _resolve_patterns(spec.raw_patterns, project_root)
    files: list[dict] = []
    integrity_values: list[str] = []
    for path in paths:
        integrity, error = _integrity(path, spec.expected_format)
        integrity_values.append(integrity)
        item = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "integrity": integrity,
        }
        if error:
            item["integrity_error"] = error
        if checksums:
            item["sha256"] = sha256_file(path)
        files.append(item)
    if not files:
        status = "missing"
    elif any(value == "incomplete" for value in integrity_values):
        status = "raw_incomplete"
    elif any(value == "invalid" for value in integrity_values):
        status = "raw_invalid"
    else:
        status = "raw_verified"
    return {
        **asdict(spec),
        "status": status,
        "matched_files": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def build_source_registry(
    *, project_root: Path = PROJECT_ROOT, checksums: bool = True
) -> dict:
    sources = [
        inspect_source(spec, project_root=project_root, checksums=checksums)
        for spec in DEFAULT_SOURCE_SPECS
    ]
    return {
        "schema_version": "immune_receptor_source_registry.v1",
        "project_root": str(project_root),
        "scope": {
            "included": ["antibody", "tcr"],
            "excluded": ["mint_ppi", "nanobody_vhh", "structure_tasks"],
        },
        "status_order": [
            "missing",
            "raw_verified",
            "parsed",
            "qc_passed",
            "canonicalized",
            "split_ready",
            "export_ready",
        ],
        "sources": sources,
    }


def freeze_benchmark_inputs(
    *, project_root: Path = PROJECT_ROOT, checksums: bool = True
) -> dict:
    """Freeze AB/TCR benchmark inputs while excluding PPI, Nb, and outputs."""

    benchmark_data_root = project_root / "downstream/benchmark/data"
    roots = [
        path
        for path in sorted(benchmark_data_root.iterdir())
        if path.is_dir()
        and (path.name.startswith("tcr_") or path.name.startswith("antibody"))
    ]
    roots.extend(
        path
        for path in (
            project_root / "data/downstream/comp_chain",
            project_root / "data/downstream/cdr_infilling/sabdab",
            project_root / "data/downstream/cdr_infilling/sab23h2_converted",
            project_root / "data/downstream/cdr_infilling/tcr",
            project_root / "data/downstream/humanization/humanisation",
            project_root / "data/downstream/flab/flab_raw",
        )
        if path.is_dir()
    )
    roots = sorted(set(path.resolve() for path in roots))
    files: list[dict] = []
    for root in roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.name.startswith("._") or "__pycache__" in path.parts:
                continue
            relative_lower = str(path.relative_to(project_root)).lower()
            if root.name == "flab_raw":
                partition = "benchmark_all"
            elif "train" in path.name.lower() or "/training" in relative_lower:
                partition = "benchmark_train"
            else:
                partition = "benchmark_eval"
            item = {
                "path": str(path.resolve()),
                "relative_path": relative_lower,
                "bytes": path.stat().st_size,
                "partition": partition,
            }
            if checksums:
                item["sha256"] = sha256_file(path)
            files.append(item)
    return {
        "schema_version": "immune_receptor_benchmark_quarantine.v1",
        "project_root": str(project_root),
        "roots": [str(path) for path in roots],
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "policy": (
            "Files are frozen byte-for-byte. benchmark_eval and benchmark_all entities "
            "are quarantined "
            "from future training; benchmark_train files remain provenance inputs for "
            "task heads and are not scanned as held-out leakage. Exact and near-neighbor "
            "audits are required before export."
        ),
    }
