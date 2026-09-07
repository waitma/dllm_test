"""Canonical adapters for IEDB, PISTE, VDJdb, McPAS, MIRA, and derived TCRs.

Run a bounded build with::

    conda run -n pllm python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py \
      build-tcr --limit-per-source 100
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..schema import (
    CanonicalRecord,
    Eligibility,
    Evidence,
    Measurement,
    SequenceEntity,
    SourceReference,
    clean_text,
    is_valid_sequence,
    normalize_sequence,
    parse_float,
    stable_digest,
)


def _mhc_class(allele: str, fallback: str = "") -> str:
    normalized = allele.upper().replace("HLA-", "")
    if fallback.upper() in {"MHCI", "I", "CLASS I"}:
        return "I"
    if fallback.upper() in {"MHCII", "II", "CLASS II"}:
        return "II"
    if normalized.startswith(("A", "B", "C", "E", "F", "G")):
        return "I"
    if normalized.startswith(("DPA", "DPB", "DQA", "DQB", "DRA", "DRB")):
        return "II"
    return "unknown"


def _entity(
    role: str,
    value: Any,
    scope: str,
    *,
    species: str = "Homo sapiens",
    observed: bool = True,
    reconstructed: bool = False,
    genes: Mapping[str, str] | None = None,
    regions: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SequenceEntity | None:
    sequence = normalize_sequence(value)
    if not is_valid_sequence(sequence):
        return None
    return SequenceEntity(
        role=role,
        sequence=sequence,
        sequence_scope=scope,
        species=species,
        observed=observed,
        reconstructed=reconstructed,
        genes=genes or {},
        regions=regions or {},
        metadata=metadata or {},
    )


def _iter_iedb_metric_rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield IEDB's two-row CSV header as stable ``Section / Field`` keys."""

    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in IEDB metric export {path}: {members}")
        with archive.open(members[0]) as raw:
            handle = io.TextIOWrapper(
                raw, encoding="utf-8-sig", errors="replace", newline=""
            )
            reader = csv.reader(handle)
            sections = next(reader)
            fields = next(reader)
            headers = [f"{section} / {field}" for section, field in zip(sections, fields)]
            for values in reader:
                if len(values) < len(headers):
                    values.extend([""] * (len(headers) - len(values)))
                yield dict(zip(headers, values))


def _iedb_assay_map(path: Path) -> dict[str, dict[str, Any]]:
    assays: dict[str, dict[str, Any]] = {}
    for row in _iter_iedb_metric_rows(path):
        assay_iri = clean_text(row.get("Assay ID / IEDB IRI"))
        assay_id = assay_iri.rstrip("/").split("/")[-1]
        qualitative = clean_text(row.get("Assay / Qualitative Measurement"))
        if qualitative.startswith("Positive"):
            label = 1
        elif qualitative == "Negative":
            label = 0
        else:
            continue
        assays[assay_id] = {
            "label": label,
            "qualitative": qualitative,
            "method": clean_text(row.get("Assay / Method")),
            "readout": clean_text(row.get("Assay / Response measured")),
            "raw_unit": clean_text(row.get("Assay / Units")),
            "raw_value": clean_text(row.get("Assay / Quantitative measurement")),
            "inequality": clean_text(row.get("Assay / Measurement Inequality")),
            "pmid": clean_text(row.get("Reference / PMID")),
            "submission_id": clean_text(row.get("Reference / Submission ID")),
            "epitope_type": clean_text(row.get("Epitope / Object Type")),
            "epitope": normalize_sequence(row.get("Epitope / Name")),
            "host": clean_text(row.get("Host / Name")),
            "mhc_restriction": clean_text(row.get("MHC Restriction / Name")),
            "mhc_evidence": clean_text(row.get("MHC Restriction / Evidence Code")),
        }
    return assays


def _iedb_chain_entity(
    row: Mapping[str, str], chain: str
) -> tuple[SequenceEntity | None, bool]:
    prefix = f"{chain} / "
    chain_type = clean_text(row.get(prefix + "Type")).lower()
    role = {"alpha": "tcr_alpha", "beta": "tcr_beta"}.get(chain_type)
    if not role:
        return None, False
    organism_iri = clean_text(row.get(prefix + "Organism IRI"))
    if organism_iri and "NCBITaxon_9606" not in organism_iri:
        return None, False
    curated_cdr3 = normalize_sequence(row.get(prefix + "CDR3 Curated"))
    calculated_cdr3 = normalize_sequence(row.get(prefix + "CDR3 Calculated"))
    junction = normalize_sequence(row.get(prefix + "Junction Calculated"))
    protein = normalize_sequence(row.get(prefix + "Protein Sequence"))
    variable = normalize_sequence(row.get(prefix + "V Domain Calculated"))
    reconstructed = False
    scope = "cdr3"
    if is_valid_sequence(protein):
        sequence = protein
        scope = "full_chain"
    elif is_valid_sequence(variable):
        sequence = variable
        scope = "variable_domain"
        reconstructed = True
    elif is_valid_sequence(curated_cdr3):
        sequence = curated_cdr3
    elif is_valid_sequence(junction):
        sequence = junction
        reconstructed = True
    elif is_valid_sequence(calculated_cdr3):
        sequence = calculated_cdr3
        reconstructed = True
    else:
        return None, False
    region_cdr3 = next(
        (
            value
            for value in (junction, curated_cdr3, calculated_cdr3)
            if is_valid_sequence(value)
        ),
        "",
    )
    entity = _entity(
        role,
        sequence,
        scope,
        species="Homo sapiens" if "NCBITaxon_9606" in organism_iri else "unknown",
        observed=not reconstructed,
        reconstructed=reconstructed,
        genes={
            "v": clean_text(row.get(prefix + "Curated V Gene"))
            or clean_text(row.get(prefix + "Calculated V Gene")),
            "d": clean_text(row.get(prefix + "Curated D Gene"))
            or clean_text(row.get(prefix + "Calculated D Gene")),
            "j": clean_text(row.get(prefix + "Curated J Gene"))
            or clean_text(row.get(prefix + "Calculated J Gene")),
        },
        regions={"CDR3": region_cdr3},
        metadata={
            "organism_iri": organism_iri,
            "protein_iri": clean_text(row.get(prefix + "Protein IRI")),
            "cdr3_curated": curated_cdr3,
            "cdr3_calculated": calculated_cdr3,
            "junction_calculated": junction,
        },
    )
    return entity, "NCBITaxon_9606" in organism_iri


def _specific_iedb_mhc(value: str) -> str:
    value = clean_text(value)
    if not value or "|" in value or value.lower() in {
        "hla class i",
        "hla class ii",
        "mhc class i",
        "mhc class ii",
    }:
        return ""
    if value.upper().startswith("HLA-") and "*" not in value:
        return ""
    return value


def iter_iedb(
    tcr_path: Path,
    tcell_path: Path,
    *,
    raw_sha256: Mapping[str, str] | None = None,
    limit: int | None = None,
) -> Iterator[CanonicalRecord]:
    """Join IEDB receptor associations to explicit T-cell assay outcomes."""

    assays = _iedb_assay_map(tcell_path)
    emitted = 0
    for line_number, row in enumerate(_iter_iedb_metric_rows(tcr_path), start=3):
        if clean_text(row.get("Receptor / Type")).lower() != "alphabeta":
            continue
        alpha, alpha_human = _iedb_chain_entity(row, "Chain 1")
        beta, beta_human = _iedb_chain_entity(row, "Chain 2")
        by_role = {entity.role: entity for entity in (alpha, beta) if entity}
        alpha = by_role.get("tcr_alpha")
        beta = by_role.get("tcr_beta")
        if not (alpha or beta):
            continue
        peptide = _entity("peptide", row.get("Epitope / Name"), "peptide")
        if not peptide:
            continue
        raw_mhc = clean_text(row.get("Assay / MHC Allele Names"))
        exact_mhc = _specific_iedb_mhc(raw_mhc)
        assay_ids = sorted(set(re.findall(r"\d+", clean_text(row.get("Assay / IEDB IDs")))))
        for assay_id in assay_ids:
            assay = assays.get(assay_id)
            if not assay or assay["epitope_type"].lower() != "linear peptide":
                continue
            if assay["epitope"] and assay["epitope"] != peptide.sequence:
                continue
            paired = bool(alpha and beta)
            human = bool(
                (not alpha or alpha_human) and (not beta or beta_human)
            )
            reconstructed = any(
                entity.reconstructed for entity in (alpha, beta) if entity
            )
            core = paired and human and bool(exact_mhc) and not reconstructed
            if core:
                pack = "tcr_core"
                reasons: tuple[str, ...] = ()
            elif paired and exact_mhc and reconstructed:
                pack = "tcr_aux_computational_sequence"
                reasons = ("one or more receptor sequences are IEDB-calculated",)
            elif paired:
                pack = "tcr_aux_low_confidence"
                reasons = tuple(
                    reason
                    for condition, reason in (
                        (not exact_mhc, "missing or non-specific MHC allele"),
                        (not human, "receptor organism is not explicitly human"),
                    )
                    if condition
                )
            else:
                pack = "tcr_aux_beta"
                reasons = ("unpaired/single-chain receptor",)
            pmid = assay["pmid"]
            study_id = (
                f"PMID:{pmid}"
                if pmid
                else (
                    f"IEDB_SUBMISSION:{assay['submission_id']}"
                    if assay["submission_id"]
                    else clean_text(row.get("Reference / IEDB IRI"))
                )
            )
            label = int(assay["label"])
            yield CanonicalRecord(
                family="tcr_pmhc" if exact_mhc else "tcr_specificity",
                record_type="experimental_specificity",
                entities=tuple(item for item in (alpha, beta, peptide) if item),
                measurement=Measurement(
                    relation="specificity",
                    label=label,
                    raw_value=assay["raw_value"],
                    raw_unit=assay["raw_unit"],
                    direction="higher",
                    assay_type=assay["method"],
                    assay_readout=assay["readout"],
                    negative_type="experimental_negative" if label == 0 else "",
                    confidence=f"iedb:{assay['qualitative']}",
                    metadata={"measurement_inequality": assay["inequality"]},
                ),
                source_records=(
                    SourceReference(
                        source_name="iedb_tcell",
                        source_version="2026-07-28_csv_metric_export",
                        source_record_id=f"receptor_row:{line_number}:assay:{assay_id}",
                        raw_path=str(tcr_path.resolve()),
                        raw_sha256=(raw_sha256 or {}).get(str(tcr_path.resolve()), ""),
                        study_id=study_id,
                        assay_id=f"IEDB:{assay_id}",
                        pmid=pmid,
                        url=f"https://www.iedb.org/assay/{assay_id}",
                        license="CC BY 4.0",
                        metadata={
                            "tcell_raw_path": str(tcell_path.resolve()),
                            "tcell_raw_sha256": (raw_sha256 or {}).get(
                                str(tcell_path.resolve()), ""
                            ),
                        },
                    ),
                ),
                evidence=Evidence(
                    tier="C" if reconstructed else "B",
                    basis="IEDB receptor linked to curated T-cell assay outcome",
                    direct_measurement=True,
                    notes=assay["mhc_evidence"],
                ),
                eligibility=Eligibility(
                    training=True,
                    evaluation=True,
                    core=core,
                    pack=pack,
                    reasons=reasons,
                ),
                context={
                    "mhc_allele": exact_mhc,
                    "mhc_restriction_raw": raw_mhc or assay["mhc_restriction"],
                    "mhc_class": _mhc_class(exact_mhc or raw_mhc),
                    "epitope_source_molecule": clean_text(
                        row.get("Epitope / Source Molecule")
                    ),
                    "epitope_source_organism": clean_text(
                        row.get("Epitope / Source Organism")
                    ),
                    "host": assay["host"],
                },
                flags=tuple(
                    flag
                    for condition, flag in (
                        (not paired, "single_chain"),
                        (reconstructed, "iedb_calculated_receptor_sequence"),
                        (not exact_mhc, "mhc_not_allele_specific"),
                    )
                    if condition
                ),
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def iter_piste(
    piste_root: Path,
    *,
    raw_sha256: Mapping[str, str] | None = None,
    limit: int | None = None,
) -> Iterator[CanonicalRecord]:
    """Parse all released random files while treating their split as provenance only."""

    emitted = 0
    for original_split, filename in (
        ("train", "train_data.csv"),
        ("valid", "val_data.csv"),
        ("test", "test_data.csv"),
    ):
        path = piste_root / filename
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), start=2):
                beta = _entity("tcr_beta", row.get("CDR3"), "cdr3")
                peptide = _entity("peptide", row.get("MT_pep"), "peptide")
                mhc = _entity(
                    "mhc_alpha", row.get("HLA_sequence"), "mhc_pseudosequence"
                )
                if not beta or not peptide or not mhc:
                    continue
                label_value = parse_float(row.get("Label"))
                if label_value not in {0.0, 1.0}:
                    continue
                label = int(label_value)
                allele = clean_text(row.get("HLA_type"))
                negative_type = "" if label else "derived_negative_source_unspecified"
                pack = "tcr_aux_beta" if label else "tcr_synthetic_negative"
                yield CanonicalRecord(
                    family="tcr_pmhc",
                    record_type="aggregate_specificity",
                    entities=(beta, peptide, mhc),
                    measurement=Measurement(
                        relation="binding",
                        label=label,
                        direction="higher",
                        assay_type="PISTE_compilation",
                        negative_type=negative_type,
                        confidence="released_label",
                    ),
                    source_records=(
                        SourceReference(
                            source_name="piste",
                            source_version="local_git_snapshot",
                            source_record_id=f"{original_split}:{line_number}",
                            raw_path=str(path.resolve()),
                            raw_sha256=(raw_sha256 or {}).get(str(path.resolve()), ""),
                            original_split=original_split,
                        ),
                    ),
                    evidence=Evidence(
                        tier="C" if label else "D",
                        basis=(
                            "released aggregate positive"
                            if label
                            else "released derived/sampled negative"
                        ),
                        direct_measurement=False,
                        derived_from_aggregate=True,
                    ),
                    eligibility=Eligibility(
                        training=True,
                        evaluation=True,
                        core=False,
                        pack=pack,
                        reasons=(
                            "beta-only receptor",
                            "PISTE original random split is provenance only",
                        ),
                    ),
                    context={
                        "mhc_allele": allele,
                        "mhc_class": _mhc_class(allele),
                        "peptide_length_reported": row.get("peplen"),
                    },
                    flags=("beta_only", "original_split_not_reused"),
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def _vdjdb_tier(score: float | None) -> str:
    if score is not None and score >= 3:
        return "A"
    if score is not None and score >= 2:
        return "B"
    if score is not None and score >= 1:
        return "C"
    return "D"


def iter_vdjdb(
    path: Path, *, raw_sha256: str = "", limit: int | None = None
) -> Iterator[CanonicalRecord]:
    emitted = 0
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=2):
            species_raw = clean_text(row.get("species"))
            if species_raw.lower().replace(" ", "") not in {
                "homosapiens",
                "human",
            }:
                continue
            alpha = _entity(
                "tcr_alpha",
                row.get("cdr3.alpha"),
                "cdr3",
                genes={"v": clean_text(row.get("v.alpha")), "j": clean_text(row.get("j.alpha"))},
            )
            beta = _entity(
                "tcr_beta",
                row.get("cdr3.beta"),
                "cdr3",
                genes={
                    "v": clean_text(row.get("v.beta")),
                    "d": clean_text(row.get("d.beta")),
                    "j": clean_text(row.get("j.beta")),
                },
            )
            peptide = _entity("peptide", row.get("antigen.epitope"), "peptide")
            if not peptide or not (alpha or beta):
                continue
            allele = clean_text(row.get("mhc.a"))
            paired = bool(alpha and beta)
            has_mhc = bool(allele)
            score = parse_float(row.get("vdjdb.score"))
            core = paired and has_mhc and (score or 0) >= 1
            source_id = clean_text(row.get("TCR_hash")) or f"row:{line_number}"
            entities = tuple(item for item in (alpha, beta, peptide) if item)
            yield CanonicalRecord(
                family="tcr_pmhc" if has_mhc else "tcr_specificity",
                record_type="curated_specificity",
                entities=entities,
                measurement=Measurement(
                    relation="specificity",
                    label=1,
                    direction="higher",
                    assay_type=clean_text(row.get("method.identification")),
                    assay_readout=clean_text(row.get("method.verification")),
                    confidence=f"vdjdb_score:{score}" if score is not None else "",
                ),
                source_records=(
                    SourceReference(
                        source_name="vdjdb",
                        source_version="2025-12-29_local_snapshot",
                        source_record_id=source_id,
                        raw_path=str(path.resolve()),
                        raw_sha256=raw_sha256,
                        study_id=clean_text(row.get("meta.study.id"))
                        or clean_text(row.get("reference.id")),
                        donor_id=clean_text(row.get("meta.subject.id")),
                        doi=clean_text(row.get("reference.id")),
                        metadata={
                            "replica_id": clean_text(row.get("meta.replica.id")),
                            "clone_id": clean_text(row.get("meta.clone.id")),
                        },
                    ),
                ),
                evidence=Evidence(
                    tier=_vdjdb_tier(score),
                    basis="VDJdb curated primary-study record",
                    direct_measurement=bool(
                        clean_text(row.get("method.identification"))
                        or clean_text(row.get("method.verification"))
                    ),
                    derived_from_aggregate=True,
                ),
                eligibility=Eligibility(
                    training=True,
                    evaluation=True,
                    core=core,
                    pack=(
                        "tcr_core"
                        if core
                        else (
                            "tcr_aux_low_confidence"
                            if paired and has_mhc
                            else "tcr_aux_beta"
                        )
                    ),
                    reasons=()
                    if paired and has_mhc
                    else tuple(
                        reason
                        for condition, reason in (
                            (not paired, "unpaired/single-chain receptor"),
                            (not has_mhc, "missing MHC allele context"),
                        )
                        if condition
                    ),
                ),
                context={
                    "mhc_allele": allele,
                    "mhc_beta": clean_text(row.get("mhc.b")),
                    "mhc_class": _mhc_class(allele, clean_text(row.get("mhc.class"))),
                    "antigen_gene": clean_text(row.get("antigen.gene")),
                    "antigen_species": clean_text(row.get("antigen.species")),
                    "cell_subset": clean_text(row.get("meta.cell.subset")),
                    "tissue": clean_text(row.get("meta.tissue")),
                    "single_cell": clean_text(row.get("method.singlecell")),
                },
                flags=() if paired else ("single_chain",),
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def iter_mcpas(
    path: Path, *, raw_sha256: str = "", limit: int | None = None
) -> Iterator[CanonicalRecord]:
    emitted = 0
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            if clean_text(row.get("Species")).lower() not in {"human", "homo sapiens"}:
                continue
            alpha = _entity(
                "tcr_alpha",
                row.get("CDR3.alpha.aa"),
                "cdr3",
                genes={"v": clean_text(row.get("TRAV")), "j": clean_text(row.get("TRAJ"))},
            )
            beta = _entity(
                "tcr_beta",
                row.get("CDR3.beta.aa"),
                "cdr3",
                genes={
                    "v": clean_text(row.get("TRBV")),
                    "d": clean_text(row.get("TRBD")),
                    "j": clean_text(row.get("TRBJ")),
                },
            )
            peptide = _entity("peptide", row.get("Epitope.peptide"), "peptide")
            if not peptide or not (alpha or beta):
                continue
            allele = clean_text(row.get("MHC"))
            paired = bool(alpha and beta)
            single_cell = clean_text(row.get("Single.cell")).lower() == "yes"
            study_id = clean_text(row.get("PubMed.ID"))
            yield CanonicalRecord(
                family="tcr_pmhc" if allele else "tcr_specificity",
                record_type="curated_specificity",
                entities=tuple(item for item in (alpha, beta, peptide) if item),
                measurement=Measurement(
                    relation="specificity",
                    label=1,
                    direction="higher",
                    assay_type=clean_text(row.get("Antigen.identification.method")),
                    confidence="curated_positive",
                ),
                source_records=(
                    SourceReference(
                        source_name="mcpas",
                        source_version="local_snapshot",
                        source_record_id=f"row:{line_number}",
                        raw_path=str(path.resolve()),
                        raw_sha256=raw_sha256,
                        study_id=f"PMID:{study_id}" if study_id else "",
                        pmid=study_id,
                    ),
                ),
                evidence=Evidence(
                    tier="B" if single_cell else "C",
                    basis="McPAS curated primary-study record",
                    direct_measurement=bool(
                        clean_text(row.get("Antigen.identification.method"))
                    ),
                    derived_from_aggregate=True,
                ),
                eligibility=Eligibility(
                    training=True,
                    evaluation=True,
                    core=paired and bool(allele),
                    pack="tcr_core" if paired and allele else "tcr_aux_beta",
                    reasons=()
                    if paired and allele
                    else tuple(
                        reason
                        for condition, reason in (
                            (not paired, "unpaired/single-chain receptor"),
                            (not allele, "missing MHC allele context"),
                        )
                        if condition
                    ),
                ),
                context={
                    "mhc_allele": allele,
                    "mhc_class": _mhc_class(allele),
                    "pathology": clean_text(row.get("Pathology")),
                    "category": clean_text(row.get("Category")),
                    "antigen_protein": clean_text(row.get("Antigen.protein")),
                    "protein_id": clean_text(row.get("Protein.ID")),
                    "tissue": clean_text(row.get("Tissue")),
                    "single_cell": single_cell,
                },
                flags=() if paired else ("single_chain",),
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def iter_mira(
    mira_root: Path,
    *,
    raw_sha256: Mapping[str, str] | None = None,
    limit: int | None = None,
) -> Iterator[CanonicalRecord]:
    emitted = 0
    for mhc_class, filename in (
        ("I", "peptide-detail-ci.csv"),
        ("II", "peptide-detail-cii.csv"),
    ):
        path = mira_root / filename
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), start=2):
                bioidentity = clean_text(row.get("TCR BioIdentity"))
                parts = bioidentity.split("+")
                beta = _entity(
                    "tcr_beta",
                    parts[0] if parts else "",
                    "cdr3",
                    genes={
                        "v": parts[1] if len(parts) > 1 else "",
                        "j": parts[2] if len(parts) > 2 else "",
                    },
                )
                pool = [
                    normalize_sequence(item)
                    for item in clean_text(row.get("Amino Acids")).split(",")
                    if is_valid_sequence(normalize_sequence(item))
                ]
                if not beta or not pool:
                    continue
                experiment = clean_text(row.get("Experiment"))
                pool_id = stable_digest(sorted(pool), prefix="pool_")
                yield CanonicalRecord(
                    family="tcr_specificity_pool",
                    record_type="pool_measurement",
                    entities=(beta,),
                    measurement=Measurement(
                        relation="specificity",
                        label=1,
                        direction="higher",
                        assay_type="MIRA peptide pool",
                        confidence="pool_level_only",
                    ),
                    source_records=(
                        SourceReference(
                            source_name="mira",
                            source_version="ImmuneCODE-MIRA-Release002.1",
                            source_record_id=f"{filename}:{line_number}",
                            raw_path=str(path.resolve()),
                            raw_sha256=(raw_sha256 or {}).get(str(path.resolve()), ""),
                            assay_id=experiment,
                        ),
                    ),
                    evidence=Evidence(
                        tier="C",
                        basis="MIRA pool-level response",
                        direct_measurement=True,
                    ),
                    eligibility=Eligibility(
                        training=True,
                        evaluation=False,
                        core=False,
                        pack="tcr_aux_pool",
                        reasons=(
                            "beta-only receptor",
                            "response resolves to a peptide pool, not one peptide",
                        ),
                    ),
                    context={
                        "mhc_class": mhc_class,
                        "peptide_pool": pool,
                        "peptide_pool_id": pool_id,
                        "experiment": experiment,
                        "orf_coverage": clean_text(row.get("ORF Coverage")),
                    },
                    groups={"peptide_pool": pool_id},
                    flags=("beta_only", "peptide_pool_ambiguous"),
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def _load_fullchain_cdr3_map(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    if not path.exists():
        return mapping
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            alpha = normalize_sequence(row.get("TRA_aa"))
            beta = normalize_sequence(row.get("TRB_aa"))
            if alpha and beta:
                mapping[(alpha, beta)] = {
                    "tcr_name": clean_text(row.get("TCR_name")),
                    "cdr3_alpha": normalize_sequence(row.get("TRA_CDR3")),
                    "cdr3_beta": normalize_sequence(row.get("TRB_CDR3")),
                    "trav": clean_text(row.get("TRAV")),
                    "traj": clean_text(row.get("TRAJ")),
                    "trbv": clean_text(row.get("TRBV")),
                    "trbj": clean_text(row.get("TRBJ")),
                }
    return mapping


def iter_fullchain_derived(
    root: Path,
    *,
    raw_sha256: Mapping[str, str] | None = None,
    limit: int | None = None,
) -> Iterator[CanonicalRecord]:
    cdr3_map = _load_fullchain_cdr3_map(root / "thimble_out.tsv")
    emitted = 0
    for original_split in ("train", "valid", "holdout"):
        path = root / f"records_{original_split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                chains = row.get("chains", [])
                if len(chains) != 5:
                    continue
                mhc, b2m, peptide, alpha, beta = map(normalize_sequence, chains)
                linkage = cdr3_map.get((alpha, beta), {})
                alpha_entity = _entity(
                    "tcr_alpha",
                    alpha,
                    "full_chain",
                    observed=False,
                    reconstructed=True,
                    genes={"v": linkage.get("trav", ""), "j": linkage.get("traj", "")},
                    regions={"CDR3": linkage.get("cdr3_alpha", "")},
                )
                beta_entity = _entity(
                    "tcr_beta",
                    beta,
                    "full_chain",
                    observed=False,
                    reconstructed=True,
                    genes={"v": linkage.get("trbv", ""), "j": linkage.get("trbj", "")},
                    regions={"CDR3": linkage.get("cdr3_beta", "")},
                )
                entities = (
                    _entity(
                        "mhc_alpha",
                        mhc,
                        "mhc_full_chain",
                        observed=False,
                        reconstructed=True,
                    ),
                    _entity(
                        "mhc_beta2m",
                        b2m,
                        "mhc_full_chain",
                        observed=False,
                        reconstructed=True,
                    ),
                    _entity("peptide", peptide, "peptide"),
                    alpha_entity,
                    beta_entity,
                )
                if any(item is None for item in entities):
                    continue
                metadata = row.get("metadata", {})
                cdr3_pair = [
                    linkage.get("cdr3_alpha", alpha),
                    linkage.get("cdr3_beta", beta),
                ]
                source_link = linkage.get("tcr_name") or (
                    f"{metadata.get('origin', 'unknown')}:{line_number}"
                )
                yield CanonicalRecord(
                    family="tcr_pmhc",
                    record_type="derived_view",
                    entities=tuple(entities),  # type: ignore[arg-type]
                    measurement=Measurement(
                        relation="binding",
                        label=1,
                        direction="higher",
                        assay_type="inherited from source specificity record",
                        confidence="derived_not_independent",
                    ),
                    source_records=(
                        SourceReference(
                            source_name="tcr_pmhc_fulllength_derived",
                            source_version="local_build",
                            source_record_id=f"{original_split}:{line_number}",
                            raw_path=str(path.resolve()),
                            raw_sha256=(raw_sha256 or {}).get(str(path.resolve()), ""),
                            original_split=original_split,
                            metadata={"origin": metadata.get("origin", "")},
                        ),
                    ),
                    evidence=Evidence(
                        tier="C",
                        basis="Thimble reconstruction of VDJdb/McPAS receptor record",
                        direct_measurement=False,
                        derived_from_aggregate=True,
                        notes="Not additive evidence; use as a sequence view only.",
                    ),
                    eligibility=Eligibility(
                        training=True,
                        evaluation=False,
                        core=False,
                        pack="tcr_fullchain_derived",
                        reasons=(
                            "reconstructed full chains",
                            "source evidence represented elsewhere",
                            "legacy random split is not reused",
                        ),
                    ),
                    context={
                        "mhc_allele": clean_text(metadata.get("mhc_a")),
                        "mhc_class": "I",
                        "source_origin": clean_text(metadata.get("origin")),
                        "source_link": source_link,
                    },
                    groups={
                        "receptor": stable_digest(cdr3_pair, prefix="rec_"),
                    },
                    derived_from=(source_link,),
                    flags=(
                        "reconstructed_tcr",
                        "derived_view_not_independent_evidence",
                        "original_split_not_reused",
                    ),
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
