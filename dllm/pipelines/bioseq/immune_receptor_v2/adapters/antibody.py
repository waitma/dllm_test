"""Canonical adapters for AbRank, CATNAP, SAbDab2, FLAb, and CoV-AbDab.

Run a bounded build with::

    conda run -n pllm python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py \
      build-antibody --limit-per-source 100
"""

from __future__ import annotations

import csv
import io
import json
import re
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..schema import (
    CanonicalRecord,
    Eligibility,
    Evidence,
    Measurement,
    SequenceEntity,
    SourceReference,
    TargetMapping,
    clean_text,
    is_valid_sequence,
    normalize_sequence,
    parse_float,
)


def _scope(sequence: str, *, antigen: bool = False) -> str:
    if antigen:
        return "antigen_construct"
    return "full_chain" if len(sequence) > 180 else "variable_domain"


def _entity(
    role: str,
    value: Any,
    *,
    sequence_scope: str | None = None,
    regions: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SequenceEntity | None:
    sequence = normalize_sequence(value)
    if not is_valid_sequence(sequence):
        return None
    return SequenceEntity(
        role=role,
        sequence=sequence,
        sequence_scope=sequence_scope
        or _scope(sequence, antigen=role == "antigen"),
        species="",
        regions=regions or {},
        metadata=metadata or {},
    )


def _censor(value: Any) -> str:
    text = clean_text(value)
    return text if text in {"=", "<", "<=", ">", ">=", "range"} else "="


def _abrank_measurement(row: Mapping[str, Any]) -> Measurement | None:
    kd_nm = parse_float(row.get("Affinity_Kd [nM]"))
    if kd_nm is not None:
        return Measurement(
            relation="affinity",
            raw_value=kd_nm,
            raw_unit="nM",
            value=kd_nm * 1e-9,
            unit="M",
            censor=_censor(row.get("Aff_op")),
            direction="lower",
            assay_type="affinity Kd",
        )
    ic50 = parse_float(row.get("IC50 [ug/mL]"))
    if ic50 is not None:
        return Measurement(
            relation="neutralization",
            raw_value=ic50,
            raw_unit="ug/mL",
            value=ic50,
            unit="ug/mL",
            direction="lower",
            assay_type="IC50",
        )
    fitness = parse_float(row.get("fitness"))
    if fitness is not None:
        return Measurement(
            relation="affinity_surrogate",
            raw_value=row.get("log_Aff") or row.get("fitness"),
            raw_unit="released transformed score",
            value=fitness,
            unit="released_fitness",
            censor=_censor(row.get("Aff_op")),
            direction="unknown",
            assay_type="AbRank transformed target",
            confidence="surrogate_only",
        )
    return None


def iter_abrank(
    path: Path, *, raw_sha256: str = "", limit: int | None = None
) -> Iterator[CanonicalRecord]:
    emitted = 0
    with zipfile.ZipFile(path) as archive:
        member = next(
            item
            for item in archive.infolist()
            if item.filename == "AbRank_dataset.csv"
        )
        with archive.open(member) as binary:
            handle = io.TextIOWrapper(
                binary, encoding="utf-8-sig", errors="replace", newline=""
            )
            for line_number, row in enumerate(csv.DictReader(handle), start=2):
                heavy = _entity("antibody_heavy", row.get("Ab_heavy_chain_seq"))
                light = _entity("antibody_light", row.get("Ab_light_chain_seq"))
                if not heavy or not light:
                    continue
                antigen = _entity("antigen", row.get("Ag_seq"))
                measurement = _abrank_measurement(row)
                if not measurement:
                    continue
                exact_target = antigen is not None
                direct_measurement = measurement.relation in {
                    "affinity",
                    "neutralization",
                }
                target_name = clean_text(row.get("Ag_name_details")) or clean_text(
                    row.get("Ag_name")
                )
                entities = (heavy, light) + ((antigen,) if antigen else ())
                flags = []
                if not exact_target:
                    flags.append("antigen_sequence_missing_or_nonsequence_expression")
                if antigen and len(antigen.sequence) > 1024:
                    flags.append("long_antigen_preserved_not_truncated")
                yield CanonicalRecord(
                    family="antibody_antigen",
                    record_type="interaction_measurement",
                    entities=entities,
                    measurement=measurement,
                    source_records=(
                        SourceReference(
                            source_name="abrank",
                            source_version="FLAb_local_snapshot",
                            source_record_id=f"row:{line_number}",
                            raw_path=str(path.resolve()),
                            raw_sha256=raw_sha256,
                            study_id=clean_text(row.get("Source")),
                            metadata={
                                "antibody_name": clean_text(row.get("Ab_name")),
                                "antigen_name": clean_text(row.get("Ag_name")),
                            },
                        ),
                    ),
                    evidence=Evidence(
                        tier="B" if direct_measurement and exact_target else "C",
                        basis="AbRank released interaction row",
                        direct_measurement=direct_measurement,
                        derived_from_aggregate=True,
                    ),
                    eligibility=Eligibility(
                        training=True,
                        evaluation=True,
                        core=direct_measurement and exact_target,
                        pack=(
                            "antibody_antigen_exact"
                            if direct_measurement and exact_target
                            else "antibody_antigen_aux"
                        ),
                        reasons=()
                        if direct_measurement and exact_target
                        else tuple(
                            reason
                            for condition, reason in (
                                (not exact_target, "no exact antigen sequence"),
                                (not direct_measurement, "released transformed target only"),
                            )
                            if condition
                        ),
                    ),
                    context={
                        "antibody_name": clean_text(row.get("Ab_name")),
                        "antigen_name": clean_text(row.get("Ag_name")),
                        "antigen_details": clean_text(row.get("Ag_name_details")),
                        "epitope_restrictions": clean_text(
                            row.get("Ag_epitope_restrictions")
                        ),
                        "antibody_cluster_lev3": clean_text(
                            row.get("Ab_Lev3_cluster")
                        ),
                        "antigen_cluster_lev3": clean_text(row.get("Ag_Lev3_cluster")),
                    },
                    target_mapping=TargetMapping(
                        target_name_raw=target_name,
                        canonical_target_id=clean_text(row.get("Ag_name")),
                        construct=target_name,
                        mapping_confidence="exact_sequence" if antigen else "name_only",
                        mapping_source="AbRank released Ag_seq",
                    ),
                    flags=tuple(flags),
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def _metadata_by_filename(root: Path) -> dict[str, dict[str, str]]:
    path = root / "flab_metadata.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        return {
            row["filename"]: row
            for row in csv.DictReader(handle)
            if clean_text(row.get("filename"))
        }


def iter_kothiwal(
    binding_root: Path,
    *,
    raw_sha256: Mapping[str, str] | None = None,
    limit: int | None = None,
) -> Iterator[CanonicalRecord]:
    metadata = _metadata_by_filename(binding_root.parent)
    emitted = 0
    for path in sorted(binding_root.glob("kothiwal2025htp_*.csv")):
        assay_kind = "spr" if path.stem.endswith("_spr") else "ec50"
        meta = metadata.get(path.name, {})
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), start=2):
                heavy = _entity("antibody_heavy", row.get("heavy"))
                light = _entity("antibody_light", row.get("light"))
                antigen = _entity("antigen", row.get("antigen_seq"))
                if not heavy or not light or not antigen:
                    continue
                if assay_kind == "spr":
                    raw_value = parse_float(row.get("SPR kinetics - KD (nM)"))
                    assay_type = "SPR Kd"
                else:
                    raw_value = parse_float(row.get("Cell Display  - EC50 (nM)"))
                    assay_type = "cell display EC50"
                if raw_value is None:
                    continue
                target_name = clean_text(row.get("Antigen"))
                flags = (
                    ("long_antigen_preserved_not_truncated",)
                    if len(antigen.sequence) > 1024
                    else ()
                )
                yield CanonicalRecord(
                    family="antibody_antigen",
                    record_type="interaction_measurement",
                    entities=(heavy, light, antigen),
                    measurement=Measurement(
                        relation="affinity",
                        raw_value=raw_value,
                        raw_unit="nM",
                        value=raw_value * 1e-9,
                        unit="M",
                        direction="lower",
                        assay_type=assay_type,
                        metadata={
                            "fitness": parse_float(row.get("fitness")),
                            "ka_1_per_Ms": parse_float(
                                row.get("SPR Kinetics - ka (1/Ms)")
                            ),
                            "kdis_1_per_s": parse_float(
                                row.get("SPR Kinetics - kdis (1/s)")
                            ),
                        },
                    ),
                    source_records=(
                        SourceReference(
                            source_name="kothiwal2025",
                            source_version="FLAb_local_snapshot",
                            source_record_id=f"{path.name}:{line_number}",
                            raw_path=str(path.resolve()),
                            raw_sha256=(raw_sha256 or {}).get(str(path.resolve()), ""),
                            study_id=clean_text(meta.get("publication_title")),
                            doi=clean_text(meta.get("doi")),
                            license=clean_text(meta.get("license")),
                            metadata={
                                "antibody_name": clean_text(row.get("Antibody_Name")),
                                "quality_control_psr": clean_text(
                                    row.get("Quality control - PSR ")
                                ),
                                "quality_control_sec": clean_text(
                                    row.get("Quality control - SEC")
                                ),
                            },
                        ),
                    ),
                    evidence=Evidence(
                        tier="A",
                        basis="released Kothiwal SPR/cell-display measurement",
                        direct_measurement=True,
                    ),
                    eligibility=Eligibility(
                        training=True,
                        evaluation=True,
                        core=True,
                        pack="antibody_antigen_exact",
                    ),
                    context={
                        "antibody_name": clean_text(row.get("Antibody_Name")),
                        "antigen_name": target_name,
                        "arm": clean_text(row.get("Antigen Recognition Module (ARM)")),
                        "light_chain_family": clean_text(row.get("LC")),
                    },
                    target_mapping=TargetMapping(
                        target_name_raw=target_name,
                        canonical_target_id=target_name,
                        construct="released antigen_seq including any tags/multimerization",
                        mapping_confidence="exact_sequence",
                        mapping_source=path.name,
                    ),
                    flags=flags,
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def _fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    header = ""
    parts: list[str] = []
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                if header:
                    yield header, "".join(parts)
                header = line[1:].strip()
                parts = []
            else:
                parts.append(line.strip())
    if header:
        yield header, "".join(parts)


def _catnap_file(root: Path, stem: str) -> Path:
    matches = sorted(root.glob(f"{stem}_*"))
    if len(matches) != 1:
        raise ValueError(f"Expected one CATNAP {stem} file under {root}: {matches}")
    return matches[0]


def _catnap_antibody_sequences(root: Path) -> dict[str, tuple[str, str]]:
    by_chain: dict[str, dict[str, set[str]]] = {}
    for chain, stem in (
        ("heavy", "heavy_seqs_aa"),
        ("light", "light_seqs_aa"),
    ):
        by_id: dict[str, set[str]] = defaultdict(set)
        for header, raw_sequence in _fasta_records(_catnap_file(root, stem)):
            match = re.search(r"ImmDBID(\d+)", header, flags=re.I)
            sequence = normalize_sequence(raw_sequence.replace("-", ""))
            if match and is_valid_sequence(sequence):
                by_id[match.group(1)].add(sequence)
        by_chain[chain] = by_id
    metadata_path = _catnap_file(root, "abs")
    sequences: dict[str, tuple[str, str]] = {}
    with metadata_path.open(
        encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = clean_text(row.get("Name"))
            immunodb_id = clean_text(row.get("Immuno DB ID"))
            heavy = by_chain["heavy"].get(immunodb_id, set())
            light = by_chain["light"].get(immunodb_id, set())
            if name and len(heavy) == 1 and len(light) == 1:
                sequences[name] = (next(iter(heavy)), next(iter(light)))
    return sequences


def _catnap_virus_sequences(root: Path) -> dict[str, tuple[str, str, dict[str, str]]]:
    by_accession: dict[str, str] = {}
    terminal_stop_by_accession: dict[str, bool] = {}
    accession_pattern = re.compile(r"(?:^|[. ])([A-Z]{1,4}\d{5,8})(?:\.\d+)?(?:$|[. ])")
    for header, raw_sequence in _fasta_records(_catnap_file(root, "virseqs_aa")):
        aligned_sequence = normalize_sequence(
            raw_sequence.replace("-", "").replace(".", "")
        )
        sequence = aligned_sequence.rstrip("*")
        if not is_valid_sequence(sequence):
            continue
        for accession in accession_pattern.findall(header.upper()):
            by_accession.setdefault(accession, sequence)
            terminal_stop_by_accession.setdefault(
                accession, aligned_sequence != sequence
            )
    viruses: dict[str, tuple[str, str, dict[str, str]]] = {}
    with _catnap_file(root, "viruses").open(
        encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = clean_text(row.get("---Virus name"))
            accessions = [
                item.split(".")[0].upper()
                for item in re.split(r"[,;\s]+", clean_text(row.get("Accession")))
                if item
            ]
            matches = [(accession, by_accession[accession]) for accession in accessions if accession in by_accession]
            unique_sequences = {sequence for _, sequence in matches}
            if name and len(unique_sequences) == 1:
                accession = next(
                    accession for accession, sequence in matches if sequence in unique_sequences
                )
                metadata = dict(row)
                metadata["terminal_stop_removed"] = str(
                    terminal_stop_by_accession.get(accession, False)
                ).lower()
                viruses[name] = (next(iter(unique_sequences)), accession, metadata)
    return viruses


def _catnap_value(value: Any) -> tuple[float | None, str]:
    text = clean_text(value)
    match = re.fullmatch(
        r"\s*(<=|>=|<|>)?\s*([0-9]+(?:\.[0-9]*)?(?:[Ee][+-]?\d+)?)\s*",
        text,
    )
    if not match:
        return None, "="
    return float(match.group(2)), match.group(1) or "="


def iter_catnap(
    root: Path,
    *,
    raw_sha256: Mapping[str, str] | None = None,
    limit: int | None = None,
) -> Iterator[CanonicalRecord]:
    """Parse independent CATNAP IC50/IC80 rows with paired H/L and Env sequence."""

    antibodies = _catnap_antibody_sequences(root)
    viruses = _catnap_virus_sequences(root)
    assay_path = _catnap_file(root, "assay")
    emitted = 0
    with assay_path.open(
        encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        for line_number, row in enumerate(
            csv.DictReader(handle, delimiter="\t"), start=2
        ):
            antibody_name = clean_text(row.get("Antibody"))
            virus_name = clean_text(row.get("Virus"))
            receptor = antibodies.get(antibody_name)
            target = viruses.get(virus_name)
            if not receptor or not target:
                continue
            heavy = _entity("antibody_heavy", receptor[0])
            light = _entity("antibody_light", receptor[1])
            antigen = _entity("antigen", target[0], sequence_scope="antigen_construct")
            if not heavy or not light or not antigen:
                continue
            _, accession, virus_metadata = target
            pmid = clean_text(row.get("Pubmed ID"))
            reference = clean_text(row.get("Reference"))
            for column in ("IC50", "IC80"):
                value, censor = _catnap_value(row.get(column))
                if value is None:
                    continue
                yield CanonicalRecord(
                    family="antibody_antigen",
                    record_type="neutralization_measurement",
                    entities=(heavy, light, antigen),
                    measurement=Measurement(
                        relation="neutralization",
                        raw_value=clean_text(row.get(column)),
                        raw_unit="ug/mL",
                        value=value,
                        unit="ug/mL",
                        censor=censor,
                        direction="lower",
                        assay_type=column,
                        assay_readout="TZM-bl pseudovirus neutralization",
                        confidence="independent_CATNAP_value",
                    ),
                    source_records=(
                        SourceReference(
                            source_name="catnap",
                            source_version=root.name,
                            source_record_id=f"assay_row:{line_number}:{column}",
                            raw_path=str(assay_path.resolve()),
                            raw_sha256=(raw_sha256 or {}).get(
                                str(assay_path.resolve()), ""
                            ),
                            study_id=f"PMID:{pmid}" if pmid else reference,
                            pmid=pmid,
                            url="https://www.hiv.lanl.gov/catnap",
                            metadata={
                                "reference": reference,
                                "snapshot_root": str(root.resolve()),
                            },
                        ),
                    ),
                    evidence=Evidence(
                        tier="B",
                        basis=(
                            "CATNAP independent pseudovirus neutralization value "
                            "joined to the released associated Env accession"
                        ),
                        direct_measurement=True,
                        notes=(
                            "The neutralization measurement is direct, but CATNAP warns "
                            "that the associated deposited Env may differ from the exact "
                            "assay clone."
                        ),
                    ),
                    eligibility=Eligibility(
                        training=True,
                        evaluation=True,
                        core=True,
                        pack="antibody_antigen_exact_accession",
                    ),
                    context={
                        "antibody_name": antibody_name,
                        "antigen_name": virus_name,
                        "virus_subtype": clean_text(virus_metadata.get("Subtype")),
                        "virus_organism": clean_text(virus_metadata.get("Organism")),
                        "virus_type": clean_text(virus_metadata.get("Virus type")),
                        "virus_tier": clean_text(virus_metadata.get("Tier")),
                        "assay_system": "TZM-bl pseudovirus",
                    },
                    target_mapping=TargetMapping(
                        target_name_raw=virus_name,
                        canonical_target_id=f"CATNAP:{virus_name}",
                        accession=accession,
                        construct="CATNAP-associated Env amino-acid sequence",
                        mapping_confidence="exact_accession",
                        mapping_source=f"CATNAP {root.name} virus metadata + Env alignment",
                        notes=(
                            "Associated Env sequence; CATNAP notes that in some studies the "
                            "assay clone may differ from the deposited accession."
                        ),
                    ),
                    flags=tuple(
                        ["catnap_associated_env_sequence"]
                        + (
                            ["terminal_stop_codon_removed"]
                            if virus_metadata.get("terminal_stop_removed") == "true"
                            else []
                        )
                    ),
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def iter_sabdab2(
    path: Path, *, raw_sha256: str = "", limit: int | None = None
) -> Iterator[CanonicalRecord]:
    """Parse paired H/L protein/peptide complexes from the antigen-aware split."""

    emitted = 0
    with tarfile.open(path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith("/abag_split.csv")
        ]
        if len(members) != 1:
            raise ValueError(
                f"Expected one antigen-aware abag_split.csv in {path}: "
                f"{[member.name for member in members]}"
            )
        binary = archive.extractfile(members[0])
        if binary is None:
            raise ValueError(f"Cannot read {members[0].name} from {path}")
        handle = io.TextIOWrapper(
            binary, encoding="utf-8-sig", errors="replace", newline=""
        )
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            heavy = _entity(
                "antibody_heavy",
                row.get("VH_numerable_seq"),
                sequence_scope="variable_domain",
                regions={
                    "CDR1": normalize_sequence(row.get("CDRH1")),
                    "CDR2": normalize_sequence(row.get("CDRH2")),
                    "CDR3": normalize_sequence(row.get("CDRH3")),
                },
                metadata={"chain_id": clean_text(row.get("Hchain"))},
            )
            light = _entity(
                "antibody_light",
                row.get("VL_numerable_seq"),
                sequence_scope="variable_domain",
                regions={
                    "CDR1": normalize_sequence(row.get("CDRL1")),
                    "CDR2": normalize_sequence(row.get("CDRL2")),
                    "CDR3": normalize_sequence(row.get("CDRL3")),
                },
                metadata={"chain_id": clean_text(row.get("Lchain"))},
            )
            if not heavy or not light:
                continue
            antigen_chains = clean_text(row.get("agchains")).split("/")
            antigen_types = clean_text(row.get("agtypes")).split("/")
            antigen_sequences = clean_text(row.get("agresolvedseqs")).split("/")
            components = []
            for component_index, (chain, antigen_type, sequence) in enumerate(
                zip(antigen_chains, antigen_types, antigen_sequences)
            ):
                antigen_type = antigen_type.upper()
                if antigen_type not in {"PROTEIN", "PEPTIDE"}:
                    continue
                antigen = _entity(
                    "antigen",
                    sequence,
                    sequence_scope="antigen_construct",
                    metadata={
                        "chain_id": chain,
                        "antigen_type": antigen_type,
                        "component_index": component_index,
                    },
                )
                if antigen:
                    components.append((component_index, chain, antigen_type, antigen))
            if not components:
                continue
            total_components = max(
                len(antigen_chains), len(antigen_types), len(antigen_sequences)
            )
            single_polymer_complex = total_components == 1 and len(components) == 1
            pdb_id = clean_text(row.get("PDB_ID"))
            instance = clean_text(row.get("INSTANCE"))
            original_split = clean_text(row.get("ab_ag_split")) or clean_text(
                row.get("abag_split")
            )
            for component_index, chain, antigen_type, antigen in components:
                yield CanonicalRecord(
                    family="antibody_antigen",
                    record_type=(
                        "structural_complex"
                        if single_polymer_complex
                        else "structural_complex_component"
                    ),
                    entities=(heavy, light, antigen),
                    measurement=Measurement(
                        relation="structural_binding",
                        label=1,
                        direction="none",
                        assay_type=clean_text(row.get("method")),
                        assay_readout="resolved antibody-antigen complex",
                        confidence="SAbDab2_curated_complex",
                        metadata={"resolution_angstrom": parse_float(row.get("resolution"))},
                    ),
                    source_records=(
                        SourceReference(
                            source_name="sabdab2_ml",
                            source_version="0.1.0_zenodo_20083995",
                            source_record_id=(
                                f"{instance}:antigen_component:{component_index}"
                            ),
                            raw_path=str(path.resolve()),
                            raw_sha256=raw_sha256,
                            original_split=original_split,
                            study_id=f"PDB:{pdb_id}" if pdb_id else instance,
                            url="https://doi.org/10.5281/zenodo.20083995",
                            metadata={
                                "csv_member": members[0].name,
                                "sabdab_id": clean_text(row.get("SABDAB_ID")),
                            },
                        ),
                    ),
                    evidence=Evidence(
                        tier="A",
                        basis="SAbDab2 curated resolved antibody-antigen complex",
                        direct_measurement=True,
                    ),
                    eligibility=Eligibility(
                        training=True,
                        evaluation=True,
                        core=single_polymer_complex,
                        pack=(
                            "antibody_antigen_exact"
                            if single_polymer_complex
                            else "antibody_antigen_aux"
                        ),
                        reasons=(
                            ()
                            if single_polymer_complex
                            else (
                                "multi-component antigen complex; component-level "
                                "association retained without treating it as the full complex",
                            )
                        ),
                    ),
                    context={
                        "pdb_id": pdb_id,
                        "sabdab_id": clean_text(row.get("SABDAB_ID")),
                        "instance": instance,
                        "antibody_type": clean_text(row.get("type")),
                        "antibody_construct": clean_text(row.get("construct")),
                        "antibody_species": clean_text(row.get("species")),
                        "antigen_chain": chain,
                        "antigen_type": antigen_type,
                        "antigen_component_index": component_index,
                        "antigen_component_count": total_components,
                        "released_antibody_cluster": clean_text(row.get("ab_cluster")),
                        "released_antigen_clusters": clean_text(row.get("agclusters")),
                        "released_antibody_antigen_cluster": clean_text(
                            row.get("ab_ag_cluster")
                        ),
                    },
                    target_mapping=TargetMapping(
                        target_name_raw=f"{pdb_id}:{chain}",
                        canonical_target_id=f"PDB:{pdb_id}:{chain}",
                        accession=pdb_id,
                        construct="resolved antigen polymer in deposited structure",
                        mapping_confidence="exact_sequence",
                        mapping_source="SAbDab2 abag_split.csv agresolvedseqs",
                    ),
                    flags=tuple(
                        ["official_split_retained_as_provenance_only"]
                        + (
                            ["multi_component_antigen"]
                            if not single_polymer_complex
                            else []
                        )
                    ),
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


_PROPERTY_EXCLUDE_COLUMNS = {
    "heavy",
    "light",
    "fitness",
    "format",
    "name",
    "antibody_name",
    "sequence",
}


def _property_value_column(
    row: Mapping[str, Any], metadata: Mapping[str, Any]
) -> str:
    preferred = clean_text(metadata.get("assay/units_raw"))
    if preferred in row and parse_float(row.get(preferred)) is not None:
        return preferred
    candidates = [
        key
        for key, value in row.items()
        if key.lower().strip() not in _PROPERTY_EXCLUDE_COLUMNS
        and parse_float(value) is not None
    ]
    return candidates[0] if candidates else ""


def iter_flab_properties(
    flab_root: Path,
    *,
    raw_sha256: Mapping[str, str] | None = None,
    limit: int | None = None,
) -> Iterator[CanonicalRecord]:
    """Parse intrinsic paired-H/L property tables; binding and VHH are excluded."""

    metadata = _metadata_by_filename(flab_root)
    emitted = 0
    categories = (
        "aggregation",
        "expression",
        "immunogenicity",
        "pharmacokinetics",
        "polyreactivity",
        "thermostability",
    )
    for category in categories:
        for path in sorted((flab_root / category).glob("*.csv")):
            meta = metadata.get(path.name, {})
            with path.open(
                encoding="utf-8-sig", errors="replace", newline=""
            ) as handle:
                for line_number, row in enumerate(csv.DictReader(handle), start=2):
                    heavy = _entity("antibody_heavy", row.get("heavy"))
                    light = _entity("antibody_light", row.get("light"))
                    if not heavy or not light:
                        continue
                    value_column = _property_value_column(row, meta)
                    raw_value = parse_float(row.get(value_column))
                    fitness = parse_float(row.get("fitness"))
                    if raw_value is None and fitness is None:
                        continue
                    assay = clean_text(meta.get("assay/units")) or value_column
                    doi = clean_text(meta.get("doi"))
                    study = clean_text(meta.get("publication_title"))
                    parent_id = doi or re.sub(r"_[^_]+$", "", path.stem)
                    yield CanonicalRecord(
                        family="antibody_property",
                        record_type="intrinsic_property_measurement",
                        entities=(heavy, light),
                        measurement=Measurement(
                            relation=category,
                            raw_value=raw_value if raw_value is not None else row.get("fitness"),
                            raw_unit=value_column,
                            value=fitness if fitness is not None else raw_value,
                            unit="released_fitness" if fitness is not None else value_column,
                            direction="unknown",
                            assay_type=assay,
                            assay_readout=value_column,
                            metadata={"released_fitness": fitness},
                        ),
                        source_records=(
                            SourceReference(
                                source_name="flab_properties",
                                source_version="local_git_snapshot",
                                source_record_id=f"{path.name}:{line_number}",
                                raw_path=str(path.resolve()),
                                raw_sha256=(raw_sha256 or {}).get(
                                    str(path.resolve()), ""
                                ),
                                study_id=study or parent_id,
                                doi=doi,
                                license=clean_text(meta.get("license")),
                            ),
                        ),
                        evidence=Evidence(
                            tier="B" if raw_value is not None else "C",
                            basis="FLAb released intrinsic-property table",
                            direct_measurement=raw_value is not None,
                            derived_from_aggregate=False,
                        ),
                        eligibility=Eligibility(
                            training=True,
                            evaluation=True,
                            core=raw_value is not None,
                            pack=f"antibody_property_{category}",
                            reasons=()
                            if raw_value is not None
                            else ("only released normalized fitness is available",),
                        ),
                        context={
                            "property_category": category,
                            "source_file": path.name,
                            "parent_id": parent_id,
                            "study": study,
                        },
                        flags=("direction_not_yet_canonicalized",),
                    )
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return


def _targets(value: Any) -> list[str]:
    return [clean_text(item) for item in str(value or "").split(";") if clean_text(item)]


def _cov_receptor_entities(row: Mapping[str, Any]) -> tuple[SequenceEntity, SequenceEntity] | None:
    heavy = _entity("antibody_heavy", row.get("VHorVHH"))
    light = _entity("antibody_light", row.get("VL"))
    if heavy and light:
        return heavy, light
    heavy = _entity(
        "antibody_heavy", row.get("CDRH3"), sequence_scope="cdr3"
    )
    light = _entity("antibody_light", row.get("CDRL3"), sequence_scope="cdr3")
    return (heavy, light) if heavy and light else None


def iter_cov_abdab(
    path: Path, *, raw_sha256: str = "", limit: int | None = None
) -> Iterator[CanonicalRecord]:
    """Expand name-only CoV-AbDab targets into an explicitly auxiliary pack."""

    emitted = 0
    relation_columns = (
        ("Binds to", "binding", 1),
        ("Doesn't Bind to", "binding", 0),
        ("Neutralising Vs", "neutralization", 1),
        ("Not Neutralising Vs", "neutralization", 0),
    )
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            if clean_text(row.get("Ab or Nb")).lower() != "ab":
                continue
            entities = _cov_receptor_entities(row)
            if not entities:
                continue
            cdr3_only = any(entity.sequence_scope == "cdr3" for entity in entities)
            for column, relation, label in relation_columns:
                for target in _targets(row.get(column)):
                    yield CanonicalRecord(
                        family="antibody_antigen",
                        record_type="target_name_proxy",
                        entities=entities,
                        measurement=Measurement(
                            relation=relation,
                            label=label,
                            direction="none",
                            assay_type="CoV-AbDab curated statement",
                            confidence="qualitative_name_only",
                            negative_type="curated_negative_statement" if not label else "",
                        ),
                        source_records=(
                            SourceReference(
                                source_name="cov_abdab",
                                source_version="2024-02-08",
                                source_record_id=(
                                    f"row:{line_number}:{column}:{target}"
                                ),
                                raw_path=str(path.resolve()),
                                raw_sha256=raw_sha256,
                                study_id=clean_text(row.get("Sources")),
                                doi=clean_text(row.get("Sources")),
                                metadata={
                                    "antibody_name": clean_text(row.get("Name")),
                                    "protein_epitope": clean_text(
                                        row.get("Protein + Epitope")
                                    ),
                                },
                            ),
                        ),
                        evidence=Evidence(
                            tier="C",
                            basis="CoV-AbDab curated qualitative statement",
                            direct_measurement=False,
                            derived_from_aggregate=True,
                        ),
                        eligibility=Eligibility(
                            training=True,
                            evaluation=False,
                            core=False,
                            pack="antibody_antigen_proxy",
                            reasons=tuple(
                                ["target has no exact antigen sequence"]
                                + (["only CDR3 H/L sequences available"] if cdr3_only else [])
                            ),
                        ),
                        context={
                            "antibody_name": clean_text(row.get("Name")),
                            "protein_epitope": clean_text(
                                row.get("Protein + Epitope")
                            ),
                            "origin": clean_text(row.get("Origin")),
                        },
                        target_mapping=TargetMapping(
                            target_name_raw=target,
                            canonical_target_id="",
                            variant=target,
                            mapping_confidence="name_only",
                            mapping_source="CoV-AbDab target column",
                        ),
                        flags=tuple(
                            ["target_sequence_missing"]
                            + (["cdr3_only_antibody"] if cdr3_only else [])
                        ),
                    )
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return


def build_target_registry(records: Iterator[CanonicalRecord] | list[CanonicalRecord]) -> dict:
    """Summarize exact versus unresolved antibody target mappings."""

    targets: dict[str, dict[str, Any]] = {}
    mapping_record_count = 0
    confidence_counts: dict[str, int] = defaultdict(int)
    for record in records:
        mapping = record.target_mapping
        if not mapping:
            continue
        mapping_record_count += 1
        confidence_counts[mapping.mapping_confidence] += 1
        key = mapping.canonical_target_id or mapping.target_name_raw
        item = targets.setdefault(
            key,
            {
                "target_key": key,
                "names": set(),
                "mapping_confidence": set(),
                "record_count": 0,
                "antigen_entity_ids": set(),
            },
        )
        item["names"].add(mapping.target_name_raw)
        item["mapping_confidence"].add(mapping.mapping_confidence)
        item["record_count"] += 1
        antigen = record.entity("antigen")
        if antigen:
            item["antigen_entity_ids"].add(antigen.entity_id)
    rows = []
    for key in sorted(targets):
        item = targets[key]
        rows.append(
            {
                **item,
                "names": sorted(item["names"]),
                "mapping_confidence": sorted(item["mapping_confidence"]),
                "antigen_entity_ids": sorted(item["antigen_entity_ids"]),
            }
        )
    antigen_entity_ids = {
        entity_id for row in rows for entity_id in row["antigen_entity_ids"]
    }
    return {
        "schema_version": "antibody_target_registry.v1",
        "mapping_record_count": mapping_record_count,
        "target_count": len(rows),
        "targets_with_sequence_entities": sum(
            bool(row["antigen_entity_ids"]) for row in rows
        ),
        "unique_antigen_entity_count": len(antigen_entity_ids),
        "mapping_confidence_record_counts": dict(sorted(confidence_counts.items())),
        "targets": rows,
    }
