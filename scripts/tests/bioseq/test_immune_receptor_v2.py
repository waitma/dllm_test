"""Regression tests for canonical immune-receptor ``bioseq.v2`` data.

Run::

    conda run -n pllm python -m pytest \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_immune_receptor_v2.py -q
"""

from __future__ import annotations

import csv
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path

from dllm.pipelines.bioseq.immune_receptor_v2.adapters.antibody import (
    iter_abrank,
    iter_catnap,
    iter_sabdab2,
)
from dllm.pipelines.bioseq.immune_receptor_v2.adapters.tcr import (
    iter_iedb,
    iter_piste,
)
from dllm.pipelines.bioseq.immune_receptor_v2.audit import merge_exact_measurements
from dllm.pipelines.bioseq.immune_receptor_v2.export import (
    ClusterResult,
    ClusterRule,
    filter_benchmark_contamination,
    generate_train_only_tcr_negatives,
    record_cluster_sequences,
)
from dllm.pipelines.bioseq.immune_receptor_v2.leakage import (
    audit_benchmark_leakage,
    build_benchmark_sequence_bank,
)
from dllm.pipelines.bioseq.immune_receptor_v2.pairing import (
    PairingSource,
    deterministic_pairing_split,
    export_pairing_source,
)
from dllm.pipelines.bioseq.immune_receptor_v2.schema import (
    BIOSEQ_V2_SCHEMA_VERSION,
    CanonicalRecord,
    Eligibility,
    Evidence,
    Measurement,
    SequenceEntity,
    SourceReference,
    TargetMapping,
    normalize_sequence,
    validate_domain_record,
)
from dllm.pipelines.bioseq.immune_receptor_v2.splits import (
    SplitProtocol,
    build_split_manifest,
)


def _tcr_record(index: int, receptor_group: str, peptide_group: str) -> CanonicalRecord:
    suffix = "A" * index + "C"
    return CanonicalRecord(
        family="tcr_pmhc",
        record_type="curated_specificity",
        entities=(
            SequenceEntity("tcr_alpha", f"CA{suffix}F", "cdr3"),
            SequenceEntity("tcr_beta", f"CB{suffix}F", "cdr3"),
            SequenceEntity("peptide", f"PEPTID{suffix}", "peptide"),
        ),
        measurement=Measurement("specificity", label=1, direction="higher"),
        source_records=(SourceReference("unit", f"row:{index}"),),
        evidence=Evidence("B", "unit test", True),
        eligibility=Eligibility(True, True, True, "tcr_core"),
        context={"mhc_allele": "HLA-A*02:01"},
        groups={"receptor": receptor_group, "peptide": peptide_group},
    )


def test_schema_ids_are_stable_and_do_not_silently_replace_j() -> None:
    assert normalize_sequence(" caJ f ") == "CAJF"
    first = _tcr_record(1, "r1", "p1")
    second = _tcr_record(1, "r1", "p1")
    assert first.schema_version == BIOSEQ_V2_SCHEMA_VERSION
    assert first.record_id == second.record_id
    assert first.biological_key == second.biological_key
    assert validate_domain_record(first) == []


def test_group_split_is_deterministic_and_disjoint() -> None:
    records = [
        _tcr_record(index, f"r{index // 2}", f"p{index % 3}")
        for index in range(1, 13)
    ]
    protocol = SplitProtocol("unit_receptor", ("receptor",), seed=42)
    forward = build_split_manifest(records, protocol)
    reverse = build_split_manifest(list(reversed(records)), protocol)
    forward_assignment = {
        row["record_id"]: row["split"] for row in forward["assignments"]
    }
    reverse_assignment = {
        row["record_id"]: row["split"] for row in reverse["assignments"]
    }
    assert forward_assignment == reverse_assignment
    assert forward["audit"]["passed"] is True
    receptor_splits: dict[str, set[str]] = {}
    for record in records:
        receptor_splits.setdefault(record.groups["receptor"], set()).add(
            forward_assignment[record.record_id]
        )
    assert all(len(splits) == 1 for splits in receptor_splits.values())


def test_cluster_candidates_keep_full_chain_and_annotated_cdr3_separate() -> None:
    record = CanonicalRecord(
        family="antibody_antigen",
        record_type="unit",
        entities=(
            SequenceEntity(
                "antibody_heavy",
                "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRSTGYYFDYWGQGTLVTVSS",
                "variable_domain",
                regions={"CDR3": "CARDRSTGYYFDY"},
            ),
            SequenceEntity("antibody_light", "CQQYNSYPYTF", "cdr3"),
            SequenceEntity("antigen", "ACDEFGHIKLMNPQRSTVWY", "antigen_construct"),
        ),
        measurement=Measurement("binding", label=1, direction="higher"),
        source_records=(SourceReference("unit", "cluster-candidates"),),
        evidence=Evidence("A", "unit", True),
        eligibility=Eligibility(True, True, True, "antibody_antigen_exact"),
    )
    assert set(record_cluster_sequences(record)) == {
        (
            "antibody_heavy_chain",
            record.entity("antibody_heavy").sequence,
        ),
        ("antibody_heavy_cdr", "CARDRSTGYYFDY"),
        ("antibody_light_cdr", "CQQYNSYPYTF"),
        ("antigen", "ACDEFGHIKLMNPQRSTVWY"),
    }


def test_cluster_blocklist_removes_near_benchmark_receptor(tmp_path: Path) -> None:
    first = _tcr_record(1, "r1", "p1")
    second = _tcr_record(2, "r2", "p2")
    beta_first = first.entity("tcr_beta").sequence
    beta_second = second.entity("tcr_beta").sequence
    cluster = ClusterResult(
        ClusterRule("tcr_beta_cdr", 0.8, 0.8),
        {beta_first: "benchmark_cluster", beta_second: "clean_cluster"},
        {"CASSBENCHMARKQYF"},
        {"benchmark_cluster"},
        tmp_path / "mapping.tsv",
        tmp_path / "report.json",
        {},
    )
    kept, blocked, report = filter_benchmark_contamination(
        [first, second], {"tcr_beta_cdr": cluster}
    )
    assert [record.record_id for record in kept] == [second.record_id]
    assert blocked[0]["record_id"] == first.record_id
    assert blocked[0]["near_buckets"] == ["tcr_beta_cdr"]
    assert report["blocked_records"] == 1


def test_train_only_negative_generation_is_deterministic_and_never_evaluation() -> None:
    records = [_tcr_record(index, f"r{index}", f"p{index}") for index in range(1, 5)]
    first, first_report = generate_train_only_tcr_negatives(
        records,
        records,
        protocol_id="unit_protocol",
        ratio=1.0,
        seed=42,
    )
    second, second_report = generate_train_only_tcr_negatives(
        list(reversed(records)),
        records,
        protocol_id="unit_protocol",
        ratio=1.0,
        seed=42,
    )
    assert [record.record_id for record in first] == [
        record.record_id for record in second
    ]
    assert first_report == second_report
    assert first_report["generated_negative_records"] == len(records)
    assert all(record.measurement.label == 0 for record in first)
    assert all(record.eligibility.evaluation is False for record in first)
    assert all(record.eligibility.pack == "tcr_train_only_synthetic_negative" for record in first)
    assert all(len(record.derived_from) == 2 for record in first)


def test_pairing_export_propagates_benchmark_hit_to_entire_upstream_group(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "train.csv"
    fieldnames = (
        "cleaned_h_sequence",
        "cleaned_l_sequence",
        "h_cdr3",
        "l_cdr3",
        "ab_cluster_key",
        "split",
    )
    rows = [
        {
            "cleaned_h_sequence": "QVQLVESGGG",
            "cleaned_l_sequence": "DIVMTQSPAS",
            "h_cdr3": "CARDRSTGYYFDY",
            "l_cdr3": "CQQYNSYPYTF",
            "ab_cluster_key": "blocked_pair",
            "split": "train",
        },
        {
            "cleaned_h_sequence": "EVQLVESGGA",
            "cleaned_l_sequence": "EIVLTQSPAT",
            "h_cdr3": "CARGGGGYYFDY",
            "l_cdr3": "CQQYYSTPYTF",
            "ab_cluster_key": "blocked_pair",
            "split": "train",
        },
        {
            "cleaned_h_sequence": "QVQLVESGAA",
            "cleaned_l_sequence": "DIVMTQSPAA",
            "h_cdr3": "CARAAAAAYFDY",
            "l_cdr3": "CQQAAAAPYTF",
            "ab_cluster_key": "clean_pair",
            "split": "train",
        },
    ]
    with input_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    valid = tmp_path / "valid.csv"
    holdout = tmp_path / "holdout.csv"
    valid.write_text(",".join(fieldnames) + "\n")
    holdout.write_text(",".join(fieldnames) + "\n")
    source = PairingSource(
        "oas",
        input_path,
        valid,
        holdout,
        ("ab_cluster_key",),
        (
            "antibody_heavy_cdr",
            "antibody_light_cdr",
            "antibody_heavy_chain",
            "antibody_light_chain",
        ),
    )
    bucket_sequences: dict[str, set[str]] = {}
    for row in rows:
        bucket_sequences.setdefault("antibody_heavy_chain", set()).add(
            row["cleaned_h_sequence"]
        )
        bucket_sequences.setdefault("antibody_light_chain", set()).add(
            row["cleaned_l_sequence"]
        )
        bucket_sequences.setdefault("antibody_heavy_cdr", set()).add(row["h_cdr3"])
        bucket_sequences.setdefault("antibody_light_cdr", set()).add(row["l_cdr3"])
    clusters = {}
    for bucket, sequences in bucket_sequences.items():
        mapping = {sequence: f"{bucket}:{sequence}" for sequence in sequences}
        benchmark = {rows[0]["cleaned_h_sequence"]} if bucket == "antibody_heavy_chain" else set()
        benchmark_clusters = {mapping[item] for item in benchmark}
        clusters[bucket] = ClusterResult(
            ClusterRule(bucket, 0.95, 0.8),
            mapping,
            benchmark,
            benchmark_clusters,
            tmp_path / f"{bucket}.tsv",
            tmp_path / f"{bucket}.json",
            {},
        )
    report = export_pairing_source(
        source,
        clusters,
        output_root=tmp_path / "export",
        seed=42,
    )
    assert report["blocked_records"] == 2
    assert report["kept_records"] == 1
    assert report["blocked_group_count"] == 1
    exported = []
    for split in ("train", "valid", "test"):
        with (tmp_path / "export" / f"{split}.csv").open() as handle:
            exported.extend(csv.DictReader(handle))
    assert len(exported) == 1
    assert exported[0]["ab_cluster_key"] == "clean_pair"
    assert exported[0]["v2_split"] in {"train", "valid", "test"}
    assert deterministic_pairing_split(
        "clean_pair", seed=42
    ) == deterministic_pairing_split("clean_pair", seed=42)


def test_exact_merge_retains_both_source_references() -> None:
    first = _tcr_record(1, "r1", "p1")
    second_source = replace(
        first.source_records[0], source_name="unit_copy", source_record_id="copy:1"
    )
    second = replace(first, source_records=(second_source,), record_id="")
    merged, report = merge_exact_measurements([first, second])
    assert report["exact_duplicate_groups"] == 1
    assert len(merged) == 1
    assert {source.source_name for source in merged[0].source_records} == {
        "unit",
        "unit_copy",
    }


def _write_piste(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("CDR3", "MT_pep", "HLA_type", "Label", "HLA_sequence", "peplen"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_piste_original_split_is_only_provenance(tmp_path: Path) -> None:
    row = {
        "CDR3": "CASSFQTGWNEQFF",
        "MT_pep": "AMDLGIHKV",
        "HLA_type": "HLA-A02:01",
        "Label": "1",
        "HLA_sequence": "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY",
        "peplen": "9",
    }
    _write_piste(tmp_path / "train_data.csv", [row])
    _write_piste(tmp_path / "val_data.csv", [row])
    _write_piste(tmp_path / "test_data.csv", [row])
    records = list(iter_piste(tmp_path))
    assert len(records) == 3
    assert {record.source_records[0].original_split for record in records} == {
        "train",
        "valid",
        "test",
    }
    assert all(record.eligibility.core is False for record in records)
    assert all("original_split_not_reused" in record.flags for record in records)


def test_abrank_exact_antigen_and_name_only_aux_are_separate(tmp_path: Path) -> None:
    csv_path = tmp_path / "AbRank_dataset.csv"
    fields = (
        "Ab_name",
        "Ag_name",
        "Ag_name_details",
        "Affinity_Kd [nM]",
        "IC50 [ug/mL]",
        "Ab_heavy_chain_seq",
        "Ab_light_chain_seq",
        "Ag_seq",
        "Aff_op",
        "Source",
        "fitness",
        "log_Aff",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "Ab_name": "exact",
                "Ag_name": "ag1",
                "Affinity_Kd [nM]": "10",
                "Ab_heavy_chain_seq": "QVQLVESGGG",
                "Ab_light_chain_seq": "DIVMTQSPAS",
                "Ag_seq": "ACDEFGHIKLMNPQRSTVWY",
                "Aff_op": "=",
                "Source": "study",
            }
        )
        writer.writerow(
            {
                "Ab_name": "proxy",
                "Ag_name": "ag2",
                "Affinity_Kd [nM]": "20",
                "Ab_heavy_chain_seq": "QVQLVESGGA",
                "Ab_light_chain_seq": "DIVMTQSPAT",
                "Ag_seq": "N501Y;E484K",
                "Aff_op": "=",
                "Source": "study",
            }
        )
    archive_path = tmp_path / "AbRank_dataset.csv.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(csv_path, arcname="AbRank_dataset.csv")
    records = list(iter_abrank(archive_path))
    assert len(records) == 2
    exact, proxy = records
    assert exact.entity("antigen") is not None
    assert exact.eligibility.core is True
    assert exact.target_mapping == TargetMapping(
        target_name_raw="ag1",
        canonical_target_id="ag1",
        construct="ag1",
        mapping_confidence="exact_sequence",
        mapping_source="AbRank released Ag_seq",
    )
    assert proxy.entity("antigen") is None
    assert proxy.eligibility.core is False
    assert proxy.target_mapping is not None
    assert proxy.target_mapping.mapping_confidence == "name_only"


def test_leakage_distinguishes_shared_peptide_from_exact_pair(tmp_path: Path) -> None:
    benchmark = tmp_path / "test.csv"
    with benchmark.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("cdr3b", "peptide", "mhc"))
        writer.writeheader()
        writer.writerow(
            {
                "cdr3b": "CASSFQTGWNEQFF",
                "peptide": "AMDLGIHKV",
                "mhc": "HLA-A*02:01",
            }
        )
    quarantine = {
        "files": [
            {
                "path": str(benchmark),
                "partition": "benchmark_eval",
            }
        ]
    }

    def record(beta: str, source_id: str) -> CanonicalRecord:
        return CanonicalRecord(
            family="tcr_pmhc",
            record_type="curated_specificity",
            entities=(
                SequenceEntity("tcr_beta", beta, "cdr3"),
                SequenceEntity("peptide", "AMDLGIHKV", "peptide"),
            ),
            measurement=Measurement("specificity", label=1, direction="higher"),
            source_records=(SourceReference("unit", source_id),),
            evidence=Evidence("B", "unit test", True),
            eligibility=Eligibility(True, True, False, "tcr_aux_beta"),
            context={"mhc_allele": "HLA-A02:01"},
        )

    report = audit_benchmark_leakage(
        [record("CASSFQTGWNEQFF", "exact"), record("CASSDIFFERENTQYF", "shared")],
        quarantine,
    )
    assert report["exact_match_record_count"] == 2
    assert report["records_with_exact_peptide_entity"] == 2
    assert report["exact_receptor_peptide_pair_records"] == 1
    assert report["exact_receptor_pmhc_records"] == 1


def test_antibody_benchmark_bank_reads_json_lines_and_humanization_cif(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "cdrh3" / "fold_0" / "test.json"
    json_path.parent.mkdir(parents=True)
    json_path.write_text(
        '{"heavy_chain_seq":"QVQLVESGGG","light_chain_seq":"DIVMTQSPAS"}\n'
        '{"heavy_chain_seq":"EVQLVESGGA","light_chain_seq":"EIVLTQSPAT"}\n'
    )
    humanization = tmp_path / "humanisation"
    pdb_dir = humanization / "test-pdb"
    pdb_dir.mkdir(parents=True)
    (humanization / "test_chains.csv").write_text("1abc,H-L\n")
    cif_path = pdb_dir / "1abc.cif"
    cif_path.write_text(
        "data_1abc\n"
        "loop_\n"
        "_pdbx_poly_seq_scheme.seq_id\n"
        "_pdbx_poly_seq_scheme.mon_id\n"
        "_pdbx_poly_seq_scheme.pdb_strand_id\n"
        + "\n".join(
            f"{index} {residue} H"
            for index, residue in enumerate(
                ("GLN", "VAL", "GLN", "LEU", "VAL", "GLU", "SER"), start=1
            )
        )
        + "\n"
        + "\n".join(
            f"{index} {residue} L"
            for index, residue in enumerate(
                ("ASP", "ILE", "VAL", "MET", "THR", "GLN", "SER"), start=1
            )
        )
        + "\n#\n"
    )
    quarantine = {
        "files": [
            {"path": str(json_path), "partition": "benchmark_eval"},
            {"path": str(cif_path), "partition": "benchmark_eval"},
        ]
    }
    bank = build_benchmark_sequence_bank(quarantine)
    assert set(bank["antibody_heavy"]) == {
        "QVQLVESGGG",
        "EVQLVESGGA",
        "QVQLVES",
    }
    assert set(bank["antibody_light"]) == {
        "DIVMTQSPAS",
        "EIVLTQSPAT",
        "DIVMTQS",
    }


def test_benchmark_bank_distinguishes_full_tcra_tcrb_and_antibody_cdrs(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "roles.csv"
    with benchmark.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("tcra", "tcrb", "cdrh3_seq", "cdrl3_seq"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "tcra": "CAVRDSNYQLIW",
                "tcrb": "CASSFQTGWNEQFF",
                "cdrh3_seq": "CARDRSTGYYFDYW",
                "cdrl3_seq": "CQQYNSYPYTF",
            }
        )
    bank = build_benchmark_sequence_bank(
        {
            "files": [
                {"path": str(benchmark), "partition": "benchmark_eval"}
            ]
        }
    )
    assert set(bank["tcr_alpha"]) == {"CAVRDSNYQLIW"}
    assert set(bank["tcr_beta"]) == {"CASSFQTGWNEQFF"}
    assert set(bank["antibody_heavy"]) == {"CARDRSTGYYFDYW"}
    assert set(bank["antibody_light"]) == {"CQQYNSYPYTF"}


def _write_two_header_zip(
    path: Path, member: str, headers: list[tuple[str, str]], rows: list[list[str]]
) -> None:
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([section for section, _ in headers])
        writer.writerow([field for _, field in headers])
        writer.writerows(rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=member)


def test_iedb_adapter_joins_experimental_negative_without_voting(
    tmp_path: Path,
) -> None:
    tcell_headers = [
        ("Assay ID", "IEDB IRI"),
        ("Assay", "Qualitative Measurement"),
        ("Assay", "Method"),
        ("Assay", "Response measured"),
        ("Reference", "PMID"),
        ("Epitope", "Object Type"),
        ("Epitope", "Name"),
        ("MHC Restriction", "Name"),
    ]
    _write_two_header_zip(
        tmp_path / "tcell.zip",
        "tcell.csv",
        tcell_headers,
        [[
            "https://www.iedb.org/assay/123",
            "Negative",
            "multimer/tetramer",
            "ligand presentation",
            "12345678",
            "Linear peptide",
            "AMDLGIHKV",
            "HLA-A*02:01",
        ]],
    )
    tcr_headers = [
        ("Receptor", "Type"),
        ("Reference", "IEDB IRI"),
        ("Epitope", "Name"),
        ("Assay", "IEDB IDs"),
        ("Assay", "MHC Allele Names"),
        ("Chain 1", "Type"),
        ("Chain 1", "Organism IRI"),
        ("Chain 1", "CDR3 Curated"),
        ("Chain 2", "Type"),
        ("Chain 2", "Organism IRI"),
        ("Chain 2", "CDR3 Curated"),
    ]
    _write_two_header_zip(
        tmp_path / "tcr.zip",
        "tcr.csv",
        tcr_headers,
        [[
            "alphabeta",
            "https://www.iedb.org/reference/9",
            "AMDLGIHKV",
            "123",
            "HLA-A*02:01",
            "alpha",
            "http://purl.obolibrary.org/obo/NCBITaxon_9606",
            "CAVRPGGAGPFF",
            "beta",
            "http://purl.obolibrary.org/obo/NCBITaxon_9606",
            "CASSFQTGWNEQFF",
        ]],
    )
    records = list(iter_iedb(tmp_path / "tcr.zip", tmp_path / "tcell.zip"))
    assert len(records) == 1
    record = records[0]
    assert record.measurement.label == 0
    assert record.measurement.negative_type == "experimental_negative"
    assert record.eligibility.core is True
    assert record.source_records[0].assay_id == "IEDB:123"
    assert record.source_records[0].pmid == "12345678"
    assert record.source_records[0].doi == ""


def test_catnap_preserves_censor_and_removes_only_terminal_stop(
    tmp_path: Path,
) -> None:
    release = tmp_path / "2026-08-01"
    release.mkdir()
    (release / "heavy_seqs_aa_2026-08-01.fasta").write_text(
        ">AB1__heavy_ImmDBID7\nQVQLVESGGGVVQPGRSLRLSCAAS\n"
    )
    (release / "light_seqs_aa_2026-08-01.fasta").write_text(
        ">AB1__light_ImmDBID7\nDIVMTQSPSSLSASVGDRVTITCRAS\n"
    )
    (release / "virseqs_aa_2026-08-01.fasta").write_text(
        ">B.US.2020.VIRUS1.AB123456\nMKTIIALSYIFCLVFADYKDDDDK-*\n"
    )
    with (release / "abs_2026-08-01.txt").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("Name", "Immuno DB ID"), delimiter="\t"
        )
        writer.writeheader()
        writer.writerow({"Name": "AB1", "Immuno DB ID": "7"})
    with (release / "viruses_2026-08-01.txt").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "---Virus name",
                "Accession",
                "Subtype",
                "Organism",
                "Virus type",
                "Tier",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "---Virus name": "VIRUS1",
                "Accession": "AB123456",
                "Subtype": "B",
                "Organism": "HIV-1",
                "Virus type": "panel",
                "Tier": "2",
            }
        )
    with (release / "assay_2026-08-01.txt").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "Antibody",
                "Virus",
                "Reference",
                "Pubmed ID",
                "IC50",
                "IC80",
                "ID50",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "Antibody": "AB1",
                "Virus": "VIRUS1",
                "Reference": "unit",
                "Pubmed ID": "12345678",
                "IC50": ">50",
            }
        )
    records = list(iter_catnap(release))
    assert len(records) == 1
    record = records[0]
    assert record.measurement.value == 50
    assert record.measurement.censor == ">"
    assert record.entity("antigen").sequence.endswith("KDDDDK")
    assert record.evidence.tier == "B"
    assert record.eligibility.pack == "antibody_antigen_exact_accession"
    assert record.target_mapping.mapping_confidence == "exact_accession"
    assert record.source_records[0].pmid == "12345678"
    assert record.source_records[0].doi == ""
    assert "terminal_stop_codon_removed" in record.flags


def test_sabdab2_uses_antigen_aware_split_and_marks_multicomponent_aux(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "abag_split.csv"
    fields = (
        "INSTANCE",
        "PDB_ID",
        "SABDAB_ID",
        "method",
        "resolution",
        "type",
        "species",
        "Hchain",
        "Lchain",
        "VH_numerable_seq",
        "VL_numerable_seq",
        "CDRH1",
        "CDRH2",
        "CDRH3",
        "CDRL1",
        "CDRL2",
        "CDRL3",
        "agchains",
        "agtypes",
        "agresolvedseqs",
        "ab_cluster",
        "agclusters",
        "ab_ag_cluster",
        "ab_ag_split",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "INSTANCE": "pdb_1abc_H_L",
                "PDB_ID": "1abc",
                "SABDAB_ID": "H1L1",
                "method": "XRAY",
                "resolution": "2.0",
                "type": "FAB",
                "species": "HOMO SAPIENS",
                "Hchain": "H",
                "Lchain": "L",
                "VH_numerable_seq": "QVQLVESGGGVVQPGRSLRLSCAAS",
                "VL_numerable_seq": "DIVMTQSPSSLSASVGDRVTITCRAS",
                "CDRH1": "GFTFSSY",
                "CDRH2": "ISSGG",
                "CDRH3": "CARDRF",
                "CDRL1": "RASQDI",
                "CDRL2": "SAS",
                "CDRL3": "QQYF",
                "agchains": "A/B",
                "agtypes": "PROTEIN/PEPTIDE",
                "agresolvedseqs": "ACDEFGHIK/AMDLGIHKV",
                "ab_ag_split": "test",
                "ab_cluster": "ab0",
                "agclusters": "ag0/ag1",
                "ab_ag_cluster": "joint0",
            }
        )
    archive_path = tmp_path / "splits.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(csv_path, arcname="splits_final/abag_split.csv")
    records = list(iter_sabdab2(archive_path))
    assert len(records) == 2
    assert all(record.eligibility.core is False for record in records)
    assert all(record.source_records[0].original_split == "test" for record in records)
    assert all("multi_component_antigen" in record.flags for record in records)
    assert all(
        record.context["released_antibody_antigen_cluster"] == "joint0"
        for record in records
    )
    assert records[0].entity("antibody_heavy").regions["CDR3"] == "CARDRF"
