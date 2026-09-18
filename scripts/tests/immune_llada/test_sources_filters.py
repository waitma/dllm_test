"""Run with ``conda run -n pllm pytest scripts/tests/immune_llada/test_sources_filters.py``."""

from __future__ import annotations

from pathlib import Path

import pytest

from dllm.pipelines.immune_llada.data.records import BioSeqRecord
from dllm.pipelines.immune_llada.data.registry import DEFAULT_SOURCES, SOURCE_REGISTRY, SourceSpec, parse_sources, source_split_path
from dllm.pipelines.immune_llada.data.sources import row_to_record
from dllm.pipelines.immune_llada.data.preprocessing.filters import (
    BLOCKLIST_NAMES,
    build_filters,
    constructed_filter_names,
    filter_reason,
    load_blocklists,
    union_filter_names,
)
from dllm.pipelines.immune_llada.data.preprocessing.pipeline import PreprocessConfig
from dllm.pipelines.immune_llada.data.records import BioSeqChain


def _pair() -> dict[str, str]:
    return {
        "cleaned_chain1_seq": "AAAA",
        "cleaned_chain2_seq": "BBBB",
        "chain1_anarci_type": "A",
        "chain2_anarci_type": "B",
        "chain2_cdr3": "CASSQF",
    }


def test_registry_defaults_and_asd_expansion(tmp_path: Path) -> None:
    assert len(DEFAULT_SOURCES) == 7
    assert set(DEFAULT_SOURCES) <= set(SOURCE_REGISTRY)
    assert parse_sources("ots+oas+ots") == ["ots", "oas"]
    assert parse_sources(["asd", "oas"]) == ["asd_antibody", "asd_nanobody", "oas"]
    assert source_split_path(SourceSpec("oas", tmp_path), "val").name == "cleaned_merged_data_step_clustered_valid_oas_label.csv"


def test_simple_sources_and_schema_invalid() -> None:
    oas = row_to_record("oas", {**_pair(), "chain1_anarci_type": "L", "chain2_anarci_type": "H"}, split="train", weight=2)
    ots = row_to_record("ots", _pair())
    nb = row_to_record("asd_nanobody", {"antigen_seq": "AAAA", "vhh_fv": "CCCCCWG"})
    asd = row_to_record("asd_antibody", {"antigen_seq": "AAAA", "heavy_fv": "CAAFFGAG", "light_fv": "LLLL"})
    native = row_to_record("tcr_native", {"epitope_seq": "AAAA", "cdr3b": "ASSQ"})
    assert isinstance(oas, BioSeqRecord) and oas.chain_roles == ["antibody_heavy", "antibody_light"]
    assert isinstance(ots, BioSeqRecord) and ots.chain_roles == ["tcr_beta", "tcr_alpha"]
    assert isinstance(nb, BioSeqRecord) and nb.source == "asd_nanobody"
    assert isinstance(asd, BioSeqRecord) and asd.chain_roles == ["antigen", "antibody_heavy", "antibody_light"]
    assert isinstance(native, BioSeqRecord) and native.source == "tcr_native"
    assert oas.weight == 2
    assert row_to_record("ots", {**_pair(), "cleaned_chain1_seq": "not valid!"}) is None


def test_recognition_relation_and_native_papers_source_parity() -> None:
    row = {"epitope_seq": "AAAA", "cdr3a": "CCCC", "cdr3b": "CASSQF", "relation": "negative"}
    trait = row_to_record("trait", row)
    papers = row_to_record("tcr_papers", row)
    assert trait is not None and trait.labels["relation"] == "nonbinding"
    assert papers is not None and papers.source == "tcr_native"
    assert papers.metadata["dataset_source"] == "tcr_papers"
    assert trait.identifiers["cdr3b_core"] == "ASSQ"
    assert papers.identifiers["cdr3b_core"] == "CASSQF"
    with pytest.raises(ValueError):
        row_to_record("trait", {**row, "relation": "maybe"})


def test_filters_use_paired_keys_and_first_reason() -> None:
    record = row_to_record("trait", {"epitope_seq": "AAAA", "cdr3a": "CCCC", "cdr3b": "CASSQF"})
    assert record is not None
    blocklists = {name: set() for name in BLOCKLIST_NAMES}
    blocklists["trait_benchmark"] = {"ASSQ"}
    filters = build_filters("trait", ["trait"], blocklists, 3, 0)
    assert filter_reason(record, filters) == "decontam.trait_benchmark"

    blocklists["trait_benchmark"] = set()
    blocklists["t4_refbinder"] = {"ASSQ|AAAA"}
    filters = build_filters("trait", ["trait"], blocklists, 0, 0)
    assert filter_reason(record, filters) == "decontam.t4_refbinder"
    filters = build_filters("trait", ["trait", "tcr_native"], blocklists, 0, 0)
    assert filter_reason(record, filters) == "decontam.t4_refbinder"


def test_repertoire_projects_pair_keys_once() -> None:
    record = row_to_record("tcr_repertoire", {"cdr3b": "ASSQ"})
    assert record is not None
    blocklists = {name: set() for name in BLOCKLIST_NAMES}
    blocklists["t4_refbinder"] = {"ASSQ|AAAA"}
    filters = build_filters("tcr_repertoire", ["tcr_repertoire"], blocklists, 0, 0)
    assert filter_reason(record, filters) == "decontam.repertoire_core_projection"


@pytest.mark.parametrize("source,chain_type", [("oas", "H"), ("oas", "L"), ("ots", "B"), ("ots", "A")])
@pytest.mark.parametrize("second_sequence", ["AAAA", "CCCC"])
def test_homotypic_pairs_are_filtered_by_role_not_sequence(source, chain_type, second_sequence):
    row = {**_pair(), "cleaned_chain2_seq": second_sequence, "chain1_anarci_type": chain_type, "chain2_anarci_type": chain_type}
    record = row_to_record(source, row)
    assert record is not None
    filters = build_filters(source, [source], {name: set() for name in BLOCKLIST_NAMES}, 0, 0)
    assert filter_reason(record, filters) == "quality.homotypic_pair"


def test_homotypic_filter_preserves_valid_pairs_and_single_chain_sources():
    blocklists = {name: set() for name in BLOCKLIST_NAMES}
    for source, row in [
        ("oas", {"cleaned_h_sequence": "AAAA", "cleaned_l_sequence": "AAAA", "l_locus": "L"}),
        ("ots", {**_pair(), "cleaned_chain2_seq": "AAAA"}),
        ("tcr_repertoire", {"cdr3b": "ASSQ"}),
    ]:
        record = row_to_record(source, row)
        assert record is not None
        assert filter_reason(record, build_filters(source, [source], blocklists, 0, 0)) is None


def test_union_filter_names_includes_source_specific_filters() -> None:
    blocklists = {name: set() for name in BLOCKLIST_NAMES}
    oas = [item.name for item in build_filters("oas", ["oas", "trait"], blocklists, 1024, 1024)]
    trait = [item.name for item in build_filters("trait", ["oas", "trait"], blocklists, 1024, 1024)]
    assert "quality.blank_epitope" in trait
    assert "quality.blank_epitope" not in oas
    assert "quality.homotypic_pair" in oas
    names = union_filter_names([oas, trait])
    assert "quality.blank_epitope" in names
    assert "quality.homotypic_pair" in names
    assert names == oas + [name for name in trait if name not in oas]


def test_blocklist_paths_fail_loudly_and_disabled_values_work(tmp_path: Path) -> None:
    disabled = {name: "off" for name in BLOCKLIST_NAMES}
    assert load_blocklists(disabled) == {name: set() for name in BLOCKLIST_NAMES}
    assert [item.name for item in build_filters("oas", ["oas"], load_blocklists(disabled), 0, 0)] == ["quality.homotypic_pair"]
    with pytest.raises(ValueError):
        load_blocklists({**disabled, "oas_benchmark": ""})
    with pytest.raises(FileNotFoundError):
        load_blocklists({**disabled, "oas_benchmark": str(tmp_path / "missing")})
    empty = tmp_path / "empty.txt"
    empty.write_text("\n")
    with pytest.raises(ValueError):
        load_blocklists({**disabled, "oas_benchmark": str(empty)})


def _relation_record(relation: str, source: str = "asd_antibody") -> BioSeqRecord:
    return BioSeqRecord(
        [BioSeqChain("ACDEF", "antigen"), BioSeqChain("GHIKL", "antibody_heavy")],
        "antibody_antigen",
        source,
        labels={"relation": relation},
        metadata={"relation_supervised": relation in {"binding", "nonbinding"}},
    )


def test_binding_only_filter_is_opt_in_and_per_source() -> None:
    blocklists = {name: set() for name in BLOCKLIST_NAMES}
    off = build_filters("asd_antibody", ["asd_antibody"], blocklists, 0, 0)
    assert "recipe.binding_only" not in constructed_filter_names(off)
    assert filter_reason(_relation_record("nonbinding"), off) is None

    on = build_filters("asd_antibody", ["asd_antibody"], blocklists, 0, 0, ["asd_antibody"])
    assert constructed_filter_names(on) == ["recipe.binding_only"]
    assert filter_reason(_relation_record("nonbinding"), on) == "recipe.binding_only"
    assert filter_reason(_relation_record("binding"), on) is None

    # A source absent from the list keeps its negatives.
    other = build_filters("trait", ["trait", "asd_antibody"], blocklists, 0, 0, ["asd_antibody"])
    assert "recipe.binding_only" not in constructed_filter_names(other)


def test_binding_only_all_sentinel_spares_unknown_relations() -> None:
    blocklists = {name: set() for name in BLOCKLIST_NAMES}
    filters = build_filters("tcr_repertoire", ["tcr_repertoire"], blocklists, 0, 0, ["all"])
    assert filter_reason(_relation_record("nonbinding", "tcr_repertoire"), filters) == "recipe.binding_only"
    # tcr_repertoire labels all 2.1M rows "unknown": unlabeled pairs are not
    # measured non-binders and must survive the binding-only recipe.
    assert filter_reason(_relation_record("unknown", "tcr_repertoire"), filters) is None


def test_drop_nonbinding_sources_config_is_validated() -> None:
    base = {
        "sources": {"asd_antibody": {"path": "/tmp/asd"}},
        "blocklists": {name: None for name in BLOCKLIST_NAMES},
    }
    assert PreprocessConfig.from_mapping(base).drop_nonbinding_sources == ()
    parsed = PreprocessConfig.from_mapping(
        {**base, "recipe": {"drop_nonbinding_sources": ["asd_antibody", "trait"]}}
    )
    assert parsed.drop_nonbinding_sources == ("asd_antibody", "trait")
    assert PreprocessConfig.from_mapping(
        {**base, "recipe": {"drop_nonbinding_sources": ["all"]}}
    ).drop_nonbinding_sources == ("all",)
    # A bare string would silently iterate into characters.
    with pytest.raises(TypeError):
        PreprocessConfig.from_mapping({**base, "recipe": {"drop_nonbinding_sources": "asd_antibody"}})
    with pytest.raises(ValueError):
        PreprocessConfig.from_mapping({**base, "recipe": {"drop_nonbinding_sources": ["nope"]}})
