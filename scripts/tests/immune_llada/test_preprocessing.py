"""Run with ``conda run -n pllm pytest scripts/tests/immune_llada/test_preprocessing.py``."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarTokenizer, load_prepared_dataset
from dllm.pipelines.immune_llada.data.preprocessing import pipeline as preprocessing_pipeline
from dllm.pipelines.immune_llada.data.preprocessing.pipeline import (
    PreprocessConfig,
    load_preprocess_config,
    preprocess_dataset,
)
from dllm.pipelines.immune_llada.data.preprocessing.validators import SCHEMA_VERSION
from dllm.pipelines.immune_llada.data.preprocessing.filters import (
    BLOCKLIST_NAMES,
    union_filter_names,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _config(tmp_path: Path) -> PreprocessConfig:
    trait_path = tmp_path / "trait.jsonl"
    if not trait_path.exists():
        trait_path.write_text(json.dumps({"epitope_seq": "AAAA", "cdr3b": "CASSQF", "relation": "binding"}) + "\n", encoding="utf-8")
    rows = {name: "off" for name in BLOCKLIST_NAMES}
    return PreprocessConfig(
        sources=[
            {"name": "oas", "path": str(tmp_path / "oas.csv"), "weight": 2.0},
            {"name": "trait", "path": str(trait_path), "weight": 1.0},
        ],
        blocklists=rows,
        max_protein_length=8,
        max_length=30,
        shard_size=1,
    )


def test_csv_jsonl_to_prepared_loader_and_collator(tmp_path: Path) -> None:
    _write_csv(tmp_path / "oas.csv", [
        {"cleaned_chain1_seq": "AAAA", "cleaned_chain2_seq": "BBBB", "chain1_anarci_type": "L", "chain2_anarci_type": "H"},
        {"cleaned_chain1_seq": "AAAA", "cleaned_chain2_seq": "INVALID!", "chain1_anarci_type": "L", "chain2_anarci_type": "H"},
    ])
    with (tmp_path / "trait.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"epitope_seq": "AAAA", "cdr3b": "CASSQF", "relation": "binding"}) + "\n")
    result = preprocess_dataset(_config(tmp_path), tmp_path / "prepared", splits=["train"])
    assert result["report"]["totals"] == {
        "raw_rows": 3, "converted_rows": 2, "kept_rows": 2,
        "dropped_schema": 1, "dropped_filters": 0, "errors": 0,
        "downgraded_all_x_mhc": 0, "beta_only_completed": 0, "alpha_only_completed": 0,
    }
    assert len(result["splits"]["train"]["shards"]) == 2
    assert (tmp_path / "prepared" / "dataset_manifest.json").is_file()
    assert json.loads((tmp_path / "prepared" / "schema.json").read_text())["token_cache"] is False

    dataset = load_prepared_dataset(tmp_path / "prepared")
    assert len(dataset) == 2
    assert dataset[0].source == "oas_paired"
    batch = GrammarBioSeqCollator(GrammarTokenizer(), max_sequence_length=128)([dataset[0], dataset[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["encoder_input_ids"].ndim == 3
    assert batch["encoder_chain_mask"].any()


def test_filter_report_and_length_filter(tmp_path: Path) -> None:
    _write_csv(tmp_path / "oas.csv", [
        {"cleaned_chain1_seq": "AAAA", "cleaned_chain2_seq": "BBBB", "chain1_anarci_type": "L", "chain2_anarci_type": "H"},
        {"cleaned_chain1_seq": "AAAAAAAAA", "cleaned_chain2_seq": "BBBB", "chain1_anarci_type": "L", "chain2_anarci_type": "H"},
    ])
    config = _config(tmp_path)
    output = tmp_path / "prepared"
    result = preprocess_dataset(config, output, splits=["train"])
    assert result["report"]["totals"]["dropped_filters"] == 1
    assert result["report"]["splits"][0]["filter_reasons"] == {"budget.max_protein_length": 1}
    assert json.loads((output / "filter_report.json").read_text())["totals"]["kept_rows"] == 2


def test_homotypic_pair_is_removed_offline_and_reported(tmp_path: Path) -> None:
    _write_csv(tmp_path / "oas.csv", [
        {"cleaned_h_sequence": "AAAA", "cleaned_l_sequence": "CCCC", "l_locus": "H"},
        {"cleaned_h_sequence": "AAAA", "cleaned_l_sequence": "CCCC", "l_locus": "L"},
    ])
    output = tmp_path / "prepared"
    result = preprocess_dataset(_config(tmp_path), output)
    assert result["report"]["splits"][0]["filter_reasons"] == {"quality.homotypic_pair": 1}
    assert result["report"]["totals"]["kept_rows"] == 2
    dataset = load_prepared_dataset(output, source="oas")
    assert len(dataset) == 1
    assert dataset[0].chain_roles == ["antibody_heavy", "antibody_light"]


def test_dry_run_does_not_publish_manifest(tmp_path: Path) -> None:
    _write_csv(tmp_path / "oas.csv", [{"cleaned_chain1_seq": "AAAA", "cleaned_chain2_seq": "BBBB", "chain1_anarci_type": "H", "chain2_anarci_type": "L"}])
    result = preprocess_dataset(_config(tmp_path), tmp_path / "dry", splits=["train"], dry_run=True)
    assert result["manifest"] is None
    assert not (tmp_path / "dry" / "dataset_manifest.json").exists()
    assert result["report"]["totals"]["kept_rows"] == 2


def test_yaml_config_loads_without_omegaconf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "schema_version: immune_llada.semantic.v1\n"
        "splits: [train]\n"
        "sources:\n"
        "  oas:\n"
        f"    path: {tmp_path / 'oas.csv'}\n"
        "blocklists:\n"
        + "\n".join(f"  {name}: off" for name in BLOCKLIST_NAMES)
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(preprocessing_pipeline, "OmegaConf", None)
    loaded = load_preprocess_config(config_path)
    assert loaded.sources[0]["name"] == "oas"
    assert loaded.splits == ("train",)


def test_resume_fails_without_mutating_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(NotImplementedError):
        preprocess_dataset(_config(tmp_path), output, splits=["train"], resume=True)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_overwrite_preflights_inputs_before_deleting_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.sources[0]["path"] = str(tmp_path / "missing.csv")
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        preprocess_dataset(config, output, splits=["train"], overwrite=True)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_filter_names_is_union_of_filters_that_actually_ran(tmp_path: Path) -> None:
    """Manifest filter_names is the union of constructed filters, not blocklist keys.

    quality.blank_epitope is source-specific (trait / tcr_native / tcr_papers).
    It must appear when trait ran, even though oas never constructed it, and
    must be listed even if a given filter dropped zero rows of some other kind.
    """
    _write_csv(tmp_path / "oas.csv", [
        {"cleaned_chain1_seq": "AAAA", "cleaned_chain2_seq": "BBBB", "chain1_anarci_type": "L", "chain2_anarci_type": "H"},
        {"cleaned_chain1_seq": "AAAA", "cleaned_chain2_seq": "CCCC", "chain1_anarci_type": "H", "chain2_anarci_type": "H"},
    ])
    with (tmp_path / "trait.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"epitope_seq": "AAAA", "cdr3b": "CASSQF", "relation": "binding"}) + "\n")
        handle.write(json.dumps({"epitope_seq": "XXXX", "cdr3b": "CASSRF", "relation": "binding"}) + "\n")
    output = tmp_path / "prepared"
    result = preprocess_dataset(_config(tmp_path), output, splits=["train"])
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    names = manifest["filter_names"]
    by_source = {item["source"]: item["filter_names"] for item in result["report"]["splits"]}

    assert "replaces_trait" not in names
    assert "trait_benchmark" not in names
    assert "quality.blank_epitope" in names
    assert "quality.blank_epitope" in by_source["trait"]
    assert "quality.blank_epitope" not in by_source["oas"]
    assert "quality.homotypic_pair" in names
    assert "quality.homotypic_pair" in by_source["oas"]
    assert names == union_filter_names(item["filter_names"] for item in result["report"]["splits"])
    for item in result["report"]["splits"]:
        for reason in item["filter_reasons"]:
            assert reason in names
            assert reason in item["filter_names"]
    assert result["report"]["splits"][1]["filter_reasons"]["quality.blank_epitope"] == 1


def test_audit_counters_land_in_filter_report(tmp_path: Path) -> None:
    path = tmp_path / "papers.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"epitope_seq": "GILGFVFTL", "mhc_seq": "X" * 34, "cdr3b": "ASSQETQY", "relation": "binding"},
        {"epitope_seq": "GILGFVFTL", "mhc_seq": "Y" * 34, "cdr3a": "AVGMNYGGSQ", "relation": "binding"},
        {"epitope_seq": "GILGFVFTL", "mhc_seq": "Y" * 34, "cdr3a": "AVGMNY", "cdr3b": "ASSQETQY", "relation": "binding"},
        {"epitope_seq": "X" * 21, "mhc_seq": "X" * 34, "cdr3b": "ASSQETQY", "relation": "nonbinding"},
    ]) + "\n", encoding="utf-8")
    output = tmp_path / "prepared"
    result = preprocess_dataset({
        "sources": {"tcr_papers": str(path)},
        "blocklists": dict.fromkeys(BLOCKLIST_NAMES, "off"),
        "tcr_region_profile": {"completion_sources": ["tcr_papers"]},
    }, output, splits=["train"])
    papers = result["report"]["splits"][0]
    assert papers["kept_rows"] == 3
    assert papers["filter_reasons"] == {"quality.blank_epitope": 1}
    # The blank-epitope drop also had all-X MHC; only kept rows are counted.
    assert papers["downgraded_all_x_mhc"] == 1
    assert papers["beta_only_completed"] == 1
    assert papers["alpha_only_completed"] == 1
    assert result["report"]["totals"]["downgraded_all_x_mhc"] == 1
    assert result["report"]["totals"]["beta_only_completed"] == 1
    assert result["report"]["totals"]["alpha_only_completed"] == 1
    published = json.loads((output / "filter_report.json").read_text())
    assert published["totals"]["downgraded_all_x_mhc"] == 1
    assert published["splits"][0]["beta_only_completed"] == 1


def test_prepared_loader_fails_on_corrupt_row(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    (output / "train").mkdir(parents=True)
    row = {"schema_version": SCHEMA_VERSION, "chains": [], "chain_roles": [], "task_type": "tcr", "source": "x"}
    (output / "train" / "oas-00000.jsonl").write_text(json.dumps(row) + "\n")
    manifest = {"schema_version": SCHEMA_VERSION, "format": "jsonl", "splits": {"train": {"shards": [{"path": "train/oas-00000.jsonl", "source": "oas", "records": 1}]}}}
    (output / "dataset_manifest.json").write_text(json.dumps(manifest))
    dataset = load_prepared_dataset(output)
    with pytest.raises(ValueError):
        _ = dataset[0]
