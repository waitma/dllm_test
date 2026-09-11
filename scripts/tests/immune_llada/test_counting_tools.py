"""Small prepared fixtures for the three standalone counting CLIs."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from dllm.pipelines.immune_llada.data.preprocessing.reports import new_source_stats, totals
from dllm.pipelines.immune_llada.data.preprocessing.validators import (
    SCHEMA_VERSION,
    add_schema_version,
)
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from scripts import count_grammar_layouts as layouts
from scripts import count_immune_drops as drops
from scripts import count_immune_mix as mix

_ROOT = Path(__file__).resolve().parents[3]
_TOOLS = (mix, drops, layouts)


def _publish(root: Path, records: dict[str, list[BioSeqRecord]]) -> Path:
    train = root / "train"
    train.mkdir(parents=True)
    shards = []
    stats = []
    for source, items in records.items():
        source_stats = new_source_stats(source, "train", "raw-input-not-present.csv")
        source_stats.update(raw_rows=len(items), converted_rows=len(items), kept_rows=len(items))
        # One record per shard exercises source membership across multiple shards.
        for index, record in enumerate(items):
            path = train / f"{source}-{index:05d}.jsonl"
            path.write_text(json.dumps(add_schema_version(record.to_dict())) + "\n", encoding="utf-8")
            shard = {"path": f"train/{path.name}", "source": source, "split": "train", "records": 1}
            shards.append(shard)
            source_stats["shards"].append(shard)
        stats.append(source_stats)
    manifest = {
        "schema_version": SCHEMA_VERSION, "format": "jsonl", "sources": list(records),
        "splits": {"train": {
            "records": sum(len(items) for items in records.values()),
            "sources": {source: {"records": len(items), "raw_rows": len(items)} for source, items in records.items()},
            "shards": shards,
        }},
        "budget": {"max_protein_length": 0, "max_length": 0},
    }
    report = {"schema_version": SCHEMA_VERSION, "dry_run": False, "splits": stats, "totals": totals(stats)}
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "filter_report.json").write_text(json.dumps(report), encoding="utf-8")
    return root


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    return _publish(tmp_path / "prepared", {
        "oas": [BioSeqRecord(
            [BioSeqChain("AAAA", "antibody_heavy"), BioSeqChain("CCCC", "antibody_light")],
            "antibody", "oas_paired", "train", weight=2.0,
        )],
        "trait": [BioSeqRecord(
            [BioSeqChain("AAAA", "peptide"), BioSeqChain("CASSQF", "tcr_beta")],
            "tcr_epitope", "trait", "train", labels={"relation": "binding"},
        )],
    })


def test_mix_reads_prepared_records_without_runtime_filters(prepared: Path, monkeypatch) -> None:
    from dllm.pipelines.immune_llada.data import sources
    from dllm.pipelines.immune_llada.data.preprocessing import filters

    def forbidden(*args, **kwargs):
        pytest.fail("Counting must never parse raw rows or execute filters")

    monkeypatch.setattr(sources, "row_to_record", forbidden)
    monkeypatch.setattr(filters, "build_filters", forbidden)
    monkeypatch.setattr(filters, "filter_reason", forbidden)
    monkeypatch.setattr(filters, "load_blocklists", forbidden)
    before = {path.relative_to(prepared): path.read_bytes() for path in prepared.rglob("*") if path.is_file()}
    result = mix.count_mix(prepared)
    assert result["totals"] == {
        "raw_rows": 2, "records": 2, "residues": 18, "generated_chain_residues": 14, "chains": 4,
    }
    assert [row["source"] for row in result["rows"]] == ["oas", "trait"]
    assert drops.load_drop_report(prepared)["totals"]["kept_rows"] == 2
    assert layouts.count_layouts(prepared)["weighted_layouts"] == {"antibody_pair": 1.0, "tcr_peptide": 1.0}
    after = {path.relative_to(prepared): path.read_bytes() for path in prepared.rglob("*") if path.is_file()}
    assert before == after


def test_manifest_sources_keep_native_and_papers_separate(tmp_path: Path) -> None:
    record = BioSeqRecord([BioSeqChain("ASSQ", "tcr_beta")], "tcr", "tcr_native")
    root = _publish(tmp_path / "prepared", {"tcr_native": [record], "tcr_papers": [record, record]})
    result = mix.count_mix(root, sources="tcr_papers+tcr_native+tcr_papers")
    assert [(row["source"], row["records"]) for row in result["rows"]] == [("tcr_papers", 2), ("tcr_native", 1)]
    assert drops.load_drop_report(root, sources="tcr_papers+tcr_papers")["totals"]["kept_rows"] == 2
    assert layouts.count_layouts(root, sources="tcr_papers")["source_rows"][0]["prepared_records"] == 2


def test_manifest_only_and_drop_report_do_not_open_shards(prepared: Path) -> None:
    next((prepared / "train").glob("oas-*.jsonl")).unlink()
    result = mix.count_mix(prepared, counts_only=True)
    assert result["basis"] == "manifest_only"
    assert result["totals"] == {"raw_rows": 2, "records": 2}
    assert drops.load_drop_report(prepared)["totals"]["kept_rows"] == 2
    for count in (mix.count_mix, layouts.count_layouts):
        with pytest.raises(FileNotFoundError, match="missing"):
            count(prepared)


@pytest.mark.parametrize("count", [mix.count_mix, layouts.count_layouts])
def test_readable_record_count_mismatch_is_fatal(prepared: Path, count) -> None:
    shard = next((prepared / "train").glob("oas-*.jsonl"))
    shard.write_text(shard.read_text() * 2)
    with pytest.raises(ValueError, match="Manifest record count mismatch"):
        count(prepared)


@pytest.mark.parametrize("tool", _TOOLS)
@pytest.mark.parametrize("corruption", ["total", "shard", "source_stats", "schema", "duplicate_shard"])
def test_incomplete_manifest_is_rejected(prepared: Path, tool, corruption: str) -> None:
    path = prepared / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    info = manifest["splits"]["train"]
    if corruption == "total":
        info["records"] = 3
    elif corruption == "shard":
        info["shards"].pop()
    elif corruption == "source_stats":
        del info["sources"]["oas"]["records"]
    elif corruption == "schema":
        manifest["schema_version"] = "retired"
    elif corruption == "duplicate_shard":
        info["shards"].append(info["shards"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit) as exc:
        tool.main(["--prepared-data-dir", str(prepared)])
    assert exc.value.code == 2


def test_requested_split_uses_its_own_counts(prepared: Path) -> None:
    manifest_path = prepared / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["valid"] = {
        "records": 0, "sources": {"oas": {"records": 0, "raw_rows": 1}}, "shards": [],
    }
    manifest_path.write_text(json.dumps(manifest))
    report_path = prepared / "filter_report.json"
    report = json.loads(report_path.read_text())
    stats = new_source_stats("oas", "valid", "not-present.csv")
    stats.update(raw_rows=1, dropped_schema=1)
    report["splits"].append(stats)
    report["totals"] = totals(report["splits"])
    report_path.write_text(json.dumps(report))
    assert mix.count_mix(prepared, split="valid")["totals"]["records"] == 0
    assert drops.load_drop_report(prepared, split="valid")["totals"]["dropped_schema"] == 1
    assert layouts.count_layouts(prepared, split="valid")["sources"] == ["oas"]
    assert drops.load_drop_report(prepared, split="train")["totals"]["dropped_schema"] == 0


def test_first_failure_comes_from_offline_order_not_length_first_stages(tmp_path: Path) -> None:
    from dllm.pipelines.immune_llada.data.preprocessing.filters import BLOCKLIST_NAMES
    from dllm.pipelines.immune_llada.data.preprocessing.pipeline import (
        preprocess_dataset,
    )

    raw = tmp_path / "trait.jsonl"
    raw.write_text("\n".join(json.dumps(row) for row in [
        {"epitope_seq": "AAAA", "cdr3b": "CASSQF"},  # blocked AND over budget
        {"epitope_seq": "AAAA", "cdr3b": "CASSRF"},  # budget only
        {"epitope_seq": "AA", "cdr3b": "AS"},       # kept
        {"epitope_seq": "invalid!", "cdr3b": "AS"}, # schema
    ]) + "\n")
    block = tmp_path / "block.txt"
    block.write_text("ASSQ\n")
    output = tmp_path / "prepared"
    preprocess_dataset({
        "sources": {"trait": str(raw)},
        "blocklists": {**dict.fromkeys(BLOCKLIST_NAMES, "off"), "trait_benchmark": str(block)},
        "budget": {"max_protein_length": 3, "max_length": 0},
    }, output)
    raw.unlink()
    block.unlink()
    result = drops.load_drop_report(output)
    assert result["attribution"] == "first_failure_from_offline_preprocessing"
    assert result["totals"] == {
        "raw_rows": 4, "converted_rows": 3, "kept_rows": 1,
        "dropped_schema": 1, "dropped_filters": 2, "errors": 0,
        "filter_reasons": {"decontam.trait_benchmark": 1, "budget.max_protein_length": 1},
    }


@pytest.mark.parametrize("corruption", ["arithmetic", "reasons", "negative", "duplicate", "missing", "total", "raw", "dry_run", "schema", "errors"])
def test_reject_inconsistent_filter_reports(prepared: Path, corruption: str) -> None:
    path = prepared / "filter_report.json"
    report = json.loads(path.read_text())
    row = report["splits"][0]
    if corruption == "arithmetic":
        row["dropped_filters"] = 1
    elif corruption == "reasons":
        row["filter_reasons"] = {"quality.homotypic_pair": 1}
    elif corruption == "negative":
        row["dropped_schema"] = -1
    elif corruption == "duplicate":
        report["splits"].append(row)
    elif corruption == "missing":
        report["splits"].pop()
    elif corruption == "total":
        report["totals"]["kept_rows"] = 100
    elif corruption == "raw":
        row["raw_rows"] += 1
        row["dropped_schema"] += 1
    elif corruption == "dry_run":
        report["dry_run"] = True
    elif corruption == "schema":
        report["schema_version"] = "retired"
    elif corruption == "errors":
        row["errors"] = 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        drops.load_drop_report(prepared)


def test_uniform_sample_spans_corpus_and_preserves_fractional_weights(tmp_path: Path) -> None:
    chosen = layouts.sample_indices(5, 2, 0)
    assert chosen == sorted(random.Random(0).sample(range(5), 2))
    assert chosen != [0, 1]
    single = BioSeqRecord([BioSeqChain("ASSQ", "tcr_beta")], "tcr", "tcr_native")
    conditional = BioSeqRecord(
        [BioSeqChain("AA", "peptide"), BioSeqChain("ASSQ", "tcr_beta")], "tcr_epitope", "tcr_native",
    )
    rows = [single] * 5
    rows[chosen[0]] = conditional
    root = _publish(tmp_path / "prepared", {"tcr_papers": rows})
    sampled = layouts.count_layouts(root, per_source=2, seed=0)
    assert sampled == layouts.count_layouts(root, per_source=2, seed=0)
    assert sampled["sampling"] == "uniform_without_replacement"
    assert sampled["weighted_layouts"] == {"tcr_peptide": 2.5, "tcr_single": 2.5}
    assert sampled["weighted_generated_tokens"] == {"tcr_peptide": 17.5, "tcr_single": 17.5}
    full = layouts.count_layouts(root, all_records=True)
    assert full["weighted_layouts"] == {"tcr_single": 4.0, "tcr_peptide": 1.0}
    assert full["source_rows"][0]["sampled_records"] == 5


def test_layouts_add_no_new_length_filter(tmp_path: Path) -> None:
    record = BioSeqRecord([BioSeqChain("A" * 1025, "tcr_beta")], "tcr", "tcr_repertoire")
    root = _publish(tmp_path / "prepared", {"tcr_repertoire": [record]})
    assert layouts.count_layouts(root, all_records=True)["weighted_generated_tokens"] == {"tcr_single": 1028.0}


def test_render_error_is_fatal_with_source_and_index(tmp_path: Path) -> None:
    record = BioSeqRecord([BioSeqChain("AAAA", "other")], "tcr", "trait")
    root = _publish(tmp_path / "prepared", {"trait": [record]})
    with pytest.raises(ValueError, match="Cannot render prepared train/trait record 0"):
        layouts.count_layouts(root, all_records=True)


@pytest.mark.parametrize("tool", _TOOLS)
def test_empty_prepared_split_is_reported_as_zero(tmp_path: Path, capsys, tool) -> None:
    root = _publish(tmp_path / "prepared", {"oas": []})
    assert tool.main(["--prepared-data-dir", str(root)]) == 0
    assert "TOTAL" in capsys.readouterr().out


@pytest.mark.parametrize("tool", _TOOLS)
@pytest.mark.parametrize("arguments", [
    ["train", "oas"], ["tcr_papers_dir=old"], ["--scenario", "current"],
    ["--tokens", "oas"], ["--tcr-papers-dir", "old"], ["--prepared-data", "old"],
    ["--sources", ""], ["--sources", "oas+"], ["--sources", "unknown"],
    ["--sources", "tcr_papers"], ["--split", "missing"],
])
def test_old_unknown_and_absent_arguments_fail(prepared: Path, tool, arguments) -> None:
    with pytest.raises(SystemExit) as exc:
        tool.main(["--prepared-data-dir", str(prepared), *arguments])
    assert exc.value.code == 2


@pytest.mark.parametrize("arguments", [
    ["--per-source", "0"], ["--per-source", "-2"],
    ["--all", "--per-source", "1"], ["--all", "--seed", "0"],
])
def test_invalid_or_ignored_sampling_arguments_fail(prepared: Path, arguments) -> None:
    with pytest.raises(SystemExit) as exc:
        layouts.main(["--prepared-data-dir", str(prepared), *arguments])
    assert exc.value.code == 2


@pytest.mark.parametrize("tool", _TOOLS)
def test_standalone_cli_without_trainer_or_model_imports(prepared: Path, tmp_path: Path, tool) -> None:
    # Execute the real script from outside the checkout, with retired imports
    # blocked before import. No PYTHONPATH or installed project is required.
    program = """
import importlib.abc
import runpy
import sys
class RejectRetired(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("protein_pretrain", "dllm.pipelines.bioseq", "dllm.pipelines.qwen3_vl_arch", "transformers", "dllm.pipelines.immune_llada.models")):
            raise AssertionError("Forbidden import: " + fullname)
sys.meta_path.insert(0, RejectRetired())
script = sys.argv.pop(1)
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
"""
    script = _ROOT / "scripts" / f"{tool.__name__.rsplit('.', 1)[-1]}.py"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(script), "--prepared-data-dir", str(prepared), "--json"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["sources"] == ["oas", "trait"]
