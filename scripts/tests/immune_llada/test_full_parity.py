"""Regression tests for the offline, independent full-dataset parity audit."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.data import run_immune_full_parity as audit
from dllm.pipelines.immune_llada.data.preprocessing.filters import BLOCKLIST_NAMES
from dllm.pipelines.immune_llada.data.preprocessing.pipeline import preprocess_dataset


def test_multiset_detects_duplicate_counts_and_ignores_order(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    left.write_text("aaa\naaa\nbbb\n")
    right.write_text("bbb\naaa\naaa\n")
    assert audit.compare_multiplicity(left, right, tmp_path)["different_fingerprints"] == 0
    right.write_text("aaa\nbbb\nbbb\n")
    result = audit.compare_multiplicity(left, right, tmp_path)
    assert result["different_fingerprints"] == 2
    assert result["missing_occurrences"] == 1
    assert result["extra_occurrences"] == 1


@pytest.fixture
def corpus(tmp_path):
    raw = tmp_path / "train.csv"
    rows = [
        {"cdr3b": "ASSQ"}, {"cdr3b": "ASSQ"}, {"cdr3b": "ASSF"},
        {"cdr3b": "CCCC"}, {"cdr3b": "bad!"}, {"cdr3b": "A" * 80},
    ]
    with raw.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cdr3b"])
        writer.writeheader()
        writer.writerows(rows)
    block = tmp_path / "block.txt"
    block.write_text("ASSF|PEP\n")
    config = {
        "sources": {"tcr_repertoire": {"path": str(raw)}},
        "splits": ["train"],
        "blocklists": {name: "off" for name in BLOCKLIST_NAMES},
        "budget": {"max_protein_length": 64, "max_length": 64},
    }
    config["blocklists"]["t4_refbinder"] = str(block)
    # JSON is valid YAML and avoids an extra test-only YAML dependency.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config))
    prepared = tmp_path / "prepared"
    preprocess_dataset(config, prepared)
    return config_path, prepared


def test_full_audit_loads_historical_code_and_matches_filters(corpus, tmp_path):
    config, prepared = corpus
    work = tmp_path / "audit"
    # Exit 2 is expected and correct here: beta-only completion turns one CDR3 row
    # into a two-chain alpha/beta layout with a relation label, which the frozen
    # baseline has no notion of. The audit stays strict -- what must still match
    # exactly is the filter/keep decision, asserted below.
    assert audit.main(["--config", str(config), "--prepared-data-dir", str(prepared), "--work-dir", str(work)]) == 2
    report = json.loads((work / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["scope"] == "full"
    assert report["totals"]["raw_rows"] == 6
    assert report["totals"]["prepared"] == 3
    # Filter decisions are untouched by the approved divergence.
    assert report["totals"]["legacy_kept"] == report["totals"]["canonical_kept"] == 3
    assert report["totals"].get("decision_mismatches", 0) == 0
    assert report["totals"]["count_mismatches"] == 0
    # The divergence is confined to record semantics and the rendered stream.
    assert report["totals"]["semantic_mismatches"] == 3
    assert report["totals"]["rendering_mismatches"] == 3
    assert {example["kind"] for example in report["pairs"][0]["examples"]} == {"semantics"}
    example = report["pairs"][0]["examples"][0]
    assert example["legacy"]["chain_roles"] == ["tcr_beta"]
    assert example["canonical"]["chain_roles"] == ["tcr_beta", "tcr_alpha"]
    assert report["pairs"][0]["rendering"]["counts"]["records"] == 3
    assert len(report["legacy_blob_sha256"]) == len(audit.LEGACY_FILES)
    assert (work / "legacy_snapshot/grammar.py").read_text().count("class GrammarRenderer") == 1
    assert Counter((work / "train.tcr_repertoire.legacy.sha256").read_text().splitlines()).most_common(1)[0][1] == 2


def test_audit_exits_nonzero_on_prepared_duplicate_corruption(corpus, tmp_path):
    config, prepared = corpus
    shard = next((prepared / "train").glob("*.jsonl"))
    lines = shard.read_text().splitlines()
    lines[-1] = lines[0]
    shard.write_text("\n".join(lines) + "\n")
    work = tmp_path / "audit"
    assert audit.main(["--config", str(config), "--prepared-data-dir", str(prepared), "--work-dir", str(work)]) == 2
    report = json.loads((work / "report.json").read_text())
    assert report["status"] == "failed"
    # Corruption is caught on the canonical (current-code) side: that is the signal
    # which must survive the approved beta-only divergence.
    assert report["totals"]["canonical_multiset_differences"] == 2
    # The legacy side carries the duplicate plus the intentional layout divergence.
    assert report["totals"]["legacy_multiset_differences"] == 3


@pytest.mark.parametrize(
    "source,row",
    [
        ("oas", {"cleaned_h_sequence": "AAAA", "cleaned_l_sequence": "CCCC", "l_locus": "H"}),
        ("ots", {"cleaned_chain1_seq": "AAAA", "cleaned_chain2_seq": "CCCC", "chain1_anarci_type": "B", "chain2_anarci_type": "B"}),
    ],
)
def test_strict_audit_reports_approved_homotypic_exclusion(tmp_path, source, row):
    # Approved exclusions must remain visible against the unmodified historical
    # baseline: policy-aware acceptance is not strict old/new dataset equality.
    raw = tmp_path / "train.csv"
    with raw.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    config = {
        "sources": {source: {"path": str(raw)}},
        "splits": ["train"],
        "blocklists": {name: "off" for name in BLOCKLIST_NAMES},
        "budget": {"max_protein_length": 64, "max_length": 64},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config))
    prepared = tmp_path / "prepared"
    preprocess_dataset(config, prepared)
    work = tmp_path / "audit"
    assert audit.main(["--config", str(config_path), "--prepared-data-dir", str(prepared), "--work-dir", str(work)]) == 2
    report = json.loads((work / "report.json").read_text())
    counts = report["totals"]
    assert counts["legacy_kept"] == 1
    assert counts["canonical_kept"] == counts.get("prepared", 0) == 0
    assert counts["decision_mismatches"] == 1
    assert counts.get("semantic_mismatches", 0) == 0
    assert counts["legacy_multiset_differences"] == 1
    assert counts["canonical_multiset_differences"] == 0
    assert counts["rendering_mismatches"] == counts["collator_mismatches"] == 0
    example = report["pairs"][0]["examples"][0]
    assert example["kind"] == "decision"
    assert example["legacy_kept"] is True
    assert example["canonical_reason"] == "quality.homotypic_pair"


def test_prefix_smoke_not_full_corpus(corpus, tmp_path):
    config, prepared = corpus
    work = tmp_path / "prefix"
    # Same approved beta-only divergence as the full-scope audit; this test pins the
    # prefix scope and row cap, not old/new equality.
    assert audit.main(["--config", str(config), "--prepared-data-dir", str(prepared), "--work-dir", str(work), "--max-rows", "2"]) == 2
    report = json.loads((work / "report.json").read_text())
    assert report["scope"] == "prefix_smoke"
    assert report["totals"]["raw_rows"] == report["totals"]["prepared"] == 2
    with pytest.raises(SystemExit):
        audit.main(["--config", str(config), "--prepared-data-dir", str(prepared), "--work-dir", str(work)])
