"""Run: python -m pytest scripts/tests/immune_llada/test_mhc.py -q."""
import json
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

from dllm.pipelines.immune_llada.data.mhc import PseudoTable, resolve_mhc, explicit_mhc_chains
from dllm.pipelines.immune_llada.data.records import BioSeqRecord
from dllm.pipelines.immune_llada.data.sources import row_to_record
from scripts.data.update_mhc_inputs import identity, non_mhc, replace_mhc, source_proposal


@pytest.fixture
def table(tmp_path):
    p = tmp_path / "reference.dat"
    p.write_text("HLA-A02:01 " + "A" * 34 + "\nHLA-B51:12 " + "C" * 34 +
                 "\nHLA-B51:12 " + "D" * 34 + "\nH-2-Db " + "E" * 34 + "\n")
    return PseudoTable(p)


@pytest.mark.parametrize("name", ["HLA-A*02:01", "A*02:01", "HLA-A02:01", "A0201"])
def test_standard_allele_aliases(name, table):
    assert table.lookup(name) == ("A" * 34, "exact_unique_pseudo")


@pytest.mark.parametrize("name", ["HLA-A*02", "HLA-A*02:01:01:01", "HLA-DRB1*01:01", "HLA-A*99:99"])
def test_no_allele_resolution_truncation_or_class_ii_guess(name, table):
    assert table.lookup(name) == ("", "allele_not_in_reference")


def test_conflict_is_not_first_row_wins(table):
    assert table.lookup("HLA-B*51:12") == ("", "conflicting_reference")


def test_supplied_sequence_overrides_lookup_and_is_not_cropped(table):
    seq = "ACDX" * 90
    result = resolve_mhc(provided=[dict(sequence=seq)], allele="HLA-A*02:01", table=table)
    assert result["chains"][0]["sequence"] == seq
    assert result["resolution"]["status"] == "provided_sequence_kept"
    assert resolve_mhc(provided=[dict(sequence="X" * 34)], allele="A*02:01", table=table)["chains"][0]["sequence"] == "A" * 34


def test_pseudo_and_partial_x_are_preserved(table):
    result = resolve_mhc(provided=[dict(sequence="C" * 33 + "X", representation="existing_pseudo34")], allele="A*02:01", table=table)
    assert result["chains"][0]["sequence"] == "C" * 33 + "X"
    assert result["chains"][0]["metadata"]["sequence_scope"] == "fragment"


def test_full_and_b2m_survive_csv_adapter_and_index_remap(table):
    entries = resolve_mhc(provided=[dict(sequence="ACDE" * 90, scope="full", component="mhc_heavy"),
                                   dict(sequence="ACD" * 33, scope="full", component="b2m")],
                          allele="A*02:01", table=table)["chains"]
    raw = dict(epitope_seq="ACDEFGHIK", cdr3b="ASSGGF", cdr3a="AVGG", mhc_seq="A" * 34,
               relation="binding", mhc_chains_json=json.dumps(entries))
    rec = row_to_record("tcr_native", raw)
    assert rec.sequences[:2] == ["ACDE" * 90, "ACD" * 33]
    old = {"chains": ["A" * 34, "ACDEFGHIK", "CASSF"], "chain_roles": ["mhc", "peptide", "tcr_beta"],
           "regions": {"2": {"CDR3": "ASS"}}, "chain_metadata": {"2": {"test": True}},
           "task_type": "tcr_pmhc", "source": "test", "split": "train"}
    updated = replace_mhc(old, entries, {"status": "source_matched"})
    assert non_mhc(old) == non_mhc(updated)
    assert updated["regions"]["3"] == old["regions"]["2"]
    assert updated["chain_metadata"]["3"] == old["chain_metadata"]["2"]
    assert BioSeqRecord.from_prepared_dict(updated).sequences[:2] == rec.sequences[:2]


def test_explicit_missing_overrides_legacy_and_plain_rows_stay_compatible():
    row = dict(epitope_seq="ACDEF", cdr3b="CASSF", mhc_seq="A" * 34)
    assert row_to_record("trait", row).chain_roles[0] == "mhc"
    assert row_to_record("trait", dict(row, mhc_chains_json="[]")).chain_roles[0] == "peptide"
    with pytest.raises(ValueError):
        explicit_mhc_chains(dict(mhc_chains_json='{"sequence":"ACD"}'))


def test_observed_identity_keeps_old_mhc_and_ignores_synthetic_padding():
    old = dict(chains=["A" * 34, "XXCASSFXX"], chain_roles=["mhc", "tcr_beta"], source="test",
               chain_metadata={"1": {"synthetic_residue_mask": [1, 1, 0, 0, 0, 0, 0, 1, 1]}})
    short = dict(chains=["A" * 34, "CASSF"], chain_roles=["mhc", "tcr_beta"], source="test")
    assert identity(old, "group") == identity(short, "group")
    assert identity(old, "group") != identity(dict(short, chains=["C" * 34, "CASSF"]), "group")


def test_explicit_source_array_wins_without_legacy_id_lookup(table):
    entries = resolve_mhc(provided=[dict(sequence="ACDE" * 90, component="mhc_heavy", scope="full")],
                          allele="A*02:01", table=table)["chains"]
    proposal = source_proposal(dict(source="tcr_pmhc_fulllength", mhc_seq="A" * 34,
                              mhc_allele_norm="A*02:01", mhc_chains_json=json.dumps(entries)), table, {})
    assert proposal["chains"] == entries


def test_full_source_ingestion_preserves_heavy_and_b2m(monkeypatch, tmp_path, table):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "scripts/data/tcr_native"))
    ingest = importlib.import_module("scripts.data.tcr_native.ingest_repo")
    raw = tmp_path / "full.jsonl"
    raw.write_text(json.dumps(dict(chains=["ACDE" * 90, "ACD" * 33, "ACDEF", "ACDEFG", "ACDEFG"],
        chain_roles=["mhc", "mhc", "peptide", "tcr_alpha", "tcr_beta"],
        metadata={"mhc_a": "HLA-A*02:01", "origin": "fixture"}, labels={"relation": "binding"})) + "\n")
    ref = tmp_path / "mhc_sequence_v1/reference/MHC_pseudo.dat"
    ref.parent.mkdir(parents=True)
    ref.write_bytes(Path(table.path).read_bytes())
    monkeypatch.setattr(ingest, "DATA", tmp_path)
    monkeypatch.setattr(ingest, "FULL_JSONL", raw)
    def number_fixture(items, **kwargs):
        return {name: dict(gate_pass=True, fv=sequence, cdr3="CASS") for name, sequence in items}
    monkeypatch.setitem(sys.modules, "anarci_fv", SimpleNamespace(segment_batch=number_fixture))
    rows, report = ingest.ingest_fulllength()
    assert report["counts"]["final"] == 1
    assert rows[0]["mhc_seq"] == "ACDE" * 90
    assert row_to_record("tcr_native", rows[0]).sequences[:2] == ["ACDE" * 90, "ACD" * 33]
