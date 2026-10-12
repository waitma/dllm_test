"""Run: python -m pytest scripts/tests/immune_llada/test_tcr_exposure.py -q."""
import csv
import json
from pathlib import Path
import sqlite3

import pytest

from scripts.data import audit_tcr_exposure as audit
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord


def test_source_core_letters_are_never_trimmed():
    core, _ = audit.benchmark_keys(dict(cdr3b="CASSF", peptide="ACDEFG"), "immrep_core")
    junction, _ = audit.benchmark_keys({"TRB.CDR3": "CCASSFF", "Epitope": "ACDEFG"}, "uptcr")
    assert core["cdr3b"] == junction["cdr3b"] == "CASSF"
    assert core["bp"] == junction["bp"]


def test_all_support_and_background_files_are_explicitly_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    base = tmp_path / "downstream/benchmark/data/tcr_binding_nm2025"
    for rel in ("seen/cdr3b/AS/test.csv", "train/cdr3b/train.csv"):
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cdr3b,peptide\n")
    specs = {name: (path, kind, usage) for name, path, kind, usage in audit.input_specs()}
    assert specs["tcr_representation/train"][1:] == ("immrep_core", "support")
    assert specs["tcr_representation_paper6/background_pool"][2] == "support_query_universe"
    assert specs["uptcr/few-shot/train"][2] == "support_or_adaptation"
    assert not any(name.startswith("nm2025/train/") for name in specs)


def test_observed_anchors_preserved_and_missing_chain_not_fabricated():
    chain = BioSeqChain("XXCCASSFFXX", "tcr_beta", regions={"CDR3": "CASSF"}, metadata={
        "synthetic_residue_mask": [1,1,0,0,0,0,0,0,0,1,1],
        "observed_regions": ["CDR3"], "observed_anchor_regions": {"FR3": "C", "FR4": "F"}})
    missing = BioSeqChain("XXXXX", "tcr_alpha", metadata={"synthetic_residue_mask": [1]*5})
    row = BioSeqRecord([chain, missing], "tcr", "toy").to_dict()
    keys, _, issues = audit.prepared_keys(row)
    assert keys == {"cdr3b": "CASSF"} and not issues
    assert row["chains"][0] == "XXCCASSFFXX"


def test_end_to_end_epitope_only_and_joint_exposure_are_distinct(tmp_path, monkeypatch):
    work, source, inventory = (tmp_path / name for name in ("work", "source", "inventory"))
    source.mkdir(); inventory.mkdir()
    benchmark = tmp_path / "support.csv"
    benchmark.write_text("cdr3a,cdr3b,peptide\nACDEFG,HIKLMN,PQRSTV\n")
    monkeypatch.setattr(audit, "input_specs", lambda: [("toy/support", benchmark, "immrep_core", "support")])
    monkeypatch.setattr(audit, "INVENTORY", inventory)
    # One row sees the receptor only; another sees the peptide only. Their
    # separate exposure must never be reported as an observed joint pair.
    records = [
        BioSeqRecord([BioSeqChain("HIKLMN", "tcr_beta", regions={"CDR3": "HIKLMN"})], "tcr", "toy").to_dict(),
        BioSeqRecord([BioSeqChain("YYYYYY", "tcr_beta", regions={"CDR3": "YYYYYY"}),
                      BioSeqChain("PQRSTV", "peptide")], "tcr_pmhc", "toy").to_dict(),
        BioSeqRecord([BioSeqChain("HIKLMQ", "tcr_beta", regions={"CDR3": "HIKLMQ"})], "tcr", "toy").to_dict(),
        BioSeqRecord([BioSeqChain("HIKL", "tcr_beta", regions={"CDR3": "HIKL"})], "tcr", "toy").to_dict(),
    ]
    shard = source / "train.jsonl"
    shard.write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest = source / "dataset_manifest.json"
    audit.write_json(manifest, dict(splits={"train": {"shards": [dict(path=shard.name, source="toy", records=4)]}}))
    audit.write_json(inventory / "inventory.json", dict(manifest_sha256=audit.sha(manifest),
        shards=[[str(shard), audit.sha(shard), 4, "{}"]]))
    with sqlite3.connect(inventory / "inventory.sqlite") as db:
        db.execute("CREATE TABLE sequences(role TEXT, sequence TEXT)")
        db.executemany("INSERT INTO sequences VALUES (?,?)", [("cdr3a", "ACDEFG"), ("cdr3b", "HIKLMN"),
            ("cdr3b", "YYYYYY"), ("cdr3b", "HIKLMQ"), ("cdr3b", "HIKL"), ("peptide", "PQRSTV")])
    audit.freeze(work)
    audit.near(work, 1)
    audit.scan(work, source)
    report = json.loads((work / "source/report.json").read_text())
    rows = report["counts"][0]["rows"]
    assert rows["exact_cdr3b"] == rows["exact_peptide"] == rows["epitope_without_near_receptor"] == 1
    assert rows["edit80_cdr3b"] == 2  # One substitution hits; a 4/6 fragment does not.
    assert rows["receptor_or_epitope"] == 3 and rows.get("exact_bp", 0) == 0
    assert report["status"] == "overlap_found"
    benchmark.write_text(benchmark.read_text() + "ACDEFG,HIKLMN,WWWWWW\n")
    with pytest.raises(ValueError, match="Benchmark input changed"):
        audit.frozen(work)
