"""Run: python -m pytest scripts/tests/immune_llada/test_v7_decontamination.py -q."""
from pathlib import Path
import json
import pytest

from scripts.data.prepare_immune_v7 import Components, extract_keys, verify_registry, sha
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain


def test_short_antigen_containment_does_not_remove_unrelated_full_proteins():
    from scripts.data.prepare_immune_v7 import covered_containment
    assert not covered_containment("ACDEFGHIKLMNPQRSTVWY","A")
    assert not covered_containment("ACDE","XXACDEXX")
    assert covered_containment("ACDE","XACDE")
    assert covered_containment("XACDE","ACDE")
    assert not covered_containment("", "ACDE")


def test_core_comparison_never_strips_real_core_end_letters():
    chain=BioSeqChain("XXCASFXX","tcr_beta",regions={"CDR3":"CASF"},metadata={
        "synthetic_residue_mask":[1,1,0,0,0,0,1,1],"observed_regions":["CDR3"]})
    row=BioSeqRecord([chain],"tcr","test").to_dict()
    keys,missing=extract_keys(row)
    assert keys["cdr3b"]=={"CASF"} and not missing
    assert row["chains"]==["XXCASFXX"]


def test_missing_chain_creates_no_fake_core():
    chain=BioSeqChain("XXXXX","tcr_alpha",metadata={"synthetic_residue_mask":[1]*5})
    keys,missing=extract_keys(BioSeqRecord([chain],"tcr","test").to_dict())
    assert not keys and not missing


def test_transitive_pair_family_components():
    c=Components()
    c.join(["heavy:a","h3:b"])
    c.join(["heavy:c","h3:b"])
    c.join(["heavy:d","h3:e"])
    assert c.find("heavy:a")==c.find("heavy:c")
    assert c.find("heavy:a")!=c.find("heavy:d")


def test_frozen_benchmark_changes_are_rejected(tmp_path):
    source=tmp_path/"source.json"
    source.write_text("{}")
    banks=tmp_path/"banks.json"
    banks.write_text("{}")
    (tmp_path/"registry.json").write_text(json.dumps({"inputs":{str(source):sha(source)},"banks_sha256":sha(banks)}))
    verify_registry(tmp_path)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError,match="changed"):
        verify_registry(tmp_path)


@pytest.mark.parametrize("missing_native",[False,True])
def test_export_independent_rescan_and_tamper_gate(tmp_path,missing_native,monkeypatch):
    from scripts.data.prepare_immune_v7 import RULES, digest, export, verify_export, write_json
    from examples.llada.immune_pretrain_v3 import assert_ready
    from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset
    work, source, out = (tmp_path/name for name in ("work","source","output"))
    work.mkdir(); source.mkdir(); (work/"clusters").mkdir()
    # Choose family hashes with known train/valid assignments; sequences are
    # arbitrary synthetic strings for plumbing, not biological annotations.
    vg = next(f"cdr3b:g{i}" for i in range(1000) if int(digest(f"cdr3b:g{i}")[:8],16)%100==0)
    tg = next(f"cdr3b:g{i}" for i in range(1000) if int(digest(f"cdr3b:g{i}")[:8],16)%100!=0)
    rows=[]
    for core in ("AAAA","CCCC","AAAA","DDDD"):
        rec=BioSeqRecord([BioSeqChain(core,"tcr_beta",regions={"CDR3":core})],"tcr","toy")
        rows.append(rec.to_dict())
    shard=source/"input.jsonl"
    shard.write_text("".join(json.dumps(r)+"\n" for r in rows))
    manifest=dict(schema_version="immune_llada.semantic.v1",format="jsonl",splits={"train":{"records":4,
        "shards":[dict(path=shard.name,source="toy",split="train",records=4)]}})
    write_json(source/"dataset_manifest.json",manifest)
    banks={r:[] for r in (*RULES,"pair","light")}; banks["cdr3b"]=["DDDD"]
    write_json(work/"banks.json",banks); write_json(work/"effective_banks.json",banks)
    benchmark=tmp_path/"nbbench/hf_data/CDRInfilling/test.csv"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("seq,label,CDR1,CDR2,CDR3\nsequence,sequence,CDR1,CDR2,CDR3\n"+
                         ("AAAAAAAAAAAAAAAAAAAA,AAAAAAAAAAAAAAAAAAAA,A,A,A\n" if missing_native else ""))
    write_json(work/"registry.json",dict(inputs={str(benchmark):sha(benchmark)},banks_sha256=sha(work/"banks.json"),missing_legacy_inputs=[]))
    write_json(work/"inventory.json",dict(manifest_sha256=sha(source/"dataset_manifest.json"),
        shards=[[str(shard),sha(shard),4,"{}"]]))
    write_json(work/"benchmark_h3_annotations.json",dict(status="complete",banks_sha256=sha(work/"banks.json"),unresolved=[]))
    write_json(work/"benchmark_h3_coverage.json",dict(status="complete",registry_sha256=sha(work/"registry.json"),
        annotations_sha256=sha(work/"benchmark_h3_annotations.json"),effective_banks_sha256=sha(work/"effective_banks.json"),
        extra_inputs={},unresolved=[],source_covered={}))
    for role in RULES:
        mapping=work/"clusters"/f"{role}.tsv"
        mapping.write_text(f"AAAA\t{tg}\t0\nCCCC\t{vg}\t0\nDDDD\tcdr3b:protected\t1\n" if role=="cdr3b" else "")
        write_json(mapping.with_suffix(".json"),dict(status="complete",mapping_sha256=sha(mapping),
            inventory_sha256=sha(work/"inventory.json"),effective_banks_sha256=sha(work/"effective_banks.json")))
    export(work,source,out)
    verify_export(work,out)
    certificate=json.loads((out/"audit_certificate.json").read_text())
    assert certificate["benchmark_repeated_header_rows"]==1
    if missing_native:
        assert "unprotected_native_benchmark_reference" in certificate["pending"]
        with pytest.raises(ValueError,match="incomplete"):
            assert_ready(out)
        return
    # The v7-only certificate is no longer sufficient for the expanded scope.
    with pytest.raises(ValueError,match="Current downstream scope audit is missing"):
        assert_ready(out)
    # This fixture tests the legacy exporter; the new scope gate is tested
    # separately with hash-bound input/evidence fixtures.
    monkeypatch.setattr("scripts.data.tcr_scope_gate.require_current_scope",lambda _: None)
    _, certificate=assert_ready(out)
    assert certificate["rows"]=={"train":1,"valid":1}
    assert len(PreparedImmuneDataset(out,split="train"))==1
    report=json.loads((out/"cleaning_report.json").read_text())
    assert report["removed_reasons"]["decontam.cdr3b"]==1
    assert report["removed_reasons"]["dedup.exact_semantic"]==1
    emitted=next((out/"train").glob("*.jsonl")); emitted.write_text(emitted.read_text()+"\n")
    with pytest.raises(ValueError,match="changed"):
        assert_ready(out)
