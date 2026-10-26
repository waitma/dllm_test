"""Activate pllm; CUDA_VISIBLE_DEVICES='' USE_TF=0 python -B -m pytest THIS_FILE -q.

Exercise cache auditing with real tensor files/SQLite and stop/reuse sequencing
without running GPU jobs or accepting structural quality from engineering checks.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import subprocess

import pytest
import torch
from safetensors.torch import save_file

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from scripts.research import bert_mlm_data, run_stage1_checks as runner
from scripts.research.audit_stage1_cache import audit
from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.stage1_data import content_sha, make_teacher_request


@pytest.fixture
def cache(tmp_path, monkeypatch):
    config = FixedChainConfig()
    record = BioSeqRecord([BioSeqChain("ACDEFGHIK", "antibody_heavy")], "antibody", "unit", split="train")
    request = make_teacher_request(record, config)
    requests = tmp_path/"requests.jsonl"
    requests.write_text(json.dumps(dict(split="train", row_index=0, request=request))+"\n")
    write_json(tmp_path/"binding.json", dict(model=asdict(config)))
    write_json(tmp_path/"index.json", dict(binding=str(tmp_path/"binding.json"), records=dict(train=10, valid=2)))
    monkeypatch.setattr(bert_mlm_data, "IndexedPreparedDataset", lambda path, split: [record]*(10 if split == "train" else 2))
    weights = tmp_path/"weights"
    teacher_plan = dict(requests=str(requests), script=str(tmp_path/"teacher.py"), model_dir=str(weights),
        references={str(tmp_path/"teacher.py"): "script", str(weights/"config.json"): "config",
                    str(weights/"model.safetensors"): "weights"}, output=str(tmp_path),
        workers=1, num_loops=3, seed=42, pairs_per_chain_pair=1024)
    contract = dict(fast_compat_source_sha256="script", all_parameters_frozen=True, mode="trunk_only",
        num_loops=3, base_seed=42, pairs_per_chain_pair=1024, config_sha256="config",
        weight_sha256={"model.safetensors": "weights"}, distogram_bins=64)
    directory = tmp_path/request["request_sha256"]
    directory.mkdir()
    save_file(dict(teacher_logits=torch.zeros(2, 64), teacher_pair_indices=torch.tensor([[0, 1], [1, 2]]),
                   student_positions=torch.tensor(request["chains"][0]["student_positions"])), str(directory/"targets.safetensors"))
    write_json(directory/"metadata.json", dict(schema="immune.structure_teacher.sparse_cache.v1",
        request_sha256=request["request_sha256"], teacher_contract=contract,
        targets_sha256=sha(directory/"targets.safetensors"), chain_lengths=[9], asym_id=[0]*9,
        pair_count=2, teacher_requires_grad=False, symmetry_max_abs=0., repeat_max_abs=0.,
        seed=(42+int(request["request_sha256"][:8], 16)) % (2**31)))
    database = tmp_path/"cache.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE targets (split TEXT,row_index INTEGER,record_sha TEXT,request TEXT,directory TEXT,metadata_sha TEXT)")
        conn.execute("INSERT INTO targets VALUES (?,?,?,?,?,?)", ("train", 0, content_sha(record.to_dict()),
            json.dumps(request), str(directory), sha(directory/"metadata.json")))
    manifest = tmp_path/"manifest.json"
    write_json(manifest, dict(schema="immune.stage1.cache_index.v1", database=str(database), database_sha256=sha(database),
        data_index=str(tmp_path/"index.json"), data_index_sha256=sha(tmp_path/"index.json"), requests=str(requests),
        requests_sha256=sha(requests), corpus_records=dict(train=10, valid=2), teacher_contract=contract,
        counts={"train/requested": 1, "train/cached": 1, "train/unit/cached": 1, "train/pairs": 2},
        max_pairs_per_record=2, coverage=dict(train=.1, valid=0.)))
    write_json(tmp_path/"verification.json", dict(status="passed", teacher_contract=contract, requests_sha256=sha(requests),
        worker_index=0, worker_count=1, counts=dict(cached=1, reused=0, mlm_only=0), loading_info={}))
    return manifest, teacher_plan, directory


def test_integrity_pass_never_implies_structural_quality(cache):
    manifest, teacher, _ = cache
    report = audit(manifest, teacher)
    assert report["status"] == "passed"
    assert report["repeat_checked_records"] == 1
    assert report["structural_accuracy"]["status"] == "unvalidated"
    assert report["formal_training_ready"] is False
    assert any("experimental structures" in x for x in report["formal_training_blockers"])


def test_audit_rejects_changed_targets(cache):
    manifest, teacher, directory = cache
    with (directory/"targets.safetensors").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="target file changed"):
        audit(manifest, teacher)


def test_audit_requires_every_eligible_request(cache):
    manifest, teacher, _ = cache
    value = json.loads(manifest.read_text())
    with sqlite3.connect(value["database"]) as conn:
        conn.execute("DELETE FROM targets")
    value["database_sha256"] = sha(value["database"])
    write_json(manifest, value)
    with pytest.raises(ValueError, match="Eligible request missing"):
        audit(manifest, teacher)


def test_replay_missing_is_reported_not_fabricated(cache):
    manifest, teacher, directory = cache
    metadata = json.loads((directory/"metadata.json").read_text())
    metadata["repeat_max_abs"] = None
    write_json(directory/"metadata.json", metadata)
    value = json.loads(manifest.read_text())
    with sqlite3.connect(value["database"]) as conn:
        conn.execute("UPDATE targets SET metadata_sha=?", (sha(directory/"metadata.json"),))
    value["database_sha256"] = sha(value["database"])
    write_json(manifest, value)
    report = audit(manifest, teacher)
    assert report["repeat_checked_records"] == 0 and report["repeat_max_abs"] is None
    assert any("no seeded replay" in x for x in report["formal_training_blockers"])


def test_failed_smoke_never_starts_teacher(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "completed_smoke", lambda plan: False)
    def fail(command):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(runner, "execute", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.sequence(dict(smoke_entrypoint="smoke.sh", teacher_entrypoint="teacher.sh"), lambda phase: None)
    assert calls == [["bash", "smoke.sh"]]


def test_combined_completion_reuses_smoke_and_retains_quality_blocker(tmp_path, monkeypatch):
    calls, phases = [], []
    manifest = tmp_path/"manifest.json"
    index, requests = tmp_path/"index.json", tmp_path/"requests.jsonl"
    index.write_text("{}")
    requests.write_text("{}\n")
    write_json(manifest, dict(data_index=str(index), data_index_sha256=sha(index),
                              requests=str(requests), requests_sha256=sha(requests)))
    audit_path = tmp_path/"audit.json"
    plan = dict(smoke_entrypoint="smoke.sh", teacher_entrypoint="teacher.sh", cache_index_dir=str(tmp_path),
        python="python", audit_script="audit.py", source_root="frozen", plan_path="plan.json", audit_output=str(audit_path),
        data_index=str(index), requests=str(requests))
    monkeypatch.setattr(runner, "completed_smoke", lambda plan: True)
    def fake(command):
        calls.append(command)
        if "audit.py" in command:
            write_json(audit_path, dict(status="passed", cache_manifest_sha256=sha(manifest),
                structural_accuracy=dict(status="unvalidated"), formal_training_blockers=["no experimental reference"]))
    monkeypatch.setattr(runner, "execute", fake)
    result = runner.sequence(plan, phases.append)
    assert len(calls) == 2 and calls[0] == ["bash", "teacher.sh", "--verify-repeat"]
    assert phases == ["student_smoke", "teacher_pilot", "cache_index", "cache_audit"]
    assert result["status"] == "checks_complete" and result["formal_training_ready"] is False


def test_completed_smoke_must_match_qualification(tmp_path):
    write_json(tmp_path/"smoke_plan.json", dict(output_dir=str(tmp_path), qualification=dict(revision="current")))
    write_json(tmp_path/"COMPLETE.json", dict(status="passed", world_size=8, qualification=dict(revision="old")))
    with pytest.raises(ValueError, match="another configuration"):
        runner.completed_smoke(dict(smoke_plan=str(tmp_path/"smoke_plan.json")))
