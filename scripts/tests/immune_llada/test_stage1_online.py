"""Activate pllm; CUDA_VISIBLE_DEVICES='' USE_TF=0 python -B -m pytest THIS_FILE -q.

Exercise atomic publication, competing writers, miss/hit recovery, student RNG,
real target alignment, corruption rejection and teacher process failures on CPU.
"""
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
from safetensors.torch import save_file

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import file_sha256
from scripts.research.launch_stage1_training import readiness
from scripts.research.run_stage1_online import completed_training, qualified_receipt
from scripts.research.stage1_data import content_sha, make_teacher_request
from scripts.research.stage1_online_cache import cache_path, ensure_cache
from scripts.research.stage1_online_teacher import OnlineTeacher


def config():
    return FixedChainConfig(d_model=32, n_layers=8, n_heads=4, mlp_hidden_size=64,
        first_chain_capacity=16, second_chain_capacity=12, full_capacity=30)


def record():
    return BioSeqRecord([BioSeqChain("ACDEFGHIK", "antibody_heavy"),
        BioSeqChain("LMNPQR", "antibody_light")], "antibody", "online_test", split="train")


def plan(root):
    return dict(teacher_contract=dict(distogram_bins=64, version="test"), max_pairs_per_record=8,
        online_teacher=dict(cache_root=str(root), request_timeout_seconds=1))


def write_target(directory, request, contract):
    positions = [p for c in request["chains"] for p in c["student_positions"]]
    save_file(dict(teacher_logits=torch.arange(64, dtype=torch.float32)[None],
        teacher_pair_indices=torch.tensor([[0, 1]]), student_positions=torch.tensor(positions)),
        str(directory / "targets.safetensors"))
    (directory / "metadata.json").write_text(json.dumps(dict(schema="immune.structure_teacher.sparse_cache.v1",
        request_sha256=request["request_sha256"], teacher_contract=contract,
        targets_sha256=file_sha256(directory / "targets.safetensors"))))


def fake_teacher(client, calls):
    def call(message):
        calls.append(message)
        request = message["request"]
        destination, created = ensure_cache(client.plan["online_teacher"]["cache_root"], request,
            lambda p: write_target(p, request, client.plan["teacher_contract"]),
            lambda p: client.load(p, request))
        return dict(status="generated" if created else "reused", directory=str(destination), seconds=.1)
    return call


def test_miss_then_hit_across_restart_preserves_student_rng(tmp_path):
    spec = plan(tmp_path)
    first = OnlineTeacher(spec, torch.device("cpu"), 0, tmp_path / "logs")
    calls = []
    first.call = fake_teacher(first, calls)
    state = torch.get_rng_state().clone()
    rows = [(4, record(), None)]
    generated, counts = first.resolve(rows, config())
    assert counts["generated"] == 1 and len(calls) == 1
    assert calls[0]["request"]["chains"][0]["sequence"] == "ACDEFGHIK"
    assert calls[0]["request"]["inter_evidence"] == []
    restarted = OnlineTeacher(spec, torch.device("cpu"), 0, tmp_path / "logs")
    restarted.call = lambda _: pytest.fail("A complete cache hit must not invoke the teacher")
    reused, counts = restarted.resolve(rows, config())
    assert counts["online_hits"] == 1 and counts["generated"] == 0
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(v, reused[0][2]["tensors"][k]) for k, v in generated[0][2]["tensors"].items())
    seed_rows, counts = restarted.resolve(generated, config())
    assert counts["seed_hits"] == 1 and seed_rows[0][2] is generated[0][2]


def test_incomplete_beta_stays_mlm_only_but_real_full_beta_is_generated(tmp_path):
    complete = BioSeqRecord([BioSeqChain("ACDEFGHIK", "tcr_beta")], "tcr", "online_test")
    partial = deepcopy(complete)
    partial.chains[0].metadata["sequence_scope"] = "cdr3"
    client = OnlineTeacher(plan(tmp_path), torch.device("cpu"), 0, tmp_path / "logs")
    calls = []
    client.call = fake_teacher(client, calls)
    rows, counts = client.resolve([(0, partial, None), (1, complete, None)], config())
    assert rows[0][2] is None and rows[1][2] is not None
    assert counts["ineligible"] == 1 and counts["generated"] == 1 and len(calls) == 1


def test_corrupt_entry_and_contract_mismatch_are_not_silently_regenerated(tmp_path):
    client = OnlineTeacher(plan(tmp_path), torch.device("cpu"), 0, tmp_path / "logs")
    calls = []
    client.call = fake_teacher(client, calls)
    source = [(0, record(), None)]
    rows, _ = client.resolve(source, config())
    request = rows[0][2]["request"]
    destination = cache_path(tmp_path, request["request_sha256"])
    (destination / "targets.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        client.resolve(source, config())
    assert len(calls) == 1
    rows[0][2]["metadata"]["teacher_contract"] = dict(distogram_bins=64, version="wrong")
    with pytest.raises(ValueError, match="contract"):
        client.resolve(rows, config())


def test_atomic_failure_leaves_no_complete_cache_and_retry_succeeds(tmp_path):
    client = OnlineTeacher(plan(tmp_path), torch.device("cpu"), 0, tmp_path / "logs")
    request = make_teacher_request(record(), config())
    destination = cache_path(tmp_path, request["request_sha256"])
    def broken(path):
        (path / "targets.safetensors").write_bytes(b"partial")
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        ensure_cache(tmp_path, request, broken, lambda _: None)
    assert not destination.exists() and not list(destination.parent.glob("*.partial.*"))
    orphan = destination.parent / (destination.name + ".partial.orphan")
    orphan.mkdir()
    result, created = ensure_cache(tmp_path, request,
        lambda p: write_target(p, request, client.plan["teacher_contract"]), lambda p: client.load(p, request))
    assert created and result == destination and not orphan.exists()


def competing_writer(root, request):
    root = Path(root)
    def build(path):
        with (root / "builds.txt").open("a") as stream:
            stream.write("build\n")
        time.sleep(.2)
        (path / "targets.safetensors").write_bytes(b"complete")
        (path / "metadata.json").write_text(json.dumps(request))
    def verify(path):
        assert (path / "targets.safetensors").read_bytes() == b"complete"
        assert json.loads((path / "metadata.json").read_text()) == request
    ensure_cache(root, request, build, verify)


def test_competing_processes_only_generate_once(tmp_path):
    payload = dict(example="duplicate across ranks")
    request = dict(**payload, request_sha256=content_sha(payload))
    context = mp.get_context("spawn")
    processes = [context.Process(target=competing_writer, args=(str(tmp_path), request)) for _ in range(2)]
    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=45)
        if p.is_alive():
            p.kill(); p.join()
        assert p.exitcode == 0
    assert (tmp_path / "builds.txt").read_text() == "build\n"


def test_bad_request_hash_rejected_before_writer(tmp_path):
    with pytest.raises(ValueError, match="hash"):
        ensure_cache(tmp_path, dict(request_sha256="../escape"), lambda _: pytest.fail(), lambda _: pytest.fail())


@pytest.mark.parametrize("response,match", [(dict(status="error", kind="ValueError", message="invalid target"), "invalid target"),
                                          (None, "exited")])
def test_teacher_error_and_dead_worker_propagate(tmp_path, response, match):
    script = tmp_path / "fake_worker.py"
    script.write_text("import json\n" + ("print(" + repr(json.dumps(response)) + ", flush=True)\n" if response else ""))
    client = OnlineTeacher(plan(tmp_path), torch.device("cpu"), 0, tmp_path / "logs")
    client.process = subprocess.Popen([sys.executable, str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        with pytest.raises(RuntimeError, match=match):
            client.receive("ensure")
    finally:
        client.close()


def test_teacher_timeout_terminates_process(tmp_path):
    script = tmp_path / "slow_worker.py"
    script.write_text("import time\ntime.sleep(.3)\n")
    spec = plan(tmp_path)
    spec["online_teacher"]["request_timeout_seconds"] = .02
    client = OnlineTeacher(spec, torch.device("cpu"), 0, tmp_path / "logs")
    process = subprocess.Popen([sys.executable, str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    client.process = process
    with pytest.raises(TimeoutError):
        client.receive("ensure")
    assert client.process is None and process.poll() is not None


def test_online_readiness_does_not_require_precomputed_train_coverage():
    spec = dict(phase="online", teacher_quality_policy="exploratory_unvalidated",
        cache_coverage_policy="generate_every_eligible_miss_before_update",
        online_teacher=dict(mode="on_demand_write_through"), minimum_valid_caches=128)
    assert readiness(spec, dict(counts={"valid/cached": 761}, coverage=dict(train=0))) == []
    assert readiness(spec, dict(counts={"valid/cached": 0}))
    del spec["teacher_quality_policy"]
    assert readiness(spec, dict(counts={"valid/cached": 761}))


def test_student_only_profile_cannot_qualify_online_training(tmp_path):
    path = tmp_path / "COMPLETE.json"
    path.write_text(json.dumps(dict(status="passed", qualification={}, world_size=8,
        micro_batch=24, scope="allocation_only; no biological targets")))
    with pytest.raises(ValueError, match="joint"):
        qualified_receipt(path, dict(qualification={}, world_size=8), "profile")


def test_completion_from_another_frozen_run_is_not_reused(tmp_path):
    spec = dict(training=dict(max_steps=10), world_size=8, online_launch_sha256="current")
    assert not completed_training(tmp_path, spec)
    (tmp_path / "train").mkdir()
    (tmp_path / "train/COMPLETE.json").write_text(json.dumps(dict(status="success", step=10, world_size=8)))
    (tmp_path / "train/run_provenance.json").write_text(json.dumps(dict(online_launch_sha256="old")))
    with pytest.raises(ValueError, match="another run"):
        completed_training(tmp_path, spec)
    (tmp_path / "train/run_provenance.json").write_text(json.dumps(dict(online_launch_sha256="current")))
    assert completed_training(tmp_path, spec)
