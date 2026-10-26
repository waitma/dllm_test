"""Activate pllm; python -B -m pytest THIS_FILE -q (all checks run on CPU).

Exercise the actual compression library, round trips, quota enforcement, resume,
corruption rejection, and a complete 16-partition offline index without models.
"""
from copy import deepcopy
import hashlib
import json
import time

import pytest
import torch

from scripts.research.stage1_ab500k_cache import (
    CODEC, PackedCacheStore, ShardWriter, WorkerTimer, atomic_json, build_cache_index,
    content_sha, decode_record, encode_record, precision_metrics, sha,
    validate_cache_manifest,
)
from scripts.research.stage1_ab500k_teacher import worker_number


def target(row=0):
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    request = dict(schema="immune.structure_teacher.request.v1", record=dict(id=row, sequence=sequence),
        config={}, chains=[dict(sequence=sequence, student_positions=list(range(2, 22)), record_chain=0, student_chain=0)],
        excluded=[], inter_evidence=[], quality_status="unvalidated")
    request["request_sha256"] = content_sha(request)
    pairs = torch.triu_indices(len(sequence), len(sequence), offset=1).T
    contract = dict(distogram_bins=64, revision="cpu-test", mode="trunk_only")
    logits = torch.randn((len(pairs), 64), generator=torch.Generator().manual_seed(row + 9)) * 5
    return dict(request=request, metadata=dict(request_sha256=request["request_sha256"], teacher_contract=contract,
        pair_count=len(pairs), quality_status="unvalidated"), tensors=dict(teacher_logits=logits,
        teacher_pair_indices=pairs, student_positions=torch.arange(2, 22)))


def envelope(cache, row=0, split="train"):
    return dict(split=split, row_index=row, request=cache["request"])


def test_real_zstd_roundtrip_preserves_indices_and_fp16_values():
    cache = target()
    restored = decode_record(encode_record(cache))
    assert restored["request"] == cache["request"]
    assert restored["metadata"] == cache["metadata"]
    for name in ("teacher_pair_indices", "student_positions"):
        assert restored["tensors"][name].dtype == torch.long
        assert torch.equal(restored["tensors"][name], cache["tensors"][name])
    assert restored["tensors"]["teacher_logits"].dtype == torch.float32
    assert torch.equal(restored["tensors"]["teacher_logits"], cache["tensors"]["teacher_logits"].half().float())
    metrics = precision_metrics(cache["tensors"]["teacher_logits"], restored["tensors"]["teacher_logits"])
    assert metrics["mean_kl"] < 1e-5
    assert metrics["max_pair_kl"] < 1e-3


def test_corrupt_and_truncated_frames_fail_closed():
    blob = encode_record(target())
    for corrupted in (blob[:20], blob[:-1], blob[:70] + bytes([blob[70] ^ 1]) + blob[71:]):
        with pytest.raises(ValueError):
            decode_record(corrupted)


def test_uint16_overflow_and_fp16_overflow_rejected():
    cache = target()
    cache["request"]["chains"][0]["student_positions"][0] = 70000
    cache["request"]["request_sha256"] = content_sha({k: v for k, v in cache["request"].items() if k != "request_sha256"})
    cache["metadata"]["request_sha256"] = cache["request"]["request_sha256"]
    cache["tensors"]["student_positions"][0] = 70000
    with pytest.raises(ValueError, match="uint16"):
        encode_record(cache)
    cache = target()
    cache["tensors"]["teacher_logits"][0, 0] = 1e10
    with pytest.raises(ValueError, match="FP16"):
        encode_record(cache)


def test_atomic_resume_discards_uncommitted_records(tmp_path):
    first, second = target(0), target(1)
    with ShardWriter(tmp_path, 0, first["metadata"]["teacher_contract"]) as writer:
        writer.append(envelope(first), first)
        writer.commit()
        writer.append(envelope(second, 1), second)
    assert list((tmp_path / "worker00").glob("*.partial.*"))
    with ShardWriter(tmp_path, 0, first["metadata"]["teacher_contract"]) as writer:
        assert writer.contains(envelope(first))
        assert not writer.contains(envelope(second, 1))
        assert not list((tmp_path / "worker00").glob("*.partial.*"))
        writer.append(envelope(second, 1), second)
        writer.finish(2, {})
    assert len(list((tmp_path / "worker00").glob("shard[0-9]*"))) == 2


def test_resume_rejects_corrupt_committed_shard(tmp_path):
    cache = target()
    with ShardWriter(tmp_path, 0, cache["metadata"]["teacher_contract"]) as writer:
        writer.append(envelope(cache), cache)
        writer.commit()
    path = tmp_path / "worker00/shard000000/records.bin"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ShardWriter(tmp_path, 0, cache["metadata"]["teacher_contract"])


def test_quota_and_shard_rotation(tmp_path):
    cache = target()
    with ShardWriter(tmp_path / "quota", 0, cache["metadata"]["teacher_contract"], budget=1000) as writer:
        with pytest.raises(RuntimeError, match="quota"):
            writer.append(envelope(cache), cache)
    with ShardWriter(tmp_path / "rotate", 0, cache["metadata"]["teacher_contract"], max_records=1) as writer:
        writer.append(envelope(cache), cache)
        other = target(1)
        writer.append(envelope(other, 1), other)
        writer.finish(2, {})
        assert len(writer.shards) == 2


def test_quota_failure_commits_already_accepted_targets(tmp_path):
    first, second = target(0), target(1)
    budget = (2 << 20) + len(encode_record(first)) + 1500
    with ShardWriter(tmp_path, 0, first["metadata"]["teacher_contract"], budget=budget) as writer:
        writer.append(envelope(first), first)
        with pytest.raises(RuntimeError, match="quota"):
            writer.append(envelope(second, 1), second)
        assert not writer.pending
        assert len(writer.shards) == 1
    with ShardWriter(tmp_path, 0, first["metadata"]["teacher_contract"], budget=budget) as writer:
        assert writer.contains(envelope(first))
        assert not writer.contains(envelope(second, 1))


def test_identical_requests_preserve_distinct_original_rows(tmp_path):
    cache = target()
    with ShardWriter(tmp_path, 0, cache["metadata"]["teacher_contract"]) as writer:
        assert writer.append(envelope(cache, 0), cache)
        assert writer.append(envelope(cache, 1), cache)
        writer.finish(2, {})
        assert len(writer.seen) == 2


def test_decoded_targets_join_existing_student_alignment():
    from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
    from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMCollator
    from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
    from scripts.research.stage1_data import batch_teacher_targets, make_teacher_request

    config = FixedChainConfig(d_model=32, n_layers=8, n_heads=4, mlp_hidden_size=64,
        first_chain_capacity=32, second_chain_capacity=32, full_capacity=66)
    record = BioSeqRecord([BioSeqChain("ACDEFGHIK", "antibody_heavy"),
        BioSeqChain("LMNPQR", "antibody_light")], "antibody", "oas", split="train")
    request = make_teacher_request(record, config)
    pairs = torch.tensor([[0, 1], [2, 4], [9, 10], [11, 13]])
    logits = torch.randn(4, 64, generator=torch.Generator().manual_seed(4))
    contract = dict(distogram_bins=64)
    original = dict(request=request, metadata=dict(request_sha256=request["request_sha256"], teacher_contract=contract),
        tensors=dict(teacher_logits=logits, teacher_pair_indices=pairs,
            student_positions=torch.tensor([p for c in request["chains"] for p in c["student_positions"]])))
    restored = decode_record(encode_record(original))
    clean = FullTokenMLMCollator(config)([record])
    joined_pairs, joined_logits, joined_contract = batch_teacher_targets([restored], clean, torch.device("cpu"))
    assert joined_pairs.shape == (4, 3)
    assert torch.equal(joined_logits, logits.half().float())
    assert joined_contract == contract


def test_four_tasks_have_sixteen_distinct_worker_indices():
    assert [worker_number(job, local) for job in range(4) for local in range(4)] == list(range(16))
    with pytest.raises(ValueError):
        worker_number(0, 0, 8)


def test_worker_timing_accumulates_and_flags_unclean_resume(tmp_path):
    with WorkerTimer(tmp_path, time.monotonic()-5, "2026-10-22T00:00:00+00:00"):
        pass
    first = json.loads((tmp_path / "timing.json").read_text())
    assert first["accumulated_seconds"] >= 5
    assert first["timing_complete"]
    with WorkerTimer(tmp_path, time.monotonic()-3, "2026-10-22T00:01:00+00:00"):
        pass
    second = json.loads((tmp_path / "timing.json").read_text())
    assert second["accumulated_seconds"] >= 8
    assert second["start_utc"] == first["start_utc"]
    second["status"] = "running"
    atomic_json(tmp_path / "timing.json", second)
    with WorkerTimer(tmp_path, time.monotonic(), "2026-10-22T00:02:00+00:00"):
        pass
    final = json.loads((tmp_path / "timing.json").read_text())
    assert not final["timing_complete"]
    assert final["timing_note"] == "lower_bound_after_unclean_exit"


def prepared_cache(tmp_path):
    data_index = tmp_path / "data_index.json"
    atomic_json(data_index, {})
    requests = tmp_path / "requests.jsonl"
    caches = [target(i) for i in range(16)]
    envs = [envelope(cache, i, "valid" if i == 15 else "train") for i, cache in enumerate(caches)]
    requests.write_text("".join(json.dumps(x) + "\n" for x in envs))
    subset_path = tmp_path / "subset.json"
    specs = [dict(worker_index=i, count=1, sha256=hashlib.sha256(str(i).encode()).hexdigest(),
        pilot_sha256=hashlib.sha256(("pilot" + str(i)).encode()).hexdigest(), pilot_count=1) for i in range(16)]
    atomic_json(subset_path, dict(requests=str(requests), requests_sha256=sha(requests), request_shards=specs,
        data_index=str(data_index), data_index_sha256=sha(data_index)))
    root = tmp_path / "cache"
    for i, (cache, env) in enumerate(zip(caches, envs)):
        with ShardWriter(root, i, cache["metadata"]["teacher_contract"]) as writer:
            writer.append(env, cache)
            pilot = writer.directory / "pilot.json"
            atomic_json(pilot, dict(status="passed", codec=CODEC, mean_kl=0., max_pair_kl=0., pilot_sha256=specs[i]["pilot_sha256"],
                count=1, teacher_contract_sha256=content_sha(cache["metadata"]["teacher_contract"]),
                measurements=[dict(repeat_max_abs=0.)]))
            writer.finish(1, dict(requests_sha256=specs[i]["sha256"], subset_manifest_sha256=sha(subset_path), pilot_sha256=sha(pilot)))
    return subset_path, root, envs


def test_full_index_lookup_and_missing_target_failure(tmp_path):
    subset, root, envs = prepared_cache(tmp_path)
    path = build_cache_index(subset, root, tmp_path / "index")
    manifest = validate_cache_manifest(path, verify_payloads=True)
    assert manifest["counts"]["train"] == 15
    assert manifest["counts"]["valid"] == 1
    assert manifest["codec"] == CODEC
    store = PackedCacheStore(path)
    try:
        for env in envs:
            cache = store.lookup(env["split"], env["row_index"], env["request"]["record"])
            assert cache["request"] == env["request"]
        with pytest.raises(ValueError, match="missing"):
            store.lookup("train", 999, {})
        with pytest.raises(ValueError, match="differs"):
            store.lookup("train", 0, {"changed": True})
    finally:
        store.close()


def test_missing_worker_prevents_index_publication(tmp_path):
    subset, root, _ = prepared_cache(tmp_path)
    (root / "worker15/complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        build_cache_index(subset, root, tmp_path / "index")
    assert not (tmp_path / "index/manifest.json").exists()
