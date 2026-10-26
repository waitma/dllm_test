"""Run in pllm: CUDA_VISIBLE_DEVICES='' USE_TF=0 python -B -m pytest THIS_FILE -q.

CPU/Gloo integration checks unequal rank workloads, an empty-KD rank, global
objective equivalence, and complete optimizer/scheduler/noise/cursor restoration.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel, PairConfig
from dllm.pipelines.immune_llada.fixed_chain.structure_losses import chain_aware_kd, normalized_objective
from scripts.research.run_stage1_distillation import make_optimizer, prepare_microbatch
from scripts.research.run_stage1_training import update, scheduler_for
from scripts.research.run_bert_mlm import save_checkpoint, load_checkpoint, capture_rng
from scripts.research.stage1_corpus import CacheStore
from scripts.research.stage1_data import content_sha
from scripts.research.stage1_teacher import iter_requests
from scripts.research.launch_stage1_training import readiness


def small_model():
    torch.manual_seed(22)
    return StructureMLMModel(FixedChainConfig(d_model=32, n_layers=8, n_heads=4,
        mlp_hidden_size=64, first_chain_capacity=16, second_chain_capacity=12, full_capacity=30),
        PairConfig(width=8, query_chunk=8))


def settings():
    return dict(backbone_lr=1e-3, pair_lr=2e-3, weight_decay=.01, warmup_steps=2,
                max_steps=8, kd_warmup_steps=3, intra_weight=.2, inter_weight=.3,
                temperature=1., clip_norm=1., precision="fp32")


def sample():
    return BioSeqRecord([BioSeqChain("ACDEFGHIK", "antibody_heavy"),
        BioSeqChain("LMNPQR", "antibody_light")], "antibody", "unit", split="train")


def inputs(model):
    result = []
    for number in range(3):
        batch, _, _, _ = prepare_microbatch(model, [sample()], [None], torch.Generator().manual_seed(52+number), "cpu")
        positions = [((batch["chain_ids"][0] == uid) & batch["residue_mask"][0]).nonzero().flatten()
                     for uid in batch["chain_ids"][0].unique() if uid >= 0]
        if number == 0:
            pairs = torch.tensor([[0, positions[0][0], positions[0][1]], [0, positions[1][0], positions[1][1]],
                                  [0, positions[0][0], positions[1][0]]])
        elif number == 1:
            pairs = torch.tensor([[0, positions[0][1], positions[0][2]]])
        else:
            pairs = torch.empty(0, 3, dtype=torch.long)
        teacher = torch.randn(len(pairs), 64, generator=torch.Generator().manual_seed(71+number))
        result.append((batch, pairs, teacher, None))
    return result


def worker(rank, temporary):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.distributed.init_process_group("gloo", init_method="file://"+str(Path(temporary)/"rendezvous"), rank=rank, world_size=2)
    try:
        model = small_model()
        ddp = DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=False)
        spec = settings()
        optimizer = make_optimizer(model, spec["backbone_lr"], spec["pair_lr"], spec["weight_decay"])
        scheduler = scheduler_for(optimizer, spec)
        micros = inputs(model)
        local = micros[:2] if rank == 0 else micros[2:]
        metrics = update(ddp, optimizer, scheduler, local, spec, 2, 0)
        if rank == 0:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), metrics=metrics), Path(temporary)/"distributed.pt")
        noise = torch.Generator().manual_seed(322+rank)
        state = dict(step=1, epoch=0, next_batch=1, seen_records=3, best_step=0, best_score=1.)
        provenance = dict(test="uneven ranks and empty KD", world=2)
        checkpoint = save_checkpoint(Path(temporary), model, optimizer, scheduler, noise, state, provenance, rank, 2)
        noisy = prepare_microbatch(model, [sample()], [None], noise, "cpu")
        update(ddp, optimizer, scheduler, [noisy], spec, 2, 1)
        expected = {k: v.detach().clone() for k,v in model.state_dict().items()}
        expected_optimizer = optimizer.state_dict()
        expected_optimizer = {p:{k:v.clone() for k,v in s.items()} for p,s in expected_optimizer["state"].items()}
        expected_noise = noise.get_state().clone()
        restored = load_checkpoint(checkpoint, model, optimizer, scheduler, noise, provenance, rank)
        assert restored == state
        replay = prepare_microbatch(model, [sample()], [None], noise, "cpu")
        assert all(torch.equal(v, replay[0][k]) for k,v in noisy[0].items())
        update(ddp, optimizer, scheduler, [replay], spec, 2, 1)
        assert all(torch.equal(v, expected[k]) for k,v in model.state_dict().items())
        assert all(torch.equal(v, expected_optimizer[p][k]) for p,s in optimizer.state_dict()["state"].items() for k,v in s.items())
        assert torch.equal(expected_noise, noise.get_state())
        assert scheduler.last_epoch == 2
        wrong = dict(provenance, world=8)
        with pytest.raises(RuntimeError, match="differs"):
            load_checkpoint(checkpoint, model, optimizer, scheduler, noise, wrong, rank)
    finally:
        torch.distributed.destroy_process_group()


def test_ddp_global_objective_and_exact_resume(tmp_path):
    torch.set_num_threads(1)
    mp.spawn(worker, args=(str(tmp_path),), nprocs=2, join=True)
    actual = torch.load(tmp_path/"distributed.pt", weights_only=False)
    model = small_model()
    spec = settings()
    optimizer = make_optimizer(model, spec["backbone_lr"], spec["pair_lr"], spec["weight_decay"])
    scheduler = scheduler_for(optimizer, spec)
    micros = inputs(model)
    batch = {key:torch.cat([x[0][key] for x in micros]) for key in micros[0][0]}
    pairs = []
    for row, item in enumerate(micros):
        pair = item[1].clone()
        pair[:, 0] = row
        pairs.append(pair)
    pairs = torch.cat(pairs)
    teacher = torch.cat([x[2] for x in micros])
    output = model(pair_indices=pairs, **batch)
    kd = chain_aware_kd(output["pair_logits"], teacher, pairs, batch["chain_ids"], batch["noise_kind"])
    loss = normalized_objective(output["loss_sum"], output["mlm_count"], kd,
        spec["intra_weight"]/spec["kd_warmup_steps"], spec["inter_weight"]/spec["kd_warmup_steps"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step(); scheduler.step()
    assert actual["metrics"]["counts"][1:] == [2, 1]
    assert actual["metrics"]["kd_coverage"] == pytest.approx(2/3)
    assert actual["metrics"]["loss"] == pytest.approx(float(loss.detach()), rel=2e-6)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, actual["model"][key], atol=3e-6, rtol=3e-5)
    for p, values in optimizer.state_dict()["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, actual["optimizer"]["state"][p][key], atol=3e-7, rtol=2e-4)


def test_cache_missing_is_mlm_only_but_bound_row_mismatch_fails(tmp_path):
    database = tmp_path/"cache.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE targets (split TEXT,row_index INTEGER,record_sha TEXT,request TEXT,directory TEXT,metadata_sha TEXT)")
    connection.execute("INSERT INTO targets VALUES ('train',0,'wrong','{}','unused','wrong')")
    connection.commit(); connection.close()
    store = CacheStore(dict(database=str(database)))
    assert store.lookup("train", 1, sample()) is None
    with pytest.raises(ValueError, match="changed"):
        store.lookup("train", 0, sample())


def test_streaming_teacher_partitions_and_hash_failure(tmp_path):
    rows = []
    for i in range(5):
        payload = dict(chains=[], ordinal=i)
        rows.append(dict(request=dict(**payload, request_sha256=content_sha(payload))))
    path = tmp_path/"requests.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    partitions = [[r["ordinal"] for r in iter_requests(path, rank, 2)] for rank in range(2)]
    assert partitions == [[0, 2, 4], [1, 3]]
    rows[0]["request"]["ordinal"] = 99
    path.write_text(json.dumps(rows[0])+"\n")
    with pytest.raises(ValueError, match="hash"):
        list(iter_requests(path))


def test_formal_training_cannot_use_smoke_quality_or_missing_profile(tmp_path):
    plan = dict(phase="train", gates=dict(profile=str(tmp_path/"profile"), smoke=str(tmp_path/"smoke")),
        qualification={}, world_size=8, training=dict(micro_batch=1), quality=None,
        minimum_train_cache_coverage=.1, minimum_valid_caches=128)
    manifest = dict(coverage=dict(train=2/6249956), counts={"train/cached":2})
    blockers = readiness(plan, manifest)
    assert len(blockers) == 5
    assert any("quality" in x for x in blockers)
    assert any("coverage" in x for x in blockers)
