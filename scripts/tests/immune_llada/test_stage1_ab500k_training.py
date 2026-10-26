"""Activate pllm; CUDA_VISIBLE_DEVICES='' USE_TF=0 python -B -m pytest THIS_FILE -q.

Tiny real CPU models check independent microbatches, both optimizer paths,
checkpoint replay, complete-epoch tails and paired held-out comparisons.
"""
from copy import deepcopy
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMModel
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel, PairConfig
from scripts.research.bert_mlm_data import CompleteEpochBatchSampler, step_budget, sha
from scripts.research.run_bert_mlm import save_checkpoint, load_checkpoint
from scripts.research.run_stage1_ab500k import (DEFAULTS, build_micros, prepare_batch,
    optimizer_update, new_optimizer, scheduler_for, training_noise_seed, validate, recover_completion,
    validate_resume_cursor, training_loader, record_training_metrics)
from scripts.research.stage1_data import make_teacher_request
from scripts.research.report_stage1_ab500k import paired_bootstrap, first_target_step


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_config():
    return FixedChainConfig(d_model=32, n_layers=8, n_heads=4, mlp_hidden_size=64,
                           first_chain_capacity=16, second_chain_capacity=12, full_capacity=30)


def make_model(arm):
    torch.manual_seed(18)
    return (FullTokenMLMModel(model_config()) if arm == "mlm" else
            StructureMLMModel(model_config(), PairConfig(width=8, query_chunk=8)))


def rows(config, count=5):
    result = []
    for number in range(count):
        record = BioSeqRecord([BioSeqChain("ACDEFGHIK"[:9-number % 3], "antibody_heavy"),
                              BioSeqChain("LMNPQR"[:6-number % 2], "antibody_light")],
                             "antibody", "unit", split="train")
        request = make_teacher_request(record, config)
        positions, pairs = [], []
        for chain in request["chains"]:
            begin = len(positions)
            positions.extend(chain["student_positions"])
            pairs.extend([[begin+i, begin+i+1] for i in range(len(chain["sequence"])-1)])
        cache = dict(request=request, metadata={"teacher_contract": {"unit": True}}, tensors=dict(
            student_positions=torch.tensor(positions), teacher_pair_indices=torch.tensor(pairs),
            teacher_logits=torch.randn(len(pairs), 64, generator=torch.Generator().manual_seed(900+number))))
        result.append((100+number, record, cache))
    return result


def settings():
    return {**DEFAULTS, "precision": "fp32", "backbone_lr": 1e-3, "pair_lr": 2e-3,
            "warmup_steps": 2, "max_steps": 8, "kd_warmup_steps": 2}


def concatenated(micros, key):
    return torch.cat([item[0][key] for item in micros])


def test_noise_is_identical_across_arms_microbatches_and_rank_partition():
    a, d = make_model("mlm"), make_model("pair_kd")
    examples = rows(a.config, 7)
    seed = training_noise_seed(42, 2, 6249956)
    reference = build_micros(a, examples, 7, seed, "mlm")
    for micro in (1, 2, 4, 7):
        actual = build_micros(d, examples, micro, seed, "pair_kd")
        for key in ("input_ids", "labels", "noise_kind"):
            assert torch.equal(concatenated(reference, key), concatenated(actual, key))
    for rank in range(3):
        part = build_micros(d, examples[rank::3], 2, seed, "pair_kd")
        assert torch.equal(concatenated(part, "labels"), concatenated(reference, "labels")[rank::3])
    next_epoch = build_micros(a, examples, 7, training_noise_seed(42, 3, 6249956), "mlm")
    assert not torch.equal(concatenated(reference, "noise_kind"), concatenated(next_epoch, "noise_kind"))


def test_parent_initialization_is_exact_and_missing_target_fails(tmp_path):
    a = make_model("mlm")
    a.save_pretrained(tmp_path)
    d = StructureMLMModel.from_bert(tmp_path, PairConfig(width=8, query_chunk=8))
    inputs = prepare_batch(a, rows(a.config), seed=10042, targets=False)[0]
    a.eval(); d.eval()
    with torch.no_grad():
        assert torch.equal(a(**inputs)["logits"], d(**inputs)["logits"])
    missing = [(0, rows(a.config, 1)[0][1], None)]
    with pytest.raises(ValueError, match="Missing offline"):
        prepare_batch(d, missing, seed=1)


@pytest.mark.parametrize("arm", ["mlm", "pair_kd"])
def test_actual_updates_are_microbatch_invariant(arm):
    model, other = make_model(arm), make_model(arm)
    data, spec = rows(model.config, 5), settings()
    first, second = new_optimizer(model, spec, arm), new_optimizer(other, spec, arm)
    scheduler, schedule2 = scheduler_for(first, spec), scheduler_for(second, spec)
    reference = {key: value.clone() for key, value in model.state_dict().items()}
    metrics = optimizer_update(model, first, scheduler, build_micros(model, data, 5, 901, arm), spec, arm, 0)
    actual = optimizer_update(other, second, schedule2, build_micros(other, data, 2, 901, arm), spec, arm, 0)
    assert metrics["records"] == 5
    assert metrics["mlm"] == pytest.approx(actual["mlm"], rel=2e-6)
    if arm == "pair_kd":
        assert metrics["kd_records"] == 5
    assert any(not torch.equal(value, reference[key]) for key, value in model.state_dict().items())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key], atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("arm", ["mlm", "pair_kd"])
def test_real_checkpoint_restores_next_update_and_scheduler(tmp_path, arm):
    rendezvous = tmp_path/"gloo"
    torch.distributed.init_process_group("gloo", init_method="file://"+str(rendezvous), rank=0, world_size=1)
    try:
        model, spec = make_model(arm), settings()
        data = rows(model.config, 3)
        optimizer = new_optimizer(model, spec, arm)
        scheduler = scheduler_for(optimizer, spec)
        noise = torch.Generator().manual_seed(42)
        optimizer_update(model, optimizer, scheduler, build_micros(model, data, 2, 10042, arm), spec, arm, 0)
        state = dict(step=1, epoch=0, next_batch=1, seen_records=3, best_step=1)
        provenance = dict(arm=arm, noise_recipe="test_per_record_epoch", world_size=1)
        checkpoint = save_checkpoint(tmp_path, model, optimizer, scheduler, noise, state, provenance, 0, 1)
        optimizer_update(model, optimizer, scheduler, build_micros(model, data, 2, 10043, arm), spec, arm, 1)
        expected_model, expected_opt = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
        restored = load_checkpoint(checkpoint, model, optimizer, scheduler, noise, provenance, 0)
        assert restored == state
        optimizer_update(model, optimizer, scheduler, build_micros(model, data, 2, 10043, arm), spec, arm, 1)
        for key, value in model.state_dict().items():
            assert torch.equal(value, expected_model[key])
        for key, values in optimizer.state_dict()["state"].items():
            for name, value in values.items():
                assert torch.equal(value, expected_opt["state"][key][name])
        assert scheduler.last_epoch == 2
        with pytest.raises(RuntimeError, match="differs"):
            load_checkpoint(checkpoint, model, optimizer, scheduler, noise, {**provenance, "arm": "changed"}, 0)
    finally:
        torch.distributed.destroy_process_group()


def test_3000_step_tail_budget_and_rank_coverage():
    budget = step_budget(500000, 512, 3000)
    assert budget == dict(steps_per_epoch=977, completed_epochs=3, next_batch=69,
        seen_records=1535328, equivalent_epochs=3.070656, tail_records=288, epochs_started=4)
    tails = [next(iter(CompleteEpochBatchSampler(500000, 512, rank, 8, 42, 0, 976, 1))) for rank in range(8)]
    assert all(len(row) == 36 for row in tails)
    assert len(set(index for row in tails for index in row)) == 288


def test_initial_best_checkpoint_zero_survives_retention_and_replays_first_update(tmp_path):
    torch.distributed.init_process_group("gloo", init_method="file://"+str(tmp_path/"initial_gloo"),
                                         rank=0, world_size=1)
    try:
        model, spec = make_model("pair_kd"), settings()
        data = rows(model.config, 3)
        optimizer = new_optimizer(model, spec, "pair_kd")
        scheduler = scheduler_for(optimizer, spec)
        noise = torch.Generator().manual_seed(42)
        state = dict(step=0, epoch=0, next_batch=0, seen_records=0, best_step=0, best_score=1.)
        provenance = dict(arm="pair_kd", purpose="initial best and first-update replay")
        checkpoint = save_checkpoint(tmp_path, model, optimizer, scheduler, noise, state, provenance, 0, 1)
        assert checkpoint.name == "checkpoint-0" and scheduler.last_epoch == 0
        optimizer_update(model, optimizer, scheduler, build_micros(model, data, 2, 10042, "pair_kd"),
                         spec, "pair_kd", 0)
        expected = deepcopy(model.state_dict())
        restored = load_checkpoint(checkpoint, model, optimizer, scheduler, noise, provenance, 0)
        validate_resume_cursor(restored, 500000, 512)
        assert restored == state and scheduler.last_epoch == 0 and not optimizer.state
        with pytest.raises(ValueError, match="cursor differs"):
            validate_resume_cursor({**restored, "seen_records": 512}, 500000, 512)
        optimizer_update(model, optimizer, scheduler, build_micros(model, data, 2, 10042, "pair_kd"),
                         spec, "pair_kd", 0)
        assert all(torch.equal(value, expected[key]) for key, value in model.state_dict().items())
        for step in (1, 2, 3):
            if step > 1:
                optimizer_update(model, optimizer, scheduler,
                    build_micros(model, data, 2, 10042+step, "pair_kd"), spec, "pair_kd", step-1)
            save_checkpoint(tmp_path, model, optimizer, scheduler, noise,
                {**state, "step": step, "next_batch": step, "seen_records": 3*step}, provenance, 0, 1)
        assert {path.name for path in tmp_path.glob("checkpoint-*")} == {"checkpoint-0", "checkpoint-2", "checkpoint-3"}
        latest = json.loads((tmp_path/"latest.json").read_text())
        assert latest["best_step"] == 0 and latest["step"] == 3
    finally:
        torch.distributed.destroy_process_group()


def ddp_worker(rank, temporary, arm):
    from pathlib import Path
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method="file://"+str(Path(temporary)/"ddp_init"),
                                         rank=rank, world_size=2)
    try:
        model, spec = make_model(arm), settings()
        wrapped = DistributedDataParallel(model, broadcast_buffers=False)
        optimizer = new_optimizer(model, spec, arm)
        scheduler = scheduler_for(optimizer, spec)
        data = rows(model.config, 5)
        # A short global tail is uneven across ranks; rank zero has two micros and rank one one.
        local = data[:3] if rank == 0 else data[3:]
        metrics = optimizer_update(wrapped, optimizer, scheduler,
            build_micros(model, local, 2, 901, arm), spec, arm, 0, world=2)
        added = {name: parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                 for name, parameter in model.named_parameters()
                 if name.startswith(("pair_encoder.", "pair_bias_outputs.", "distogram_head."))}
        if rank == 0:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), metrics=metrics,
                            pair_gradients=added), Path(temporary)/"distributed.pt")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("arm", ["mlm", "pair_kd"])
def test_two_rank_ddp_uneven_tokens_micros_and_tail_match_one_batch(tmp_path, arm):
    mp.spawn(ddp_worker, args=(str(tmp_path), arm), nprocs=2, join=True)
    actual = torch.load(tmp_path/"distributed.pt", weights_only=False)
    model, spec = make_model(arm), settings()
    optimizer = new_optimizer(model, spec, arm)
    scheduler = scheduler_for(optimizer, spec)
    expected = optimizer_update(model, optimizer, scheduler,
        build_micros(model, rows(model.config, 5), 5, 901, arm), spec, arm, 0)
    assert actual["metrics"]["selected_tokens"] == expected["selected_tokens"]
    assert actual["metrics"]["records"] == expected["records"] == 5
    assert actual["metrics"]["mlm"] == pytest.approx(expected["mlm"], rel=3e-6)
    assert actual["metrics"]["intra_kl"] == pytest.approx(expected["intra_kl"], rel=3e-6)
    if arm == "pair_kd":
        assert len(actual["pair_gradients"]) > 8 and all(actual["pair_gradients"].values())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, actual["model"][key], atol=3e-6, rtol=3e-5)
    for key, values in optimizer.state_dict()["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, actual["optimizer"]["state"][key][name], atol=3e-7, rtol=2e-4)


def test_three_mask_validation_and_bootstrap_preserve_record_counts(tmp_path):
    model = make_model("mlm")
    data = rows(model.config, 7)
    first = validate(model, data, settings(), "mlm", torch.device("cpu"), 0, 1, 2, tmp_path, 0)
    second = validate(model, data, settings(), "mlm", torch.device("cpu"), 0, 1, 7, tmp_path, 250)
    assert first["masks"] == 3 and first["records"] == 7
    assert first["selected_tokens"] == second["selected_tokens"]
    assert first["aa_ce"] == pytest.approx(second["aa_ce"], rel=1e-6)
    result = paired_bootstrap([first]*3, [first]*3, repeats=20)
    assert result["relative_aa_ce_improvement_ci95"] == [0., 0.]
    assert first_target_step([dict(step=0, aa_ce=2.), dict(step=250, aa_ce=1.9),
                              dict(step=500, aa_ce=1.8)], 1.95) == 250
    assert first_target_step([dict(step=0, aa_ce=2.), dict(step=250, aa_ce=1.9),
                              dict(step=500, aa_ce=1.99)], 1.95) is None


def test_final_checkpoint_can_reconstruct_completion_without_training(tmp_path):
    for step in range(0, 3001, 250):
        data = tmp_path/f"validation_step{step}.records.npz"
        np.savez_compressed(data, row_ids=np.arange(4096), statistics=np.zeros((4096, 8)))
        (tmp_path/f"validation_step{step}.json").write_text(json.dumps(dict(step=step, records=4096, masks=3,
            record_statistics=dict(path=str(data), sha256=sha(data)))))
    state = dict(step=3000, seen_records=1535328, timing_complete=False)
    recover_completion(tmp_path, state, "mlm", DEFAULTS)
    assert json.loads((tmp_path/"COMPLETE.json").read_text())["completion_reconstructed_from_checkpoint"]
    (tmp_path/"validation_step3000.records.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified validation"):
        recover_completion(tmp_path, state, "mlm", DEFAULTS)


def test_report_does_not_accept_faster_aa_with_full_token_regression(monkeypatch):
    from scripts.research import report_stage1_ab500k as report
    monkeypatch.setattr(report, "paired_bootstrap", lambda a, d: {"test": True})
    provenance = {key: "same" for key in ("plan_sha256", "subset_manifest_sha256", "cache_manifest_sha256",
        "parent_model_sha256", "training", "noise_recipe", "noise_parent_records", "recipe", "validation_seeds", "world_size")}
    a = dict(provenance={**provenance, "cache_manifest_sha256": None}, validation=[dict(step=i*250, aa_ce=loss, full_token_ce=2.)
        for i, loss in enumerate((2., 1.99, 1.97, 1.95, 1.94))])
    d = dict(provenance=provenance, validation=[dict(step=i*250, aa_ce=loss, full_token_ce=2.1)
        for i, loss in enumerate((2., 1.94, 1.93, 1.92, 1.91))])
    result = report.compare(a, d)
    assert result["convergence"]["first_target_step"] == {"mlm": 750, "pair_kd": 250}
    assert not result["lower_ce_signal"] and not result["convergence"]["twenty_percent_fewer_steps_signal"]


def test_training_metrics_keep_cumulative_and_batch_counts_separate(tmp_path):
    state = dict(step=2, selected_tokens=321, eligible_tokens=999, seen_records=8)
    scores = dict(selected_tokens=123, eligible_tokens=400, records=4, mlm=.2)
    record_training_metrics(tmp_path, "mlm", state, scores)
    saved = json.loads((tmp_path/"metrics.jsonl").read_text())
    assert saved["selected_tokens"] == 321 and saved["eligible_tokens"] == 999
    assert saved["batch_selected_tokens"] == 123 and saved["batch_eligible_tokens"] == 400
    assert json.loads((tmp_path/"status.json").read_text())["step"] == 2


def loader_loop_worker(rank, temporary, arm):
    """Exercise the production loading/logging boundary after DDP and checkpoint 0."""
    from scripts.research.run_bert_mlm import rank_zero
    torch.set_num_threads(1)
    out = Path(temporary)
    torch.distributed.init_process_group("gloo", init_method="file://"+str(out/"loader_ddp_init"),
                                         rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        model, spec = make_model(arm), settings()
        wrapped = DistributedDataParallel(model, broadcast_buffers=False)
        optimizer = new_optimizer(model, spec, arm)
        scheduler = scheduler_for(optimizer, spec)
        data = rows(model.config, 6)
        state = dict(step=0, selected_tokens=0, eligible_tokens=0, seen_records=0,
                     epoch=0, next_batch=0, best_step=0, best_score=1e30)
        save_checkpoint(out, model, optimizer, scheduler, torch.Generator().manual_seed(rank),
                        state, dict(test="loader_and_logging", arm=arm), rank, 2)
        sampler = CompleteEpochBatchSampler(6, 4, rank, 2, 42, 0)
        loader = training_loader(data, sampler, spec, rank, 0)
        assert loader.multiprocessing_context.get_start_method() == "spawn"
        assert loader.timeout > 0
        for batch in loader:
            scores = optimizer_update(wrapped, optimizer, scheduler,
                build_micros(model, batch, 2, 10042, arm), spec, arm, state["step"], world=2)
            state["step"] += 1
            state["selected_tokens"] += scores["selected_tokens"]
            state["eligible_tokens"] += scores["eligible_tokens"]
            state["seen_records"] += scores["records"]
            rank_zero(lambda: record_training_metrics(out, arm, state, scores), rank)
        # Failure in rank-zero logging must reach every rank instead of stranding peers.
        try:
            rank_zero(lambda: record_training_metrics(out/"absent", arm, state, scores), rank)
        except RuntimeError as exc:
            assert "FileNotFoundError" in str(exc)
        else:
            raise AssertionError("Logging failure was not propagated")
        (out/f"loader_rank{rank}.json").write_text(json.dumps(state))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("arm", ["mlm", "pair_kd"])
def test_spawn_loader_checkpoint_to_training_logging_two_ranks(tmp_path, arm):
    mp.spawn(loader_loop_worker, args=(str(tmp_path), arm), nprocs=2, join=True)
    logs = [json.loads(line) for line in (tmp_path/"metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in logs] == [1, 2]
    assert logs[-1]["seen_records"] == 6
    assert logs[-1]["selected_tokens"] == sum(row["batch_selected_tokens"] for row in logs)
    assert logs[-1]["eligible_tokens"] == sum(row["batch_eligible_tokens"] for row in logs)
    assert json.loads((tmp_path/"loader_rank0.json").read_text()) == json.loads((tmp_path/"loader_rank1.json").read_text())


def test_report_reads_all_bound_teacher_sessions_and_marks_lower_bound(tmp_path):
    from scripts.research.report_stage1_ab500k import teacher_timing
    subset_path, cache_path = tmp_path/"subset.json", tmp_path/"cache.json"
    specs = [dict(worker_index=i, count=1, sha256=f"request{i}") for i in range(16)]
    subset_path.write_text(json.dumps(dict(request_shards=specs)))
    bindings = []
    for spec in specs:
        directory = tmp_path/f"worker{spec['worker_index']:02d}"
        directory.mkdir()
        receipt = directory/"complete.json"
        receipt.write_text(json.dumps(dict(status="passed", count=1, requests_sha256=spec["sha256"],
            subset_manifest_sha256=sha(subset_path), teacher_contract={"teacher": "frozen"},
            seconds=120., session_seconds=60., start_utc="2026-10-22T00:00:00+00:00",
            end_utc="2026-10-22T00:02:00+00:00", timing_complete=spec["worker_index"] != 0,
            timing_note="test")))
        bindings.append(dict(path=str(receipt), sha256=sha(receipt)))
    cache_path.write_text(json.dumps(dict(teacher_contract={"teacher": "frozen"}, workers=bindings)))
    plan = dict(subset_manifest=str(subset_path), cache_manifest=str(cache_path), cache_root=str(tmp_path))
    result = teacher_timing(plan)
    assert result["gpu_hours"] == pytest.approx(16*120/3600)
    assert result["wall_seconds"] == 120
    assert result["status"] == "lower_bound_after_unclean_exit"
    (tmp_path/"worker00"/"complete.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        teacher_timing(plan)
