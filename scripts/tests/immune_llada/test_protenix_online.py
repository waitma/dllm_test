"""Activate pllm; USE_TF=0 python -B -m pytest THIS_FILE -q.

Tiny CPU teachers test online semantics, not scientific teacher accuracy.
"""
import random
import weakref
from dataclasses import asdict

import numpy as np
import pytest
import torch

from scripts.tests.immune_llada.test_structure_mlm import config, record
from scripts.research.stage1_ab500k_initialization import build_scratch_model, shared_state_hash
from scripts.research.stage1_protenix_online import (
    OnlineProtenixTeacher, teacher_rng, make_request, build_micros, optimizer_update)
from scripts.research.run_stage1_training import scheduler_for
from scripts.research.run_stage1_ab500k import DEFAULTS
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel


class FakeAdapter:
    def __init__(self, device):
        self.device = device
        self.contract = dict(model="test", distogram_bins=64)
        random.random(); np.random.random(); torch.rand(1)
        self.model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        self.examples = []

    def prepare(self, example):
        self.examples.append(example)
        random.random(); np.random.random(); torch.rand(1)
        return example

    def predict_with_confidence(self, prepared, threshold):
        size = sum(prepared["chain_lengths"])
        random.random(); np.random.random()
        return dict(logits=torch.randn(size, size, 64), confidence=torch.rand(size, size))


@pytest.fixture(autouse=True)
def small_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_rng_loading_prepare_and_prediction_are_isolated_and_repeatable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def draws():
        return random.random(), np.random.random(), torch.rand(8)
    with teacher_rng(123, "cpu"):
        expected = draws()
    with teacher_rng(123, "cpu"):
        teacher = OnlineProtenixTeacher("cpu", adapter_factory=FakeAdapter)
        request = make_request(record(), config())
        first = teacher.targets([request])
        second = teacher.targets([request])
        actual = draws()
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert all(not t.requires_grad for t in first)
    assert not list(tmp_path.iterdir())
    assert len(teacher.adapter.examples) == 2  # Recompute; no memoized feature cache.


def test_complete_cross_chain_mapping_uses_clean_sequences():
    model = build_scratch_model(asdict(config()), "pair_kd", 42, dict(width=8))
    rows = [(10, record(), None)]
    batch, pairs, requests = build_micros(model, rows, 1, 10042, "pair_kd")[0]
    assert len(pairs) == 15*14//2
    r, i, j = pairs.unbind(-1)
    same = batch["chain_ids"][r, i] == batch["chain_ids"][r, j]
    assert (~same).sum() == 9*6
    assert batch["residue_mask"][r, i].all() and batch["residue_mask"][r, j].all()
    assert [c["sequence"] for c in requests[0]["chains"]] == ["ACDEFGHIK", "LMNPQR"]
    teacher = OnlineProtenixTeacher("cpu", adapter_factory=FakeAdapter)
    targets, weights = teacher.targets(requests)
    assert targets.shape == (105, 64) and weights.shape == (105,)
    with pytest.raises(ValueError, match="complete native"):
        make_request(record(synthetic=True), config())


def test_streamed_micro_gradients_match_full_batch_and_release_targets():
    settings = {**DEFAULTS, "precision": "fp32", "inter_weight": .01, "confidence_weighting": True}
    models = [build_scratch_model(asdict(config()), "pair_kd", 42, dict(width=8, independent_layers=True)) for _ in range(2)]
    rows = [(10+i, record(), None) for i in range(3)]
    outcomes = []
    for model, micro in zip(models, (3, 1)):
        teacher = OnlineProtenixTeacher("cpu", adapter_factory=FakeAdapter)
        previous = []
        original = teacher.targets
        def checked(requests):
            assert all(ref() is None for ref in previous)
            values = original(requests)
            previous[:] = [weakref.ref(x) for x in values]
            return values
        teacher.targets = checked
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        scheduler = scheduler_for(optimizer, settings)
        inputs = build_micros(model, rows, micro, 10042, "pair_kd")
        metrics = optimizer_update(model, optimizer, scheduler, inputs, settings, "pair_kd", 0, teacher=teacher)
        assert all(ref() is None for ref in previous)
        assert metrics["inter_records"] == 3 and metrics["pairs"] == 315
        assert metrics["teacher_target_disk_bytes"] == 0
        assert metrics["inter_kl"] > 0
        assert all(p.grad is None for p in teacher.adapter.model.parameters())
        outcomes.append(metrics)
    for name, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[name], rtol=1e-5, atol=1e-7)
    assert outcomes[0]["objective"] == pytest.approx(outcomes[1]["objective"], abs=1e-5)


def test_independent_layers_shared_backbone_and_checkpoint(tmp_path):
    pair = dict(width=8, independent_layers=True, distogram_chunk_size=13)
    baseline = build_scratch_model(asdict(config()), "mlm", 42)
    model = build_scratch_model(asdict(config()), "pair_kd", 42, pair)
    assert shared_state_hash(model) == shared_state_hash(baseline)
    assert len({id(e.single_projection.weight) for e in model.layer_pair_encoders}) == config().n_layers
    model.save_pretrained(tmp_path)
    restored = StructureMLMModel.from_pretrained(tmp_path)
    assert restored.pair_config.independent_layers
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    # Zero exits deliberately block layer encoder gradients on the first step.
    # Nonzero exits verify that every independent layer can learn subsequently.
    for output in restored.pair_bias_outputs:
        torch.nn.init.normal_(output.weight, std=.1)
    batch, pairs, _ = build_micros(restored, [(0, record(), None)], 1, 10042, "pair_kd")[0]
    result = restored(pair_indices=pairs, **batch)
    (result["loss_sum"] + result["pair_logits"].square().sum()).backward()
    assert all(e.single_projection.weight.grad.abs().sum() > 0 for e in restored.layer_pair_encoders)


def distributed_worker(rank, rendezvous, output):
    """Actual Gloo collectives with unequal local record/micro counts."""
    from pathlib import Path
    from torch.nn.parallel import DistributedDataParallel
    from scripts.research.run_stage1_protenix_online import replay_smoke, validate
    from scripts.research.stage1_ab500k_initialization import canonical_hash
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        settings = {**DEFAULTS, 'precision': 'fp32', 'inter_weight': .01, 'confidence_weighting': True}
        spec, pair = asdict(config()), dict(width=8, independent_layers=True, distogram_chunk_size=13)
        model = build_scratch_model(spec, 'pair_kd', 42, pair)
        shared = shared_state_hash(model)
        teacher = OnlineProtenixTeacher('cpu', adapter_factory=FakeAdapter)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        scheduler = scheduler_for(optimizer, settings)
        rows = [(10+i, record(), None) for i in range(5)]
        local = rows[:2] if rank == 0 else rows[2:]
        ddp = DistributedDataParallel(model, broadcast_buffers=False)
        optimizer_update(ddp, optimizer, scheduler, build_micros(model, local, 2, 10042, 'pair_kd'),
                         settings, 'pair_kd', 0, 2, teacher=teacher)
        if rank == 0:
            reference = build_scratch_model(spec, 'pair_kd', 42, pair)
            opt = torch.optim.SGD(reference.parameters(), lr=.01)
            sched = scheduler_for(opt, settings)
            optimizer_update(reference, opt, sched, build_micros(reference, rows, 5, 10042, 'pair_kd'),
                             settings, 'pair_kd', 0, teacher=teacher)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, reference.state_dict()[name], rtol=1e-5, atol=1e-7)
        torch.distributed.barrier()
        plan = dict(model=spec, pair=pair, training=settings, teacher_contract=teacher.contract,
            initialization=dict(schema='immune.stage1.random_init.v1', mode='scratch', seed=42,
                                config_sha256=canonical_hash(spec), shared_state_sha256=shared))
        replay_smoke(plan, 'pair_kd', rows, Path(output), torch.device('cpu'), rank, 2,
                     dict(test='online_two_rank'), teacher)
        before = len(teacher.adapter.examples)
        metrics = validate(model, rows, settings, 'pair_kd', torch.device('cpu'), rank, 2, 2,
                           Path(output), 2, teacher)
        assert len(teacher.adapter.examples)-before == len(rows[rank::2])
        assert metrics['records'] == 5 and metrics['inter_kl'] > 0
        assert metrics['teacher_target_disk_bytes'] == 0
    finally:
        torch.distributed.destroy_process_group()


def test_real_two_rank_reduction_resume_and_validation(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(distributed_worker, args=('file://'+str(tmp_path/'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    import json
    result = json.loads((tmp_path/'smoke_rank0.json').read_text())
    assert result['teacher_targets_exact'] and result['corruption_exact']
    from safetensors.torch import load_file
    state = load_file(str(tmp_path/'smoke/checkpoint-1/model.safetensors'))
    assert all(not name.startswith(('teacher', 'esm')) for name in state)


def test_runner_referenced_globals_are_bound():
    """Catch unresolved entrypoint names before an expensive teacher startup."""
    import builtins
    import dis
    from types import CodeType, FunctionType
    from scripts.research import run_stage1_protenix_online as entry

    def check(code):
        for instruction in dis.get_instructions(code):
            if instruction.opname == 'LOAD_GLOBAL':
                assert instruction.argval in vars(entry) or instruction.argval in vars(builtins), (
                    code.co_name, instruction.argval)
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                check(constant)

    for value in vars(entry).values():
        if isinstance(value, FunctionType) and value.__module__ == entry.__name__:
            check(value.__code__)


def test_spawn_loader_after_teacher_adds_conflicting_scripts_package(tmp_path):
    """Reexecute the real entrypoint in four spawned workers after path mutation."""
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    teacher_root = tmp_path/'teacher'
    shadow = teacher_root/'scripts'
    shadow.mkdir(parents=True)
    (shadow/'__init__.py').write_text('"""Unrelated teacher script package."""\n')
    driver = tmp_path/'spawn_loader.py'
    driver.write_text('''"""CPU regression driver; run with the training Python."""
from pathlib import Path
import json
import runpy
import sys

root = Path(sys.argv[1])
entry = runpy.run_path(str(root/'scripts/research/run_stage1_protenix_online.py'))

if __name__ == '__main__':
    import scripts
    assert Path(scripts.__file__).resolve() == root/'scripts/__init__.py'
    # Protenix does this after the parent has already imported our scripts.
    # Spawn inherits sys.path and imports the entrypoint afresh in each worker.
    sys.path.insert(0, sys.argv[2])
    sampler = entry['CompleteEpochBatchSampler'](1040, 512, 0, 8, 42, 0)
    loader = entry['training_loader'](list(range(1040)), sampler,
                                    dict(loader_workers=4, seed=42), 0, 0)
    actual = list(loader)
    assert actual == list(sampler)
    assert [len(batch) for batch in actual] == [64, 64, 2]
    print(json.dumps(dict(status='passed', workers=4, batch_sizes=[64, 64, 2])))
''')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'USE_TF': '0',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    process = subprocess.Popen([sys.executable, '-B', str(driver), str(root), str(teacher_root)],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail('Spawned loader did not complete:\n'+output)
    assert process.returncode == 0, output
    assert json.loads(output.splitlines()[-1]) == dict(
        status='passed', workers=4, batch_sizes=[64, 64, 2])


def test_formal_run_first_update_stop_resume_and_epoch_tail(tmp_path, monkeypatch):
    """Exercise run(), real loader/updates/checkpoints; emulate only GPU orchestration.

    One actual CPU/Gloo rank represents rank zero of the eight-rank launch.
    This catches missing entrypoint imports and cursor/log/checkpoint failures;
    the separate two-rank test checks collective and gradient correctness.
    """
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from safetensors.torch import load_file
    from scripts.research import run_stage1_protenix_online as entry
    from scripts.research.stage1_ab500k_initialization import canonical_hash

    settings = {**DEFAULTS, 'max_steps': 3, 'warmup_steps': 1,
                'precision': 'fp32', 'loader_workers': 0, 'eval_every': 2,
                'save_every': 1, 'inter_weight': .01, 'confidence_weighting': True}
    spec = asdict(config())
    pair = dict(width=8, independent_layers=True, distogram_chunk_size=13)
    initial = build_scratch_model(spec, 'pair_kd', 42, pair)
    initial_state = {k: v.clone() for k, v in initial.state_dict().items()}
    teacher_contract = OnlineProtenixTeacher('cpu', adapter_factory=FakeAdapter).contract
    plan_path, index_path = tmp_path/'plan.json', tmp_path/'index.json'
    index_path.write_text(json.dumps({'records': {'train': 520}}))
    plan = dict(model=spec, pair=pair, training=settings, source_root=str(tmp_path),
                sources={}, teacher_assets={}, teacher_contract=teacher_contract,
                subset_manifest=str(index_path), parent_model_sha256=None,
                output_root=str(tmp_path/'resumed'), initialization=dict(
                    schema='immune.stage1.random_init.v1', mode='scratch', seed=42,
                    config_sha256=canonical_hash(spec), shared_state_sha256=shared_state_hash(initial)))
    plan_path.write_text(json.dumps(plan))
    rows = [(i, record(), None) for i in range(520)]
    valid = rows[:2]
    subset = dict(data_index=str(index_path), student_max_length=spec['max_length'])

    class CpuRuntime:
        cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda _: None,
            empty_cache=lambda: None, get_device_name=lambda _: 'CPU test',
            get_device_properties=lambda _: SimpleNamespace(total_memory=1))
        distributed = SimpleNamespace(init_process_group=lambda *a, **k: None,
                                      destroy_process_group=lambda: None,
                                      all_gather_object=torch.distributed.all_gather_object)

        @staticmethod
        def device(*args):
            return torch.device('cpu')

        def __getattr__(self, name):
            return getattr(torch, name)

    monkeypatch.setattr(entry, 'torch', CpuRuntime())
    monkeypatch.setenv('WORLD_SIZE', '8')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(entry, 'verify_plan', lambda *a: dict(
        status='passed', pending=[], plan=plan, subset=subset, settings=settings))
    monkeypatch.setattr(entry, 'PilotDataset', lambda *a, **k: valid if len(a) > 2 else rows)
    monkeypatch.setattr(entry, 'OnlineProtenixTeacher', lambda device, **kw:
        OnlineProtenixTeacher('cpu', adapter_factory=FakeAdapter, **kw))
    monkeypatch.setattr(entry, 'distributed_model', lambda model, device: model)

    def qualified_profile(plan, arm, dataset, out, *args):
        result = dict(micro_batch=64, seconds=0., sequence_length=spec['max_length'])
        entry.write_json(out/'profile.json', result)
        return result

    monkeypatch.setattr(entry, 'profile', qualified_profile)
    monkeypatch.setattr(entry, 'replay_smoke', lambda *a: dict(seconds=0.))
    validate, save, update, load = entry.validate, entry.save_checkpoint, entry.optimizer_update, entry.load_checkpoint
    monkeypatch.setattr(entry, 'validate', lambda model, dataset, settings, arm, device,
        rank, world, micro, output, step, teacher:
        validate(model, dataset, settings, arm, device, rank, 1, micro, output, step, teacher))
    monkeypatch.setattr(entry, 'save_checkpoint', lambda output, model, optimizer, scheduler,
        noise, state, provenance, rank, world:
        save(output, model, optimizer, scheduler, noise, state, provenance, rank, 1))
    resumed_states, update_steps = [], []

    def restored(*args):
        state = load(*args)
        resumed_states.append(dict(state))
        return state

    def one_cpu_update(model, optimizer, scheduler, micros, settings, arm, step, world, **kwargs):
        scores = update(model, optimizer, scheduler, micros, settings, arm, step, 1, **kwargs)
        for name in ('records', 'selected_tokens', 'eligible_tokens'):
            scores[name] *= world
        update_steps.append(step)
        return scores

    monkeypatch.setattr(entry, 'load_checkpoint', restored)
    monkeypatch.setattr(entry, 'optimizer_update', one_cpu_update)
    torch.distributed.init_process_group('gloo', init_method='file://'+str(tmp_path/'run_rendezvous'),
                                        rank=0, world_size=1)
    try:
        out = Path(plan['output_root'])/'pair_kd'
        out.mkdir(parents=True)
        (out/'STOP').touch()
        entry.run(plan_path, 'pair_kd', 'latest')
        stopped = json.loads((out/'STOPPED.json').read_text())
        assert (stopped['step'], stopped['epoch'], stopped['next_batch'], stopped['seen_records']) == (1, 0, 1, 512)
        first = json.loads((out/'metrics.jsonl').read_text().splitlines()[0])
        assert first['step'] == 1 and first['inter_kl'] > 0 and first['teacher_target_disk_bytes'] == 0
        learned = load_file(str(out/'checkpoint-1/model.safetensors'))
        assert any(not torch.equal(v, learned[k]) for k, v in initial_state.items())
        (out/'STOP').unlink()
        entry.run(plan_path, 'pair_kd', 'latest')
        assert resumed_states[0]['step'] == 1 and update_steps == [0, 1, 2]
        complete = json.loads((out/'COMPLETE.json').read_text())
        assert (complete['step'], complete['epoch'], complete['next_batch'], complete['seen_records']) == (3, 1, 1, 1032)
        metrics = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
        assert [m['records'] for m in metrics] == [512, 8, 512]
        assert all(m['teacher_target_disk_bytes'] == 0 for m in metrics)
        plan['output_root'] = str(tmp_path/'continuous')
        entry.run(plan_path, 'pair_kd', 'latest')
        expected = load_file(str(Path(plan['output_root'])/'pair_kd/checkpoint-3/model.safetensors'))
        actual = load_file(str(out/'checkpoint-3/model.safetensors'))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()
