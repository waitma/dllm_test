"""Focused CPU acceptance for the new generic GPU control/restoration engine.

Activate conda pllm, then run from the project directory:
USE_TF=0 CUDA_VISIBLE_DEVICES='' python -m pytest -q
scripts/tests/immune_llada/test_generic_joint_gpu_engine.py
The tests never initialize CUDA, install a runtime, enqueue or read real data.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.architecture_prototypes.generic_joint_model import ModelConfig
from scripts.architecture_prototypes.generic_joint_experiment import JointSystem, evaluate
from scripts.architecture_prototypes import generic_joint_gpu_protocol_v1 as protocol
from scripts.architecture_prototypes import generic_joint_gpu_engine_v1 as engine


@pytest.fixture(autouse=True)
def CPU_only():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(before)


def CPU_binding(method='g00_shared_joint'):
    return protocol.binding(method, micro_batch=128, package_sha256='a'*64,
                            plan_sha256='b'*64, profile_job_id='CPU_CONTROL_not_GPU_acceptance')


def CPU_system(method='g00_shared_joint', seed=42):
    torch.manual_seed(seed)
    system = JointSystem(ModelConfig(variant=method, d_model=12))
    optimizer, scheduler = engine.make_optimizer_scheduler(system)
    return system, optimizer, scheduler


def identical(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            identical(a, b)
    else:
        assert left == right


def test_independent_transport_mapping_is_honest_and_has_unique_predecessors():
    assert len(protocol.WORKLOADS) == 48
    for method in protocol.METHODS:
        profile = protocol.WORKLOADS[method+'__profile4']
        assert profile.actual_step_budget == 4
        previous = None
        for start in (0, 100, 200, 300, 400):
            work = protocol.WORKLOADS[method+f'__s{start+1:03d}_{start+100:03d}']
            assert work.start_step == start and work.stop_step == start+100
            assert work.actual_step_budget == 100 and work.previous_transport == previous
            assert work.method == method and not work.transport.startswith(('r', 'd'))
            previous = work.transport


@pytest.mark.parametrize('field,value', [('global_batch', 7), ('seed', 43),
    ('real_corpus_allowed', True), ('external_pretrained_encoder', True), ('method', 'r20_refined_control')])
def test_binding_rejects_changed_scope_and_training_identity(field, value):
    data = CPU_binding()
    data[field] = value
    with pytest.raises(ValueError):
        protocol.validate_binding(data)


def test_schedule_is_one_continuous_warmup_and_horizon():
    assert engine.schedule_factor(0) == 0
    assert engine.schedule_factor(24) == 24/25
    assert engine.schedule_factor(25) == 1
    assert engine.schedule_factor(100) > engine.schedule_factor(200) > engine.schedule_factor(400)
    assert engine.schedule_factor(500) == 0
    with pytest.raises(ValueError):
        engine.schedule_factor(501)


def transport(tmp_path, monkeypatch, method='g00_shared_joint', suffix='__s001_100', steps=100):
    pool = tmp_path/'pool'
    (pool/'runs').mkdir(parents=True)
    monkeypatch.setattr(protocol, 'POOL', pool)
    name = method+suffix
    output = pool/'runs'/('123_'+name+'_abcd1234')
    argv = ['--variant', name, '--mode', 'profile-synthetic', '--steps', str(steps),
            '--profile-dir', str(pool/'profiles'/name), '--output-dir', str(output)]
    return argv, output


def test_transport_states_actual_100_steps_and_requires_installed_package(tmp_path, monkeypatch):
    argv, output = transport(tmp_path, monkeypatch)
    args, work, actual_output, _ = protocol.validate_transport(argv)
    assert args.steps == 100 and work.actual_step_budget == 100 and actual_output == output
    with pytest.raises(ValueError, match='package not installed'):
        protocol.validate_installed_entry(argv)
    assert not output.exists()


@pytest.mark.parametrize('extra', [['--data-dir', '/absolute/corpus'],
    ['--resume-from-checkpoint', '/absolute/ckpt'], ['--command', 'anything'],
    ['--variant', 'g00_shared_joint__s001_100']])
def test_unknown_real_data_resume_command_and_duplicate_options_refused(tmp_path, monkeypatch, extra):
    argv, output = transport(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        protocol.validate_transport(argv+extra)
    assert not output.exists()


@pytest.mark.parametrize('steps', [1, 101, 500])
def test_segment_budget_is_not_silently_reinterpreted(tmp_path, monkeypatch, steps):
    argv, output = transport(tmp_path, monkeypatch, steps=steps)
    with pytest.raises(ValueError, match='actual per-job optimizer budget'):
        protocol.validate_transport(argv)
    assert not output.exists()


def test_fresh_entry_refusal_happens_without_Torch_import():
    root = Path(__file__).resolve().parents[3]
    code = ("import sys; sys.path.insert(0, "+repr(str(root))+"); "
            "from scripts.architecture_prototypes.generic_joint_gpu_entry_v1 import main; "
            "\ntry: main(['--data-dir','/absolute/corpus'])"
            "\nexcept ValueError: pass"
            "\nelse: raise AssertionError('accepted real data')"
            "\nassert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, '-I', '-c', code], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('method', ['g00_shared_joint', 'g06_self_alignment'])
def test_continuous_102_matches_100_plus_restored_two_windows_including_EMA(tmp_path, method):
    binding = CPU_binding(method)
    uninterrupted, full_optimizer, full_scheduler = CPU_system(method)
    full = engine.train_windows(uninterrupted, full_optimizer, full_scheduler, binding,
                                start_step=0, stop_step=102, device='cpu')
    full_rng = torch.random.get_rng_state().clone()
    partial, optimizer, scheduler = CPU_system(method)
    first = engine.train_windows(partial, optimizer, scheduler, binding,
                                 start_step=0, stop_step=100, device='cpu')
    state = engine.capture_state(partial, optimizer, scheduler, binding,
                                 completed_steps=100, previous_job_id='CPU_CONTROL_s001_100')
    receipt = engine.save_state(tmp_path/'owned_CPU_resume_s100.pt', state)
    resumed, next_optimizer, next_scheduler = CPU_system(method, seed=1337)
    engine.restore_state(Path(receipt['path']), receipt['sha256'], resumed, next_optimizer,
                         next_scheduler, binding, expected_completed_steps=100,
                         expected_previous_job_id='CPU_CONTROL_s001_100')
    assert torch.equal(torch.random.get_rng_state(), state['CPU_RNG'])
    last = engine.train_windows(resumed, next_optimizer, next_scheduler, binding,
                                start_step=100, stop_step=102, device='cpu')
    identical(uninterrupted.state_dict(), resumed.state_dict())
    identical(full_optimizer.state_dict(), next_optimizer.state_dict())
    identical(full_scheduler.state_dict(), next_scheduler.state_dict())
    assert torch.equal(full_rng, torch.random.get_rng_state())
    assert full == first+last
    assert last[0]['data_digest'] != first[0]['data_digest']
    assert len(first) == 100 and len(last) == 2
    assert sum(w['records'] for w in full) == 128*102
    assert not state['CUDA_RNG'] and state['execution_device_type'] == 'cpu'
    if method == 'g06_self_alignment':
        assert any(key.startswith('ema.') for key in state['system'])
        assert any(key.startswith('alignment_projection.') for key in state['system'])


def test_restore_rejects_hash_corruption_before_deserialization(tmp_path, monkeypatch):
    system, optimizer, scheduler = CPU_system()
    path = tmp_path/'corrupt.pt'
    path.write_bytes(b'not-a-state')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('must reject before deserialization'))
    with pytest.raises(ValueError, match='bytes differ'):
        engine.restore_state(path, '0'*64, system, optimizer, scheduler, CPU_binding(),
                             expected_completed_steps=100, expected_previous_job_id='previous')


@pytest.mark.parametrize('changed', ['job', 'step', 'config', 'missing_teacher'])
def test_restore_rejects_cross_job_skipped_step_config_or_incomplete_teacher(tmp_path, changed):
    method = 'g06_self_alignment'
    binding = CPU_binding(method)
    system, optimizer, scheduler = CPU_system(method)
    engine.train_windows(system, optimizer, scheduler, binding, start_step=0, stop_step=1, device='cpu')
    state = engine.capture_state(system, optimizer, scheduler, binding,
                                 completed_steps=1, previous_job_id='CPU_CONTROL_previous')
    if changed == 'job':
        state['previous_job_id'] = 'some_other_job'
    elif changed == 'step':
        state['completed_steps'] = 100
    elif changed == 'config':
        state['model_config'] = {**state['model_config'], 'd_model': 192}
    else:
        state['system'].pop(next(k for k in state['system'] if k.startswith('ema.')))
    path = tmp_path/'owned_invalid_state.pt'
    receipt = engine.save_state(path, state)
    restored, new_optimizer, new_scheduler = CPU_system(method)
    with pytest.raises(ValueError):
        engine.restore_state(path, receipt['sha256'], restored, new_optimizer, new_scheduler, binding,
                             expected_completed_steps=1, expected_previous_job_id='CPU_CONTROL_previous')


def test_GPU_over_100_updates_refused_before_CUDA_initialization():
    system, optimizer, scheduler = CPU_system()
    with pytest.raises(ValueError, match='100 actual updates'):
        engine.train_windows(system, optimizer, scheduler, CPU_binding(), start_step=0,
                             stop_step=101, device='cuda')
    assert not torch.cuda.is_initialized()


def test_device_reporting_proxy_preserves_actual_packets_RNG_and_training_mode():
    system, _, _ = CPU_system()
    reference = evaluate(system, records=8)
    system.train()
    rng = torch.random.get_rng_state().clone()
    result = engine.evaluate_device(system, records=8)
    for key in ('relation_packet', 'mask_packets', 'generation_packets', 'groups',
                'segment_permutation_logit_difference', 'representation_variance'):
        assert result[key] == reference[key]
    assert system.training and torch.equal(rng, torch.random.get_rng_state())
    assert result['model_execution_device'] == 'cpu'
    assert result['model_autocast'] == 'disabled_CPU_control'


def test_strict_JSON_preserves_failure_on_duplicates_and_nonfinite(tmp_path):
    path = tmp_path/'bad.json'
    for content in ('{"key":1,"key":2}', '{"value":NaN}'):
        path.write_text(content)
        with pytest.raises(ValueError):
            protocol.read_json(path)


@pytest.mark.parametrize('method', ['g00_shared_joint', 'g06_self_alignment'])
def test_complete_restore_validates_Adam_and_scheduler_without_changing_payload(tmp_path, method):
    binding = CPU_binding(method)
    system, optimizer, scheduler = CPU_system(method)
    engine.train_windows(system, optimizer, scheduler, binding, start_step=0, stop_step=2, device='cpu')
    state = engine.capture_state(system, optimizer, scheduler, binding,
                                 completed_steps=2, previous_job_id='CPU_CONTROL_valid_previous')
    receipt = engine.save_state(tmp_path/'complete_CPU_state.pt', state)
    restored, new_optimizer, new_scheduler = CPU_system(method, seed=902)
    engine.restore_state(Path(receipt['path']), receipt['sha256'], restored, new_optimizer,
                         new_scheduler, binding, expected_completed_steps=2,
                         expected_previous_job_id='CPU_CONTROL_valid_previous')
    identical(restored.state_dict(), state['system'])
    identical(new_optimizer.state_dict(), state['optimizer'])
    identical(new_scheduler.state_dict(), state['scheduler'])


@pytest.mark.parametrize('changed', ['missing_state', 'missing_moment', 'wrong_step',
    'wrong_shape', 'wrong_dtype', 'scheduler_count', 'scheduler_base', 'optimizer_hyperparameter'])
def test_restore_refuses_incomplete_or_changed_Adam_scheduler_state(tmp_path, changed):
    binding = CPU_binding()
    system, optimizer, scheduler = CPU_system()
    engine.train_windows(system, optimizer, scheduler, binding, start_step=0, stop_step=1, device='cpu')
    state = engine.capture_state(system, optimizer, scheduler, binding,
                                 completed_steps=1, previous_job_id='CPU_CONTROL_previous')
    first = next(iter(state['optimizer']['state']))
    if changed == 'missing_state':
        state['optimizer']['state'] = {}
    elif changed == 'missing_moment':
        state['optimizer']['state'][first].pop('exp_avg')
    elif changed == 'wrong_step':
        state['optimizer']['state'][first]['step'] = torch.tensor(0.)
    elif changed == 'wrong_shape':
        state['optimizer']['state'][first]['exp_avg'] = state['optimizer']['state'][first]['exp_avg'].flatten()
    elif changed == 'wrong_dtype':
        state['optimizer']['state'][first]['exp_avg'] = state['optimizer']['state'][first]['exp_avg'].double()
    elif changed == 'scheduler_count':
        state['scheduler']['_step_count'] = 1
    elif changed == 'scheduler_base':
        state['scheduler']['base_lrs'] = [1.]
    else:
        state['optimizer']['param_groups'][0]['weight_decay'] = 0
    receipt = engine.save_state(tmp_path/'invalid_Adam_CPU_state.pt', state)
    restored, new_optimizer, new_scheduler = CPU_system()
    with pytest.raises(ValueError, match='Adam|scheduler|Optimizer'):
        engine.restore_state(Path(receipt['path']), receipt['sha256'], restored, new_optimizer,
                             new_scheduler, binding, expected_completed_steps=1,
                             expected_previous_job_id='CPU_CONTROL_previous')
    assert not new_optimizer.state and new_scheduler.last_epoch == 0


def test_completion_distinguishes_execution_from_independent_acceptance():
    work = protocol.WORKLOADS['g00_shared_joint__s001_100']
    result = {'transport': work.transport, 'method': work.method, 'cuda_measured': True,
              'actual_optimizer_steps': 100, 'cumulative_start': 0, 'cumulative_stop': 100,
              'real_corpus_training_started': False, 'biological_quality_claim': False}
    receipt = protocol.execution_completion(work, result)
    assert receipt['status'] == 'execution_completed_independent_review_pending'
    assert not receipt['independent_root_verified'] and not receipt['data_acceptance']
    assert not receipt['five_segments_independently_verified']
    for changed in ({**result, 'actual_optimizer_steps': 99}, {**result, 'cuda_measured': False},
                    {**result, 'cumulative_stop': 500}, {**result, 'real_corpus_training_started': True}):
        with pytest.raises(ValueError):
            protocol.execution_completion(work, changed)


def test_evidence_rejects_file_and_parent_symlink_before_any_content_read(tmp_path):
    folder = tmp_path/'owned'
    folder.mkdir()
    path = folder/'record.json'
    path.write_text('{"CPU_fixture":true}')
    assert protocol.evidence_file(path) == path
    alias = tmp_path/'alias.json'
    alias.symlink_to(path)
    with pytest.raises(ValueError, match='canonical and nonsymlink'):
        protocol.evidence_file(alias)
    alias_parent = tmp_path/'alias_parent'
    alias_parent.symlink_to(folder, target_is_directory=True)
    with pytest.raises(ValueError, match='canonical and nonsymlink'):
        protocol.evidence_file(alias_parent/'record.json')


@pytest.mark.parametrize('failure', ['before_update', 'after_update', 'evaluation', 'none'])
def test_profile_failure_keeps_completed_updates_and_names_failure_phase(tmp_path, monkeypatch, failure):
    """Fault injection only: no model, optimizer update, CUDA or real data executes."""
    method = 'g00_shared_joint'
    work = protocol.WORKLOADS[method+'__profile4']
    config = ModelConfig(variant=method, d_model=12)
    fake_system = SimpleNamespace(to=lambda device: fake_system)
    optimizer = SimpleNamespace(param_groups=[{'lr': 0.}])
    current = {'micro': None}
    monkeypatch.setattr(engine, 'checked_plan', lambda name: ({}, config))
    monkeypatch.setattr(engine, 'JointSystem', lambda config: fake_system)
    monkeypatch.setattr(engine, 'make_optimizer_scheduler', lambda system: (optimizer, None))
    monkeypatch.setattr(engine, 'sha', lambda path: 'b'*64)
    for name in ('set_device', 'reset_peak_memory_stats', 'synchronize', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(total_memory=1000))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 200)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: 300)

    def simulated_training(system, optimizer, scheduler, binding, **kwargs):
        current['micro'] = binding['micro_batch']
        failing_trial = current['micro'] != 16 and failure != 'none'
        if failing_trial and failure == 'before_update':
            raise torch.OutOfMemoryError('CPU fault injection before completed update')
        kwargs['on_optimizer_step'](1)
        if failing_trial and failure == 'after_update':
            raise torch.OutOfMemoryError('CPU fault injection after completed update')
        return [{'microsteps': 128//current['micro'], 'records': 128}]

    def simulated_evaluation(system):
        if current['micro'] != 16 and failure == 'evaluation':
            raise torch.OutOfMemoryError('CPU fault injection during evaluation')
        return {'CPU_fault_injection_only': True}

    monkeypatch.setattr(engine, 'train_windows', simulated_training)
    monkeypatch.setattr(engine, 'evaluate_device', simulated_evaluation)
    result = engine.run_profile(work, tmp_path/'owned_profile', {}, {'package_sha256': 'a'*64})
    assert result['actual_optimizer_steps'] == (1 if failure == 'before_update' else 4)
    assert result['recommended_micro_batch'] == (128 if failure == 'none' else 16)
    assert len(result['measurements']) == 4
    assert result['measurements'][0]['actual_optimizer_steps'] == 1
    for trial in result['measurements'][1:]:
        if failure == 'none':
            assert trial['status'] == 'passed' and trial['actual_optimizer_steps'] == 1
        else:
            assert trial['status'] == 'expected_profile_OOM'
            assert trial['actual_optimizer_steps'] == (0 if failure == 'before_update' else 1)
            assert trial['failure_phase'] == ('evaluation' if failure == 'evaluation' else 'training')
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('scheduler_fails', [False, True])
def test_completed_update_callback_survives_later_scheduler_failure(monkeypatch, scheduler_fails):
    """One genuine tiny CPU update proves the counter precedes later failures."""
    system, optimizer, scheduler = CPU_system()
    completed = []
    if scheduler_fails:
        def fail_after_optimizer():
            raise torch.OutOfMemoryError('CPU fault injection after the optimizer returned')
        monkeypatch.setattr(scheduler, 'step', fail_after_optimizer)
        with pytest.raises(torch.OutOfMemoryError, match='optimizer returned'):
            engine.train_windows(system, optimizer, scheduler, CPU_binding(), start_step=0,
                stop_step=1, device='cpu', on_optimizer_step=completed.append)
        assert scheduler.last_epoch == 0
    else:
        windows = engine.train_windows(system, optimizer, scheduler, CPU_binding(), start_step=0,
            stop_step=1, device='cpu', on_optimizer_step=completed.append)
        assert len(windows) == scheduler.last_epoch == 1
    assert completed == [1]
    assert all(float(state['step']) == 1 for state in optimizer.state.values())
