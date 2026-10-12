"""Artificial integer training engine with exact continuous-state restoration.

Activate conda pllm; CPU acceptance:
python -m pytest scripts/tests/immune_llada/test_generic_joint_gpu_engine.py -q
GPU execution is only through the separately checked fixed package entry.
This module has no biological corpus reader, pretrained weights or queue API.
"""
from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch

from .generic_joint_data import make_examples, corrupt, move_batch, Inputs
from .generic_joint_experiment import JointSystem, checked_plan, evaluate
from .generic_joint_gpu_protocol_v1 import (POOL, PACKAGE, PROFILE_MICROBATCHES,
    TOTAL_STEPS, WARMUP_STEPS, SCHEDULE, require, sha, read_json, validate_binding)
from .generic_joint_gpu_protocol_v1 import evidence_file


def schedule_factor(completed_steps):
    require(type(completed_steps) is int and 0 <= completed_steps <= TOTAL_STEPS,
            'Schedule is bound to the cumulative 500-step horizon')
    if completed_steps < WARMUP_STEPS:
        return completed_steps/WARMUP_STEPS
    phase = (completed_steps-WARMUP_STEPS)/(TOTAL_STEPS-WARMUP_STEPS)
    return (1.0+math.cos(math.pi*phase))/2.0


def make_optimizer_scheduler(system):
    optimizer = torch.optim.AdamW((p for p in system.parameters() if p.requires_grad),
                                 lr=SCHEDULE['learning_rate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_factor)
    return optimizer, scheduler


def artificial_window(completed_steps, training_binding):
    validate_binding(training_binding)
    require(type(completed_steps) is int and 0 <= completed_steps < TOTAL_STEPS,
            'Artificial data stream is indexed by cumulative optimizer steps')
    seed = training_binding['seed']
    examples = make_examples(training_binding['global_batch'],
                             seed=seed+17*completed_steps, split='train')
    batch = corrupt(examples, seed=1000+seed+19*completed_steps)
    digest = hashlib.sha256()
    for value in (batch.generation.tokens, batch.relation_labels, batch.selected):
        digest.update(value.contiguous().numpy().tobytes())
    return batch, digest.hexdigest()


def train_windows(system, optimizer, scheduler, training_binding, *, start_step,
                  stop_step, device, on_window=None, on_optimizer_step=None):
    """CPU control experiments may span the horizon; each GPU job is <=100 steps."""
    validate_binding(training_binding)
    require(type(start_step) is int and type(stop_step) is int
            and 0 <= start_step < stop_step <= TOTAL_STEPS, 'Invalid cumulative range')
    device = torch.device(device)
    require(device.type in ('cpu', 'cuda'), 'Only CPU control or assigned GPU')
    require(device.type != 'cuda' or stop_step-start_step <= 100, 'GPU job exceeds 100 actual updates')
    require(scheduler.last_epoch == start_step, 'Scheduler is not at the declared continuation point')
    windows = []
    micro_batch = training_binding['micro_batch']
    system.train()
    for completed in range(start_step, stop_step):
        batch, data_digest = artificial_window(completed, training_binding)
        optimizer.zero_grad(set_to_none=True)
        totals = {'relation_contribution': 0.0, 'generation_contribution': 0.0,
                  'alignment_contribution': 0.0, 'teacher_forwards': 0,
                  'selected_tokens': 0, 'records': 0}
        microsteps = 0
        for begin in range(0, batch.records, micro_batch):
            current = move_batch(batch.rows(begin, min(begin+micro_batch, batch.records)), device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda'):
                loss, statistics = system.objective(current, window_records=batch.records)
            require(bool(torch.isfinite(loss)), 'Nonfinite artificial training loss')
            loss.backward()
            for name in totals:
                totals[name] += statistics[name]
            microsteps += 1
        norm = torch.nn.utils.clip_grad_norm_(system.parameters(), SCHEDULE['gradient_clip_norm'],
                                              error_if_nonfinite=True)
        require(float(norm) > 0, 'Artificial joint objective produced no gradient')
        learning_rate = optimizer.param_groups[0]['lr']
        optimizer.step()
        if on_optimizer_step is not None:
            on_optimizer_step(completed+1)
        scheduler.step()
        system.update_ema()
        record = {'cumulative_step': completed+1, 'optimizer_steps': 1,
                  'microsteps': microsteps, 'records': batch.records,
                  'learning_rate': learning_rate, 'preclip_gradient_norm': float(norm),
                  'gradient_finite': True, 'data_digest': data_digest, **totals}
        windows.append(record)
        if on_window is not None:
            on_window(record)
            system.train()
    return windows


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def capture_state(system, optimizer, scheduler, training_binding, *, completed_steps, previous_job_id):
    validate_binding(training_binding)
    require(type(completed_steps) is int and scheduler.last_epoch == completed_steps
            and 0 < completed_steps <= TOTAL_STEPS, 'State and cumulative scheduler disagree')
    require(type(previous_job_id) is str and bool(previous_job_id), 'State requires its producing job')
    device = next(system.parameters()).device
    return {'schema': 'generic.joint_internal_continuation_state.v1',
            'binding': dict(training_binding), 'completed_steps': completed_steps,
            'previous_job_id': previous_job_id, 'model_config': system.model.config.__dict__,
            'system': cpu_tree(system.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
            'scheduler': cpu_tree(scheduler.state_dict()),
            'CPU_RNG': torch.random.get_rng_state().clone(),
            'CUDA_RNG': torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None,
            'execution_device_type': device.type, 'scaler': None,
            'data_stream': 'pure_cumulative_step_seed_no_mutable_sampler'}


def save_state(path, payload):
    require(path.is_absolute() and path == path.resolve() and not path.exists(), 'Fresh owned state path required')
    require(path.parent.is_dir() and not path.parent.is_symlink(), 'State parent must be canonical')
    temporary = path.with_name(path.name+'.writing')
    with temporary.open('xb') as stream:
        torch.save(payload, stream)
    temporary.replace(path)
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'completed_steps': payload['completed_steps'], 'previous_job_id': payload['previous_job_id']}


def validate_optimizer_scheduler(payload, optimizer, scheduler, completed_steps):
    """Reject missing Adam moments, changed hyperparameters or a reset schedule."""
    saved = payload['optimizer']
    current = optimizer.state_dict()
    require(type(saved) is dict and set(saved) == set(current)
            and len(saved['param_groups']) == len(current['param_groups']), 'Optimizer schema differs')
    parameter_ids = [item for group in current['param_groups'] for item in group['params']]
    parameters = [item for group in optimizer.param_groups for item in group['params']]
    require(type(saved['state']) is dict and set(saved['state']) == set(parameter_ids),
            'Adam state coverage is incomplete')
    learning_rate = SCHEDULE['learning_rate']*schedule_factor(completed_steps)
    for previous, fresh in zip(saved['param_groups'], current['param_groups']):
        expected_group = {**fresh, 'lr': learning_rate}
        require(previous == expected_group, 'Optimizer parameter mapping or hyperparameters differ')
    for identity, parameter in zip(parameter_ids, parameters):
        state = saved['state'][identity]
        require(type(state) is dict and set(state) == {'step', 'exp_avg', 'exp_avg_sq'},
                'Adam moment fields are incomplete')
        step = state['step']
        require(isinstance(step, torch.Tensor) and step.ndim == 0
                and step.dtype == torch.float32 and float(step) == completed_steps,
                'Adam cumulative step differs')
        for name in ('exp_avg', 'exp_avg_sq'):
            moment = state[name]
            require(isinstance(moment, torch.Tensor) and moment.shape == parameter.shape
                    and moment.dtype == parameter.dtype and bool(torch.isfinite(moment).all()),
                    'Adam moment shape, dtype or finiteness differs')
    expected_scheduler = {**scheduler.state_dict(), 'last_epoch': completed_steps,
                          '_step_count': completed_steps+1, '_last_lr': [learning_rate]*len(current['param_groups'])}
    require(payload['scheduler'] == expected_scheduler, 'Complete cumulative scheduler fields differ')


def restore_state(path, expected_sha256, system, optimizer, scheduler, training_binding, *,
                  expected_completed_steps, expected_previous_job_id):
    """Hash and full identity are checked before loading this owned Torch state."""
    validate_binding(training_binding)
    require(path.is_absolute() and path == path.resolve() and path.is_file()
            and not path.is_symlink() and sha(path) == expected_sha256, 'Continuation bytes differ')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    fields = {'schema', 'binding', 'completed_steps', 'previous_job_id', 'model_config',
              'system', 'optimizer', 'scheduler', 'CPU_RNG', 'CUDA_RNG',
              'execution_device_type', 'scaler', 'data_stream'}
    require(type(payload) is dict and set(payload) == fields
            and payload['schema'] == 'generic.joint_internal_continuation_state.v1'
            and payload['binding'] == training_binding
            and payload['completed_steps'] == expected_completed_steps
            and payload['previous_job_id'] == expected_previous_job_id
            and payload['model_config'] == system.model.config.__dict__
            and payload['scaler'] is None
            and payload['data_stream'] == 'pure_cumulative_step_seed_no_mutable_sampler',
            'Continuation identity, complete schema or schedule differs')
    expected = system.state_dict()
    require(set(payload['system']) == set(expected)
            and all(value.shape == expected[key].shape for key, value in payload['system'].items()),
            'Student, EMA or alignment state is incomplete')
    require(payload['scheduler']['last_epoch'] == expected_completed_steps,
            'Saved scheduler has another cumulative step')
    validate_optimizer_scheduler(payload, optimizer, scheduler, expected_completed_steps)
    device = next(system.parameters()).device
    require(payload['execution_device_type'] == device.type
            and (payload['CUDA_RNG'] is not None) == (device.type == 'cuda'), 'RNG device coverage differs')
    system.load_state_dict(payload['system'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    require(all(group['lr'] == SCHEDULE['learning_rate']*schedule_factor(expected_completed_steps)
                for group in optimizer.param_groups), 'Optimizer learning rate differs from cumulative schedule')
    # Restore after all object construction/loading, including teacher and optimizer.
    torch.random.set_rng_state(payload['CPU_RNG'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(payload['CUDA_RNG'], device)
    return payload


class ReportModel:
    """Run the existing evaluator on the model device; return actual outputs to CPU."""
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, task):
        require(isinstance(inputs, Inputs), 'Evaluation accepts only the four input fields')
        device = next(self.model.parameters()).device
        view = Inputs(*(value.to(device) for value in
                        (inputs.tokens, inputs.roles, inputs.positions, inputs.valid)))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == 'cuda'):
            result = self.model(view, task)
        return {name: result[name].detach().float().cpu()
                for name in ('token_logits', 'relation_logits', 'features')}


class ReportSystem:
    def __init__(self, system):
        self.system = system
        self.model = ReportModel(system.model)

    def eval(self):
        self.system.eval()
        return self


def evaluate_device(system, *, records=96):
    CPU_RNG = torch.random.get_rng_state().clone()
    device = next(system.parameters()).device
    CUDA_RNG = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    was_training = system.training
    try:
        result = evaluate(ReportSystem(system), records=records)
        result['scope'] = 'artificial_integer_fixed_validation_not_biological_quality'
        result['model_execution_device'] = device.type
        result['model_autocast'] = 'bfloat16' if device.type == 'cuda' else 'disabled_CPU_control'
        result['reporting_CE'] = 'FP64_from_actual_logits_cast_to_FP32'
        return result
    finally:
        system.train(was_training)
        torch.random.set_rng_state(CPU_RNG)
        if device.type == 'cuda':
            torch.cuda.set_rng_state(CUDA_RNG, device)


def write_json(path, value):
    temporary = path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verified_profile(training_binding):
    """Require the specific independently reviewed artificial profile, not a flag."""
    validate_binding(training_binding)
    job_id = training_binding['profile_job_id']
    require(re.fullmatch(r'[0-9]+_'+re.escape(training_binding['method']+'__profile4')+r'_[a-f0-9]{8}', job_id),
            'Invalid profile job identity')
    directory = POOL/'runs'/job_id
    proof = directory/'GENERIC_PROFILE_RESULT.json'
    review = read_json(evidence_file(directory/'ROOT_REVIEW.json'))
    result = read_json(evidence_file(proof))
    require(review.get('status') == 'synthetic_profile_verified'
            and review.get('result_sha256') == sha(proof)
            and result.get('job_id') == job_id
            and result.get('method') == training_binding['method']
            and result.get('package_sha256') == training_binding['package_sha256']
            and result.get('plan_sha256') == training_binding['plan_sha256']
            and result.get('recommended_micro_batch') == training_binding['micro_batch']
            and result.get('cuda_measured') is True, 'Actual fixed profile has not been independently verified')
    return result


def previous_segment(work, training_binding, admission, series):
    if work.start_step == 0:
        require(admission.get('previous') is None, 'First segment cannot restore another series')
        return None
    previous = admission.get('previous')
    require(type(previous) is dict and set(previous) == {'job_id', 'result_sha256', 'review_sha256', 'state_sha256'},
            'Missing explicit prior segment evidence')
    job_id = previous['job_id']
    require(re.fullmatch(r'[0-9]+_'+re.escape(work.previous_transport)+r'_[a-f0-9]{8}', job_id),
            'Previous job is not the unique fixed predecessor')
    directory = POOL/'runs'/job_id
    result_path, review_path = directory/'GENERIC_SEGMENT_RESULT.json', directory/'ROOT_REVIEW.json'
    evidence_file(result_path)
    evidence_file(review_path)
    require(sha(result_path) == previous['result_sha256'] and sha(review_path) == previous['review_sha256'],
            'Prior reviewed result bytes differ')
    result, review = read_json(result_path), read_json(review_path)
    state_path = series/f'resume_s{work.start_step:03d}.pt'
    evidence_file(state_path)
    require(review.get('status') == 'synthetic_segment_verified'
            and review.get('result_sha256') == previous['result_sha256']
            and result.get('status') == 'segment_completed_pending_independent_review'
            and result.get('binding') == training_binding
            and result.get('job_id') == job_id
            and result.get('transport') == work.previous_transport
            and result.get('cumulative_start') == work.start_step-100
            and result.get('cumulative_stop') == work.start_step
            and result.get('actual_optimizer_steps') == 100
            and result.get('cuda_measured') is True
            and result.get('internal_state', {}).get('path') == str(state_path)
            and result['internal_state']['sha256'] == previous['state_sha256']
            and sha(state_path) == previous['state_sha256'], 'Previous segment or restoration state is unverified')
    return state_path, previous['state_sha256'], job_id


def run_segment(work, output, manifest, admission):
    training_binding = validate_binding(admission['binding'])
    plan, config = checked_plan(work.method)
    require(training_binding['method'] == work.method
            and training_binding['package_sha256'] == admission['package_sha256']
            and training_binding['plan_sha256'] == sha(PACKAGE/'train_jobs/immune_cv_joint_exploration_20261001'/(work.method+'.yml')),
            'Installed method/config binding differs')
    require(plan['training_plan']['steps'] == TOTAL_STEPS
            and plan['training_plan']['global_batch'] == 128
            and plan['training_plan']['warmup_steps'] == WARMUP_STEPS, 'Planned continuous schedule differs')
    verified_profile(training_binding)
    series = POOL/'generic_series'/training_binding['package_sha256'][:16]/work.method/'seed42'
    require(series == series.resolve(), 'Series state path is noncanonical')
    prior = previous_segment(work, training_binding, admission, series)
    series.mkdir(parents=True, exist_ok=True)
    series_identity = series/'SERIES.json'
    if series_identity.exists():
        require(read_json(series_identity) == training_binding, 'Series belongs to another binding')
    else:
        require(work.start_step == 0, 'Cannot invent a missing predecessor series')
        write_json(series_identity, training_binding)
    with (series/f'segment_s{work.stop_step:03d}.claimed').open('x') as claim:
        claim.write(output.name+'\n')
    output.mkdir()
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    require(torch.cuda.is_bf16_supported(), 'Assigned GPU must support BF16')
    torch.manual_seed(training_binding['seed'])
    system = JointSystem(config).to(device)
    optimizer, scheduler = make_optimizer_scheduler(system)
    if prior is not None:
        restore_state(prior[0], prior[1], system, optimizer, scheduler, training_binding,
                      expected_completed_steps=work.start_step, expected_previous_job_id=prior[2])
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    evaluations = {}
    if work.start_step == 0:
        evaluations['0'] = evaluate_device(system)
    def observe(window):
        with (output/'optimizer_windows.jsonl').open('a') as stream:
            stream.write(json.dumps(window, allow_nan=False)+'\n')
        if window['cumulative_step'] % 50 == 0:
            key = str(window['cumulative_step'])
            evaluations[key] = evaluate_device(system)
            write_json(output/('evaluation_s'+key+'.json'), evaluations[key])
        write_json(output/'progress.json', {'transport': work.transport, 'method': work.method,
            'cumulative_step': window['cumulative_step'], 'actual_optimizer_steps': window['cumulative_step']-work.start_step,
            'actual_microsteps': (window['cumulative_step']-work.start_step)*window['microsteps'],
            'actual_records': (window['cumulative_step']-work.start_step)*window['records'],
            'status': 'running_artificial_segment', 'real_corpus_training_started': False})
    windows = train_windows(system, optimizer, scheduler, training_binding,
                            start_step=work.start_step, stop_step=work.stop_step,
                            device=device, on_window=observe)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic()-started
    state = capture_state(system, optimizer, scheduler, training_binding,
                          completed_steps=work.stop_step, previous_job_id=output.name)
    state_receipt = save_state(series/f'resume_s{work.stop_step:03d}.pt', state)
    result = {'schema': 'generic.joint_GPU_segment_result.v1',
        'status': 'segment_completed_pending_independent_review', 'job_id': output.name,
        'transport': work.transport, 'method': work.method, 'binding': training_binding,
        'cumulative_start': work.start_step, 'cumulative_stop': work.stop_step,
        'actual_optimizer_steps': len(windows), 'actual_microsteps': sum(w['microsteps'] for w in windows),
        'actual_records': sum(w['records'] for w in windows), 'windows': windows,
        'evaluations': evaluations, 'internal_state': state_receipt, 'cuda_measured': True,
        'GPU_wall_seconds_including_evaluation': elapsed,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'total_memory_bytes': torch.cuda.get_device_properties(device).total_memory,
        'model': config.__dict__, 'dtype': 'FP32_parameters_AdamW_BF16_autocast_no_scaler',
        'five_segments_independently_verified': False, 'biological_quality_claim': False,
        'real_corpus_training_started': False, 'regular_checkpoint_save': False,
        'temporary_internal_state_retained_for_review_and_continuation': True}
    write_json(output/'GENERIC_SEGMENT_RESULT.json', result)
    return result


def run_profile(work, output, manifest, admission):
    plan, config = checked_plan(work.method)
    require(admission.get('binding') is None and admission.get('previous') is None, 'Profile cannot restore state')
    output.mkdir()
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    require(torch.cuda.is_bf16_supported(), 'Assigned GPU must support BF16')
    from .generic_joint_gpu_protocol_v1 import binding
    measurements = []
    for micro in PROFILE_MICROBATCHES:
        system = optimizer = scheduler = None
        actual_optimizer_steps = 0
        failure_phase = 'model_and_optimizer_setup'

        def count_completed_update(cumulative_step):
            nonlocal actual_optimizer_steps
            actual_optimizer_steps += 1

        try:
            torch.manual_seed(42)
            system = JointSystem(config).to(device)
            optimizer, scheduler = make_optimizer_scheduler(system)
            # Profile one genuine update at LR3e-4; outside the continuous series.
            optimizer.param_groups[0]['lr'] = SCHEDULE['learning_rate']
            training_binding = binding(work.method, micro_batch=micro,
                package_sha256=admission['package_sha256'],
                plan_sha256=sha(PACKAGE/'train_jobs/immune_cv_joint_exploration_20261001'/(work.method+'.yml')),
                profile_job_id=output.name)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.monotonic()
            failure_phase = 'training'
            windows = train_windows(system, optimizer, scheduler, training_binding,
                                    start_step=0, stop_step=1, device=device,
                                    on_optimizer_step=count_completed_update)
            require(actual_optimizer_steps == len(windows) == 1, 'Profile update accounting differs')
            failure_phase = 'evaluation'
            evaluation = evaluate_device(system)
            failure_phase = 'post_evaluation_synchronize'
            torch.cuda.synchronize(device)
            measurements.append({'micro_batch': micro, 'status': 'passed', 'actual_optimizer_steps': actual_optimizer_steps,
                'actual_microsteps': windows[0]['microsteps'], 'actual_records': windows[0]['records'],
                'window': windows[0], 'evaluation': evaluation,
                'elapsed_seconds_including_evaluation': time.monotonic()-started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
        except torch.OutOfMemoryError as error:
            measurements.append({'micro_batch': micro, 'status': 'expected_profile_OOM',
                'actual_optimizer_steps': actual_optimizer_steps,
                'failure_phase': failure_phase, 'error': str(error)})
        finally:
            system = optimizer = scheduler = None
            gc.collect()
            torch.cuda.empty_cache()
        write_json(output/'profile_progress.json', {'method': work.method, 'measurements': measurements,
                                                     'status': 'profiling', 'real_corpus_training_started': False})
    total = torch.cuda.get_device_properties(device).total_memory
    eligible = [m for m in measurements if m['status'] == 'passed' and m['peak_reserved_bytes'] <= 0.85*total]
    require(eligible, 'No complete artificial profile within the memory envelope')
    result = {'schema': 'generic.joint_GPU_profile_result.v1', 'job_id': output.name,
        'method': work.method, 'transport': work.transport,
        'package_sha256': admission['package_sha256'],
        'plan_sha256': training_binding['plan_sha256'], 'model': config.__dict__,
        'measurements': measurements, 'actual_optimizer_steps': sum(m['actual_optimizer_steps'] for m in measurements),
        'profile_memory_includes_fixed_96_record_length24_evaluation': True,
        'recommended_micro_batch': max(m['micro_batch'] for m in eligible),
        'total_memory_bytes': total, 'cuda_measured': True,
        'regular_checkpoint_save': False, 'temporary_internal_state_written': False,
        'real_corpus_training_started': False, 'biological_quality_claim': False,
        'status': 'profile_completed_pending_independent_review'}
    write_json(output/'GENERIC_PROFILE_RESULT.json', result)
    return result
