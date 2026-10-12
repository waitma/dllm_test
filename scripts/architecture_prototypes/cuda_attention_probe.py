"""Isolated random-tensor CUDA SDPA probe; never imports models or datasets.

Activate pllm. CPU planning only:
python scripts/architecture_prototypes/cuda_attention_probe.py --dry-run --output-dir /tmp/attention-plan
After the current GPU work ends, omit --dry-run to execute the explicit plan.
CPU tests: python -m pytest -q scripts/tests/immune_llada/test_cuda_attention_probe_cpu.py
See CUDA_ATTENTION_PROBE_README.md beside this file. No live release integration.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import warnings

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.architecture_prototypes import attention_backends as oracle


SCHEMA = 'isolated_random_cuda_attention.v1'
BACKENDS = {'math': SDPBackend.MATH, 'efficient': SDPBackend.EFFICIENT_ATTENTION}
PRECISIONS = {'float32': torch.float32, 'bfloat16': torch.bfloat16}
STATUSES = {'passed', 'unsupported', 'oom', 'unavailable', 'timeout', 'failed',
            'numerical_mismatch', 'backend_mismatch', 'not_run_budget', 'dry_run'}
TOLERANCES = {'float32': {'atol': 2e-4, 'rtol': 3e-3},
              'bfloat16': {'atol': 3e-3, 'rtol': 8e-2}}


class BackendMismatch(RuntimeError):
    pass


class NumericalMismatch(RuntimeError):
    pass


class ProbeInterrupted(BaseException):
    """A parent termination request; cleanup runs before CLI exit 128+signal."""
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f'Probe interrupted by signal {signum}')


class CaseCleanupError(RuntimeError):
    pass


@contextmanager
def case_supervision():
    """Linux/main-thread only; import never changes signal or subreaper state."""
    if sys.platform != 'linux' or threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Case supervision requires Linux and the main thread')
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:  # PR_GET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot read child-subreaper state')
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot enable child subreaper')
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(number, frame):
        raise ProbeInterrupted(number)
    try:
        for number in handlers:
            signal.signal(number, interrupted)
        yield
    finally:
        # Restore even after failure/interruption; never reap unrelated children.
        for number, handler in handlers.items():
            signal.signal(number, handler)
        if libc.prctl(36, previous.value, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot restore child-subreaper state')


def wait_case_leader(pid, deadline):
    while True:
        found, status = os.waitpid(pid, os.WNOHANG)
        if found:
            return os.waitstatus_to_exitcode(status)
        if time.monotonic() >= deadline:
            return None
        time.sleep(min(.01, max(0., deadline-time.monotonic())))


def terminate_case_group(pgid, deadline):
    """TERM, then KILL if needed; reap only children in this case's process group."""
    def signal_group(number):
        try:
            os.killpg(pgid, number)
            return True
        except ProcessLookupError:
            return False
    def reaped_and_gone():
        while True:
            try:
                found, _ = os.waitpid(-pgid, os.WNOHANG)
            except ChildProcessError:
                break
            if not found:
                break
        return not signal_group(0)
    signal_group(signal.SIGTERM)
    gentle_until = min(time.monotonic()+.3, deadline)
    while time.monotonic() < gentle_until:
        if reaped_and_gone():
            return
        time.sleep(.01)
    signal_group(signal.SIGKILL)
    while time.monotonic() < deadline:
        if reaped_and_gone():
            return
        time.sleep(.01)
    if not reaped_and_gone():
        raise CaseCleanupError(f'Case process group {pgid} could not be terminated/reaped within budget')


def run_case_process(command, *, stdout, timeout):
    """Run a fresh process in the same session, with bounded descendant cleanup.

    posix_spawn(setpgroup=0) works on Python 3.10/3.12 without preexec_fn in a
    process that has imported Torch. Cleanup reserves at most 2 seconds inside
    the supplied timeout, below the pool's 10-second TERM-to-KILL grace.
    SIGKILL of this supervisor cannot be intercepted and is not guaranteed.
    Trusted case descendants must remain in their inherited process group.
    """
    if not command or not all(isinstance(value, str) for value in command):
        raise ValueError('A case requires an explicit executable argument list')
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('A case requires a finite positive timeout')
    started = time.monotonic()
    cleanup_budget = min(2., timeout/2)
    deadline = started+timeout
    pid = None
    with case_supervision():
        try:
            # Block only around spawn/assignment so a signal cannot lose the PID.
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
            try:
                pid = os.posix_spawn(command[0], command, dict(os.environ), setpgroup=0,
                    setsigmask=mask, setsigdef=(signal.SIGTERM, signal.SIGINT), file_actions=[
                        (os.POSIX_SPAWN_DUP2, stdout.fileno(), 1),
                        (os.POSIX_SPAWN_DUP2, stdout.fileno(), 2)])
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            returncode = wait_case_leader(pid, deadline-cleanup_budget)
            if returncode is None:
                raise subprocess.TimeoutExpired(command, timeout)
            return subprocess.CompletedProcess(command, returncode)
        finally:
            # Defer rather than discard a signal arriving during normal cleanup.
            deferred = []
            def defer(number, frame):
                deferred.append(number)
            for number in (signal.SIGTERM, signal.SIGINT):
                signal.signal(number, defer)
            if pid is not None:
                terminate_case_group(pid, min(deadline, time.monotonic()+cleanup_budget))
            if deferred:
                raise ProbeInterrupted(deferred[0])


@dataclass(frozen=True)
class ProbeCase:
    backend: str = 'efficient'
    precision: str = 'bfloat16'
    batch_size: int = 1
    heads: int = 12
    length: int = 256
    head_dim: int = 64
    mask: str = 'global'
    bias_kind: str = 'table'
    padding: bool = True
    local_radius: int = 32
    seed: int = 8421
    warmup: int = 5
    iterations: int = 20
    device_index: int = 0

    def validate(self):
        if self.backend not in BACKENDS or self.precision not in PRECISIONS:
            raise ValueError('Unknown forced backend or precision')
        if self.mask not in {'global', 'chain', 'local'} or self.bias_kind not in {'table', 'dense'}:
            raise ValueError('Unknown mask or bias kind')
        limits = {'batch_size': (1, 32), 'heads': (1, 32), 'length': (8, 4096),
                  'head_dim': (8, 256), 'local_radius': (0, 4096), 'seed': (0, 2**31-1),
                  'warmup': (3, 20), 'iterations': (20, 100), 'device_index': (0, 31)}
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in [{low}, {high}]')
        if type(self.padding) is not bool:
            raise ValueError('padding must be boolean')
        if self.batch_size*self.heads*self.length*self.length > 2**29:
            raise ValueError('Bounded probe: at most 2**29 dense attention entries (2 GiB FP32)')
        return self

    @property
    def case_id(self):
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]
        return f'{self.backend}_{self.precision}_{self.mask}_{self.bias_kind}_{digest}'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def fingerprints():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), Path(oracle.__file__).resolve())}


def make_inputs(case):
    """Always generate FP32 CPU inputs with a private generator, irrespective of backend."""
    case.validate()
    generator = torch.Generator(device='cpu').manual_seed(case.seed)
    shape = (case.batch_size, case.heads, case.length, case.head_dim)
    values = {name: torch.randn(shape, generator=generator, device='cpu', dtype=torch.float32)
              for name in ('q', 'k', 'v', 'weights')}
    bias_shape = (2, case.heads) if case.bias_kind == 'table' else shape[:3]+(case.length,)
    values['parameter'] = .2*torch.randn(bias_shape, generator=generator, device='cpu')
    groups = ((torch.arange(case.length)-2).clamp_min(0)*3 // (case.length-2)).clamp_max(2)
    groups[:2] = -1
    values['groups'] = groups.expand(case.batch_size, -1).clone()
    values['valid'] = torch.ones((case.batch_size, case.length), dtype=torch.bool)
    if case.padding:
        for row in range(case.batch_size):
            tail = 1 + row % max(1, case.length//8)
            values['valid'][row, -tail:] = False
    return values


def input_digest(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(memoryview(tensor.contiguous().numpy()).cast('B'))
    return digest.hexdigest()


def prepare_state(values, device):
    state = {name: tensor.to(device) for name, tensor in values.items()}
    for name in ('q', 'k', 'v', 'parameter'):
        state[name] = state[name].detach().requires_grad_(True)
    return state


def compute_tensors(state, case):
    dtype = PRECISIONS[case.precision]
    q, k, v = [state[name].to(dtype) for name in ('q', 'k', 'v')]
    allowed = oracle.allowed_mask(state['groups'], state['valid'], case.mask, case.local_radius)
    bias = oracle.effective_bias(state['parameter'].to(dtype), state['groups'], allowed, case.bias_kind)
    return q, k, v, bias


def forced_attention(q, k, v, bias, backend):
    """A single enabled backend: no catch/retry or silent math fallback."""
    with sdpa_kernel(BACKENDS[backend]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                            dropout_p=0., is_causal=False)


def forward_loss(state, case, backend=None):
    q, k, v, bias = compute_tensors(state, case)
    out = forced_attention(q, k, v, bias, backend or case.backend)
    # FP32 scalar reduction; the attention activation dtype is explicit above.
    loss = (out.float()*state['weights']).sum()/math.sqrt(out.numel())
    return out, loss


def backward(loss, state):
    return torch.autograd.grad(loss, tuple(state[name] for name in ('q', 'k', 'v', 'parameter')))


def snapshot_step(state, case, backend):
    out, loss = forward_loss(state, case, backend)
    gradients = backward(loss, state)
    return {'output': out.detach().float().cpu(),
            **{name: grad.detach().float().cpu() for name, grad in
               zip(('q_grad', 'k_grad', 'v_grad', 'bias_parameter_grad'), gradients)}}


def compare_snapshots(reference, actual, precision):
    limit = TOLERANCES[precision]
    results = {}
    for name in ('output', 'q_grad', 'k_grad', 'v_grad', 'bias_parameter_grad'):
        a, r = actual[name], reference[name]
        if a.shape != r.shape:
            raise NumericalMismatch(f'{name}: shape mismatch')
        finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
        delta = (a.double()-r.double()).abs()
        difference_norm = float(torch.linalg.vector_norm(delta)) if finite else None
        reference_norm = float(torch.linalg.vector_norm(r.double())) if finite else None
        norm_limit = limit['atol']+limit['rtol']*reference_norm if finite else None
        results[name] = dict(finite=finite,
            passed=finite and torch.allclose(a, r, **limit) and difference_norm <= norm_limit,
            max_abs_error=float(delta.max()) if finite else None,
            rms_error=float(delta.square().mean().sqrt()) if finite else None,
            difference_l2=difference_norm, reference_l2=reference_norm, allowed_difference_l2=norm_limit,
            reference_abs_max=float(r.abs().max()) if finite else None,
            candidate_abs_max=float(a.abs().max()) if finite else None,
            candidate_l1=float(a.double().abs().sum()) if finite else None)
    return {'passed': all(item['passed'] for item in results.values()),
            'tolerances': limit, 'comparisons': results,
            'bias_gradient_present': True,
            'note': 'Zero bias gradient can be valid for row-constant table bias; existence and equality are checked.'}


def observed_backend(operators, cuda_kernels, requested):
    operators = set(operators)
    expected = {'math': {'aten::_scaled_dot_product_attention_math'},
                'efficient': {'aten::_scaled_dot_product_efficient_attention',
                              'aten::_scaled_dot_product_efficient_attention_backward'}}[requested]
    forbidden = ('flash_attention', 'cudnn_attention') + (('efficient_attention',) if requested == 'math' else ('attention_math',))
    if not expected.issubset(operators) or any(any(key in op for key in forbidden) for op in operators):
        raise BackendMismatch(f'Requested {requested}; actual SDPA operators: {sorted(operators)}')
    if not cuda_kernels:
        raise BackendMismatch('No CUDA kernel events captured; backend timing is not certified')
    return requested


def profile_candidate(state, case):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=False) as profiler:
        candidate = snapshot_step(state, case, case.backend)
        torch.cuda.synchronize(case.device_index)
    operators = sorted({event.key for event in profiler.key_averages()
                        if 'scaled_dot_product' in event.key})
    kernel_names = sorted({event.name for event in profiler.events()
                           if event.device_type == torch.autograd.DeviceType.CUDA
                           and not event.name.startswith(('Memcpy', 'Memset'))})
    observed = observed_backend(operators, kernel_names, case.backend)
    return candidate, dict(requested=case.backend, observed=observed, operators=operators,
                           cuda_kernel_names=kernel_names,
                           timing_scope='Profiler only identifies kernels; timing uses a separate unprofiled loop.')


def timing_summary(values):
    ordered = sorted(values)
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'p95': ordered[math.ceil(.95*len(ordered))-1], 'minimum': ordered[0], 'maximum': ordered[-1]}


def measure_candidate(state, case):
    for _ in range(case.warmup):
        out, loss = forward_loss(state, case)
        gradients = backward(loss, state)
        del out, loss, gradients
    torch.cuda.synchronize(case.device_index)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(case.device_index)
    baseline = torch.cuda.memory_allocated(case.device_index)
    events = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(case.iterations)]
    started = time.monotonic()
    for begin, forward_end, backward_end in events:
        begin.record()
        out, loss = forward_loss(state, case)
        forward_end.record()
        gradients = backward(loss, state)
        backward_end.record()
        del out, loss, gradients
    torch.cuda.synchronize(case.device_index)
    wall_seconds = time.monotonic()-started
    forward_ms = [a.elapsed_time(b) for a, b, _ in events]
    backward_ms = [b.elapsed_time(c) for _, b, c in events]
    total_ms = [a.elapsed_time(c) for a, _, c in events]
    peak = torch.cuda.max_memory_allocated(case.device_index)
    return {'warmup': case.warmup, 'iterations': case.iterations,
            'forward_ms': timing_summary(forward_ms), 'backward_ms': timing_summary(backward_ms),
            'total_ms': timing_summary(total_ms), 'wall_seconds': wall_seconds,
            'scope': 'Forward includes activation casts, exact bias/mask construction, attention and scalar loss; no optimizer or model.',
            'raw_total_ms': total_ms}, {
                'baseline_allocated_bytes': baseline, 'peak_allocated_bytes': peak,
                'incremental_peak_allocated_bytes': peak-baseline,
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(case.device_index),
                'scope': 'Unprofiled warm benchmark; reference/profiler tensors discarded before reset.'}


def classify_exception(error):
    if isinstance(error, torch.cuda.OutOfMemoryError) or 'cuda out of memory' in str(error).lower():
        return 'oom'
    if isinstance(error, BackendMismatch):
        return 'backend_mismatch'
    if isinstance(error, NumericalMismatch):
        return 'numerical_mismatch'
    if isinstance(error, subprocess.TimeoutExpired):
        return 'timeout'
    message = str(error).lower()
    if isinstance(error, RuntimeError) and any(phrase in message for phrase in
            ('no available kernel', 'no viable backend', 'not compiled with', 'not supported',
             'unsupported', 'only supports', 'derivative for')):
        return 'unsupported'
    return 'failed'


def empty_result(case, status, phase, source_hashes):
    return dict(schema_version=SCHEMA, case=asdict(case), case_id=case.case_id,
                status=status, phase=phase, source_fingerprints=source_hashes,
                gpu_execution='not_started', environment=None, correctness=None,
                backend_evidence=None, timing=None, memory=None,
                created_at=datetime.now(timezone.utc).isoformat())


def run_cuda_case(case, source_hashes):
    case.validate()
    result = empty_result(case, 'failed', 'environment', source_hashes)
    started = time.monotonic()
    try:
        if fingerprints() != source_hashes:
            raise RuntimeError('Probe/oracle source changed after planning')
        result['environment'] = dict(torch_version=torch.__version__, python=platform.python_version(),
                                     platform=platform.platform(), device_index=case.device_index,
                                     cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if not result['environment']['cuda_available']:
            result['status'] = 'unavailable'
            result['error'] = {'type': 'CudaUnavailable', 'message': 'CUDA is unavailable; no CPU fallback.'}
            return result
        torch.cuda.set_device(case.device_index)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        properties = torch.cuda.get_device_properties(case.device_index)
        result['environment'].update(gpu=properties.name, total_memory_bytes=properties.total_memory,
            gpu_uuid=str(properties.uuid) if getattr(properties, 'uuid', None) is not None else None,
            capability=list(torch.cuda.get_device_capability(case.device_index)), cudnn=torch.backends.cudnn.version(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            backend_flags={name: bool(getattr(torch.backends.cuda, name+'_sdp_enabled')())
                           for name in ('flash', 'mem_efficient', 'math', 'cudnn')},
            activation_dtype=case.precision, leaf_dtype='float32',
            precision_policy='Explicit activation/bias casts; FP32 leaf tensors and scalar loss, TF32 disabled.')
        result['phase'] = 'inputs'
        values = make_inputs(case)
        result['input_sha256'] = input_digest(values)
        result['gpu_execution'] = 'attempted'
        state = prepare_state(values, torch.device('cuda', case.device_index))
        del values
        q, k, v, bias = compute_tensors(state, case)
        result['attention_input'] = dict(q_shape=list(q.shape), bias_shape=list(bias.shape),
            q_dtype=str(q.dtype), bias_dtype=str(bias.dtype), bias_stride=list(bias.stride()),
            bias_requires_grad=bias.requires_grad, dropout_p=0., is_causal=False)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            params = torch.backends.cuda.SDPAParams(q, k, v, bias, 0., False, False)
            result['efficient_eligibility'] = {'can_use': torch.backends.cuda.can_use_efficient_attention(params, debug=True)}
            result['efficient_eligibility']['warnings'] = [str(item.message) for item in captured]
        del q, k, v, bias, params
        result['phase'] = 'math_reference'
        reference = snapshot_step(state, case, 'math')
        torch.cuda.synchronize(case.device_index)
        result['phase'] = 'candidate_profile_and_gradients'
        candidate, result['backend_evidence'] = profile_candidate(state, case)
        result['correctness'] = compare_snapshots(reference, candidate, case.precision)
        del reference, candidate
        if not result['correctness']['passed']:
            raise NumericalMismatch('Output or Q/K/V/bias gradient differs from forced MATH at the same precision')
        result['phase'] = 'warmup_and_timing'
        result['timing'], result['memory'] = measure_candidate(state, case)
        if fingerprints() != source_hashes:
            raise RuntimeError('Probe/oracle source changed during case execution')
        result.update(status='passed', phase='complete', gpu_execution='completed')
    except Exception as error:
        result['status'] = classify_exception(error)
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
        traceback.print_exc()
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
    return result


def validate_result(result, case, source_hashes):
    if (result.get('schema_version') != SCHEMA or result.get('case') != asdict(case)
            or result.get('case_id') != case.case_id or result.get('status') not in STATUSES
            or result.get('source_fingerprints') != source_hashes):
        raise ValueError('Mismatched or incomplete worker result identity')
    if result['status'] == 'passed':
        if (result.get('gpu_execution') != 'completed' or not (result.get('environment') or {}).get('cuda_available')
                or not (result.get('correctness') or {}).get('passed')
                or not (result.get('correctness') or {}).get('bias_gradient_present')):
            raise ValueError('Passing result lacks CUDA/gradient evidence')
        comparisons = result['correctness'].get('comparisons', {})
        if (set(comparisons) != {'output','q_grad','k_grad','v_grad','bias_parameter_grad'}
                or any(not item.get('passed') or not item.get('finite') for item in comparisons.values())
                or result['correctness'].get('tolerances') != TOLERANCES[case.precision]):
            raise ValueError('Passing result lacks complete numerical comparisons')
        evidence = result['backend_evidence']
        observed_backend(evidence['operators'], evidence['cuda_kernel_names'], case.backend)
        if evidence['observed'] != case.backend or evidence['requested'] != case.backend:
            raise ValueError('Passing result reports another backend')
        timing = result['timing']
        if timing['iterations'] != case.iterations or timing['warmup'] != case.warmup:
            raise ValueError('Incomplete timing protocol')
        for name in ('forward_ms', 'backward_ms', 'total_ms'):
            if any(not math.isfinite(value) or value <= 0 for value in timing[name].values()):
                raise ValueError('Invalid timing')
        if (len(timing['raw_total_ms']) != case.iterations
                or any(not math.isfinite(value) or value <= 0 for value in timing['raw_total_ms'])):
            raise ValueError('Incomplete timing samples')
        memory = result['memory']
        if any(type(memory[key]) is not int or memory[key] < 0 for key in
               ('baseline_allocated_bytes', 'peak_allocated_bytes', 'incremental_peak_allocated_bytes', 'peak_reserved_bytes')):
            raise ValueError('Invalid memory evidence')
        if (memory['peak_allocated_bytes']-memory['baseline_allocated_bytes'] != memory['incremental_peak_allocated_bytes']
                or memory['peak_reserved_bytes'] < memory['peak_allocated_bytes']):
            raise ValueError('Inconsistent memory evidence')
    return result


def matched_comparisons(results):
    groups = {}
    for result in results:
        config = {key: value for key, value in result['case'].items() if key != 'backend'}
        key = json.dumps(config, sort_keys=True)
        groups.setdefault(key, {})[result['case']['backend']] = result
    comparisons = []
    for key, pair in groups.items():
        if set(pair) != {'math', 'efficient'}:
            continue
        a, b = pair['math'], pair['efficient']
        row = {'case_without_backend': json.loads(key), 'status': 'not_comparable',
               'backend_statuses': {name: item['status'] for name, item in pair.items()}}
        if a['status'] == b['status'] == 'passed':
            same_environment = all(a['environment'].get(field) == b['environment'].get(field)
                                   for field in ('torch_version','cuda_build','gpu','gpu_uuid','capability','cudnn','total_memory_bytes'))
            if same_environment and a.get('input_sha256') == b.get('input_sha256') and a.get('input_sha256'):
                row.update(status='comparable_passed',
                    total_median_speedup_math_over_efficient=a['timing']['total_ms']['median']/b['timing']['total_ms']['median'],
                    peak_allocated_bytes_math_minus_efficient=a['memory']['peak_allocated_bytes']-b['memory']['peak_allocated_bytes'])
            else:
                row['reason'] = 'Input digest or runtime environment differs'
        comparisons.append(row)
    return comparisons


def worker_command(spec_path, result_path):
    return [sys.executable, str(Path(__file__).resolve()), '--worker-spec', str(spec_path),
            '--worker-result', str(result_path)]


def execute_plan(cases, output_dir, *, dry_run=False, case_timeout=180, total_timeout=1200):
    if not 1 <= len(cases) <= 24 or not 1 <= case_timeout <= 3600 or not 1 <= total_timeout <= 7200:
        raise ValueError('Limit cases to 24, case timeout to 3600s, total timeout to 7200s')
    for case in cases:
        case.validate()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = fingerprints()
    write_json(output_dir/'plan.json', {'schema_version': SCHEMA, 'source_fingerprints': sources,
        'cases': [asdict(case) for case in cases], 'case_timeout_seconds': case_timeout,
        'total_timeout_seconds': total_timeout, 'dry_run': dry_run, 'execution': 'sequential fresh subprocess per case',
        'process_supervision': 'Linux same-session process groups; parent TERM/INT forwarding; temporary subreaper; cleanup reserved inside case timeout'})
    started = time.monotonic()
    results = []
    for index, case in enumerate(cases):
        directory = output_dir/f'{index:02d}_{case.case_id}'
        directory.mkdir()
        spec_path, result_path = directory/'spec.json', directory/'result.json'
        write_json(spec_path, {'case': asdict(case), 'source_fingerprints': sources})
        command = worker_command(spec_path, result_path)
        remaining = total_timeout-(time.monotonic()-started)
        if dry_run:
            result = empty_result(case, 'dry_run', 'not_executed', sources)
            result['command'] = command
        elif remaining <= 0:
            result = empty_result(case, 'not_run_budget', 'overall_deadline', sources)
        else:
            child_process = None
            try:
                if fingerprints() != sources:
                    raise RuntimeError('Source changed between cases; validation rejected')
                with (directory/'worker.log').open('w') as log:
                    completed = run_case_process(command, stdout=log, timeout=min(case_timeout, remaining))
                # A normal return also proves run_case_process's finally block
                # reaped this case's process group. Never infer this from JSON.
                child_process = {'returncode': completed.returncode, 'cleanup_completed': True}
                if not result_path.exists():
                    raise RuntimeError(f'Worker exit {completed.returncode} without result JSON')
                result = validate_result(json.loads(result_path.read_text()), case, sources)
                child_process['reported_status'] = result['status']
                expected_code = 0 if result['status'] == 'passed' else 2
                child_process['expected_returncode'] = expected_code
                child_process['protocol_valid'] = completed.returncode == expected_code
                if not child_process['protocol_valid']:
                    # A protocol failure must not be reclassified by a backend
                    # keyword (e.g. "unsupported") embedded in the message.
                    raise ValueError(f'Worker reported {result["status"]} but exited '
                                     f'{completed.returncode}; expected {expected_code}')
            except CaseCleanupError:
                # Never run another case while this group's cleanup is unproven.
                raise
            except Exception as error:
                result = empty_result(case, classify_exception(error), 'parent_execution', sources)
                result['error'] = {'type': type(error).__name__, 'message': str(error)}
            if child_process is not None:
                result['child_process'] = child_process
        write_json(result_path, result)
        results.append(result)
        print(json.dumps({'case_id': case.case_id, 'status': result['status'], 'result': str(result_path)}), flush=True)
    status = ('dry_run_not_executed' if dry_run else 'all_passed' if all(r['status'] == 'passed' for r in results)
              else 'completed_with_nonpassing_cases')
    summary = {'schema_version': SCHEMA, 'status': status, 'source_fingerprints': sources,
               'elapsed_seconds': time.monotonic()-started,
               'status_counts': {name: sum(row['status'] == name for row in results)
                                 for name in sorted({row['status'] for row in results})},
               'results': results, 'matched_comparisons': matched_comparisons(results),
               'limits': ['Random operator probe only; not a model/training profile.',
                   'A failing backend is never retried with a different backend.',
                   'No CPU test or dry run constitutes a CUDA pass.']}
    write_json(output_dir/'summary.json', summary)
    return summary


def split_choices(value, allowed):
    selected = value.split(',')
    if not selected or len(selected) != len(set(selected)) or any(item not in allowed for item in selected):
        raise argparse.ArgumentTypeError(f'Use unique comma-separated values from {sorted(allowed)}')
    return selected


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--output-dir', type=Path)
    value.add_argument('--dry-run', action='store_true')
    value.add_argument('--backends', default='math,efficient')
    value.add_argument('--dtypes', default='float32,bfloat16')
    value.add_argument('--masks', default='global')
    value.add_argument('--bias-kinds', default='table')
    value.add_argument('--padding', action=argparse.BooleanOptionalAction, default=True)
    for name, default in (('batch-size',1), ('heads',12), ('length',256), ('head-dim',64),
                          ('local-radius',32), ('seed',8421), ('warmup',5), ('iterations',20), ('device-index',0)):
        value.add_argument('--'+name, type=int, default=default)
    value.add_argument('--case-timeout', type=int, default=180)
    value.add_argument('--total-timeout', type=int, default=1200)
    value.add_argument('--worker-spec', type=Path, help=argparse.SUPPRESS)
    value.add_argument('--worker-result', type=Path, help=argparse.SUPPRESS)
    return value


def cases_from_args(args):
    combinations = itertools.product(split_choices(args.backends, BACKENDS), split_choices(args.dtypes, PRECISIONS),
        split_choices(args.masks, {'global','chain','local'}), split_choices(args.bias_kinds, {'table','dense'}))
    shared = {name: getattr(args, name) for name in ('batch_size','heads','length','head_dim','padding',
              'local_radius','seed','warmup','iterations','device_index')}
    return [ProbeCase(backend=backend, precision=precision, mask=mask, bias_kind=kind, **shared).validate()
            for backend, precision, mask, kind in combinations]


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.worker_spec is not None:
        if args.worker_result is None or args.dry_run:
            arguments.error('Worker requires --worker-result and cannot be a dry run')
        spec = json.loads(args.worker_spec.read_text())
        case = ProbeCase(**spec['case']).validate()
        result = run_cuda_case(case, spec['source_fingerprints'])
        write_json(args.worker_result, result)
        return 0 if result['status'] == 'passed' else 2
    if args.worker_result is not None or args.output_dir is None:
        arguments.error('--output-dir is required for an isolated plan')
    cases = cases_from_args(args)
    summary = execute_plan(cases, args.output_dir, dry_run=args.dry_run,
                           case_timeout=args.case_timeout, total_timeout=args.total_timeout)
    return 0 if summary['status'] in {'all_passed','dry_run_not_executed'} else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ProbeInterrupted as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(128+error.signum)
