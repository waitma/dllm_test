"""Undeployed R21 artificial-input GPU runtime acceptance, never a data certificate.

Activate pllm; USE_TF=0 pytest -q scripts/tests/immune_llada/test_pilot_gpu_release_producer_r21_v1.py
Only after a verified immutable runtime is built, run its fixed file in fresh
Python: python -I <runtime>/scripts/architecture_prototypes/pilot_gpu_release_producer_r21_v1.py
--profile-dir /absolute/pool/profiles/r21_refined_head_gate --output-dir /absolute/new/output
--runtime-root /absolute/fixed/runtime
No submit, biological input, data gate, resume, checkpoint, or arbitrary variant.
The parent alone writes GPU_RELEASE_COMPLETE after a successful real CUDA child.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import weakref
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
# -I ignores PYTHONDONTWRITEBYTECODE; protect the immutable runtime explicitly.
sys.dont_write_bytecode = True
VARIANT = 'r21_refined_head_gate'
SCHEMA = 'immune.pilot_gpu_producer.v1'
EXTENSION_SCHEMA = 'immune.pilot_gpu_release_producer.v1'
PARENT_SHA = '46399204c7018d00726e3ae08142bd4eb28475c8a9a2991b3a5c90725c8824f7'
NAME = 'r21_pilot_v1_20261001'
SELF = 'scripts/architecture_prototypes/pilot_gpu_release_producer_r21_v1.py'
ENTRY = 'scripts/architecture_prototypes/pilot_runtime_entry_r21_v1.py'
GLOBAL_BATCH, STEPS, TIMEOUT = 256, 3, 3600
PHASE_SECONDS = dict(gradient_oracle=1200,train_optimizer_step=1800,four_mask_evaluation=600)
STAGE_EVIDENCE = 'producer_stage_evidence.json'
LIFETIME_PHASE, LIFETIME_SECONDS = 'lifetime_preflight', 120
LIFETIME_REFERENTS = ('model','trainer','accelerator','callback_handler','observer','optimizer_wrapper',
                      'optimizer','scheduler','parameter','moment','amp_original_forward')
GRAD_ATOL, GRAD_RTOL = 2e-5, 5e-3
LOSS_ATOL, LOSS_RTOL = 2e-5, 2e-5
TOTAL_ATOL, TOTAL_RTOL = 2e-4, 2e-6
RATIOS = (.25, .5, .75, 1.)
ARTIFACTS = ('gpu_release_raw.json', 'gpu_release_summary.json', 'PRODUCER_SUCCESS.json',
             'GPU_RELEASE_COMPLETE.json', 'COMPLETE.json', 'FAILED.json', 'producer_progress.json',STAGE_EVIDENCE)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(text)
    temp.replace(path)


def runtime_context(profile_dir, runtime_root=None):
    """Verify immutable bytes before importing ML; never redirects any module ROOT."""
    require(ROOT.name == NAME and ROOT.parent.name == 'immune_pilot_runtimes',
            'GPU producer must run inside the fixed derived runtime')
    require(runtime_root is None or Path(runtime_root) == ROOT,
            'Runtime argument must identify this fixed producer source')
    original = ROOT.parents[2]
    parent = original/'output/immune_research_releases/r20_r27_20261001'
    require(ROOT == ROOT.resolve() and sha(parent/'release_manifest.json') == PARENT_SHA,
            'Required parent release SHA differs')
    manifest = json.loads((ROOT/'pilot_runtime_manifest.json').read_text())
    require(manifest.get('base_snapshot') == str(parent) and manifest.get('source_root') == str(original),
            'Runtime parent/source identity differs')
    inspector_path = ROOT/'scripts/architecture_prototypes/pilot_runtime_inspect.py'
    parent_map = json.loads((parent/'release_manifest.json').read_text())['copied_files_sha256']
    require(sha(inspector_path) == parent_map[str(inspector_path.relative_to(ROOT))], 'Bootstrap inspector differs')
    require(manifest.get('candidate_variant') == VARIANT and manifest.get('package_name') == NAME,
            'Candidate runtime identity differs')
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in ('scripts', 'dllm', 'examples'):
            continue
        file = getattr(module, '__file__', None)
        require(file is None or Path(file).resolve().is_relative_to(ROOT), 'Preloaded live project module')
        require(all(Path(path).resolve().is_relative_to(ROOT) for path in getattr(module, '__path__', [])),
                'Preloaded external project namespace')
        if name in ('scripts', 'examples') and file is None:
            require(list(getattr(module, '__path__', [])) == [str(ROOT/name)],
                    'Preloaded namespace must use the exact fixed runtime package')
    spec = importlib.util.spec_from_file_location('_fixed_runtime_inspector', inspector_path)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    sys.path[:] = bootstrap.isolated_search_path(ROOT, original, sys.path, [])
    for name, namespace in bootstrap.project_namespace_specs(ROOT).items():
        if name not in sys.modules:
            sys.modules[name] = importlib.util.module_from_spec(namespace)
    from scripts.architecture_prototypes import pilot_candidate_runtime_package_v1 as package
    checked = package.verify_runtime(ROOT)
    require(checked == manifest and checked['base_manifest_sha256'] == PARENT_SHA,
            'Runtime manifest changed during bootstrap')
    require(SELF in checked['additions_sha256'] and ENTRY in checked['additions_sha256'],
            'Runtime must bind both new fixed producer and pilot wrapper')
    profile = Path(profile_dir)
    require(profile.is_absolute() and profile == profile.resolve() and profile.name == VARIANT
            and profile.parent.name == 'profiles', 'Expected canonical R21 profile directory')
    from scripts.architecture_prototypes import certified_pilot as pilot
    from scripts.architecture_prototypes import certified_pilot_execution as execution
    require(pilot.ROOT == ROOT and Path(execution.__file__).resolve().is_relative_to(ROOT),
            'Pilot/executor escaped runtime')
    measured = pilot.profile_checks.verify_profile_evidence(profile.parent.parent, VARIANT)
    sources = execution.source_binding()
    require(sources['model'] == checked['measured_source_sha256'], 'Model profile map differs from runtime')
    for mapping in sources.values():
        require(all(checked['runtime_files_sha256'].get(name) == value for name, value in mapping.items()),
                'Adapter/execution source escaped runtime map')
    return SimpleNamespace(package=package, pilot=pilot, execution=execution, manifest=checked,
        manifest_sha=sha(ROOT/'pilot_runtime_manifest.json'), profile=measured,
        profile_dir=profile, sources=sources)


def source_evidence(context):
    return dict(runtime_root=str(ROOT), runtime_manifest_sha256=context.manifest_sha,
        parent_manifest_sha256=PARENT_SHA, producer_sha256=sha(ROOT/SELF), wrapper_sha256=sha(ROOT/ENTRY),
        runtime_files_sha256=context.manifest['runtime_files_sha256'], source_binding=context.sources)


def stable(context):
    require(context.package.verify_runtime(ROOT) == context.manifest
            and sha(ROOT/'pilot_runtime_manifest.json') == context.manifest_sha
            and context.execution.source_binding() == context.sources, 'Source changed during GPU acceptance')
    require(context.pilot.profile_checks.verify_profile_evidence(context.profile_dir.parent.parent, VARIANT)
            == context.profile, 'Model profile changed')


def fixed_plan(context):
    p = context.profile
    micro = p['recommended_micro_batch']
    require(type(micro) is int and micro == 32, 'Fixed R21 requires measured microbatch32')
    return dict(schema=SCHEMA, variant=VARIANT, config=p['config'], micro_batch=micro,
        eval_batch=min(micro, 16), global_batch=GLOBAL_BATCH, grad_accum=GLOBAL_BATCH//micro,
        optimizer_steps=STEPS, scheduler_horizon=10000, seed=42, timeout_seconds=TIMEOUT,
        gradient_tolerance=dict(atol=GRAD_ATOL, rtol=GRAD_RTOL, relative_l2=GRAD_RTOL, scale=GRAD_RTOL),
        loss_tolerance=dict(atol=LOSS_ATOL, rtol=LOSS_RTOL),
        total_tolerance=dict(atol=TOTAL_ATOL, rtol=TOTAL_RTOL),
        phase_seconds=dict(PHASE_SECONDS),
        lifetime_preflight=dict(kind='additional_tiny_native_trainer_lifetime_only',
            config=dict(p['config'],d_model=16,n_layers=2,n_heads=2,mlp_hidden_size=32,
                        max_length=64,full_capacity=8,fragment_capacity=4),
            micro_batch=1,grad_accum=8,global_batch=8,optimizer_steps=1,dataset_records=24,
            manual_microsteps=8,timeout_seconds=LIFETIME_SECONDS,counts_toward_formal=False),
        oracle='one isolated unclipped real-HF window versus independent manual CE/GA replay',
        formal='fresh model; three complete real-HF windows with original clipping/AdamW/BF16',
        mask_levels=list(RATIOS), input_kind='procedural_artificial_only', biological_quality_claim=False,
        data_certificate_issued=False)


def installed_sources(context):
    import torch
    from accelerate import Accelerator
    from dllm.pipelines.immune_llada.research.model import ResearchModel
    classes = (*context.pilot.metrics.TokenValidationTrainer.__mro__[:-1],
               ResearchModel, torch.optim.AdamW, Accelerator)
    return {str(Path(inspect.getfile(cls)).resolve()): sha(inspect.getfile(cls)) for cls in classes}


def artificial_records(config, count, *, stress=True, offset=0):
    """Generate integers then map to vocabulary letters; no input files are opened."""
    from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
    result = []
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    for i in range(count):
        rng = random.Random(42000+offset+i)
        def letters(length):
            return ''.join(alphabet[rng.randrange(len(alphabet))] for _ in range(length))
        if stress:
            context_length = config.max_length-config.full_capacity-config.fragment_capacity-6
            require(context_length > 0, 'Configuration cannot hold both capacities and context')
            chains = [BioSeqChain(letters(context_length), 'antigen'),
                BioSeqChain(letters(config.full_capacity-1), 'antibody_heavy'),
                BioSeqChain(letters(config.fragment_capacity-1), 'antibody_light', metadata={'sequence_scope':'fragment'})]
        else:
            chains = [BioSeqChain(letters(2+i%3), 'peptide'),
                BioSeqChain(letters(min(config.fragment_capacity-1, 2+i%4)), 'tcr_beta',
                            metadata={'sequence_scope':'fragment'})]
            if i % 2:
                chains.append(BioSeqChain(letters(2), 'tcr_alpha', metadata={'sequence_scope':'fragment'}))
        relation = not stress and i % 2 == 0
        result.append(BioSeqRecord(chains, 'generic', 'procedural_stress' if stress else 'procedural_mixed',
            metadata={'synthetic_only':True, 'relation_supervised':relation},
            labels={'relation':'binding' if relation else 'unknown'}))
    return result


def tensor_digest(batch):
    hasher = hashlib.sha256()
    for name, value in sorted(batch.items()):
        value = value.detach().cpu().contiguous()
        hasher.update(name.encode()); hasher.update(str(value.dtype).encode())
        hasher.update(str(tuple(value.shape)).encode()); hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def rng_state(device):
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None)


def restore_rng(state, device):
    import numpy as np
    import torch
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state[3], device)


def rng_digest(state):
    # Stable serialization for comparison only; never deserializes external RNG files.
    return digest([state[0], [state[1][0], state[1][1].tolist(), *state[1][2:]],
                   state[2].tolist(), None if state[3] is None else state[3].tolist()])


def observed_batch(batch):
    import torch
    scopes = {'full_capacity':1, 'fragment_capacity':4}
    capacities = {}
    for name, scope in scopes.items():
        lengths = []
        for row in range(batch['input_ids'].shape[0]):
            slots = batch['target_slot_mask'][row].bool() & batch['scope_ids'][row].eq(scope)
            for uid in batch['chain_ids'][row, slots].unique().tolist():
                lengths.append(int((slots & batch['chain_ids'][row].eq(uid)).sum()))
        capacities[name] = max(lengths, default=0)
    return dict(input_shape=list(batch['input_ids'].shape), device=str(batch['input_ids'].device),
        target_capacities=capacities, attended_tokens=int(batch['attention_mask'].sum()),
        target_slots=int(batch['target_slot_mask'].sum()), eligible_tokens=int(batch['eligible_mask'].sum()),
        padding_tokens=int((~batch['attention_mask'].bool()).sum()), input_sha256=tensor_digest(batch),
        tensors={name:dict(shape=list(value.shape), device=str(value.device), dtype=str(value.dtype))
                 for name,value in batch.items()})


def memory_start(device):
    import torch
    if device.type != 'cuda':
        return None
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    return dict(baseline_allocated_bytes=torch.cuda.memory_allocated(device),
                baseline_reserved_bytes=torch.cuda.memory_reserved(device), started=time.monotonic())


def memory_end(device, before):
    import torch
    if before is None:
        return dict(cuda_measured=False)
    torch.cuda.synchronize(device)
    return dict(cuda_measured=True, baseline_allocated_bytes=before['baseline_allocated_bytes'],
        baseline_reserved_bytes=before['baseline_reserved_bytes'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        elapsed_seconds=time.monotonic()-before['started'])


class PhaseBudgetExceeded(ValueError):
    """A time limit failure; never loosens an objective or numerical check."""
    def __init__(self,phase,elapsed,limit):
        self.phase,self.elapsed_seconds,self.limit_seconds = phase,elapsed,limit
        super().__init__(f'{phase} exceeded fixed budget: {elapsed:.6f}s > {limit}s')


def compact_stage_statistics(phase,value):
    """Keep observed scalars/hashes on failure, excluding logits and CPU gradients."""
    result = {}
    keys = ('optimizer_steps','microsteps','records','training_loss','windows','adamw',
            'initial_parameters_sha256','final_parameters_sha256','precision_observations','memory','cleanup_observations')
    if phase == 'four_mask_evaluation':
        keys = ('source_records','callback_counts','rng_before_sha256','rng_after_sha256',
                'rng_preserved','optimizer_state_resident','adamw_state_parameters','adamw_state_bytes','memory')
    result.update({key:value[key] for key in keys if key in value})
    if 'batches' in value and value['batches']:
        batches = value['batches']
        result['actual_batch_count'] = len(batches)
        result['first_input_shape'] = batches[0]['input_shape']
        result['first_target_capacities'] = batches[0]['target_capacities']
        result['input_batch_sha256'] = digest([batch['input_sha256'] for batch in batches])
    changes = value.get('parameter_max_abs_changes')
    if changes:
        result['parameter_max_abs_change'] = max(changes.values())
        result['parameters_changed'] = sum(change > 0 for change in changes.values())
    for name in ('oracle','first_window_preclip_match'):
        comparison = value.get(name)
        if comparison:
            result[name] = {key:comparison[key] for key in ('passed','gradient_scale','relative_l2_error',
                'microsteps','records','method','native_losses','manual_losses','rng_replay_equal') if key in comparison}
            parameters = comparison.get('parameters',{})
            result[name]['parameters_compared'] = len(parameters)
            result[name]['max_abs_error'] = max((p['max_abs_error'] for p in parameters.values()),default=0.)
    if phase == 'four_mask_evaluation' and 'metrics' in value:
        result['mask_losses'] = {f'mask{int(r*100)}':value['metrics'][f'eval_mask{int(r*100)}_token_micro_loss']
                                for r in RATIOS}
        result['record_evaluations'] = value['metrics']['eval_record_evaluations']
        recount = value.get('cpu_recount_oracle',{})
        result['cpu_recount'] = {key:recount[key] for key in ('counts_exact','max_loss_sum_error') if key in recount}
    return result


def record_stage(output,plan,phase,status,statistics=None,elapsed_seconds=None):
    """Observations only: never a raw-completed report or GPU acceptance marker."""
    budgets = dict(plan['phase_seconds'],**{LIFETIME_PHASE:plan['lifetime_preflight']['timeout_seconds']})
    require(phase in budgets, 'Unknown observed producer phase')
    output = Path(output);path = output/STAGE_EVIDENCE
    evidence = json.loads(path.read_text()) if path.exists() else dict(schema=SCHEMA,
        kind='artificial_runtime_partial_stage_observations',runtime_package_name=NAME,
        plan_sha256=digest(plan),phase_seconds=budgets,timeout_seconds=plan['timeout_seconds'],
        gpu_accepted=False,raw_completed=False,phases={})
    require(evidence.get('plan_sha256') == digest(plan) and evidence.get('gpu_accepted') is False
            and evidence.get('raw_completed') is False,'Stage observation plan/scope changed')
    phase_value = {**evidence['phases'].get(phase,{}),**(statistics or {}),'status':status,
                   'budget_seconds':budgets[phase]}
    if elapsed_seconds is not None:
        require(type(elapsed_seconds) in (int,float) and math.isfinite(elapsed_seconds) and elapsed_seconds >= 0,
                'Invalid observed stage elapsed time')
        phase_value['elapsed_seconds'] = elapsed_seconds
    evidence['phases'][phase] = phase_value
    write_json(path,evidence)
    write_json(output/'producer_progress.json',dict(schema=SCHEMA,status=status,stage=phase,
        gpu_accepted=False,raw_completed=False,**{key:phase_value[key] for key in
            ('optimizer_steps','microsteps','records','windows','elapsed_seconds','budget_seconds') if key in phase_value}))
    return phase_value


def observed_phase_budget(output,plan,phase,value):
    elapsed = value['phase_elapsed_seconds'] if phase == LIFETIME_PHASE else value['memory']['elapsed_seconds']
    limit = plan['lifetime_preflight']['timeout_seconds'] if phase == LIFETIME_PHASE else plan['phase_seconds'][phase]
    status = 'budget_failed' if elapsed > limit else 'observed_within_budget'
    observed = record_stage(output,plan,phase,status,compact_stage_statistics(phase,value),elapsed)
    if elapsed > limit:
        raise PhaseBudgetExceeded(phase,elapsed,limit)
    return observed


def lifetime_observation(references,device,initial_allocated_bytes):
    """Actual weak-reference and allocator observations; CPU results cannot prove CUDA."""
    import torch
    cuda = device.type == 'cuda'
    alive = {name:reference() is not None for name,reference in references.items()}
    types = {name:type(references[name]()).__module__+'.'+type(references[name]()).__name__
             for name,present in alive.items() if present}
    return dict(cuda_measured=cuda,actual_allocated_before_model_bytes=initial_allocated_bytes,
        actual_allocated_after_cleanup_bytes=torch.cuda.memory_allocated(device) if cuda else None,
        actual_reserved_after_cleanup_bytes=torch.cuda.memory_reserved(device) if cuda else None,
        weakrefs_alive_after_cleanup=alive,alive_object_types=types)


def cleanup_stage(output,plan,phase,references,device,initial_allocated_bytes):
    """Release our child cuBLAS workspace only after every recorded owner is dead.

    PyTorch 2.8 CUDA notes document that del/empty_cache do not free these native
    workspaces. Record both measurements; never clear with surviving owners.
    """
    import torch
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.synchronize(device); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    observation = lifetime_observation(references,device,initial_allocated_bytes)
    observation.update(actual_allocated_after_gc_empty_cache_bytes=observation['actual_allocated_after_cleanup_bytes'],
        actual_reserved_after_gc_empty_cache_bytes=observation['actual_reserved_after_cleanup_bytes'],
        actual_allocated_after_workspace_clear_bytes=None,native_workspace_bytes_released=None,
        cublas_workspace_clear_called=False)
    record_stage(output,plan,phase,'gc_cleanup_observed',dict(cleanup_observations=observation,**observation))
    owners_dead = set(observation['weakrefs_alive_after_cleanup']) == set(LIFETIME_REFERENTS) and all(
        present is False for present in observation['weakrefs_alive_after_cleanup'].values())
    if device.type == 'cuda' and owners_dead and initial_allocated_bytes == 0:
        clear = getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
        require(callable(clear),'Required PyTorch 2.8 cuBLAS workspace cleanup API is unavailable')
        torch.cuda.synchronize(device)
        clear()
        torch.cuda.synchronize(device); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
        observation.update(cublas_workspace_clear_called=True,
            actual_allocated_after_workspace_clear_bytes=torch.cuda.memory_allocated(device),
            actual_allocated_after_cleanup_bytes=torch.cuda.memory_allocated(device),
            actual_reserved_after_cleanup_bytes=torch.cuda.memory_reserved(device))
        observation['native_workspace_bytes_released'] = (observation['actual_allocated_after_gc_empty_cache_bytes']
                                                        -observation['actual_allocated_after_workspace_clear_bytes'])
    record_stage(output,plan,phase,'cleanup_observed',dict(cleanup_observations=observation,**observation))
    return observation


def validate_cuda_cleanup(value):
    """Exact-zero allocator and released owners are both required; CPU cannot pass."""
    require(value.get('cuda_measured') is True,'Cleanup requires actual CUDA observations')
    require(set(value['weakrefs_alive_after_cleanup']) == set(LIFETIME_REFERENTS)
            and all(alive is False for alive in value['weakrefs_alive_after_cleanup'].values())
            and value['alive_object_types'] == {},'CUDA stage owners remain alive or were not observed')
    require(type(value['actual_allocated_before_model_bytes']) is int and value['actual_allocated_before_model_bytes'] == 0
            and type(value['actual_allocated_after_gc_empty_cache_bytes']) is int
            and value['actual_allocated_after_gc_empty_cache_bytes'] >= 0
            and value['cublas_workspace_clear_called'] is True
            and type(value['actual_allocated_after_workspace_clear_bytes']) is int
            and type(value['actual_allocated_after_cleanup_bytes']) is int
            and value['actual_allocated_after_workspace_clear_bytes'] == value['actual_allocated_after_cleanup_bytes'] == 0
            and type(value['native_workspace_bytes_released']) is int and value['native_workspace_bytes_released'] >= 0
            and value['native_workspace_bytes_released'] == value['actual_allocated_after_gc_empty_cache_bytes']
                                                        -value['actual_allocated_after_workspace_clear_bytes']
            and type(value['actual_reserved_after_cleanup_bytes']) is int and value['actual_reserved_after_cleanup_bytes'] >= 0,
            'CUDA allocations remain after model and native workspace cleanup')


def assert_close(actual, expected, message):
    if expected is None:
        require(actual is None, message)
    elif type(expected) is int:
        require(type(actual) in (int,float) and actual == expected, message)
    else:
        require(type(actual) in (int,float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=TOTAL_RTOL, abs_tol=LOSS_ATOL), message)


def independent_mask(batch, ratio):
    """Recreate the frozen record hash mask without calling its implementation."""
    import torch
    ids = batch['input_ids'].detach().cpu()
    eligible = batch['eligible_mask'].detach().cpu().bool()
    attention = batch['attention_mask'].detach().cpu().bool()
    selected = torch.zeros_like(eligible)
    for row in range(len(ids)):
        visible = ids[row, attention[row]].tolist()
        seed = int.from_bytes(hashlib.sha256(json.dumps(visible).encode()).digest()[:8], 'little') % (2**63-1)
        positions = eligible[row].nonzero().flatten()
        draws = torch.rand(len(positions), generator=torch.Generator().manual_seed(seed))
        chosen = draws < ratio
        if len(positions) and not chosen.any():
            chosen[draws.argmin()] = True
        selected[row, positions[chosen]] = True
    return selected & eligible & attention


def independent_training_loss(model, batch):
    """Manual absorbing corruption and log-softmax CE; never calls forward/loss."""
    import torch
    ids = batch['input_ids']; eligible = batch['eligible_mask'].bool() & batch['attention_mask'].bool()
    times = torch.empty(ids.shape[0], device=ids.device, dtype=torch.float32).uniform_(1e-3, 1.)
    selected = (torch.rand(*ids.shape, device=ids.device) < times[:, None]) & eligible
    for row in range(len(ids)):
        if eligible[row].any() and not selected[row].any():
            positions = eligible[row].nonzero().flatten()
            selected[row, positions[torch.randint(len(positions), (1,), device=ids.device)]] = True
    features = model.feature_args(batch)
    features['input_ids'] = ids.masked_fill(selected, model.tokenizer.mask_token_id)
    logits = model.denoise(**features)[0].float()
    values = -torch.log_softmax(logits[selected], dim=-1).gather(1, ids[selected].unsqueeze(1)).squeeze(1)
    return values.sum()/selected.sum().clamp_min(1)


def gradient_comparison(actual, reference):
    """Record numerical sufficient statistics, preserving the GA magnitude test."""
    import torch
    require(set(actual) == set(reference) and actual, 'Oracle gradient membership differs')
    checks = {}; native_square = oracle_square = error_square = dot = 0.
    for name in actual:
        a, b = actual[name].double(), reference[name].double(); difference = a-b
        require(torch.isfinite(a).all() and torch.isfinite(b).all(), 'Nonfinite oracle gradients')
        aa, bb, ee, ab = (float(value) for value in (a.square().sum(), b.square().sum(),
                          difference.square().sum(), (a*b).sum()))
        checks[name] = dict(max_abs_error=float(difference.abs().max()), native_square=aa,
            reference_square=bb, error_square=ee, dot=ab,
            allclose=torch.allclose(actual[name], reference[name], atol=GRAD_ATOL, rtol=GRAD_RTOL))
        require(checks[name]['allclose'], 'Independent CE/GA gradient replay differs: '+name)
        native_square += aa; oracle_square += bb; error_square += ee; dot += ab
    require(oracle_square > 0, 'Oracle gradients are all zero')
    scale, relative = dot/oracle_square, (error_square/oracle_square)**.5
    require(abs(scale-1) <= GRAD_RTOL and relative <= GRAD_RTOL, 'GA gradient magnitude differs')
    return dict(passed=True, parameters=checks, gradient_scale=scale, relative_l2_error=relative,
                native_square=native_square, reference_square=oracle_square, error_square=error_square, dot=dot)


def evaluation_packet(batch, logits, ratio):
    """Compact CPU recount evidence: selected logits plus END positions, no tensors."""
    import torch
    ids = batch['input_ids'].detach().cpu(); clean = {k:v.detach().cpu() for k,v in batch.items()}
    native_shape,native_dtype = list(logits.shape),str(logits.dtype)
    selected = independent_mask(clean, ratio); logits = logits.detach().float().cpu()
    values = logits[selected]; predictions = logits.argmax(-1)
    encoded = [[float(value) if math.isfinite(float(value)) else None for value in row] for row in values]
    require(not torch.isnan(values).any() and not torch.isposinf(values).any(), 'Invalid observed evaluation logits')
    return dict(input_sha256=tensor_digest(clean),logits_shape=native_shape,logits_dtype=native_dtype,
        selected_positions=selected.nonzero().tolist(),
        selected_labels=ids[selected].tolist(), selected_logits=encoded,
        slot_predictions=[predictions[row, clean['target_slot_mask'][row].bool()].tolist()
                          for row in range(len(ids))])


def recount_packet(batch, packet, ratio, terminal_id, relation_ids, vocab_size=None):
    """Independent float64 CPU CE, token populations, and first-END accounting."""
    import numpy as np
    clean = {k:v.detach().cpu() for k,v in batch.items()}; selected = independent_mask(clean, ratio)
    require(packet['input_sha256'] == tensor_digest(clean)
            and packet['selected_positions'] == selected.nonzero().tolist()
            and packet['selected_labels'] == clean['input_ids'][selected].tolist(), 'CPU oracle input/mask binding differs')
    logits = np.asarray([[float('-inf') if value is None else value for value in row]
                         for row in packet['selected_logits']], dtype=np.float64)
    labels = np.asarray(packet['selected_labels'], dtype=np.int64)
    require(logits.ndim == 2 and len(logits) == len(labels) and len(labels) > 0,
            'CPU oracle selected-logit population differs')
    require(packet['logits_shape'] == [*clean['input_ids'].shape,logits.shape[1]]
            and (vocab_size is None or logits.shape[1] == vocab_size), 'CPU oracle actual logits shape differs')
    require(np.isfinite(logits[np.isfinite(logits)]).all() and not np.isnan(logits).any()
            and not np.isposinf(logits).any(), 'Invalid CPU oracle logits')
    losses = np.logaddexp.reduce(logits, axis=1)-logits[np.arange(len(labels)), labels]
    require(np.isfinite(losses).all(), 'CPU oracle has nonfinite CE')
    predictions = logits.argmax(1); totals = [0.]*16
    for index, (row, position) in enumerate(packet['selected_positions']):
        aa, end = bool(clean['residue_mask'][row, position]), bool(clean['terminal_mask'][row, position])
        require(not (aa and end), 'CPU oracle AA/END overlap')
        correct = int(predictions[index] == labels[index]); loss = float(losses[index])
        totals[0] += loss; totals[1] += 1
        if aa:
            require(labels[index] != terminal_id, 'CPU oracle AA label is END')
            totals[2] += loss; totals[3] += 1; totals[6] += correct
            totals[8] += int(predictions[index] == terminal_id)
        elif end:
            require(labels[index] == terminal_id, 'CPU oracle END label differs')
            totals[4] += loss; totals[5] += 1; totals[7] += correct
        else:
            require(labels[index] in relation_ids and not clean['target_slot_mask'][row, position]
                    and clean['chain_ids'][row, position] < 0, 'CPU oracle relation label differs')
            totals[13] += loss; totals[14] += 1; totals[15] += correct
    for row in range(len(clean['input_ids'])):
        slots = clean['target_slot_mask'][row].bool() & clean['attention_mask'][row].bool()
        positions = slots.nonzero().flatten().tolist()
        guesses = packet['slot_predictions'][row]
        require(len(guesses) == len(positions), 'CPU oracle slot prediction count differs')
        predicted = dict(zip(positions, guesses))
        for index,(selected_row,position) in enumerate(packet['selected_positions']):
            if row == selected_row and position in predicted:
                require(predicted[position] == int(predictions[index]),'CPU oracle selected/slot predictions disagree')
        if ratio != 1.:
            continue
        chains = clean['chain_ids'][row]
        truth = clean['terminal_mask'][row].bool() & clean['eligible_mask'][row].bool() & clean['attention_mask'][row].bool()
        for uid in chains[slots].unique().tolist():
            interval = [position for position in positions if int(chains[position]) == uid]
            ends = ((chains == uid) & truth).nonzero().flatten().tolist()
            require(uid >= 0 and len(ends) == 1 and ends[0] in interval, 'CPU oracle chain END contract differs')
            true_offset = interval.index(ends[0]); offsets = [i for i,p in enumerate(interval) if predicted[p] == terminal_id]
            totals[9] += 1
            if not offsets:
                totals[11] += 1
            else:
                totals[10] += int(offsets[0] == true_offset); totals[12] += abs(offsets[0]-true_offset)
    return totals


def cpu_summary(totals, protocol):
    """Independent arithmetic for v2 totals, empty metrics, and derived counts."""
    values = dict(zip(protocol.FIELDS,totals))
    values.update(terminal_true_positive_count=values['terminal_correct'],
        terminal_prediction_count=values['terminal_correct']+values['terminal_false_positive_count'],
        length_compared_count=values['chain_count']-values['terminal_missing_count'])
    result = {name:int(values[name]) for name in protocol.COUNTS}
    fractions = dict(loss=('loss_sum','selected_count'),aa_loss=('aa_loss_sum','aa_count'),
        terminal_loss=('terminal_loss_sum','terminal_count'),aa_accuracy=('aa_correct','aa_count'),
        terminal_accuracy=('terminal_correct','terminal_count'),terminal_precision=('terminal_correct','terminal_prediction_count'),
        terminal_false_positive_rate=('terminal_false_positive_count','aa_count'),
        relation_loss=('relation_loss_sum','relation_count'),relation_accuracy=('relation_correct','relation_count'),
        first_terminal_accuracy=('first_terminal_correct_count','chain_count'),
        terminal_missing_rate=('terminal_missing_count','chain_count'),length_mae=('length_absolute_error_sum','length_compared_count'))
    for name,(numerator,denominator) in fractions.items():
        result[name] = values[numerator]/values[denominator] if values[denominator] else None
    return result


def cpu_macro(sources, protocol):
    result = {name:sum(source[name] for source in sources.values()) for name in protocol.COUNTS}
    for name in protocol.METRICS:
        values = [source[name] for source in sources.values() if source[name] is not None]
        result[name] = sum(values)/len(values) if values else None; result[name+'_sources'] = len(values)
    return result


def cpu_mask_mean(levels, protocol):
    result = {}
    for name in protocol.TOKEN_METRICS:
        values = [level[name] for level in levels.values() if level[name] is not None]
        result[name] = sum(values)/len(values) if values else None; result[name+'_levels'] = len(values)
    return result


def verify_evaluation(raw, reported, datasets, collate, metrics, base, *, packets=None, eval_batch=16):
    """Independently count CPU masks, then recompute aggregates from raw totals."""
    import torch
    levels = {'source_macro':{}, 'token_micro':{}}
    expected_keys = {f'eval_mask{int(r*100)}_{source}' for r in RATIOS for source in datasets}
    require(set(raw) == expected_keys, 'Missing source/mask raw totals')
    population = {}; oracle_totals = {}; maximum_error = 0.
    if packets is not None:
        require(set(packets) == expected_keys, 'Missing CPU recount source/mask packets')
    for ratio in RATIOS:
        level = 'mask'+str(int(ratio*100)); summaries = {}; total_rows = []
        for source, records in datasets.items():
            prefix = 'eval_'+level+'_'+source
            counts = dict(selected_count=0, aa_count=0, terminal_count=0, relation_count=0, chain_count=0)
            for record in records:
                batch = collate([record])
                selected = independent_mask(batch, ratio)
                aa, end = selected & batch['residue_mask'], selected & batch['terminal_mask']
                counts['selected_count'] += int(selected.sum()); counts['aa_count'] += int(aa.sum())
                counts['terminal_count'] += int(end.sum()); counts['relation_count'] += int((selected & ~(aa|end)).sum())
                if ratio == 1.:
                    counts['chain_count'] += int(batch['chain_ids'][batch['target_slot_mask']].unique().numel())
            totals = raw[prefix]
            metrics.summarize(totals)  # Reuse only its input consistency rejection.
            summary = cpu_summary(totals,metrics)
            if packets is not None:
                expected_batches = [records[start:start+eval_batch] for start in range(0,len(records),eval_batch)]
                require(len(packets[prefix]) == len(expected_batches), 'CPU recount batch count differs')
                token = collate.tokenizer
                parts = [recount_packet(collate(rows), packet, ratio, token.special_id('<protd>'),
                         [token.special_id('<binding>'),token.special_id('<nonbinding>')],token.vocab_size)
                         for rows,packet in zip(expected_batches,packets[prefix])]
                oracle = [sum(part[i] for part in parts) for i in range(len(metrics.FIELDS))]
                require(len(totals) == len(oracle), 'CPU recount total width differs')
                for index,(actual,expected) in enumerate(zip(totals,oracle)):
                    if index in (0,2,4,13):
                        require(math.isfinite(actual) and math.isclose(actual,expected,rel_tol=TOTAL_RTOL,abs_tol=TOTAL_ATOL),
                                'CPU recount loss sum differs')
                        maximum_error = max(maximum_error,abs(actual-expected))
                    else:
                        require(type(actual) in (int,float) and actual == expected, 'CPU recount count differs')
                oracle_totals[prefix] = oracle
            require(all(summary[name] == value for name,value in counts.items()), 'Evaluation population differs from CPU masks')
            for name, value in summary.items():
                assert_close(reported[prefix+'_'+name], value, 'Per-source metric differs from raw totals')
            summaries[source] = summary; total_rows.append(torch.tensor(totals, dtype=torch.float64))
            population[prefix] = counts
        aggregate = dict(source_macro=cpu_macro(summaries,metrics),
                         token_micro=cpu_summary([sum(float(row[i]) for row in total_rows) for i in range(len(metrics.FIELDS))],metrics))
        for weighting, values in aggregate.items():
            levels[weighting][level] = values
            for name, value in values.items():
                assert_close(reported['eval_'+level+'_'+weighting+'_'+name], value, 'Macro/micro aggregation differs')
        for name,value in aggregate['source_macro'].items():
            assert_close(reported['eval_'+level+'_'+name],value,'Full/source alias differs')
    for weighting, by_level in levels.items():
        for name, value in cpu_mask_mean(by_level,metrics).items():
            assert_close(reported['eval_mask_mean_'+weighting+'_'+name], value, 'Mask mean differs')
    require(reported['eval_records_per_mask'] == sum(map(len,datasets.values()))
            and reported['eval_record_evaluations'] == 4*sum(map(len,datasets.values())), 'Evaluation record count differs')
    for name in metrics.METRICS:
        assert_close(reported['eval_'+name], reported['eval_mask100_'+name], 'Full-mask endpoint alias differs')
    return dict(populations=population, totals=oracle_totals, max_loss_sum_error=maximum_error,
                counts_exact=packets is not None)


def exercise(model, settings, training, validation, collate, *, optimizer_steps, global_batch, oracle=False,
             reference_gradients=None, progress_path=None,progress_plan=None,progress_stage='train_optimizer_step',
             lifetime_references=None):
    """Actual inherited HF loops. CPU callers exercise mechanics, never issue GPU proof."""
    import numpy as np
    import torch
    from transformers import Trainer, TrainerCallback
    from scripts.architecture_prototypes import validation_metrics as metrics
    from scripts import run_immune_research as base
    device = next(model.parameters()).device
    accum = settings.gradient_accumulation_steps; micro = settings.per_device_train_batch_size
    require(micro*accum == global_batch and len(training) >= optimizer_steps*global_batch
            and len(training) % global_batch == 0, 'Only full finite windows are supported')
    require(settings.max_steps == 10000 and settings.optim.value == 'adamw_torch', 'Optimizer/horizon differs')
    observations, windows, replay = [], [], []
    active = [True]; first_step = [True]; oracle_result = {}
    train_losses = []; raw_eval = {}; eval_batches = []; eval_details = []; precision_observations = []
    packets = {}; current_eval = [None]; last_logits = [None]; preclip = []; reference_match = {}
    captured_gradients = {}; step_initial = [{}]
    callback_counts = dict(prediction_steps=0, logs=0, evaluations=0)

    class ObservedTrainer(metrics.TokenValidationTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if not model.training:
                eval_details.append(observed_batch(inputs))
                last_logits[0] = None
            result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            if not model.training:
                require(last_logits[0] is not None and current_eval[0] is not None, 'Missing observed native denoise logits')
                packet = evaluation_packet(inputs,last_logits[0],self._validation_ratio)
                packets[current_eval[0]].append(packet); last_logits[0] = None
            return result

        def evaluation_loop(self, *args, metric_key_prefix='eval', **kwargs):
            require(metric_key_prefix not in raw_eval, 'Duplicate evaluation source/mask')
            current_eval[0] = metric_key_prefix; packets[metric_key_prefix] = []
            try:
                result = super().evaluation_loop(*args, metric_key_prefix=metric_key_prefix, **kwargs)
                raw_eval[metric_key_prefix] = self._last_token_validation_totals.tolist()
                return result
            finally:
                current_eval[0] = None

    trainer = ObservedTrainer(model=model, args=settings, train_dataset=training,
        eval_dataset=validation, data_collator=collate)
    if lifetime_references is not None:
        lifetime_references.update(model=weakref.ref(model),trainer=weakref.ref(trainer),
            accelerator=weakref.ref(trainer.accelerator),callback_handler=weakref.ref(trainer.callback_handler))
    require(type(trainer).train is Trainer.train and type(trainer).training_step is Trainer.training_step,
            'Training must use actual inherited HF methods')
    require(trainer.model_accepts_loss_kwargs is False and trainer.compute_loss_func is None
            and trainer.accelerator.gradient_accumulation_steps == 1 and settings.world_size == 1,
            'Unsupported accumulation/runtime contract')
    require(not trainer.is_deepspeed_enabled and not trainer.is_fsdp_enabled, 'Single ordinary Trainer only')

    original_denoise = model.denoise
    def observed_denoise(**features):
        result = original_denoise(**features)
        if not model.training and active[0]:
            last_logits[0] = result[0].detach().cpu()
        return result
    model.denoise = observed_denoise

    original_clip = trainer.accelerator.clip_grad_norm_
    def observe_clip(parameters, max_norm, norm_type=2):
        parameters = list(parameters)
        gradients = {name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
        require(gradients and all(torch.isfinite(g).all() for g in gradients.values()), 'Missing/nonfinite preclip gradient')
        norm = sum(float(g.double().square().sum()) for g in gradients.values())**.5
        require(norm > 0, 'Preclip gradients are all zero')
        preclip.append(norm)
        if len(preclip) == 1 and reference_gradients is not None:
            reference_match.update(gradient_comparison(gradients,reference_gradients))
        return original_clip(parameters,max_norm,norm_type)
    trainer.accelerator.clip_grad_norm_ = observe_clip

    def before_forward(module, args, batch):
        if not active[0]:
            return
        if module.training:
            observations.append(observed_batch(batch))
            if oracle and first_step[0]:
                replay.append(dict(batch={k:v.detach().cpu().clone() for k,v in batch.items()},
                                   rng_before=rng_state(device),rng_after_sha256=None))

    def after_forward(module, args, batch, output):
        if active[0] and module.training:
            loss = float(output['loss'].detach())
            require(math.isfinite(loss), 'Nonfinite training loss')
            train_losses.append(loss)
            if oracle and first_step[0]:
                replay[-1]['rng_after_sha256'] = rng_digest(rng_state(device))

    # Evaluation uses denoise rather than forward: observe features via the first
    # token embedding, without replacing denoise or passing labels as features.
    def embedding_shape(module, args):
        if not model.training and args and hasattr(args[0], 'shape'):
            eval_batches.append(list(args[0].shape))

    def head_precision(module, args, output):
        if active[0]:
            item = dict(phase='train' if model.training else 'eval',
                autocast_enabled=torch.is_autocast_enabled(device.type),
                autocast_dtype=str(torch.get_autocast_dtype(device.type)), output_dtype=str(output.dtype))
            if item not in precision_observations:
                precision_observations.append(item)

    handles = [model.register_forward_pre_hook(before_forward, with_kwargs=True),
               model.register_forward_hook(after_forward, with_kwargs=True),
               model.backbone.transformer.wte.register_forward_pre_hook(embedding_shape),
               model.backbone.transformer.ff_out.register_forward_hook(head_precision)]

    class Observe(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            start = state.global_step*accum
            current = observations[start:start+accum]
            require(len(current) == accum and all(x['input_shape'][0] == micro for x in current), 'Incomplete Trainer window')
            gradients = {name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
            require(gradients and all(torch.isfinite(g).all() for g in gradients.values()), 'Missing/nonfinite gradient')
            norm = sum(float(g.double().square().sum()) for g in gradients.values())**.5
            require(norm > 0, 'All gradients are zero')
            windows.append(dict(step=state.global_step+1, microsteps=len(current), records=sum(x['input_shape'][0] for x in current),
                gradient_parameters=len(gradients), finite_gradients=True, observed_gradient_norm=norm,
                nonzero_gradients=True, preclip_gradient_norm=preclip[-1] if preclip else norm,
                learning_rate=trainer.optimizer.param_groups[0]['lr'],
                mean_micro_loss=sum(train_losses[start:start+accum])/accum))
            step_initial[0] = {name:p.detach().cpu().clone() for name,p in model.named_parameters()}
            if oracle and first_step[0]:
                require(args.max_grad_norm == 0 and len(replay) == accum, 'Oracle requires an unclipped complete first window')
                saved = rng_state(device); active[0] = False
                model.zero_grad(set_to_none=True)
                try:
                    manual_losses = []; rng_matches = []
                    for item in replay:
                        restore_rng(item['rng_before'], device)
                        transferred = {k:v.to(device) for k,v in item['batch'].items()}
                        with trainer.accelerator.autocast():
                            loss = independent_training_loss(model,transferred)
                        manual_losses.append(float(loss.detach())); (loss/accum).backward()
                        rng_matches.append(rng_digest(rng_state(device)))
                        require(rng_matches[-1] == item['rng_after_sha256'], 'Independent replay RNG differs')
                    reference = {}
                    for name, parameter in model.named_parameters():
                        require((parameter.grad is not None) == (name in gradients), 'Oracle gradient membership differs')
                        if parameter.grad is None:
                            continue
                        reference[name] = parameter.grad.detach().cpu().clone()
                    oracle_result.update(gradient_comparison(gradients,reference))
                    require(all(math.isclose(a,b,rel_tol=LOSS_RTOL,abs_tol=LOSS_ATOL)
                                for a,b in zip(train_losses[:accum],manual_losses)), 'Independent training CE differs')
                    oracle_result.update(microsteps=accum,
                        records=global_batch, clipping_disabled_only_in_oracle=True,
                        method='independent_manual_corruption_logsoftmax_ce_micro_mean_ga_replay',
                        native_losses=train_losses[:accum],manual_losses=manual_losses,
                        rng_replay_equal=True,native_rng_after_sha256=[item['rng_after_sha256'] for item in replay],
                        native_rng_before_sha256=[rng_digest(item['rng_before']) for item in replay],
                        replay_rng_after_sha256=rng_matches,
                        tolerance=dict(atol=GRAD_ATOL,rtol=GRAD_RTOL,relative_l2=GRAD_RTOL,scale=GRAD_RTOL))
                    captured_gradients.update(gradients)
                finally:
                    for name, parameter in model.named_parameters():
                        parameter.grad = gradients[name].to(device) if name in gradients else None
                    restore_rng(saved, device); active[0] = True; first_step[0] = False
                replay.clear()

        def on_optimizer_step(self,args,state,control,model=None,optimizer=None,**kwargs):
            changes = [float((p.detach().cpu()-step_initial[0][name]).double().square().sum())
                       for name,p in model.named_parameters()]
            window = windows[-1]; window['parameter_delta_l2'] = sum(changes)**.5
            window['adamw_state_parameters'] = len(optimizer.state)
            window['adamw_state_bytes'] = sum(v.numel()*v.element_size() for values in optimizer.state.values()
                for v in values.values() if torch.is_tensor(v))
            step_initial[0] = {}

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= optimizer_steps:
                control.should_training_stop = True
            if progress_path is not None:
                progress = dict(optimizer_steps=state.global_step,microsteps=len(observations),
                    records=sum(batch['input_shape'][0] for batch in observations),windows=windows)
                if progress_plan is not None:
                    progress.update(oracle=oracle_result,first_window_preclip_match=reference_match)
                    record_stage(Path(progress_path).parent,progress_plan,progress_stage,'running',
                        compact_stage_statistics(progress_stage,progress),time.monotonic()-train_started)
                else:
                    write_json(progress_path,dict(schema=SCHEMA,status='running',stage=progress_stage,
                    optimizer_steps=state.global_step,microsteps=len(observations),
                    records=sum(batch['input_shape'][0] for batch in observations),windows=windows))
            return control

        def draw(self):
            random.random(); np.random.random(); torch.rand(1)
            if device.type == 'cuda':
                torch.rand(1, device=device)

        def on_prediction_step(self,args,state,control,**kwargs):
            callback_counts['prediction_steps'] += 1; self.draw()

        def on_log(self,args,state,control,**kwargs):
            if not model.training:
                callback_counts['logs'] += 1; self.draw()

        def on_evaluate(self, args, state, control, **kwargs):
            callback_counts['evaluations'] += 1; self.draw()
            return control

    observer = Observe()
    trainer.add_callback(observer)
    if lifetime_references is not None:
        lifetime_references['observer'] = weakref.ref(observer)
    initial = {name:p.detach().cpu().clone() for name,p in model.named_parameters()}
    initial_sha = tensor_digest(initial)
    train_started = time.monotonic()
    before = memory_start(device)
    try:
        result = trainer.train()
        train_memory = memory_end(device, before)
        require(trainer.state.global_step == optimizer_steps and len(windows) == optimizer_steps
                and len(observations) == optimizer_steps*accum, 'Trainer did not complete exact full windows')
        changes = {name:float((p.detach().cpu()-initial[name]).abs().max()) for name,p in model.named_parameters()}
        require(oracle or any(value > 0 for value in changes.values()), 'No actual parameter update')
        require(all(p.dtype == torch.float32 for p in model.parameters()), 'Parameters are not FP32')
        optimizer = trainer.optimizer.optimizer if hasattr(trainer.optimizer,'optimizer') else trainer.optimizer
        require(isinstance(optimizer, torch.optim.AdamW), 'Actual optimizer is not AdamW')
        if lifetime_references is not None:
            lifetime_references.update(optimizer_wrapper=weakref.ref(trainer.optimizer),optimizer=weakref.ref(optimizer),
                scheduler=weakref.ref(trainer.lr_scheduler),parameter=weakref.ref(next(model.parameters())),
                moment=weakref.ref(next(iter(optimizer.state.values()))['exp_avg']))
            if hasattr(model,'_original_forward'):
                lifetime_references['amp_original_forward'] = weakref.ref(model._original_forward)
        adam_steps = []; state_count = 0
        for state in optimizer.state.values():
            require(set(state) >= {'step','exp_avg','exp_avg_sq'}, 'Missing AdamW moments')
            adam_steps.append(int(state['step'])); state_count += 1
            require(all(state[key].dtype == torch.float32 and state[key].device == device and torch.isfinite(state[key]).all()
                        for key in ('exp_avg','exp_avg_sq')), 'AdamW states are nonfinite or not FP32')
        require(state_count > 0 and set(adam_steps) == {optimizer_steps}, 'AdamW step counter differs')
        summary = dict(optimizer_steps=optimizer_steps, microsteps=len(observations),
            records=sum(batch['input_shape'][0] for batch in observations),windows=windows, batches=observations,
            parameter_max_abs_changes=changes, adamw=dict(state_parameters=state_count, steps=sorted(set(adam_steps)),
                dtype='torch.float32',state_bytes=sum(value.numel()*value.element_size() for state in optimizer.state.values()
                    for value in state.values() if torch.is_tensor(value)),
                state_tensors=sum(torch.is_tensor(value) for state in optimizer.state.values() for value in state.values())),
            parameter_dtype='torch.float32',initial_parameters_sha256=initial_sha,
            final_parameters_sha256=tensor_digest(dict(model.named_parameters())),
            accelerator_accumulation=1, model_accepts_loss_kwargs=False,
            compute_loss_func=None, scheduler_horizon=trainer.state.max_steps,
            scheduler_last_epoch=trainer.lr_scheduler.last_epoch, learning_rates=trainer.lr_scheduler.get_last_lr(),
            training_loss=float(result.training_loss), memory=train_memory, oracle=oracle_result,
            first_window_preclip_match=reference_match, native_train=True,native_training_step=True,
            trainer_base='validation_metrics.TokenValidationTrainer',
            observer_overrides=['compute_loss_batch_observation_only','evaluation_loop_raw_totals_only'],
            instrumentation=['forward_input_loss_hooks','denoise_return_observation_for_cpu_recount',
                             'delegating_accelerator_clip_grad_norm_preclip_observer','optimizer_callbacks'],
            precision_observations=precision_observations,
            model_class=type(model).__module__+'.'+type(model).__name__,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            actual_training_settings=dict(micro_batch=micro,grad_accum=accum,max_steps=settings.max_steps,
                warmup_steps=settings.warmup_steps,learning_rate=settings.learning_rate,
                lr_scheduler_type=settings.lr_scheduler_type.value,max_grad_norm=settings.max_grad_norm,
                optimizer=settings.optim.value,dataloader_num_workers=settings.dataloader_num_workers,
                dataloader_drop_last=settings.dataloader_drop_last,seed=settings.seed,data_seed=settings.data_seed))
        summary['test_saving_disabled'] = settings.save_strategy.value == 'no'
        require(not list(Path(settings.output_dir).glob('checkpoint-*')), 'Diagnostic unexpectedly saved checkpoints')
        del initial
        if not oracle:
            if progress_path is not None and progress_plan is not None:
                record_stage(Path(progress_path).parent,progress_plan,progress_stage,'training_observed',
                    compact_stage_statistics(progress_stage,summary),time.monotonic()-train_started)
                record_stage(Path(progress_path).parent,progress_plan,'four_mask_evaluation','running')
            state_before = rng_state(device); before = memory_start(device)
            evaluated = trainer.evaluate()
            eval_memory = memory_end(device, before); state_after = rng_state(device)
            require(rng_digest(state_before) == rng_digest(state_after), 'Evaluation/callback advanced RNG')
            populations = verify_evaluation(raw_eval,evaluated,validation,collate,metrics,base,
                packets=packets,eval_batch=settings.per_device_eval_batch_size)
            summary['evaluation'] = dict(metrics=evaluated, raw_totals=raw_eval, fields=list(metrics.FIELDS),
                cpu_recount_oracle=populations,cpu_recount_packets=packets,
                actual_embedding_input_shapes=eval_batches, memory=eval_memory,callback_counts=callback_counts,
                actual_batches=eval_details, adamw_state_parameters=len(optimizer.state),
                adamw_state_bytes=summary['adamw']['state_bytes'],
                rng_before_sha256=rng_digest(state_before), rng_after_sha256=rng_digest(state_after),
                rng_preserved=True, source_records={name:len(rows) for name,rows in validation.items()},
                optimizer_state_resident=True)
            if progress_path is not None and progress_plan is not None:
                record_stage(Path(progress_path).parent,progress_plan,'four_mask_evaluation','evaluation_observed',
                    compact_stage_statistics('four_mask_evaluation',summary['evaluation']))
        if oracle:
            summary['_reference_gradients'] = captured_gradients
        return summary
    finally:
        for handle in handles:
            handle.remove()
        model.denoise = original_denoise
        trainer.accelerator.clip_grad_norm_ = original_clip


def verify_gradient_statistics(result):
    require(result.get('passed') is True and result.get('parameters'), 'Missing gradient comparison statistics')
    sums = dict(native_square=0.,reference_square=0.,error_square=0.,dot=0.)
    for parameter in result['parameters'].values():
        require(parameter.get('allclose') is True and type(parameter.get('max_abs_error')) in (int,float)
                and math.isfinite(parameter['max_abs_error']) and parameter['max_abs_error'] >= 0,
                'Invalid per-parameter gradient comparison')
        for name in sums:
            value = parameter.get(name)
            require(type(value) in (int,float) and math.isfinite(value)
                    and (name == 'dot' or value >= 0),'Invalid gradient sufficient statistic')
            sums[name] += value
    require(sums['reference_square'] > 0,'Zero reference gradients')
    for name,value in sums.items():
        require(result.get(name) == value,'Gradient aggregate statistics differ')
    scale = sums['dot']/sums['reference_square']; relative = (sums['error_square']/sums['reference_square'])**.5
    require(result.get('gradient_scale') == scale and result.get('relative_l2_error') == relative
            and abs(scale-1) <= GRAD_RTOL and relative <= GRAD_RTOL,'GA gradient magnitude differs')


def validate_worker_report(report, plan, sources):
    """Strict raw numerical/source checks; never publishes acceptance markers."""
    require(report.get('schema') == SCHEMA and report.get('status') == 'completed'
            and report.get('plan') == plan and report.get('sources') == sources
            and report.get('cuda_execution') is True, 'Worker identity/source/CUDA evidence differs')
    require(report.get('artificial_inputs_only') is True and report.get('biological_data_read') is False,
            'Producer is artificial-only')
    require(report.get('source_after') == sources and report.get('checkpoint_written') is False
            and report.get('data_gate_executed') is False and report.get('biological_quality_claim') is False,
            'Runtime source/diagnostic scope differs')
    validate_lifetime_preflight(report['lifetime_preflight'],plan)
    require(report.get('additional_lifetime_preflight_optimizer_steps') == 1
            and report.get('additional_lifetime_preflight_manual_microsteps') == 8,
            'Additional tiny lifetime workload differs')
    oracle, train = report['gradient_oracle'], report['formal_training']
    validate_cuda_cleanup(oracle['cleanup_observations']); validate_cuda_cleanup(train['cleanup_observations'])
    require(oracle['oracle'].get('passed') is True and oracle['oracle']['records'] == GLOBAL_BATCH
            and oracle['oracle']['tolerance'] == plan['gradient_tolerance']
            and oracle['oracle']['clipping_disabled_only_in_oracle'] is True, 'Gradient oracle is missing')
    verify_gradient_statistics(oracle['oracle']); verify_gradient_statistics(train['first_window_preclip_match'])
    reference = oracle['oracle']
    require(reference.get('method') == 'independent_manual_corruption_logsoftmax_ce_micro_mean_ga_replay'
            and reference.get('rng_replay_equal') is True
            and reference.get('replay_rng_after_sha256') == reference.get('native_rng_after_sha256')
            and len(reference['native_rng_before_sha256']) == plan['grad_accum']
            and len(reference['manual_losses']) == len(reference['native_losses']) == plan['grad_accum']
            and all(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=LOSS_RTOL,abs_tol=LOSS_ATOL)
                    for a,b in zip(reference['manual_losses'],reference['native_losses'])),
            'Independent CE/RNG replay is incomplete')
    require(oracle['optimizer_steps'] == 1 and train['optimizer_steps'] == STEPS
            and train['scheduler_horizon'] == 10000 and train['scheduler_last_epoch'] == STEPS,
            'Actual optimizer/scheduler counts differ')
    require(oracle['batches'] == train['batches'][:plan['grad_accum']]
            and report.get('first_window_inputs_match') is True,
            'Oracle/formal first-window inputs or sampling differ')
    require(oracle['initial_parameters_sha256'] == train['initial_parameters_sha256']
            and train['initial_parameters_sha256'] != train['final_parameters_sha256'],
            'Scratch initialization/native parameter update differs')
    require(train['actual_training_settings'] == dict(micro_batch=plan['micro_batch'],grad_accum=plan['grad_accum'],
        max_steps=10000,warmup_steps=100,learning_rate=1e-4,lr_scheduler_type='cosine',max_grad_norm=1.,
        optimizer='adamw_torch',dataloader_num_workers=2,dataloader_drop_last=False,seed=42,data_seed=42),
        'Formal training settings differ from the original pilot')
    require(train['accelerator_accumulation'] == 1 and train['model_accepts_loss_kwargs'] is False
            and train['compute_loss_func'] is None and train['native_train'] is True
            and train['native_training_step'] is True and train['test_saving_disabled'] is True,
            'HF scaling/native training/test-saving protocol differs')
    require(type(train['microsteps']) is int and train['microsteps'] == 24
            and type(train['records']) is int and train['records'] == 768, 'Actual record/microstep counts differ')
    require(len(train['windows']) == STEPS and len(train['batches']) == STEPS*plan['grad_accum'], 'Missing actual windows')
    for i, window in enumerate(train['windows'], 1):
        require(window['step'] == i and window['records'] == GLOBAL_BATCH
                and window['microsteps'] == plan['grad_accum'] and window['finite_gradients'] is True
                and window.get('nonzero_gradients') is True
                and math.isfinite(window['observed_gradient_norm']) and window['observed_gradient_norm'] > 0
                and math.isfinite(window['preclip_gradient_norm']) and window['preclip_gradient_norm'] > 0
                and type(window['adamw_state_parameters']) is int and window['adamw_state_parameters'] > 0
                and type(window['adamw_state_bytes']) is int and window['adamw_state_bytes'] > 0
                and math.isfinite(window['parameter_delta_l2']) and window['parameter_delta_l2'] >= 0,
                'Incomplete/nonfinite optimizer window')
    require(train['windows'][0]['learning_rate'] == 0
            and any(window['parameter_delta_l2'] > 0 for window in train['windows'][1:]),
            'Original warmup or later AdamW updates differ')
    expected_shape = [plan['micro_batch'],plan['config']['max_length']]
    capacities = {name:plan['config'][name] for name in ('full_capacity','fragment_capacity')}
    require(all(batch['input_shape'] == expected_shape and batch['target_capacities'] == capacities
                and batch['device'].startswith('cuda') for batch in train['batches']), 'Full configured training shape not observed')
    require(train['parameter_dtype'] == 'torch.float32' and train['adamw']['dtype'] == 'torch.float32'
            and train['adamw']['steps'] == [STEPS] and train['adamw']['state_parameters'] > 0
            and train['adamw']['state_bytes'] > 0 and train['adamw']['state_tensors'] >= 3*train['adamw']['state_parameters'],
            'FP32 AdamW state evidence differs')
    changes = train['parameter_max_abs_changes'].values()
    require(all(math.isfinite(v) and v >= 0 for v in changes) and any(v > 0 for v in changes), 'No finite parameter update')
    evaluation = train['evaluation']
    require(evaluation['rng_preserved'] is True and evaluation['rng_before_sha256'] == evaluation['rng_after_sha256']
            and evaluation['optimizer_state_resident'] is True, 'Evaluation RNG/state contract differs')
    require(evaluation['adamw_state_bytes'] == train['adamw']['state_bytes']
            and evaluation['cpu_recount_oracle']['counts_exact'] is True
            and evaluation['callback_counts'] == dict(prediction_steps=12,logs=1,evaluations=1),
            'Evaluation resident AdamW/recount/callback evidence differs')
    require([plan['eval_batch'],plan['config']['max_length']] in evaluation['actual_embedding_input_shapes'],
            'Full configured evaluation shape not observed')
    full_eval = [batch for batch in evaluation['actual_batches']
                 if batch['input_shape'] == [plan['eval_batch'],plan['config']['max_length']]]
    require(len(full_eval) == 4 and all(batch['target_capacities'] == capacities
                and batch['device'].startswith('cuda') for batch in full_eval)
            and evaluation['adamw_state_parameters'] == train['adamw']['state_parameters'],
            'Full evaluation capacities/resident optimizer state not observed')
    require(set(evaluation['source_records']) == {'procedural_mixed','full_capacity'}
            and evaluation['source_records'] == dict(procedural_mixed=7,full_capacity=17),
            'Unequal artificial sources required')
    require(len(evaluation['actual_batches']) == 12 and any(batch['padding_tokens'] > 0 for batch in evaluation['actual_batches']),
            'Actual uneven evaluation batches/padding were not observed')
    for phase in ('train','eval'):
        observed = [item for item in train['precision_observations'] if item['phase'] == phase]
        require(observed and all(item['autocast_enabled'] is True and item['autocast_dtype'] == 'torch.bfloat16'
                and item['output_dtype'] == 'torch.bfloat16' for item in observed), 'Actual BF16 path differs')
    for memory in (train['memory'], evaluation['memory']):
        require(memory.get('cuda_measured') is True and type(memory['peak_allocated_bytes']) is int
                and 0 < memory['peak_allocated_bytes'] <= .85*memory['total_memory_bytes']
                and memory['peak_reserved_bytes'] >= memory['peak_allocated_bytes'], 'GPU memory margin failed')
    for batch in [*train['batches'],*evaluation['actual_batches']]:
        require(type(batch['input_sha256']) is str and len(batch['input_sha256']) == 64
                and set(batch['tensors']) == {'input_ids','labels','attention_mask','role_ids','scope_ids','chain_ids',
                    'position_ids','target_slot_mask','eligible_mask','residue_mask','terminal_mask'},
                'Actual tensor observation is incomplete')
        for name,tensor in batch['tensors'].items():
            expected_dtype = 'torch.bool' if name.endswith('_mask') else 'torch.int64'
            require(tensor['shape'] == batch['input_shape'] and tensor['device'] == batch['device']
                    and tensor['dtype'] == expected_dtype,'Actual CUDA tensor shape/device/dtype differs')
    require(report['cleanup'] == dict(retained_checkpoints=[],measured_model_and_optimizer_released=True,
            oracle_model_released_before_formal=True,final_cuda_allocated_bytes=0)
            and report['additional_oracle_optimizer_steps'] == 1
            and report['additional_oracle_manual_microsteps'] == plan['grad_accum'],
            'Additional reference work or cleanup evidence differs')
    # Complete JSON finite check includes all raw metric totals, losses and errors.
    json.dumps(report, allow_nan=False)


def validate_lifetime_preflight(value,plan):
    """Tiny CUDA lifecycle evidence is additional work, never the full acceptance workload."""
    fixed = plan['lifetime_preflight']; cfg = fixed['config']
    require(value.get('device','').startswith('cuda') and value['memory'].get('cuda_measured') is True,
            'Lifetime preflight requires actual CUDA observations; CPU is not GPU proof')
    require(value.get('kind') == fixed['kind'] and value.get('model_config') == cfg
            and value.get('model_variant') == VARIANT and value.get('counts_toward_formal') is False
            and value.get('whole_layer_checkpointing') == 'whole_layer', 'Tiny lifetime identity/config differs')
    require(type(value['optimizer_steps']) is int and value['optimizer_steps'] == fixed['optimizer_steps']
            and type(value['microsteps']) is int and value['microsteps'] == fixed['grad_accum']
            and type(value['records']) is int and value['records'] == fixed['global_batch']
            and value['scheduler_horizon'] == 10000 and value['scheduler_last_epoch'] == 1
            and value['native_train'] is True and value['native_training_step'] is True,
            'Tiny lifetime native complete window differs')
    expected_settings = dict(micro_batch=1,grad_accum=8,max_steps=10000,warmup_steps=100,learning_rate=1e-4,
        lr_scheduler_type='cosine',max_grad_norm=0.,optimizer='adamw_torch',dataloader_num_workers=2,
        dataloader_drop_last=False,seed=42,data_seed=42)
    require(value['actual_training_settings'] == expected_settings and value['test_saving_disabled'] is True
            and value['accelerator_accumulation'] == 1 and value['model_accepts_loss_kwargs'] is False
            and value['compute_loss_func'] is None,'Tiny lifetime original Trainer/AdamW settings differ')
    require(len(value['windows']) == 1 and len(value['batches']) == 8,'Missing tiny lifetime complete window')
    window = value['windows'][0]
    require(window['step'] == 1 and window['microsteps'] == 8 and window['records'] == 8
            and window['finite_gradients'] is True and window['nonzero_gradients'] is True
            and math.isfinite(window['observed_gradient_norm']) and window['observed_gradient_norm'] > 0
            and math.isfinite(window['preclip_gradient_norm']) and window['preclip_gradient_norm'] > 0
            and window['learning_rate'] == 0 and window['parameter_delta_l2'] == 0
            and window['adamw_state_parameters'] > 0 and window['adamw_state_bytes'] > 0,
            'Tiny lifetime gradient/state/warmup observations differ')
    require(value['parameter_dtype'] == value['adamw']['dtype'] == 'torch.float32'
            and value['adamw']['steps'] == [1] and value['adamw']['state_parameters'] > 0
            and value['adamw']['state_bytes'] > 0
            and value['initial_parameters_sha256'] == value['final_parameters_sha256'],
            'Tiny lifetime FP32 state or original zero-LR first step differs')
    oracle = value['oracle']; verify_gradient_statistics(oracle)
    require(oracle['records'] == 8 and oracle['microsteps'] == 8 and oracle['tolerance'] == plan['gradient_tolerance']
            and oracle['clipping_disabled_only_in_oracle'] is True and oracle['rng_replay_equal'] is True
            and oracle['method'] == 'independent_manual_corruption_logsoftmax_ce_micro_mean_ga_replay'
            and oracle['replay_rng_after_sha256'] == oracle['native_rng_after_sha256']
            and len(oracle['native_rng_before_sha256']) == 8
            and len(oracle['manual_losses']) == len(oracle['native_losses']) == 8
            and all(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=LOSS_RTOL,abs_tol=LOSS_ATOL)
                    for a,b in zip(oracle['manual_losses'],oracle['native_losses'])),
            'Tiny lifetime independent CE/GA/RNG differs')
    capacities = {name:cfg[name] for name in ('full_capacity','fragment_capacity')}
    for batch in value['batches']:
        require(batch['input_shape'] == [1,64] and batch['target_capacities'] == capacities
                and batch['device'] == value['device'],'Tiny lifetime actual CUDA input shape differs')
        require(set(batch['tensors']) == {'input_ids','labels','attention_mask','role_ids','scope_ids','chain_ids',
                    'position_ids','target_slot_mask','eligible_mask','residue_mask','terminal_mask'},
                'Tiny lifetime actual input tensor set differs')
        for name,tensor in batch['tensors'].items():
            require(tensor['shape'] == [1,64] and tensor['device'] == value['device']
                    and tensor['dtype'] == ('torch.bool' if name.endswith('_mask') else 'torch.int64'),
                    'Tiny lifetime actual CUDA input dtype differs')
    precision = value['precision_observations']
    require(precision and all(item['phase'] == 'train' and item['autocast_enabled'] is True
            and item['autocast_dtype'] == 'torch.bfloat16' and item['output_dtype'] == 'torch.bfloat16'
            for item in precision),'Tiny lifetime actual BF16 path differs')
    validate_cuda_cleanup(value['cleanup_observations'])
    require(math.isfinite(value['phase_elapsed_seconds']) and 0 <= value['phase_elapsed_seconds'] <= LIFETIME_SECONDS,
            'Tiny lifetime phase exceeded fixed budget')
    from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
    expected = artificial_records(ImmuneConfig(**cfg),fixed['dataset_records'])
    require(value['fixture_sha256'] == digest([row.to_dict() for row in expected]),'Tiny lifetime artificial fixture differs')


def run_lifetime_preflight(context,output,plan,device):
    """One fixed tiny native window. CPU callers may observe mechanics, never satisfy CUDA validation."""
    import torch
    from transformers.trainer_utils import IntervalStrategy
    from dllm.pipelines.immune_llada.research_refined.model import RefinedModel
    from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
    from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
    fixed = plan['lifetime_preflight']; started = time.monotonic(); references = {}; result = None
    initial = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
    record_stage(output,plan,LIFETIME_PHASE,'starting',dict(actual_allocated_before_model_bytes=initial))
    cfg = ImmuneConfig(**fixed['config']); rows = artificial_records(cfg,fixed['dataset_records'])
    validation = dict(procedural_mixed=artificial_records(cfg,7,stress=False,offset=100),
                      full_capacity=artificial_records(cfg,3,offset=200))
    request = SimpleNamespace(variant=VARIANT,output_dir=output/'hf_lifetime_preflight')
    settings = context.pilot.training_arguments(request,dict(micro_batch=1,grad_accum=8))
    settings.save_strategy = settings.eval_strategy = settings.logging_strategy = IntervalStrategy.NO
    settings.max_grad_norm = 0.
    torch.manual_seed(42)
    model = RefinedModel(cfg,VARIANT)
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    model = model.to(device)
    try:
        result = exercise(model,settings,rows,validation,context.pilot.base.collator_for(cfg),
            optimizer_steps=1,global_batch=8,oracle=True,progress_path=output/'producer_progress.json',
            progress_plan=plan,progress_stage=LIFETIME_PHASE,lifetime_references=references)
        result.pop('_reference_gradients')
        result.update(kind=fixed['kind'],model_config=asdict(model.config),model_variant=model.variant,
            device=str(device),counts_toward_formal=False,
            whole_layer_checkpointing=model.backbone.activation_checkpointing_strategy.value,
            fixture_sha256=digest([row.to_dict() for row in rows]))
    finally:
        del model
        cleanup = cleanup_stage(output,plan,LIFETIME_PHASE,references,device,initial)
        if result is not None:
            result.update(cleanup_observations=cleanup,phase_elapsed_seconds=time.monotonic()-started)
    return result


def worker(context, output):
    """Called only in the fixed isolated child. Does not emit a GPU COMPLETE."""
    import torch
    from accelerate import Accelerator
    from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
    pilot = context.pilot; plan = fixed_plan(context); started = time.monotonic()
    sources_before = source_evidence(context); installed_before = installed_sources(context)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
            'Exactly one visible BF16 CUDA device is required')
    require(pilot.runtime_gpu_name() == context.profile['gpu'], 'CUDA device differs from original model profile')
    device = torch.device('cuda', 0); properties = torch.cuda.get_device_properties(device)
    require(getattr(properties, 'uuid', None) is not None, 'GPU UUID is unavailable')
    preflight = run_lifetime_preflight(context,output,plan,device)
    observed_phase_budget(output,plan,LIFETIME_PHASE,preflight)
    validate_lifetime_preflight(preflight,plan)
    fixture,training,validation,collate = generated_fixture(plan)
    provenance = dict(micro_batch=plan['micro_batch'],grad_accum=plan['grad_accum'])
    request = SimpleNamespace(variant=VARIANT, output_dir=output/'hf_artificial')
    def run_stage(oracle, reference=None):
        phase = 'gradient_oracle' if oracle else 'train_optimizer_step'
        references = {}
        before_model = torch.cuda.memory_allocated(device)
        record_stage(output,plan,phase,'starting',dict(actual_allocated_before_model_bytes=before_model))
        model = pilot.make_model(request)
        require(model.variant == VARIANT and asdict(model.config) == plan['config']
                and model.options == pilot.VARIANTS[VARIANT], 'Factory returned a different model')
        settings = pilot.training_arguments(request, provenance)
        from transformers.trainer_utils import IntervalStrategy
        settings.save_strategy = IntervalStrategy.NO
        settings.eval_strategy = IntervalStrategy.NO
        settings.logging_strategy = IntervalStrategy.NO
        if oracle:
            settings.max_grad_norm = 0.
        stage_result = None
        try:
            stage_result = exercise(model,settings,training,validation,collate,
                optimizer_steps=1 if oracle else STEPS,global_batch=GLOBAL_BATCH,oracle=oracle,
                reference_gradients=reference,progress_path=output/'producer_progress.json',
                progress_plan=plan,progress_stage=phase,lifetime_references=references)
        finally:
            del model
            cleanup = cleanup_stage(output,plan,phase,references,device,before_model)
            if stage_result is not None:
                stage_result['cleanup_observations'] = cleanup
        return stage_result
    oracle = run_stage(True)
    reference = oracle.pop('_reference_gradients')
    observed_phase_budget(output,plan,'gradient_oracle',oracle)
    validate_cuda_cleanup(oracle['cleanup_observations'])
    allocated = torch.cuda.memory_allocated(device)
    require(allocated == 0, f'Oracle left GPU tensors resident before formal measurement: {allocated} allocated bytes')
    formal = run_stage(False,reference)
    del reference
    observed_phase_budget(output,plan,'train_optimizer_step',formal)
    observed_phase_budget(output,plan,'four_mask_evaluation',formal['evaluation'])
    stable(context)
    sources = source_evidence(context)
    installed_after = installed_sources(context)
    require(sources == sources_before and installed_after == installed_before, 'Full runtime/framework sources changed')
    require(time.monotonic()-started <= TIMEOUT,'Producer exceeded fixed total budget')
    report = dict(schema=SCHEMA,status='completed',plan=plan,sources=sources_before,source_after=sources,cuda_execution=True,
        artificial_inputs_only=True, biological_data_read=False, data_gate_executed=False,
        training_kind='three_window_artificial_runtime_check', biological_quality_claim=False,
        first_window_inputs_match=oracle['batches'] == formal['batches'][:plan['grad_accum']],
        fixture=fixture,lifetime_preflight=preflight, gradient_oracle=oracle, formal_training=formal,
        gpu=dict(name=properties.name, uuid=str(properties.uuid), total_memory_bytes=properties.total_memory),
        runtime_versions=pilot.software_versions(), cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        installed_source_sha256=installed_before,installed_source_after_sha256=installed_after,
        python_version=sys.version,numpy_version=__import__('numpy').__version__,
        torch_initial_seed=42,checkpoint_written=False,oracle_model_released_before_formal=True,
        additional_oracle_optimizer_steps=1,additional_oracle_manual_microsteps=plan['grad_accum'],
        additional_lifetime_preflight_optimizer_steps=1,additional_lifetime_preflight_manual_microsteps=8,
        elapsed_seconds=time.monotonic()-started,
        cleanup=dict(retained_checkpoints=[],measured_model_and_optimizer_released=True,
                     oracle_model_released_before_formal=True,final_cuda_allocated_bytes=torch.cuda.memory_allocated(device)))
    validate_worker_report(report, plan, sources)
    write_json(output/'gpu_release_raw.json', report)
    return report



def summary_from_report(context, report, raw_sha):
    """Old executor contract plus separately validated full proof extensions."""
    plan = report['plan']; train = report['formal_training']; evaluation = train['evaluation']; p = context.pilot
    full_eval = next(batch for batch in evaluation['actual_batches']
                     if batch['input_shape'] == [plan['eval_batch'],plan['config']['max_length']])
    phases = dict(train_optimizer_step=dict(status='completed',input_shape=train['batches'][0]['input_shape'],
        target_capacities=train['batches'][0]['target_capacities'], **train['memory'],
        losses=[w['mean_micro_loss'] for w in train['windows']],optimizer_steps=STEPS,finite_gradients=True),
        four_mask_evaluation=dict(status='completed',input_shape=full_eval['input_shape'],
            target_capacities=full_eval['target_capacities'],**evaluation['memory'],
            losses=[evaluation['metrics'][f'eval_mask{int(r*100)}_token_micro_loss'] for r in RATIOS],
            mask_levels=list(RATIOS),trainer='validation_metrics.TokenValidationTrainer'))
    training = {key:value for key,value in train.items() if key != 'evaluation'}
    evaluated = {key:value for key,value in evaluation.items() if key != 'cpu_recount_packets'}
    sources = report['sources']
    extension = dict(schema=EXTENSION_SCHEMA,plan=plan,plan_sha256=digest(plan),sources=sources,
        source_after=report['source_after'],runtime_manifest_sha256=sources['runtime_manifest_sha256'],
        producer_sha256=sources['producer_sha256'],wrapper_sha256=sources['wrapper_sha256'],
        fixture_sha256=digest(report['fixture']),raw_sha256=raw_sha,training=training,evaluation=evaluated,
        gradient_oracle=report['gradient_oracle'],lifetime_preflight=report['lifetime_preflight'],
        additional_lifetime_preflight_optimizer_steps=report['additional_lifetime_preflight_optimizer_steps'],
        additional_lifetime_preflight_manual_microsteps=report['additional_lifetime_preflight_manual_microsteps'],
        execution=dict(producer_return_ready=True,checkpoint_written=False))
    summary = dict(schema=context.execution.GPU_SCHEMA,kind='full_pilot_runtime_gpu_acceptance',variant=VARIANT,
        config=plan['config'],options=p.VARIANTS[VARIANT],source_binding=context.sources,metric_protocol=p.metric_protocol(),
        profile_complete_sha256=sha(context.profile_dir/'COMPLETE.json'),
        profile_summary_sha256=sha(context.profile_dir/'summary.json'),runtime_versions=p.software_versions(),
        gpu=context.profile['gpu'],global_batch=GLOBAL_BATCH,micro_batch=plan['micro_batch'],grad_accum=plan['grad_accum'],
        eval_batch=plan['eval_batch'],precision='fp32_parameters_adam_bf16_autocast',optimizer='adamw_torch',
        objective='unchanged model.forward selected-token CE per microbatch; same variant options',
        cuda_execution=True,phases=phases,proof_extension=extension,
        biological_quality_claim=False,data_certificate_issued=False)
    context.execution.verify_gpu_phases(phases,summary)
    return summary


def generated_fixture(plan):
    from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
    from scripts import run_immune_research as base
    config = ImmuneConfig(**plan['config'])
    training = artificial_records(config,STEPS*GLOBAL_BATCH)
    validation = {'procedural_mixed':artificial_records(config,7,stress=False,offset=1000),
                  'full_capacity':artificial_records(config,plan['eval_batch']+1,offset=2000)}
    collate = base.collator_for(config)
    order = list(base.EpochRandomSampler(training,42))
    expected_batches = [tensor_digest(collate([training[i] for i in order[start:start+plan['micro_batch']]]))
                        for start in range(0,len(order),plan['micro_batch'])]
    fixture = dict(generator_sha256=sha(Path(__file__)),training_records=len(training),
        validation_records={k:len(v) for k,v in validation.items()},
        training_sha256=digest([row.to_dict() for row in training]),
        validation_sha256={name:digest([row.to_dict() for row in rows]) for name,rows in validation.items()},
        train_epoch_batch_sha256=expected_batches,
        biological_data_read=False,procedural_only=True)
    return fixture,training,validation,collate


def verify_candidate(directory, profile_dir, runtime_root):
    """Read-only strict verification; cannot grant CUDA acceptance or write markers."""
    context = runtime_context(profile_dir,runtime_root)
    directory = Path(directory)
    require(directory.is_absolute() and directory == directory.resolve() and directory.is_dir()
            and not directory.is_relative_to(ROOT),'Candidate output must be canonical and outside runtime')
    raw_path = directory/'gpu_release_raw.json'; summary_path = directory/'gpu_release_summary.json'
    report = json.loads(raw_path.read_text()); summary = json.loads(summary_path.read_text())
    success = json.loads((directory/'PRODUCER_SUCCESS.json').read_text())
    plan,sources = fixed_plan(context),source_evidence(context)
    validate_worker_report(report,plan,sources)
    fixture,training,validation,collate = generated_fixture(plan)
    require(report['fixture'] == fixture,'Artificial fixture generator/content changed')
    require([batch['input_sha256'] for batch in report['formal_training']['batches']]
            == fixture['train_epoch_batch_sha256'], 'Observed full epoch artificial inputs differ from native sampler')
    evaluation = report['formal_training']['evaluation']
    recount = verify_evaluation(evaluation['raw_totals'],evaluation['metrics'],validation,collate,
        context.pilot.metrics,context.pilot.base,packets=evaluation['cpu_recount_packets'],eval_batch=plan['eval_batch'])
    require(recount == evaluation['cpu_recount_oracle'],'Stored CPU recount differs from independent recomputation')
    require(report['installed_source_sha256'] == installed_sources(context)
            and report['installed_source_after_sha256'] == report['installed_source_sha256'],
            'Installed HF/Accelerate/model/optimizer source changed')
    require(report['runtime_versions'] == context.pilot.software_versions()
            and report['gpu']['name'] == context.profile['gpu'],'GPU/software profile changed')
    require(summary == summary_from_report(context,report,sha(raw_path)),
            'GPU summary extension/raw/contract binding differs')
    expected_success = dict(schema=EXTENSION_SCHEMA,status='candidate_ready',summary_sha256=sha(summary_path),
        raw_sha256=sha(raw_path),plan_sha256=digest(plan),source_binding=context.sources)
    require(success == expected_success,'Candidate success/hash binding differs')
    stable(context)
    return summary


def produce(profile_dir, output_dir, runtime_root=None):
    """One fixed CUDA child; only a normal-exit supervising wrapper may accept it."""
    context = runtime_context(profile_dir,runtime_root)
    output = Path(output_dir)
    require(output.is_absolute() and output == output.resolve() and not output.is_relative_to(ROOT)
            and (not output.exists() or output.is_dir()),'Use canonical output outside immutable runtime')
    require(not output.is_symlink() and not any((output/name).exists() for name in ARTIFACTS),
            'Producer artifacts already exist; never overwrite candidate evidence')
    output.mkdir(parents=True,exist_ok=True)
    stage = 'gpu_worker'
    try:
        report = worker(context,output)
        stage = 'candidate_validation'
        raw_path = output/'gpu_release_raw.json'
        summary = summary_from_report(context,report,sha(raw_path))
        write_json(output/'gpu_release_summary.json',summary)
        write_json(output/'PRODUCER_SUCCESS.json',dict(schema=EXTENSION_SCHEMA,status='candidate_ready',
            summary_sha256=sha(output/'gpu_release_summary.json'),raw_sha256=sha(raw_path),
            plan_sha256=digest(report['plan']),source_binding=context.sources))
        verify_candidate(output,context.profile_dir,ROOT)
        write_json(output/'producer_progress.json',dict(schema=SCHEMA,status='candidate_ready',
            optimizer_steps=3,microsteps=24,records=768,raw_bytes=raw_path.stat().st_size))
        return summary
    except BaseException as error:
        (output/'PRODUCER_SUCCESS.json').unlink(missing_ok=True)
        failed_stage = error.phase if isinstance(error,PhaseBudgetExceeded) else stage
        stage_path = output/STAGE_EVIDENCE
        if stage_path.exists():
            partial = json.loads(stage_path.read_text())
            progress = json.loads((output/'producer_progress.json').read_text())
            phase = error.phase if isinstance(error,PhaseBudgetExceeded) else progress.get('stage')
            if phase in partial['phase_seconds']:
                record_stage(output,fixed_plan(context),phase,
                    'budget_failed' if isinstance(error,PhaseBudgetExceeded) else 'execution_failed',
                    dict(error_type=type(error).__name__,message=str(error)),
                    error.elapsed_seconds if isinstance(error,PhaseBudgetExceeded) else None)
                failed_stage = phase
        write_json(output/'FAILED.json',dict(schema=SCHEMA,status='failed',stage=stage,
            failed_stage=failed_stage,
            category=('budget_exceeded' if isinstance(error,PhaseBudgetExceeded) else
                'oom' if type(error).__name__ == 'OutOfMemoryError' else 'validation_or_execution_failure'),
            error_type=type(error).__name__,message=str(error),smaller_batch_fallback=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--profile-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--runtime-root',type=Path,required=True)
    args = parser.parse_args()
    produce(args.profile_dir,args.output_dir,args.runtime_root)


if __name__ == '__main__':
    main()
