"""Undeployed pilot lifecycle; existing data and future GPU-release gates are mandatory.

Activate pllm, then run artificial CPU interface tests only:
  USE_TF=0 python -m pytest scripts/tests/immune_llada/test_certified_pilot_execution.py -q
API: execute(PilotRequest(...), gpu_release_dir). If future independent evidence
and the original data gates pass, this API will train on GPU. It is undeployed:
no CLI, queue registration, submission, or certificate producer. This change was
validated only with artificial CPU tests; no GPU or corpus training was run.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from transformers import TrainerCallback

from scripts.architecture_prototypes import certified_pilot as pilot


GPU_SCHEMA = 'immune.certified_pilot_gpu_release.v1'
EXECUTION_SCHEMA = 'immune.certified_pilot_execution.v1'
CHECKPOINT_FILES = ('run_provenance.json', 'research_config.json', 'immune_config.json',
    'vocabulary.json', 'trainer_state.json', 'model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    # Serialize before touching disk: NaN cannot produce partial JSON evidence.
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def source_binding():
    return dict(model=pilot.refined.fingerprints(), adapter=pilot.protocol_sources(),
        execution={str(Path(__file__).resolve().relative_to(pilot.ROOT)): pilot.base.sha(Path(__file__))})


def invariant(provenance):
    return {key: value for key, value in provenance.items() if key != 'pilot_stop_step'}


def verify_gpu_phases(phases, provenance):
    """Validate measured shapes/peaks; this helper does not certify GPU execution."""
    p = provenance
    require(isinstance(phases, dict) and set(phases) == {'train_optimizer_step', 'four_mask_evaluation'},
            'Missing new train/eval GPU phases')
    for name, batch_size in (('train_optimizer_step', p['micro_batch']), ('four_mask_evaluation', min(p['micro_batch'],16))):
        phase = phases[name]
        shape, capacities = phase.get('input_shape'), phase.get('target_capacities')
        require(isinstance(shape,list) and all(type(value) is int for value in shape)
            and isinstance(capacities,dict) and all(type(value) is int for value in capacities.values())
            and phase.get('status') == 'completed' and shape == [batch_size,p['config']['max_length']]
            and capacities == {key:p['config'][key] for key in ('full_capacity','fragment_capacity')},
            'GPU phase lacks full configured input shape/capacity evidence')
        used, total = phase.get('peak_allocated_bytes'), phase.get('total_memory_bytes')
        require(type(used) is int and type(total) is int and 0 < used <= .85*total,
                'GPU phase exceeds the memory margin or lacks measured bytes')
        losses = phase.get('losses')
        require(isinstance(losses,list) and losses and all(type(value) in (int,float)
            and math.isfinite(value) and value >= 0 for value in losses), 'GPU phase lacks finite measured loss')
        if name == 'train_optimizer_step':
            require(type(phase.get('optimizer_steps')) is int and phase['optimizer_steps'] == len(losses)
                and phase.get('finite_gradients') is True, 'Missing actual optimizer/gradient evidence')
        else:
            require(phase.get('mask_levels') == [.25,.5,.75,1.] and len(losses) == 4
                and phase.get('trainer') == 'validation_metrics.TokenValidationTrainer', 'GPU evaluation protocol differs')


def verify_gpu_release(directory, checked, sources):
    """Read a future independent GPU acceptance; old memory profiles cannot pass.

    There is deliberately no producer here. CPU tests mock this interface for
    lifecycle wiring, never emit a passing GPU/data certificate.
    """
    directory = Path(directory).resolve()
    complete_path, summary_path = directory/'GPU_RELEASE_COMPLETE.json', directory/'gpu_release_summary.json'
    proof = pilot.json_checks.read_json(complete_path)
    summary = pilot.json_checks.read_json(summary_path)
    p = checked.provenance
    require(proof.get('schema') == GPU_SCHEMA and proof.get('status') == 'passed'
        and proof.get('summary_sha256') == pilot.base.sha(summary_path), 'Missing independent GPU release acceptance or bad summary SHA')
    contract = dict(schema=GPU_SCHEMA, kind='full_pilot_runtime_gpu_acceptance', variant=p['variant'],
        config=p['config'], options=p['options'], source_binding=sources, metric_protocol=pilot.metric_protocol(),
        profile_complete_sha256=p['profile_complete_sha256'], profile_summary_sha256=p['profile_summary_sha256'],
        runtime_versions=p['runtime_versions'], gpu=p['profile_gpu'], global_batch=256,
        micro_batch=p['micro_batch'], grad_accum=p['grad_accum'], eval_batch=min(p['micro_batch'],16),
        precision=p['precision'], optimizer=p['optimizer'], objective=p['objective'])
    require(all(summary.get(key) == value for key, value in contract.items()), 'GPU release source/runtime/objective contract differs')
    require(proof.get('source_binding') == sources and summary.get('cuda_execution') is True,
            'GPU acceptance must bind the complete current source maps and actual CUDA execution')
    verify_gpu_phases(summary.get('phases', {}), p)
    return dict(directory=str(directory), complete_sha256=pilot.base.sha(complete_path),
                summary_sha256=pilot.base.sha(summary_path), contract=contract)


@dataclass(frozen=True)
class ExecutionPreflight:
    checked: pilot.PilotPreflight
    provenance: dict


def verify_checkpoint_sidecar(checkpoint, provenance, step, pilot_provenance):
    checkpoint = Path(checkpoint)
    require(all((checkpoint/name).is_file() for name in CHECKPOINT_FILES), 'Incomplete full-state checkpoint')
    state = pilot.json_checks.read_json(checkpoint/'trainer_state.json')
    require(type(state.get('global_step')) is int and state['global_step'] == step
        and checkpoint.name == f'checkpoint-{step}', 'Checkpoint step identity differs')
    require(invariant(pilot.json_checks.read_json(checkpoint/'run_provenance.json')) == invariant(pilot_provenance),
        'Checkpoint base pilot provenance differs')
    sidecar = pilot.json_checks.read_json(checkpoint/'execution_provenance.json')
    require(type(sidecar.get('checkpoint_global_step')) is int and sidecar['checkpoint_global_step'] == step
        and invariant(sidecar.get('execution', {})) == invariant(provenance), 'Checkpoint execution/GPU-release sidecar differs')
    require(sidecar.get('checkpoint_sha256') == {name:pilot.base.sha(checkpoint/name) for name in CHECKPOINT_FILES},
            'Checkpoint file bytes changed after its sidecar was saved')


def preflight(request, gpu_release_dir):
    # This delegated preflight begins with the original unchanged assert_ready.
    checked = pilot.preflight(request)
    p = checked.provenance
    diagnostic = pilot.profile_checks.verify_diagnostic(checked.request.profile_dir.parent.parent,
        checked.request.variant, 500, require_learning=True)
    sources = source_binding()
    require(sources['model'] == p['source_sha256'] and sources['adapter'] == p['adapter_source_sha256'],
            'Sources changed after the base pilot preflight')
    release = verify_gpu_release(gpu_release_dir, checked, sources)
    provenance = dict(schema=EXECUTION_SCHEMA, source_binding=sources,
        pilot_invariant_sha256=digest(invariant(p)), pilot_stop_step=checked.request.steps,
        diagnostic500_payload_sha256=digest(diagnostic), gpu_release=release,
        metric_protocol=pilot.metric_protocol(), production_loss_changed=False)
    if checked.request.resume_from_checkpoint is not None:
        verify_checkpoint_sidecar(checked.request.resume_from_checkpoint, provenance,
            checked.completed_checkpoint_step, p)
    return ExecutionPreflight(checked, provenance)


class ExecutionCheckpointSidecar(TrainerCallback):
    def __init__(self, provenance, pilot_provenance):
        self.provenance, self.pilot_provenance = deepcopy(provenance), deepcopy(pilot_provenance)

    def on_save(self, args, state, control, **kwargs):
        directory = Path(args.output_dir)/f'checkpoint-{state.global_step}'
        require(all((directory/name).is_file() for name in CHECKPOINT_FILES), 'Checkpoint save omitted full-state files')
        require(pilot.json_checks.read_json(directory/'run_provenance.json') == self.pilot_provenance,
                'Saved pilot provenance differs')
        saved_state = pilot.json_checks.read_json(directory/'trainer_state.json')
        require(type(saved_state.get('global_step')) is int and saved_state['global_step'] == state.global_step,
                'Saved checkpoint step differs from callback state')
        write_json(directory/'execution_provenance.json',
            dict(execution=self.provenance, checkpoint_global_step=state.global_step,
                 checkpoint_sha256={name:pilot.base.sha(directory/name) for name in CHECKPOINT_FILES}))


def assert_stable(provenance):
    require(source_binding() == provenance['source_binding'], 'Source changed during pilot execution')
    release = provenance['gpu_release']
    directory = Path(release['directory'])
    require(pilot.base.sha(directory/'GPU_RELEASE_COMPLETE.json') == release['complete_sha256']
        and pilot.base.sha(directory/'gpu_release_summary.json') == release['summary_sha256'],
        'GPU release evidence changed during pilot execution')


def execute(request: pilot.PilotRequest, gpu_release_dir):
    """Gate, then execute an admitted pilot; never register or submit a job."""
    admitted = preflight(request, gpu_release_dir)  # Failure here never creates output.
    request = admitted.checked.request
    output, stage, trainer = request.output_dir, 'trainer_construction', None
    try:
        assert_stable(admitted.provenance)
        bundle = pilot.build_trainer(request)
        trainer = bundle.trainer
        require(bundle.preflight.provenance == admitted.checked.provenance, 'Pilot evidence changed during construction')
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'run_provenance.json', admitted.checked.provenance)
        write_json(output/'execution_provenance.json', admitted.provenance)
        write_json(output/'execution_request.json', dict(variant=request.variant,
            resume_from_checkpoint=str(request.resume_from_checkpoint) if request.resume_from_checkpoint else None,
            requested_cumulative_step=request.steps, validation_uses_test=False))
        trainer.add_callback(ExecutionCheckpointSidecar(admitted.provenance, admitted.checked.provenance))
        assert_stable(admitted.provenance)
        if request.resume_from_checkpoint is None:
            stage = 'initial_evaluation'
            initial = trainer.evaluate()
            digest(initial)  # Reject nonfinite JSON before persisting metrics.
            trainer.save_metrics('initial_eval', initial)
        stage = 'training'
        result = trainer.train(resume_from_checkpoint=str(request.resume_from_checkpoint) if request.resume_from_checkpoint else None)
        require(type(trainer.state.global_step) is int and trainer.state.global_step == request.steps,
                'Training did not reach the exact requested cumulative step')
        assert_stable(admitted.provenance)
        stage = 'final_evaluation'
        final = trainer.evaluate()
        digest(final); digest(result.metrics)
        trainer.save_metrics('final_eval', final)
        trainer.save_metrics('train', result.metrics)
        stage = 'final_save'
        trainer.save_model()
        trainer.save_state()
        checkpoint = output/f'checkpoint-{request.steps}'
        verify_checkpoint_sidecar(checkpoint, admitted.provenance, request.steps, admitted.checked.provenance)
        require(pilot.json_checks.read_json(output/'trainer_state.json').get('global_step') == request.steps,
                'Final saved state differs from the requested step')
        assert_stable(admitted.provenance)
        artifacts = ('run_provenance.json','execution_provenance.json','execution_request.json',
                     'trainer_state.json','model.safetensors','final_eval_results.json','train_results.json')
        complete = dict(schema=EXECUTION_SCHEMA, status='completed', variant=request.variant,
            global_step=request.steps, execution_provenance_sha256=pilot.base.sha(output/'execution_provenance.json'),
            artifact_sha256={name:pilot.base.sha(output/name) for name in artifacts},
            checkpoint=str(checkpoint), checkpoint_sidecar_sha256=pilot.base.sha(checkpoint/'execution_provenance.json'),
            production_loss_changed=False)
        write_json(output/'COMPLETE.json', complete)  # Last write; never emitted on failure.
        return complete
    except BaseException as error:
        if output.is_dir():
            (output/'COMPLETE.json').unlink(missing_ok=True)
            write_json(output/'FAILED.json', dict(schema=EXECUTION_SCHEMA, status='failed', stage=stage,
                error_type=type(error).__name__, message=str(error),
                global_step=getattr(getattr(trainer,'state',None),'global_step',None)))
        raise
