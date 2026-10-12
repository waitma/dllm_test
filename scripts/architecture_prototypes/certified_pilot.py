"""Undeployed adapter for a future, independently certified corpus pilot.

Activate pllm, then run CPU interface tests (no training):
  USE_TF=0 python -m pytest scripts/tests/immune_llada/test_certified_pilot.py -q
API: preflight(PilotRequest(...)); build_trainer(PilotRequest(...)).
There is deliberately no training CLI, submission, certificate writer, checkpoint
loader execution, or training loop. The existing double-certificate assert_ready
runs before profile/selection/model/GPU/output work. Model memory evidence and
this still-undeployed metric protocol have distinct source bindings.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import ConcatDataset, Subset
from transformers import TrainingArguments

from examples.llada.immune_pretrain_v3 import assert_ready, PilotStop
from scripts import run_immune_research as base
from scripts import run_immune_refined as refined
from scripts import enqueue_immune_refined as profile_checks
from scripts import enqueue_immune_architecture as json_checks
from scripts.architecture_prototypes import validation_selection as selections
from scripts.architecture_prototypes import validation_metrics as metrics
from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.immune_llada.v3.vocabulary import make_tokenizer
from dllm.pipelines.immune_llada.research_refined.model import VARIANTS, BASE_VARIANTS

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PilotRequest:
    variant: str
    data_dir: Path
    selection_path: Path
    profile_dir: Path
    output_dir: Path
    steps: int = 1000
    resume_from_checkpoint: Path | None = None


@dataclass(frozen=True)
class PilotPreflight:
    request: PilotRequest
    manifest: dict
    selection: dict
    profile: dict
    provenance: dict
    completed_checkpoint_step: int | None


@dataclass(frozen=True)
class PilotTrainerBundle:
    preflight: PilotPreflight
    trainer: object
    training_started: bool = False


def require(condition, message):
    if not condition:
        raise ValueError(message)


def protocol_sources():
    """Dynamic file hashes: metric fields are owned solely by validation_metrics."""
    files = (Path(__file__), Path(selections.__file__), Path(metrics.__file__),
             Path(profile_checks.__file__), Path(json_checks.__file__))
    return {str(path.resolve().relative_to(ROOT)): base.sha(path.resolve()) for path in files}


def metric_protocol():
    # The source digest is authoritative even when the metric module evolves.
    return dict(implementation='validation_metrics.TokenValidationTrainer',
                descriptor=deepcopy(metrics.METRIC_PROTOCOL),
                source_sha256=base.sha(Path(metrics.__file__)))


def software_versions():
    return {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate')}


def validate_pilot_resume(checkpoint, provenance, requested_step):
    """Reuse legacy complete-state/provenance checks, then enforce staged resume."""
    checkpoint = Path(checkpoint).resolve()
    completed = base.validate_research_resume(checkpoint, provenance, requested_step)
    saved = json_checks.read_json(checkpoint/'run_provenance.json')
    prior_stop = saved['pilot_stop_step']
    state = json_checks.read_json(checkpoint/'trainer_state.json')
    require(type(state.get('global_step')) is int and type(prior_stop) is int,
            'Checkpoint steps must be exact integers')
    if requested_step == 1000:
        require(prior_stop == 1000 and completed < 1000, 'Only an interrupted first-stage pilot can resume to 1000')
    elif requested_step == 5000:
        require((prior_stop == 1000 and completed == 1000) or
                (prior_stop == 5000 and 1000 <= completed < 5000),
                '5000 is cumulative: require checkpoint-1000 or an interrupted continuation')
    else:
        raise ValueError('Only cumulative 1000/5000 pilot stages are supported')
    require(checkpoint.name == f'checkpoint-{completed}', 'Checkpoint directory and saved step differ')
    for filename in ('immune_config.json', 'vocabulary.json'):
        require((checkpoint/filename).is_file(), 'Missing checkpoint contract: ' + filename)
    require(json_checks.read_json(checkpoint/'immune_config.json') == provenance['config'], 'Checkpoint model configuration differs')
    research = json_checks.read_json(checkpoint/'research_config.json')
    variant = provenance['variant']
    require(research == dict(variant=variant, parent_variant=BASE_VARIANTS[variant], options=VARIANTS[variant],
                             base=provenance['config'], schema='immune_architecture.refined.v1'),
            'Checkpoint architecture/schema differs')
    tokenizer = make_tokenizer('A4')
    require(json.loads((checkpoint/'vocabulary.json').read_text()) == [tokenizer.token(i) for i in range(tokenizer.vocab_size)],
            'Checkpoint vocabulary differs')
    return completed


def preflight(request: PilotRequest):
    """Read-only; the original double-certificate gate is always the first check."""
    data_dir = Path(request.data_dir).resolve()
    manifest, _ = assert_ready(data_dir)
    # No earlier profile, selection, GPU/model, output mkdir, or certificate edit.
    raw_output = Path(request.output_dir).absolute()
    require(not raw_output.exists() and not raw_output.is_symlink(), 'Use a fresh output directory; never overwrite a pilot')
    request = PilotRequest(request.variant, data_dir, Path(request.selection_path).resolve(),
        Path(request.profile_dir).resolve(), Path(request.output_dir).resolve(), request.steps,
        Path(request.resume_from_checkpoint).resolve() if request.resume_from_checkpoint is not None else None)
    require(request.variant in VARIANTS, 'Only the R20--R27 refined family is supported')
    require(type(request.steps) is int and request.steps in (1000, 5000), 'Only cumulative 1000/5000 pilot stages are supported')
    require(request.steps == 1000 or request.resume_from_checkpoint is not None, '5000 requires a resumable 1000-step pilot')
    require(os.environ.get('WORLD_SIZE', '1') == '1', 'Prototype requires single-process global batch 256')
    require(not request.output_dir.is_relative_to(data_dir), 'Output must be outside prepared data')
    require(not request.output_dir.exists(), 'Use a fresh output directory; never overwrite a pilot')
    if request.resume_from_checkpoint is not None:
        require(not request.output_dir.is_relative_to(request.resume_from_checkpoint), 'Output must not overwrite the resume checkpoint')
    require(request.profile_dir.name == request.variant and request.profile_dir.parent.name == 'profiles',
            'Expected the existing pool/profiles/variant profile location')
    manifest_sha = base.sha(data_dir/'dataset_manifest.json')
    selected = selections.load_verified_validation_selection(request.selection_path,
        expected_data_dir=data_dir, expected_manifest_sha256=manifest_sha)
    require(selected['config']['seed'] == 42 and selected['config']['max_records_per_source'] == 256,
            'Pilot requires the common seed42/up-to-256 selection')
    train_sources = sorted({item['source'] for item in manifest['splits']['train']['shards']})
    valid_sources = sorted({item['source'] for item in manifest['splits']['valid']['shards']})
    require(train_sources == valid_sources == sorted(selected['sources']), 'Training/validation source membership differs')
    require(all(item['selected']['records'] > 0 for item in selected['sources'].values()), 'Every validation source must have records')
    profile = profile_checks.verify_profile_evidence(request.profile_dir.parent.parent, request.variant)
    release_sources = refined.fingerprints()
    proof = json_checks.read_json(request.profile_dir/'COMPLETE.json')
    require(proof.get('source_sha256') == release_sources, 'Profile model release changed during preflight')
    config = asdict(ImmuneConfig(ablation='A4'))
    require(profile['config'] == config, 'Profile is not the production backbone/configuration')
    batch = profile['recommended_micro_batch']
    require(type(batch) is int and batch > 0 and 256 % batch == 0, 'Profile microbatch does not divide global256')
    provenance = dict(schema='immune.certified_pilot.prototype.v1', kind='certified_pilot_adapter_not_deployed',
        variant=request.variant, options=deepcopy(VARIANTS[request.variant]), config=config,
        data_dir=str(data_dir), data_manifest_sha256=manifest_sha,
        certificate_sha256=base.sha(data_dir/'audit_certificate.json'),
        scope_certificate_sha256=base.sha(data_dir/'downstream_scope_certificate.json'),
        selection_path=str(request.selection_path), selection_file_sha256=base.sha(request.selection_path),
        selection_sha256=selected['selection_sha256'],
        validation_indices={source: item['source_local_indices'] for source, item in selected['sources'].items()},
        selection_warnings=selected['warnings'], validation_uses_test=False,
        source_sha256=release_sources, adapter_source_sha256=protocol_sources(), metric_protocol=metric_protocol(),
        profile_dir=str(request.profile_dir), profile_complete_sha256=base.sha(request.profile_dir/'COMPLETE.json'),
        profile_summary_sha256=base.sha(request.profile_dir/'summary.json'), profile_gpu=profile['gpu'],
        profile_scope='existing model/training graph; new metric adapter is separately bound and not yet GPU released',
        runtime_versions=software_versions(), source_names=train_sources, training_sampling='natural',
        seed=42, data_seed=42, micro_batch=batch, grad_accum=256//batch, global_batch=256,
        max_steps=10000, pilot_stop_step=request.steps, learning_rate=1e-4, warmup_steps=100, scheduler='cosine',
        objective='unchanged model.forward selected-token CE per microbatch; same variant options',
        precision='fp32_parameters_adam_bf16_autocast', optimizer='adamw_torch',
        eval_steps=100, save_steps=100, save_total_limit=3, save_only_model=False,
        checkpoint_retention='unchanged HF limit 3; existing pool cleanup protects latest two, an actually tracked best, and queued/running resume references; adapter never deletes checkpoints')
    completed = None
    if request.resume_from_checkpoint is not None:
        completed = validate_pilot_resume(request.resume_from_checkpoint, provenance, request.steps)
    return PilotPreflight(request, manifest, selected, profile, provenance, completed)


def exact_validation_subsets(data_dir, selected):
    """Construct Subsets from verified saved indices, with no new random sampling."""
    require(str(Path(data_dir).resolve()) == selected['data_dir'], 'Selection dataset differs')
    result = {}
    for source, item in selected['sources'].items():
        data = PreparedImmuneDataset(data_dir, split='valid', source=source)
        indices = item['source_local_indices']
        require(len(data) == item['available']['records'] and len(indices) == item['selected']['records'] and
                len(set(indices)) == len(indices) and all(type(i) is int and 0 <= i < len(data) for i in indices),
                'Selected indices or available record count differ')
        result[source] = Subset(data, indices)
    return result


def training_arguments(request, provenance):
    """Pin the worker's AdamW implementation and preserve its original LR horizon."""
    return TrainingArguments(output_dir=str(request.output_dir),
        per_device_train_batch_size=provenance['micro_batch'], per_device_eval_batch_size=min(provenance['micro_batch'], 16),
        gradient_accumulation_steps=provenance['grad_accum'], max_steps=10000,
        learning_rate=1e-4, warmup_steps=100, lr_scheduler_type='cosine', optim='adamw_torch',
        seed=42, data_seed=42, bf16=True, remove_unused_columns=False, label_names=['labels'],
        eval_strategy='steps', eval_steps=100, save_strategy='steps', save_steps=100,
        save_total_limit=3, save_only_model=False, logging_steps=10, report_to='none',
        prediction_loss_only=True, dataloader_num_workers=2, dataloader_persistent_workers=False, disable_tqdm=True)


def runtime_gpu_name():
    require(torch.cuda.is_available(), 'Future trainer construction requires its measured GPU')
    return torch.cuda.get_device_name()


def make_model(request):
    # Reuse protocol-v2 initialization/RNG reset and registered refined factory.
    return refined.previous.make_model(SimpleNamespace(variant=request.variant, cpu_test=False))


def build_trainer(request: PilotRequest):
    """Gate and construct only; HF may initialize output after gates, never train."""
    checked = preflight(request)
    request, provenance = checked.request, checked.provenance
    require(runtime_gpu_name() == checked.profile['gpu'], 'Runtime GPU differs from memory profile')
    model = make_model(request)
    require(model.variant == request.variant and asdict(model.config) == provenance['config'] and
            model.options == provenance['options'], 'Constructed model differs from bound profile/options')
    training = ConcatDataset([PreparedImmuneDataset(request.data_dir, split='train', source=source)
                              for source in provenance['source_names']])
    validation = exact_validation_subsets(request.data_dir, checked.selection)
    settings = training_arguments(request, provenance)
    require(settings.world_size == 1, 'Prototype trainer requires single process')
    trainer = metrics.TokenValidationTrainer(model=model, args=settings, train_dataset=training,
        eval_dataset=validation, data_collator=base.collator_for(model.config))
    trainer.run_provenance = deepcopy(provenance)
    trainer.add_callback(PilotStop(request.steps))
    return PilotTrainerBundle(checked, trainer)
