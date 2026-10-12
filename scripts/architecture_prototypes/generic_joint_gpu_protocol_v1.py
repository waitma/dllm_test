"""Fixed artificial-only GPU transport and continuous 500-step protocol.

Activate conda pllm; run the focused acceptance with:
python -m pytest scripts/tests/immune_llada/test_generic_joint_gpu_engine.py -q
This module uses only the standard library. It neither queues nor runs jobs.
Each training transport performs exactly 100 steps; five distinct transports
form one continuous series. No real corpus, arbitrary command, or resume CLI.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys

PROJECT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
POOL = PROJECT/'output/immune_research_pool_c20250601_20260930'
PACKAGE = PROJECT/'output/generic_joint_runtimes/generic_joint_gpu_v1_20261001'
ROUTE = 'generic_joint_gpu_v1'
METHODS = ('g00_shared_joint', 'g01_task_norm', 'g02_late_adapters', 'g03_register4',
           'g04_bottleneck4', 'g05_bottleneck16', 'g06_self_alignment',
           'g07_alignment_compute_control')
TOTAL_STEPS = 500
SEGMENT_STEPS = 100
WARMUP_STEPS = 25
PROFILE_MICROBATCHES = (16, 32, 64, 128)
SCHEDULE = {'kind': 'linear_warmup_cosine', 'total_steps': TOTAL_STEPS,
            'warmup_steps': WARMUP_STEPS, 'learning_rate': 0.0003,
            'gradient_clip_norm': 1.0}
FIXED_FILES = frozenset({
    'scripts/architecture_prototypes/generic_joint_data.py',
    'scripts/architecture_prototypes/generic_joint_model.py',
    'scripts/architecture_prototypes/generic_joint_experiment.py',
    'scripts/architecture_prototypes/record_loss.py',
    'scripts/architecture_prototypes/generic_joint_gpu_protocol_v1.py',
    'scripts/architecture_prototypes/generic_joint_gpu_engine_v1.py',
    'scripts/architecture_prototypes/generic_joint_gpu_entry_v1.py',
    *(f'train_jobs/immune_cv_joint_exploration_20261001/{method}.yml' for method in METHODS)})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    result = json.loads(path.read_text(), object_pairs_hook=pairs)
    json.dumps(result, allow_nan=False)
    return result


def evidence_file(path):
    require(path.is_absolute() and path == path.resolve()
            and path.is_file() and not path.is_symlink(), 'Evidence must be canonical and nonsymlink')
    return path


def execution_completion(work, result):
    """Worker execution receipt only; independent verification stays false."""
    require(result.get('transport') == work.transport and result.get('method') == work.method
            and result.get('cuda_measured') is True
            and result.get('real_corpus_training_started') is False
            and result.get('biological_quality_claim') is False,
            'Cannot write completion for a different workload or unmeasured execution')
    steps = result.get('actual_optimizer_steps')
    require(type(steps) is int and 0 < steps <= work.actual_step_budget, 'Invalid observed optimizer count')
    if work.kind == 'segment':
        require(steps == 100 and result.get('cumulative_start') == work.start_step
                and result.get('cumulative_stop') == work.stop_step, 'Incomplete segment')
    return {'schema': 'generic.joint_worker_execution_complete.v1',
            'status': 'execution_completed_independent_review_pending',
            'transport': work.transport, 'method': work.method,
            'actual_optimizer_steps': steps, 'independent_root_verified': False,
            'five_segments_independently_verified': False,
            'real_corpus_training_started': False, 'data_acceptance': False}


@dataclass(frozen=True)
class Workload:
    method: str
    kind: str
    start_step: int
    stop_step: int
    transport: str
    previous_transport: str | None

    @property
    def actual_step_budget(self):
        return len(PROFILE_MICROBATCHES) if self.kind == 'profile' else self.stop_step-self.start_step


def workloads():
    result = {}
    for method in METHODS:
        name = method+'__profile4'
        result[name] = Workload(method, 'profile', 0, 0, name, None)
        previous = None
        for start in range(0, TOTAL_STEPS, SEGMENT_STEPS):
            name = method+f'__s{start+1:03d}_{start+SEGMENT_STEPS:03d}'
            result[name] = Workload(method, 'segment', start, start+SEGMENT_STEPS, name, previous)
            previous = name
    return result


WORKLOADS = workloads()


def binding(method, *, micro_batch, package_sha256, plan_sha256, profile_job_id):
    require(method in METHODS and type(micro_batch) is int
            and micro_batch in PROFILE_MICROBATCHES, 'Unknown method or unmeasured microbatch')
    require(all(re.fullmatch('[a-f0-9]{64}', value or '') for value in
                (package_sha256, plan_sha256)), 'Missing fixed package/plan hash')
    require(type(profile_job_id) is str and bool(profile_job_id), 'Missing profile job identity')
    return {'schema': 'generic.joint_continuous_training_binding.v1', 'method': method,
            'seed': 42, 'global_batch': 128, 'micro_batch': micro_batch,
            'schedule': dict(SCHEDULE), 'package_sha256': package_sha256,
            'plan_sha256': plan_sha256, 'profile_job_id': profile_job_id,
            'data_kind': 'artificial_integer_objects_only', 'real_corpus_allowed': False,
            'external_pretrained_encoder': False}


def validate_binding(value):
    require(type(value) is dict, 'Training binding must be an object')
    expected = binding(value.get('method'), micro_batch=value.get('micro_batch'),
                       package_sha256=value.get('package_sha256'), plan_sha256=value.get('plan_sha256'),
                       profile_job_id=value.get('profile_job_id'))
    require(value == expected, 'Training identity, schedule or artificial-only scope differs')
    return value


def validate_transport(argv):
    names = {'--variant', '--mode', '--steps', '--profile-dir', '--output-dir'}
    options = [arg.partition('=')[0] for arg in argv if arg.startswith('--')]
    require(set(options) == names and len(options) == len(names), 'Exactly five unique transport options required')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in sorted(names):
        parser.add_argument(name, required=True, type=int if name == '--steps' else str)
    args = parser.parse_args(argv)
    require(args.variant in WORKLOADS and args.mode == 'profile-synthetic', 'Only fixed artificial GPU workloads')
    work = WORKLOADS[args.variant]
    require(args.steps == work.actual_step_budget, 'Transport must state the actual per-job optimizer budget')
    output, profile = Path(args.output_dir), Path(args.profile_dir)
    require(output.is_absolute() and profile.is_absolute()
            and output == output.resolve() and profile == profile.resolve(), 'Canonical absolute paths required')
    require(output.parent == POOL/'runs' and output.parent.is_dir() and not output.exists(), 'Fresh existing-pool run required')
    require(profile == POOL/'profiles'/work.transport, 'Profile path belongs to another workload')
    require(re.fullmatch(r'[0-9]+_'+re.escape(work.transport)+r'_[a-f0-9]{8}', output.name), 'Run identity differs')
    return args, work, output, profile


def validate_installed_entry(argv):
    args, work, output, profile = validate_transport(argv)
    require(Path(__file__).resolve().parents[2] == PACKAGE and PACKAGE == PACKAGE.resolve(), 'Fixed package not installed')
    marker = getattr(sys, '_immune_release_isolation', None)
    require(type(marker) is dict and marker.get('root') == str(PACKAGE)
            and marker.get('route') == ROUTE and marker.get('pid') == os.getpid(), 'Missing fixed startup isolation')
    require(re.fullmatch('[0-7]', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'Existing worker must assign one GPU')
    manifest_path = PACKAGE/'generic_gpu_manifest.json'
    require(manifest_path.is_file() and not manifest_path.is_symlink(), 'Missing source manifest')
    package_sha = sha(manifest_path)
    require(package_sha == os.environ.get('GENERIC_JOINT_PACKAGE_SHA256'), 'Dispatcher package binding differs')
    manifest = read_json(manifest_path)
    require(manifest.get('schema') == 'generic.joint_gpu_source_package.v1'
            and manifest.get('runtime_root') == str(PACKAGE)
            and manifest.get('status') == 'source_only_not_GPU_or_data_acceptance'
            and manifest.get('data_kind') == 'artificial_integer_objects_only'
            and manifest.get('real_corpus_allowed') is False, 'Source package scope differs')
    files = manifest.get('files_sha256')
    require(type(files) is dict and set(files) == FIXED_FILES, 'Fixed complete source map differs')
    actual = {str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob('*')
              if path.is_file() and path.name != 'generic_gpu_manifest.json'}
    require(actual == FIXED_FILES and not any(path.is_symlink() for path in PACKAGE.rglob('*')),
            'Package contains unrecorded files or aliases')
    for name, expected in files.items():
        path = PACKAGE/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and path.is_file() and not path.is_symlink() and path == path.resolve()
                and sha(path) == expected, 'Packaged source differs: '+name)
    for name in ('scripts', 'examples'):
        require(list(getattr(sys.modules.get(name), '__path__', [])) == [str(PACKAGE/name)], 'Namespace isolation differs')
    admission_path = POOL/'generic_admissions'/(output.name+'.json')
    admission = read_json(evidence_file(admission_path))
    require(admission.get('schema') == 'generic.joint_artificial_job_preparation.v1'
            and admission.get('job_id') == output.name
            and admission.get('transport') == work.transport
            and admission.get('package_sha256') == package_sha
            and admission.get('actual_steps') == args.steps
            and admission.get('real_corpus_allowed') is False, 'Job preparation identity differs')
    # This record is a general engineering binding, never a dataset certificate.
    return work, output, profile, manifest, admission
