"""Fixed runtime entry for an artificial Trainer check and admitted corpus pilots.

Activate pllm; CPU refusal/interface tests:
  USE_TF=0 python -m pytest scripts/tests/immune_llada/test_pilot_runtime_entry.py -q
The pool CLI accepts only the fixed R24 profile-synthetic/1 envelope. It runs a
bounded artificial producer, observes its real exit/cleanup, then verifies its
reports. execute_admitted(request, gpu_release_dir) keeps the original data gate
first. No resource submission, certificate creation, checkpoint deletion, or
arbitrary executable/target is supported. Install only in a verified runtime.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VARIANT = 'r24_refined_fixed_scale'
MODEL_VARIANT = 'r24_refined_fixed_scale'
RUNTIME_NAME = 'r24_pilot_v1_20261001'
BASE_SHA256 = '46399204c7018d00726e3ae08142bd4eb28475c8a9a2991b3a5c90725c8824f7'
SCHEMA = 'immune.pool_trainer_gpu_runtime.v1'
KIND = 'trainer_gpu_runtime_functional_diagnostic'
PROCESS_SCHEMA = 'immune.pilot_gpu_process_receipt.v1'
EXTENSION_SCHEMA = 'immune.pilot_gpu_release_producer.v1'
TIMEOUT_SECONDS = 3600
PRODUCER = 'scripts/architecture_prototypes/pilot_gpu_release_producer_r24_v1.py'
ENTRY = 'scripts/architecture_prototypes/pilot_runtime_entry_r24_v1.py'
MANIFEST = 'pilot_runtime_manifest.json'
ARTIFACTS = ('GPU_RELEASE_COMPLETE.json', 'gpu_release_summary.json',
             'gpu_release_raw.json', 'process_evidence.json', 'PRODUCER_SUCCESS.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or symlink JSON: '+path.name)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: '+key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    value = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=constant)
    json.dumps(value, allow_nan=False)
    require(type(value) is dict, 'Expected JSON object: '+path.name)
    return value


def write_json(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    temporary = path.with_name(path.name+'.tmp')
    require(not temporary.exists(), 'Unexpected temporary evidence file')
    temporary.write_text(text)
    temporary.replace(path)


def install_runtime_search(manifest):
    """Validate package bytes and search paths before any ML/project import."""
    require(ROOT == ROOT.resolve() and ROOT.name == RUNTIME_NAME,
            'This entry must run from the fixed immutable pilot runtime')
    original = Path(manifest.get('source_root', ''))
    require(original.is_absolute() and original == original.resolve()
            and ROOT == original/'output/immune_pilot_runtimes'/RUNTIME_NAME,
            'Runtime source identity differs')
    require(manifest.get('schema') == 'immune.pilot_runtime_source_package.v1'
            and manifest.get('runtime_root') == str(ROOT)
            and manifest.get('candidate_variant') == MODEL_VARIANT
            and manifest.get('package_name') == RUNTIME_NAME
            and manifest.get('base_manifest_sha256') == BASE_SHA256,
            'Runtime does not inherit the accepted R24 source archive')
    files = manifest.get('runtime_files_sha256')
    require(type(files) is dict and ENTRY in files and PRODUCER in files,
            'Runtime lacks the fixed entry or producer binding')
    for name, expected in files.items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Unsafe runtime source path')
        path = ROOT/name
        require(path == path.resolve() and path.is_file() and not path.is_symlink()
                and sha(path) == expected, 'Runtime source changed: '+name)
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in ('scripts', 'examples', 'dllm'):
            continue
        # The inherited startup guard may install these two empty namespaces.
        require(name in ('scripts', 'examples') and getattr(module, '__file__', None) is None
                and list(getattr(module, '__path__', [])) == [str(ROOT/name)],
                'A project module was loaded before runtime isolation: '+name)
    sys.path[:] = [str(ROOT), *(value for value in sys.path if value
        and Path(value).resolve() != ROOT and (not Path(value).resolve().is_relative_to(original)
        or Path(value).resolve().is_relative_to(ROOT)))]
    for name in ('scripts', 'examples'):
        directory = ROOT/name
        require(directory.is_dir() and not (directory/'__init__.py').exists(),
                'Expected fixed namespace: '+name)
        spec = ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(directory)]
        sys.modules[name] = module_from_spec(spec)
    return manifest


def load_runtime():
    manifest = install_runtime_search(strict_json(ROOT/MANIFEST))
    from scripts.architecture_prototypes import pilot_candidate_runtime_package_v1 as package
    require(package.verify_runtime(ROOT) == manifest, 'Runtime identity changed during isolation')
    from scripts.architecture_prototypes import certified_pilot_execution as execution
    from scripts.architecture_prototypes import pilot_gpu_release_producer_r24_v1 as producer
    from scripts.architecture_prototypes import cuda_attention_probe as processes
    # Only its audited generic process cleanup is used; no attention probe runs.
    return manifest, package, execution, producer, processes


def validate_transport(argv):
    allowed = {'--variant', '--mode', '--steps', '--profile-dir', '--output-dir'}
    options = [value.partition('=')[0] for value in argv if value.startswith('--')]
    require(set(options) == allowed and len(options) == len(allowed),
            'R24 requires exactly five unique transport options')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in sorted(allowed):
        parser.add_argument(name, required=True, type=int if name == '--steps' else str)
    args = parser.parse_args(argv)
    require(args.variant == VARIANT and args.mode == 'profile-synthetic' and args.steps == 1,
            'Only the fixed R24 profile-synthetic/1 invocation is accepted')
    output, profile = Path(args.output_dir), Path(args.profile_dir)
    require(output.is_absolute() and output == output.resolve()
            and profile.is_absolute() and profile == profile.resolve(), 'Use canonical absolute pool paths')
    require(output.parent.name == 'runs' and output.parent.is_dir() and not output.exists(),
            'R24 requires a fresh pool/runs output directory')
    require(profile == output.parent.parent/'profiles'/VARIANT,
            'R24 transport profile belongs to another pool')
    pool = output.parent.parent
    require((pool/'pool.json').is_file(), 'Uninitialized pool')
    return pool, output, pool/'profiles'/MODEL_VARIANT


def validate_observed_exit(receipt, command, elapsed):
    require(type(receipt) is dict and receipt.get('schema') == PROCESS_SCHEMA
            and receipt.get('argv') == command and type(receipt.get('returncode')) is int
            and receipt.get('returncode') == 0
            and receipt.get('cleanup_completed') is True
            and receipt.get('completion_observed_by_parent') is True,
            'Producer did not exit normally with verified descendant cleanup')
    require(type(elapsed) in (int, float) and math.isfinite(elapsed)
            and 0 <= elapsed <= TIMEOUT_SECONDS,
            'Producer exceeded its fixed total deadline')


def verified_candidate(directory, profile, manifest, producer, execution):
    candidate = producer.verify_candidate(directory, profile, ROOT)
    require(candidate == strict_json(Path(directory)/'gpu_release_summary.json'),
            'Candidate verifier returned different summary bytes')
    extension = candidate.get('proof_extension', {})
    require(extension.get('schema') == EXTENSION_SCHEMA
            and extension.get('runtime_manifest_sha256') == sha(ROOT/MANIFEST)
            and extension.get('producer_sha256') == manifest['runtime_files_sha256'][PRODUCER]
            and extension.get('wrapper_sha256') == manifest['runtime_files_sha256'][ENTRY],
            'GPU candidate is outside this exact runtime/producer/wrapper')
    require(candidate.get('schema') == execution.GPU_SCHEMA
            and candidate.get('variant') == MODEL_VARIANT
            and candidate.get('cuda_execution') is True
            and candidate.get('source_binding') == execution.source_binding(),
            'GPU candidate is not the current corrected R24 Trainer')
    training = extension.get('training', {})
    windows = training.get('windows', [])
    require(type(training.get('optimizer_steps')) is int and training['optimizer_steps'] == 3
            and type(training.get('microsteps')) is int and training['microsteps'] == 24
            and type(training.get('records')) is int and training['records'] == 768
            and type(windows) is list and len(windows) == 3
            and all(type(window.get('microsteps')) is int and window['microsteps'] == 8
                    and type(window.get('records')) is int and window['records'] == 256 for window in windows),
            'Missing three observed complete global256 optimizer windows')
    return candidate


def verify_completion(directory, profile, manifest, producer, execution):
    directory = Path(directory)
    summary = verified_candidate(directory, profile, manifest, producer, execution)
    proof = strict_json(directory/'GPU_RELEASE_COMPLETE.json')
    receipt = strict_json(directory/'process_evidence.json')
    expected_command = [sys.executable, '-u', str(ROOT/PRODUCER),
        '--profile-dir', str(profile), '--output-dir', str(directory), '--runtime-root', str(ROOT)]
    validate_observed_exit(receipt, expected_command, receipt.get('elapsed_seconds'))
    require(proof.get('schema') == execution.GPU_SCHEMA and proof.get('status') == 'passed'
            and proof.get('summary_sha256') == sha(directory/'gpu_release_summary.json')
            and proof.get('source_binding') == execution.source_binding()
            and proof.get('runtime_manifest_sha256') == sha(ROOT/MANIFEST)
            and proof.get('producer_success_sha256') == sha(directory/'PRODUCER_SUCCESS.json')
            and proof.get('raw_sha256') == sha(directory/'gpu_release_raw.json')
            and proof.get('process_evidence_sha256') == sha(directory/'process_evidence.json'),
            'GPU completion no longer matches producer/parent evidence')
    return summary


def run_artificial(argv):
    pool, output, profile = validate_transport(argv)
    manifest, package, execution, producer, processes = load_runtime()
    # All checks above are CPU/source checks. The child alone constructs a model.
    require(package.read_json(pool/'pool.json').get('source_root') == manifest['source_root'],
            'Pool belongs to another original source root')
    output.mkdir()
    stage = 'producer_execution'
    started = time.monotonic()
    try:
        write_json(output/'request.json', dict(schema=SCHEMA, kind=KIND,
            variant=VARIANT, transport=dict(mode='profile-synthetic', steps=1),
            runtime_root=str(ROOT), runtime_manifest_sha256=sha(ROOT/MANIFEST),
            profile_dir=str(profile), output_dir=str(output), timeout_seconds=TIMEOUT_SECONDS,
            expected_optimizer_windows=3, biological_quality_claim=False))
        command = [sys.executable, '-u', str(ROOT/PRODUCER), '--profile-dir', str(profile),
                   '--output-dir', str(output), '--runtime-root', str(ROOT)]
        with (output/'producer.log').open('w') as log:
            child = processes.run_case_process(command, stdout=log, timeout=TIMEOUT_SECONDS)
        elapsed = time.monotonic()-started
        receipt = dict(schema=PROCESS_SCHEMA, argv=command, returncode=child.returncode,
            cleanup_completed=True, completion_observed_by_parent=True, elapsed_seconds=elapsed,
            timeout_seconds=TIMEOUT_SECONDS, producer_log_sha256=sha(output/'producer.log'),
            process_supervision='audited same-session child process group with scoped Linux subreaper')
        write_json(output/'process_evidence.json', receipt)
        validate_observed_exit(receipt, command, elapsed)
        stage = 'candidate_verification'
        candidate = verified_candidate(output, profile, manifest, producer, execution)
        require(package.verify_runtime(ROOT) == manifest, 'Runtime changed during GPU execution')
        complete = dict(schema=execution.GPU_SCHEMA, status='passed',
            summary_sha256=sha(output/'gpu_release_summary.json'), source_binding=execution.source_binding(),
            runtime_manifest_sha256=sha(ROOT/MANIFEST), producer_success_sha256=sha(output/'PRODUCER_SUCCESS.json'),
            raw_sha256=sha(output/'gpu_release_raw.json'), process_evidence_sha256=sha(output/'process_evidence.json'))
        write_json(output/'GPU_RELEASE_COMPLETE.json', complete)
        stage = 'pool_presentation'
        training = candidate['proof_extension']['training']
        summary = dict(schema=SCHEMA, kind=KIND, variant=VARIANT,
            transport=dict(mode='profile-synthetic', steps=1), dataset_kind='artificial_trainer_fixture',
            biological_quality_claim=False, corpus_training_started=False, training_started=True,
            optimizer_window_count=3, completed_optimizer_window_count=training['optimizer_steps'],
            optimizer_windows=training['windows'], execution_completed=True, functional_gpu_passed=True,
            runtime_root=str(ROOT), runtime_manifest_sha256=sha(ROOT/MANIFEST),
            source_sha256=manifest['runtime_files_sha256'],
            artifact_sha256={name:sha(output/name) for name in ARTIFACTS},
            elapsed_seconds=time.monotonic()-started)
        write_json(output/'summary.json', summary)
        write_json(output/'COMPLETE.json', dict(schema=SCHEMA, kind=KIND, status='completed',
            variant=VARIANT, execution_completed=True, functional_gpu_passed=True,
            completed_optimizer_window_count=3, runtime_manifest_sha256=sha(ROOT/MANIFEST),
            summary_sha256=sha(output/'summary.json'), artifact_sha256=summary['artifact_sha256']))
        return summary
    except BaseException as error:
        (output/'COMPLETE.json').unlink(missing_ok=True)
        (output/'GPU_RELEASE_COMPLETE.json').unlink(missing_ok=True)
        write_json(output/'FAILED.json', dict(schema=SCHEMA, kind=KIND, status='failed',
            stage=stage, error_type=type(error).__name__, message=str(error),
            biological_quality_claim=False, corpus_training_started=False))
        raise


def execute_admitted(request, gpu_release_dir):
    """Original data gate first; then strengthen the original execution admission."""
    # This API is called from an already isolated verified source package. Never
    # read a real prepared dataset here before the unchanged original data gate.
    from scripts.architecture_prototypes import certified_pilot_execution as execution
    admitted = execution.preflight(request, gpu_release_dir)
    from scripts.architecture_prototypes import pilot_candidate_runtime_package_v1 as package
    from scripts.architecture_prototypes import pilot_gpu_release_producer_r24_v1 as producer
    manifest = package.verify_runtime(ROOT)
    require(request.variant == MODEL_VARIANT,
            'This candidate admits only its own measured model variant')
    verify_completion(gpu_release_dir, admitted.checked.request.profile_dir, manifest, producer, execution)
    return execution.execute(request, gpu_release_dir)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--help']:
        print('Fixed R24 artificial Trainer GPU acceptance. Real pilots use execute_admitted after original gates.')
        return 0
    try:
        run_artificial(argv)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print('Trainer runtime rejected: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
