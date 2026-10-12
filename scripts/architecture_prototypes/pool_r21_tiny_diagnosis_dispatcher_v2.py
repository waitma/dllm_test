"""Dispatch one R21 seed43 tiny numerical observation through the existing pool.

Activate pllm before root runs its frozen CPU control budget:
  python -B -m pytest -p no:cacheprovider /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/test_seed43_gpu_observation_v2_tmp_control.py -q
Only root may deploy/enqueue the single fixed alias after source review. This
module does not submit jobs, accept a model, or admit training/data. Its child
observes the original tiny assertion unchanged; a COMPLETE file means the
numerical observations, normal child exit and process cleanup were collected.
Other aliases delegate to the unchanged SHA-pinned D02 dispatcher. The paused
operator branch has no route here. Cold import and CPU metadata checks use only
the standard library. The generic process helper is imported after isolation.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
import time
from types import SimpleNamespace


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OWN = ROOT/'output/immune_seed43_gpu_observation_20261002'
POOL = ROOT/'output/immune_research_pool_c20250601_20260930'
RUNTIME = ROOT/'output/immune_multiseed_pilot_engineering_20261001/runtime_packages/r21_seed_runtime_v2'
ALIAS = 'd03_r21_seed43_tiny_gpu_observation_v1'
PAUSED = 'd00_cuda_attention_probe_v1'
SELF = 'pool_r21_tiny_diagnosis_dispatcher_v2.py'
DRIVER = OWN/'diagnose_r21_seed43_gpu_v2.py'
DRIVER_SHA256 = 'fb35b5501c2a8cbee09bd142d55d68bd5db7c058404a63dc2be3296720a1aa8a'
BUDGET = OWN/'execution_budget_v2.json'
BUDGET_SHA256 = 'be159b84b00886cb2752809b90147a9773b5dd4623b2ca69bb8ec436135b0fbe'
LEGACY = ROOT/'scripts/architecture_prototypes/pool_seed_release_dispatcher_v2.py'
LEGACY_SHA256 = 'd5ba656d12965436494d00ef9df662a6d1956e23404c9343cc42517c21109a8d'
MANIFEST = 'seed_runtime_manifest_v2.json'
MANIFEST_SHA256 = 'b6772cc422afe7753046730dc097f1a056a9b314c92ce7027b8b369dd1905c62'
PACKAGE = 'scripts/architecture_prototypes/pilot_seed_runtime_package_v2.py'
PACKAGE_SHA256 = 'e1f3286c24b4f7dc39ab6fe20a649400d3ddc2d07805a4f9ae6cf67506620964'
PROCESS_HELPER = 'scripts/architecture_prototypes/cuda_attention_probe.py'
PROCESS_HELPER_SHA256 = '193ef514bc29683d6579e9c94ec72215df77a5b0aa51ba81e153ba6a881183a0'
CPU_HELPER = ROOT/'output/immune_seed_failure_diagnosis_20261002/diagnose_r21_seed43_cpu.py'
CPU_HELPER_SHA256 = '0f24916d193e4cbcdf563219c9ea474614fa07d99c7d55166d85fd9201c8fe45'
PRODUCER_SHA256 = '0f9dc066b3c9bc972eda4d6c8fdeac39576e5c625bde44d0dbdcf28e227a14b9'
FAILED_RUN = POOL/'runs/1790906045312184986_d02_pilot_seed_gpu_r21_seed43_v2_1e0b6229'
PROFILE = FAILED_RUN/'evidence_root/profiles/r21_refined_head_gate'
PROFILE_SIDECAR_SHA256 = '3e8d1f70ecece8eccd70211ff241e8d537b80066d64505c0e4e781a83f5ad9ad'
REPORT_SCHEMA = 'immune.r21_seed43_gpu_failure_observation.v2'
COMPLETE_SCHEMA = 'immune.r21_seed43_gpu_numerical_observation_completion.v2'
TIMEOUT_SECONDS = 180
FALSE_FLAGS = ('gpu_accepted', 'data_accepted', 'training_admitted',
               'biological_quality_claim', 'checkpoint_written')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    path = Path(value)
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink(),
            'Canonical absolute nonsymlink path required')
    return path


def sha(path):
    return hashlib.sha256(canonical(path).read_bytes()).hexdigest()


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Actual frozen SHA binding required')
    return value


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate metadata key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON metadata')
    result = json.loads(canonical(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)
    require(type(result) is dict, 'Expected metadata object')
    return result


def selected_variant(argv):
    values = []
    for index, item in enumerate(argv):
        if item == '--variant':
            require(index+1 < len(argv) and not argv[index+1].startswith('-'), 'Missing variant')
            values.append(argv[index+1])
        elif item.startswith('--variant='):
            values.append(item.partition('=')[2])
    require(len(values) == 1 and values[0], 'Exactly one nonempty variant required')
    require(values[0] != PAUSED, 'Independent CUDA operator branch remains paused')
    return values[0]


def validate_argv(argv):
    allowed = {'--variant', '--mode', '--steps', '--profile-dir', '--output-dir'}
    options = [item.partition('=')[0] for item in argv if item.startswith('--')]
    require(set(options) == allowed and len(options) == 5, 'Exactly five unique observation options required')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in sorted(allowed):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)
    require(args.variant == ALIAS and args.mode == 'profile-synthetic' and args.steps == '1',
            'Only the fixed observation alias/profile-synthetic/literal1 envelope is allowed')
    output, profile = canonical(args.output_dir), canonical(args.profile_dir)
    require(output.parent == POOL/'runs' and output.parent.is_dir() and not output.exists()
            and re.fullmatch('[1-9][0-9]*_'+ALIAS+'_[0-9a-f]{8}', output.name), 'Fresh legal fixed-pool job output required')
    require(profile == POOL/'profiles'/ALIAS and profile.parent.is_dir(), 'Transport profile belongs to a different identity')
    require(strict_json(POOL/'pool.json').get('source_root') == str(ROOT), 'Pool source root differs')
    args.output_dir, args.profile_dir = output, profile
    return args


def source_evidence():
    current = canonical(__file__)
    frozen = canonical(OWN/SELF)
    require(current in (frozen, ROOT/'scripts/architecture_prototypes'/SELF) and sha(current) == sha(frozen),
            'Dispatcher differs from its frozen private source')
    return dict(dispatcher_sha256=sha(frozen), driver_sha256=digest(DRIVER_SHA256),
        budget_sha256=BUDGET_SHA256, runtime_manifest_sha256=MANIFEST_SHA256,
        runtime_package_sha256=PACKAGE_SHA256, process_helper_sha256=PROCESS_HELPER_SHA256,
        reused_CPU_helper_sha256=CPU_HELPER_SHA256, legacy_dispatcher_sha256=LEGACY_SHA256)


def verify_sources():
    evidence = source_evidence()
    for path, expected in ((BUDGET, BUDGET_SHA256), (LEGACY, LEGACY_SHA256), (DRIVER, evidence['driver_sha256']),
                           (CPU_HELPER, CPU_HELPER_SHA256), (RUNTIME/MANIFEST, MANIFEST_SHA256),
                           (RUNTIME/PACKAGE, PACKAGE_SHA256), (RUNTIME/PROCESS_HELPER, PROCESS_HELPER_SHA256),
                           (PROFILE/'seed_profile_sidecar.json', PROFILE_SIDECAR_SHA256)):
        require(canonical(path).is_file() and sha(path) == expected, 'Frozen source/budget SHA differs: '+str(path))
    budget = strict_json(BUDGET)
    require(budget.get('alias') == ALIAS and budget.get('runtime') == str(RUNTIME)
            and type(budget.get('seed')) is int and budget['seed'] == 43
            and type(budget.get('data_seed')) is int and budget['data_seed'] == 43
            and budget.get('whole_observation_hard_seconds') == TIMEOUT_SECONDS
            and budget.get('real_shards_read') is False and budget.get('all_diagnostic_admission_flags') is False,
            'Fixed observation budget/identity differs')
    manifest = strict_json(RUNTIME/MANIFEST)
    require(manifest.get('schema') == 'immune.seed_pilot_runtime_source_package.v2'
            and manifest.get('method') == 'r21' and manifest.get('runtime_root') == str(RUNTIME)
            and manifest.get('source_root') == str(ROOT), 'Fixed runtime identity differs')
    files = manifest.get('runtime_files_sha256')
    require(type(files) is dict and files.get(PACKAGE) == PACKAGE_SHA256
            and files.get(PROCESS_HELPER) == PROCESS_HELPER_SHA256, 'Missing actual verifier/process binding')
    for name, expected in files.items():
        require(type(name) is str and str(Path(name)) == name and not Path(name).is_absolute()
                and '..' not in Path(name).parts, 'Unsafe runtime source name')
        path = canonical(RUNTIME/name)
        require(path.is_relative_to(RUNTIME) and path.is_file() and sha(path) == digest(expected), 'Runtime source SHA differs: '+name)
    sys.dont_write_bytecode = True
    package = SimpleNamespace(**runpy.run_path(str(RUNTIME/PACKAGE), run_name='_observation_package_verify'))
    require(package.ROOT == RUNTIME and package.verify_runtime(RUNTIME) == manifest,
            'Actual runtime verification differs')
    require(source_evidence() == evidence and sha(RUNTIME/MANIFEST) == MANIFEST_SHA256
            and sha(RUNTIME/PACKAGE) == PACKAGE_SHA256, 'Sources changed during metadata verification')
    return manifest


def exact_count(value, expected):
    return type(value) is int and value == expected


def validate_observation_report(report):
    require(type(report) is dict and report.get('schema') == REPORT_SCHEMA and report.get('observation_complete') is True,
            'Numerical observation is incomplete')
    require(report.get('method') == 'r21' and exact_count(report.get('seed'), 43) and exact_count(report.get('data_seed'), 43)
            and all(report.get(name) is False for name in FALSE_FLAGS), 'Observation identity/admission flags differ')
    for name, expected in dict(runtime_root=str(RUNTIME), runtime_manifest_sha256=MANIFEST_SHA256,
        producer_sha256=PRODUCER_SHA256, driver_sha256=digest(DRIVER_SHA256),
        reused_CPU_helper_sha256=CPU_HELPER_SHA256, failed_run=str(FAILED_RUN), profile_dir=str(PROFILE),
        actual_profile_sidecar_sha256=PROFILE_SIDECAR_SHA256).items():
        require(report.get(name) == expected, 'Observation source binding differs: '+name)
    require(report.get('temporary_cleanup_completed') is True and exact_count(report.get('gradient_call_count'), 1),
            'Temporary cleanup or original gradient capture is incomplete')
    status, error = report.get('status'), report.get('original_error')
    require(status in ('original_tiny_raised', 'original_tiny_returned') and
            ((status == 'original_tiny_returned' and error is None) or
             (status == 'original_tiny_raised' and type(error) is dict and
              all(type(error.get(key)) is str and error[key] for key in ('type', 'message', 'traceback')))),
            'Original tiny status/error is missing or inconsistent')
    require(report.get('model_test_passed') is (error is None), 'Original assertion outcome differs')
    settings = report.get('actual_tiny_settings', {})
    require(settings.get('use_cpu') is False and settings.get('device') == 'cuda:0' and settings.get('bf16') is True
            and exact_count(settings.get('seed'), 43) and exact_count(settings.get('data_seed'), 43)
            and exact_count(settings.get('micro_batch'), 1) and exact_count(settings.get('grad_accum'), 8)
            and exact_count(settings.get('optimizer_steps'), 1) and settings.get('save_strategy') == 'no',
            'Actual tiny configuration differs')
    replay = report.get('replay_observations', {})
    for key in ('native_losses', 'manual_losses', 'native_rng_before_sha256', 'native_rng_after_sha256',
                'manual_rng_after_sha256', 'batch_sha256'):
        values = replay.get(key)
        require(type(values) is list and len(values) == 8, 'Incomplete replay capture: '+key)
        for value in values:
            if key.endswith('losses'):
                require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite observed loss')
            else:
                digest(value)
    gradients = report.get('all_parameter_gradient_statistics', {})
    parameters = gradients.get('parameters')
    require(gradients.get('membership_equal') is True and type(parameters) is dict and len(parameters) == 26
            and all(type(value) is dict and value.get('membership_equal') is True
                    and type(value.get('original_allclose')) is bool and 'max_abs_error' in value
                    and type(value.get('failed_elements')) is int for value in parameters.values()),
            'Full original tiny parameter-gradient capture is incomplete')
    layout = report.get('post_tiny_loss_layout_comparison', {})
    require(exact_count(layout.get('captured_native_count'), 8) and exact_count(layout.get('captured_manual_count'), 8)
            and exact_count(layout.get('extra_calls'), 0) and type(layout.get('per_capture')) is list
            and len(layout['per_capture']) == 16 and type(layout.get('native_manual_pairs')) is list
            and len(layout['native_manual_pairs']) == 8, 'Detached post-tiny CE observations are incomplete')
    for index, value in enumerate(layout['per_capture']):
        require(type(value) is dict and value.get('capture') == ('native_logits' if index < 8 else 'manual_logits')
                and exact_count(value.get('microstep'), index%8+1)
                and type(value.get('dloss_dlogits')) is dict, 'CE capture ordering/details differ')
    gpu = report.get('gpu_assignment', {})
    require(gpu.get('logical_device') == 'cuda:0' and gpu.get('original_failure_physical_gpu') == '2'
            and all(type(gpu.get(key)) is str and gpu[key] for key in ('cuda_visible_devices', 'name', 'uuid'))
            and len(gpu['cuda_visible_devices'].split(',')) == 1, 'Actual single-GPU identity is missing')


def validate_child_exit(receipt, command):
    require(type(command) is list and len(command) == 7 and all(type(value) is str for value in command)
            and command[:6] == [sys.executable, '-I', '-B', '-u', str(DRIVER), '--output-dir'],
            'Child command differs from the fixed fresh isolated driver')
    numerical = canonical(command[6])
    require(numerical.name == 'numerical_evidence' and numerical.parent.parent == POOL/'runs'
            and re.fullmatch('[1-9][0-9]*_'+ALIAS+'_[0-9a-f]{8}', numerical.parent.name),
            'Child command output escaped the fixed pool identity')
    require(type(receipt) is dict and receipt.get('command') == command
            and exact_count(receipt.get('returncode'), 0) and receipt.get('normal_exit') is True
            and receipt.get('cleanup_completed') is True, 'Child exit/actual process-group cleanup is unproven')
    elapsed, timeout = receipt.get('elapsed_seconds'), receipt.get('timeout_seconds')
    require(type(elapsed) in (int, float) and type(timeout) in (int, float)
            and math.isfinite(elapsed) and math.isfinite(timeout) and 0 <= elapsed <= timeout <= TIMEOUT_SECONDS,
            'Child exceeded its remaining whole-observation budget')


def completion_metadata(report, receipt, sources):
    validate_observation_report(report)
    validate_child_exit(receipt, receipt.get('command'))
    require(type(sources) is dict and sources == source_evidence(), 'Completion software binding differs')
    return dict(schema=COMPLETE_SCHEMA, status='observed', numerical_observation_completed=True,
        alias=ALIAS, method='r21', seed=43, data_seed=43,
        original_tiny_status=report['status'], original_tiny_error=report['original_error'],
        gpu_accepted=False, data_accepted=False, training_accepted=False, training_admitted=False,
        model_accepted=False, biological_quality_claim=False, checkpoint_written=False,
        source_evidence=sources, child_process=receipt,
        limitation='Numerical collection only; original thresholds and real-data/full-training gates are unchanged.')


def write_json(path, value):
    with canonical(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def run_observation(args):
    started = time.monotonic()
    args = validate_argv(['--variant', args.variant, '--mode', args.mode, '--steps', str(args.steps),
                          '--profile-dir', str(args.profile_dir), '--output-dir', str(args.output_dir)])
    require('torch' not in sys.modules, 'Observation parent must enter cold before ML imports')
    manifest = verify_sources()
    sources = source_evidence()
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    require(type(mask) is str and re.fullmatch('[0-9]+', mask), 'Original worker single-GPU assignment is missing')
    package = SimpleNamespace(**runpy.run_path(str(RUNTIME/PACKAGE), run_name='_observation_package_bootstrap'))
    package.install_runtime_search(RUNTIME, manifest)
    process = importlib.import_module('scripts.architecture_prototypes.cuda_attention_probe')
    require(canonical(process.__file__) == RUNTIME/PROCESS_HELPER and sha(process.__file__) == PROCESS_HELPER_SHA256
            and not process.torch.cuda.is_initialized(), 'Generic helper escaped the runtime or initialized parent CUDA')
    package.verify_loaded_runtime(RUNTIME, manifest)
    remaining = TIMEOUT_SECONDS-(time.monotonic()-started)
    require(remaining > 0 and verify_sources() == manifest, 'Source preflight exceeded budget or changed')
    args.output_dir.mkdir()
    numerical = args.output_dir/'numerical_evidence'
    command = [sys.executable, '-I', '-B', '-u', str(DRIVER), '--output-dir', str(numerical)]
    receipt = dict(command=command, normal_exit=False, cleanup_completed=False, returncode=None)
    try:
        remaining = TIMEOUT_SECONDS-(time.monotonic()-started)
        require(remaining > 0, 'No whole-observation budget remains')
        child_started = time.monotonic()
        with (args.output_dir/'numerical_child.log').open('x') as log:
            completed = process.run_case_process(command, stdout=log, timeout=remaining)
        # Only normal return proves the original helper's finally reaped the
        # entire same-session process group. Never infer cleanup from child JSON.
        receipt.update(returncode=completed.returncode, normal_exit=True, cleanup_completed=True,
                       timeout_seconds=remaining, elapsed_seconds=time.monotonic()-child_started)
        write_json(args.output_dir/'child_process.json', receipt)
        validate_child_exit(receipt, command)
        report_path = canonical(numerical/'gpu_diagnosis.json')
        report = strict_json(report_path)
        validate_observation_report(report)
        require(report['gpu_assignment']['cuda_visible_devices'] == mask, 'Observed GPU differs from the original worker assignment')
        require(not any(path.name.startswith('checkpoint-') or path.name in ('GPU_RELEASE_COMPLETE.json', 'PRODUCER_SUCCESS.json')
                        for path in args.output_dir.rglob('*')), 'Unexpected acceptance marker/checkpoint')
        require(verify_sources() == manifest and source_evidence() == sources, 'Source changed during observation')
        package.verify_loaded_runtime(RUNTIME, manifest)
        require(not process.torch.cuda.is_initialized(), 'Observation parent initialized CUDA')
        result = completion_metadata(report, receipt, sources)
        result.update(report_sha256=sha(report_path), log_sha256=sha(args.output_dir/'numerical_child.log'),
                      elapsed_seconds=time.monotonic()-started)
        require(result['elapsed_seconds'] <= TIMEOUT_SECONDS, 'Whole observation exceeded the frozen 180 seconds')
        write_json(args.output_dir/'COMPLETE.json', result)
        return result
    except BaseException as error:
        write_json(args.output_dir/'observation_error.json', dict(status='observation_incomplete',
            error=dict(type=type(error).__name__, message=str(error)), child_process=receipt,
            elapsed_seconds=time.monotonic()-started, **{name: False for name in FALSE_FLAGS}))
        raise


def delegate_legacy(argv):
    selected_variant(argv)
    require(sha(LEGACY) == LEGACY_SHA256, 'Unchanged D02 dispatcher SHA differs')
    return runpy.run_path(str(LEGACY), run_name='_tiny_observation_legacy')['main'](argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--help']:
        print('One fixed R21 seed43 tiny numerical observation; all admission flags false; legacy routes SHA-pinned.')
        return 0
    try:
        if selected_variant(argv) != ALIAS:
            return delegate_legacy(argv)
        result = run_observation(validate_argv(argv))
        print(json.dumps(dict(status=result['status'], output=str(Path(result['child_process']['command'][-1]).parent)), allow_nan=False), flush=True)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
