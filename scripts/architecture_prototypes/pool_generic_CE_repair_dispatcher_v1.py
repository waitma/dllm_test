"""Run one synthetic CE-repair tiny check through the existing eight-GPU pool.

After activating pllm, root reviews this file, builds the pinned runtime, and
publishes a byte-identical copy at the absolute project scripts directory.
The unchanged pool worker invokes /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_research.py.
This dispatcher never submits resources or admits real data/model training.
Its only new envelope is the existing D03 profile-synthetic/1 transport.
Normal child exit, process cleanup and the original tiny assertion are separate
facts; even a passing tiny check grants no complete GPU or training admission.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time
from types import SimpleNamespace

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OWN = ROOT/'output/generic_ce_layout_repair_20261002'
BASE = ROOT/'output/immune_seed43_gpu_observation_20261002/pool_r21_tiny_diagnosis_dispatcher_v5.py'
BASE_SHA = 'cad2625c361f93daf45da39a7a52f9abdd75b01faf6d7b62716d66beb629cbe7'
DRIVER = OWN/'run_generic_CE_tiny_repair_v1.py'
DRIVER_SHA = 'fd8aa33ec74915ef9661ff13b3da28b0cb1f2cfa75a8c2eea10aa094dafc983f'
RUNTIME = OWN/'runtime_packages/r21_seed_runtime_v2'
PACKAGE = 'scripts/architecture_prototypes/pilot_seed_runtime_package_v2.py'
PACKAGE_SHA = '714438276cb88002aeef5a8df5dc3af1f3f3b7efa820182fc791cf59f975a873'
MANIFEST = 'seed_runtime_manifest_v2.json'
MANIFEST_SHA = '2c2c9ee3ca2f17c29da402b937902d158d82eeedb5ba7753dcf1e0be1307c148'
ALIAS = 'd03_r21_seed43_tiny_gpu_observation_v1'
TIMEOUT = 180
FALSE_FLAGS = ('gpu_accepted', 'data_accepted', 'training_accepted', 'training_admitted',
               'model_accepted', 'biological_quality_claim', 'checkpoint_written',
               'real_training_started', 'data_certificate_issued', 'full_release_accepted')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(path):
    require(Path(path).is_absolute() and Path(path) == Path(path).resolve()
            and not Path(path).is_symlink(), 'Canonical absolute metadata required')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate metadata key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON metadata')
    result = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)
    require(type(result) is dict, 'Expected metadata object')
    return result


def binding():
    for value in (DRIVER_SHA, MANIFEST_SHA):
        require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Frozen actual source binding missing')
    current = Path(__file__).resolve()
    require(current in (OWN/Path(__file__).name, ROOT/'scripts/architecture_prototypes'/Path(__file__).name)
            and sha(current) == sha(OWN/Path(__file__).name), 'Dispatcher differs from its frozen source')
    require(sha(BASE) == BASE_SHA and sha(DRIVER) == DRIVER_SHA
            and sha(RUNTIME/PACKAGE) == PACKAGE_SHA and sha(RUNTIME/MANIFEST) == MANIFEST_SHA,
            'Frozen original helper/new source runtime changed')
    package = SimpleNamespace(**runpy.run_path(str(RUNTIME/PACKAGE), run_name='_CE_repair_parent_package'))
    package.verify_runtime(RUNTIME)
    return dict(driver_sha256=DRIVER_SHA, runtime_manifest_sha256=MANIFEST_SHA,
                package_sha256=PACKAGE_SHA, dispatcher_sha256=sha(OWN/Path(__file__).name))


def verify_report(report, command, receipt):
    require(type(report) is dict and report.get('schema') == 'immune.generic_CE_layout_tiny_repair.v1'
            and report.get('tiny_assertion_passed') is True and report.get('status') == 'original_tiny_passed'
            and report.get('observation_complete') is True and report.get('profile_present') is False
            and report.get('original_error') is None and report.get('diagnostic_error') is None,
            'Original repaired-source tiny assertion is not proven passed')
    require(report.get('method') == 'r21' and type(report.get('seed')) is int and report['seed'] == 43
            and type(report.get('data_seed')) is int and report['data_seed'] == 43
            and all(report.get(name) is False for name in FALSE_FLAGS), 'Tiny identity/admission differs')
    require(report.get('runtime_manifest_sha256') == MANIFEST_SHA
            and report.get('driver_sha256') == DRIVER_SHA
            and report.get('runtime_root') == str(RUNTIME), 'Tiny source binding differs')
    require(command == [sys.executable, '-I', '-B', '-u', str(DRIVER), '--output-dir', command[-1]]
            and command[-1] == str(Path(command[-1]).resolve()), 'Isolated child command differs')
    require(report.get('job_id') == Path(command[-1]).parent.name
            and receipt.get('command') == command and receipt.get('returncode') == 0
            and receipt.get('normal_exit') is True and receipt.get('cleanup_completed') is True,
            'Child job binding/normal exit/actual process cleanup missing')
    require(report.get('temporary_cleanup_completed') is True, 'Child owned temporary cleanup incomplete')
    require(0 <= receipt['elapsed_seconds'] <= receipt['timeout_seconds'] <= TIMEOUT,
            'Child exceeded the remaining parent budget')


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main(argv=None):
    started = time.monotonic()
    require(sha(BASE) == BASE_SHA, 'Original dispatcher helper changed')
    base = SimpleNamespace(**runpy.run_path(str(BASE), run_name='_CE_repair_original_dispatcher'))
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--help'] or base.selected_variant(argv) != ALIAS:
        return base.main(argv)
    args = base.validate_argv(argv)
    require('torch' not in sys.modules, 'Parent must enter cold')
    before = binding()
    original_manifest = base.verify_sources()
    package = SimpleNamespace(**runpy.run_path(str(base.RUNTIME/base.PACKAGE), run_name='_CE_repair_process_bootstrap'))
    package.install_runtime_search(base.RUNTIME, original_manifest)
    process = importlib.import_module('scripts.architecture_prototypes.cuda_attention_probe')
    require(Path(process.__file__) == base.RUNTIME/base.PROCESS_HELPER
            and sha(process.__file__) == base.PROCESS_HELPER_SHA256
            and not process.torch.cuda.is_initialized(), 'Generic process supervisor identity differs')
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(re.fullmatch('[0-7]', mask), 'One original claimed GPU required')
    command = [sys.executable, '-I', '-B', '-u', str(DRIVER), '--output-dir', str(args.output_dir/'numerical_evidence')]
    args.output_dir.mkdir()
    receipt = dict(command=command, normal_exit=False, cleanup_completed=False, returncode=None)
    try:
        remaining = TIMEOUT-(time.monotonic()-started)
        require(remaining > 0, 'No parent budget remains')
        child_started = time.monotonic()
        with (args.output_dir/'numerical_child.log').open('x') as log:
            result = process.run_case_process(command, stdout=log, timeout=remaining)
        receipt.update(returncode=result.returncode, normal_exit=True, cleanup_completed=True,
                       elapsed_seconds=time.monotonic()-child_started, timeout_seconds=remaining)
        write_json(args.output_dir/'child_process.json', receipt)
        report_path = args.output_dir/'numerical_evidence/tiny_repair_report.json'
        report = metadata(report_path)
        verify_report(report, command, receipt)
        require(report['gpu_assignment']['cuda_visible_devices'] == mask, 'Child GPU differs from worker assignment')
        require(binding() == before and not process.torch.cuda.is_initialized(), 'Source changed or parent initialized CUDA')
        require(not any(p.name.startswith('checkpoint-') or p.name in ('GPU_RELEASE_COMPLETE.json', 'PRODUCER_SUCCESS.json')
                        for p in args.output_dir.rglob('*')), 'Unexpected full-admission marker/checkpoint')
        elapsed = time.monotonic()-started
        require(elapsed <= TIMEOUT, 'Whole parent budget exceeded')
        completion = dict(schema='generic.ce_row_major_tiny_completion.v1', status='tiny_assertion_passed',
            job_id=args.output_dir.name, source_evidence=before, report_sha256=sha(report_path),
            log_sha256=sha(args.output_dir/'numerical_child.log'), child_process=receipt,
            elapsed_seconds=elapsed, **{name: False for name in FALSE_FLAGS})
        write_json(args.output_dir/'COMPLETE.json', completion)
        print(json.dumps(dict(status=completion['status'], output=str(args.output_dir))), flush=True)
        return 0
    except BaseException as error:
        write_json(args.output_dir/'repair_error.json', dict(error_type=type(error).__name__, error=str(error),
            child_process=receipt, elapsed_seconds=time.monotonic()-started,
            **{name: False for name in FALSE_FLAGS}))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
