"""Supervise original artificial25/500 learning in the existing FULL evidence root.

Activate pllm; root alone runs the new frozen controls:
  python -B -m pytest -p no:cacheprovider /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/test_generic_CE_learning_binding.py -q
The unchanged worker supplies d04_r21_CE_learning_v1 with profile-synthetic/25
or train/500 plus the fixed artificial data directory. Publication, the single
launcher-only registry binding, fixture preparation and enqueue are separate
root actions. This file never submits resources or admits real training.
Only actual child exit/cleanup and original artifacts create an inner result;
the outer transport result remains explicitly different from the R21 report.
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
OWN = ROOT/'output/generic_CE_learning_integration_20261002'
SELF = 'pool_generic_CE_learning_dispatcher_v1.py'
CHILD = OWN/'cold_learning_child_v1.py'
CHILD_SHA256 = '7ec9d4fe098384d969ed3dd09117ba5a3921db19d2760f0f65be8e0ab5e8bc47'
LEGACY = ROOT/'output/generic_CE_full_release_20261002/pool_generic_CE_full_dispatcher_v1.py'
LEGACY_SHA256 = '487997673629e39c92fcdfe2c72cc3f74f2ebcbf99de3e1b4a17cedee1b3f909'
ALIAS = 'd04_r21_CE_learning_v1'
POOL = ROOT/'output/immune_research_pool_c20250601_20260930'
TIMEOUTS = {'profile_diagnostic25': 1800, 'synthetic500': 28800}
PROCESS_HELPER = 'scripts/architecture_prototypes/cuda_attention_probe.py'
PROCESS_HELPER_SHA256 = '193ef514bc29683d6579e9c94ec72215df77a5b0aa51ba81e153ba6a881183a0'
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    path = Path(value)
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink(), 'Canonical absolute nonsymlink path required')
    return path


def child_module():
    require(type(CHILD_SHA256) is str and re.fullmatch('[0-9a-f]{64}', CHILD_SHA256), 'Actual frozen learning child SHA missing')
    current, source = canonical(__file__), canonical(OWN/SELF)
    require(current in (source, ROOT/'scripts/architecture_prototypes'/SELF) and sha(current) == sha(source),
            'Learning dispatcher differs from its frozen private source')
    require(sha(canonical(CHILD)) == CHILD_SHA256, 'Frozen learning child changed')
    return SimpleNamespace(**runpy.run_path(str(CHILD), run_name='_learning_fixed_child'))


def legacy_module():
    require(sha(canonical(LEGACY)) == LEGACY_SHA256, 'Frozen FULL dispatcher changed')
    return SimpleNamespace(**runpy.run_path(str(LEGACY), run_name='_learning_fixed_legacy'))


def validate_argv(argv):
    allowed = {'--variant', '--mode', '--steps', '--profile-dir', '--output-dir'}
    if '--data-dir' in argv or any(value.startswith('--data-dir=') for value in argv):
        allowed.add('--data-dir')
    options = [value.partition('=')[0] for value in argv if value.startswith('--')]
    require(set(options) == allowed and len(options) == len(allowed), 'Only the exact learning options are allowed; no resume or seed field')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in sorted(allowed):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)
    require(args.variant == ALIAS and ((args.mode, args.steps) == ('profile-synthetic', '25')
            or (args.mode, args.steps) == ('train', '500')), 'Only fixed artificial25 then artificial500 transport allowed')
    args.stage = 'profile_diagnostic25' if args.steps == '25' else 'synthetic500'
    args.steps = int(args.steps)
    child = child_module()
    if args.steps == 25:
        require('--data-dir' not in allowed, 'Artificial25 uses only the original built-in fixture')
    else:
        require('--data-dir' in allowed and canonical(args.data_dir) == child.DATA, 'Artificial500 requires the fixed artificial directory; real data is rejected')
    args.output_dir, args.profile_dir = canonical(args.output_dir), canonical(args.profile_dir)
    require(args.output_dir.parent == POOL/'runs' and args.output_dir.parent.is_dir() and not args.output_dir.exists(), 'Fresh fixed pool/runs transport output required')
    require(re.fullmatch(r'[1-9][0-9]*_'+re.escape(ALIAS)+r'_[0-9a-f]{8}', args.output_dir.name), 'Transport task identity differs')
    require(args.profile_dir == POOL/'profiles'/ALIAS and args.profile_dir.parent.is_dir(), 'Worker profile transport identity differs')
    require(child.read_json(POOL/'pool.json').get('source_root') == str(ROOT), 'Fixed pool source differs')
    child.validate_output(child.EVIDENCE_ROOT/'runs'/args.output_dir.name, args.stage)
    return args


def inner_job(args):
    return child_module().fixed_stage_job(args.stage)


def software_binding():
    child = child_module()
    require(sha(canonical(LEGACY)) == LEGACY_SHA256, 'Frozen FULL dispatcher changed')
    return dict(child.software_binding(), dispatcher_sha256=sha(OWN/SELF), child_sha256=CHILD_SHA256)


def validate_child_exit(receipt, command, timeout):
    require(type(receipt) is dict and receipt.get('command') == command
            and type(receipt.get('returncode')) is int and receipt['returncode'] == 0
            and receipt.get('normal_exit') is True and receipt.get('cleanup_completed') is True,
            'Actual child normal exit and descendant cleanup missing')
    elapsed, observed_timeout = receipt.get('elapsed_seconds'), receipt.get('timeout_seconds')
    require(type(elapsed) in (int, float) and math.isfinite(elapsed)
            and type(observed_timeout) in (int, float) and math.isfinite(observed_timeout)
            and 0 < observed_timeout <= timeout and 0 <= elapsed <= observed_timeout, 'Actual child exceeded its stage budget')


def result_metadata(args, receipt, summary, complete, sources):
    child = child_module()
    run = child.EVIDENCE_ROOT/'runs'/args.output_dir.name
    command = [sys.executable, '-I', '-B', '-u', str(CHILD), '--stage', args.stage, '--output-dir', str(run)]
    validate_child_exit(receipt, command, TIMEOUTS[args.stage])
    require(sources == software_binding(), 'Learning source changed before genuine result index')
    require(summary.get('variant') == child.VARIANT and summary.get('steps') == args.steps
            and summary.get('status') == 'completed' and type(summary.get('seed')) is int and summary['seed'] == 42
            and summary.get('kind') == 'synthetic_generalization_diagnostic'
            and summary.get('biological_quality_claim') is False and summary.get('validation_uses_test') is False
            and summary.get('micro_batch') == 32 and summary.get('grad_accum') == 8 and summary.get('global_batch') == 256,
            'Actual original artificial learning report/seed differs')
    require(complete.get('status') == 'completed' and complete.get('global_step') == args.steps
            and complete.get('kind') == summary['kind'] and complete.get('biological_quality_claim') is False
            and complete.get('summary_sha256') == sha(canonical(run/'summary.json'))
            and child.read_json(run/'summary.json') == summary and child.read_json(run/'COMPLETE.json') == complete,
            'Actual original COMPLETE/summary bytes differ')
    execution = child.read_json(run/'learning_execution.json')
    require(execution.get('stage') == args.stage and execution.get('actual_model_job') == inner_job(args)
            and execution.get('source_evidence') == child.software_binding()
            and execution.get('summary_sha256') == sha(run/'summary.json')
            and execution.get('complete_sha256') == sha(run/'COMPLETE.json')
            and all(execution.get(name) is False for name in child.FALSE_FLAGS), 'Actual learning child source receipt differs')
    return dict(job_id=args.output_dir.name, job=inner_job(args), state='succeeded', returncode=0,
        actual_child_command=command, observed_child_process=receipt, source_evidence=sources,
        outer_transport=dict(variant=ALIAS, mode=args.mode, steps=args.steps), learning_only=True,
        learning_seed=42, formal_source_profile_seed=43, summary_sha256=sha(run/'summary.json'),
        complete_sha256=sha(run/'COMPLETE.json'), **{name: False for name in child.FALSE_FLAGS})


def write_json(path, value):
    with canonical(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def run_learning(args):
    started = time.monotonic()
    require('torch' not in sys.modules, 'Learning supervisor must enter cold')
    child = child_module()
    sources = software_binding()
    runtime, package_path = child.RUNTIME, child.RUNTIME/child.PACKAGE
    package = SimpleNamespace(**runpy.run_path(str(package_path), run_name='_learning_supervisor_runtime'))
    manifest = package.verify_runtime(runtime)
    package.install_runtime_search(runtime, manifest)
    process = importlib.import_module('scripts.architecture_prototypes.cuda_attention_probe')
    checks = importlib.import_module('scripts.enqueue_immune_refined')
    require(canonical(process.__file__) == runtime/PROCESS_HELPER and sha(process.__file__) == PROCESS_HELPER_SHA256
            and not process.torch.cuda.is_initialized(), 'Original generic supervisor source/CUDA state differs')
    checks.validate_job(child.EVIDENCE_ROOT, inner_job(args), args.stage)  # No output/index before original25 gate.
    require(re.fullmatch('[0-7]', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'One claimed pool GPU required')
    inner = child.validate_output(child.EVIDENCE_ROOT/'runs'/args.output_dir.name, args.stage)
    index = canonical(child.EVIDENCE_ROOT/'results'/(args.output_dir.name+'.json'))
    require(not index.exists(), 'Never overwrite an actual learning result')
    child.EVIDENCE_ROOT.joinpath('runs').mkdir(exist_ok=True)
    child.EVIDENCE_ROOT.joinpath('results').mkdir(exist_ok=True)
    args.output_dir.mkdir()
    command = [sys.executable, '-I', '-B', '-u', str(CHILD), '--stage', args.stage, '--output-dir', str(inner)]
    receipt = dict(command=command, returncode=None, normal_exit=False, cleanup_completed=False)
    index_written = False
    index_identity = None
    try:
        remaining = TIMEOUTS[args.stage]-(time.monotonic()-started)
        require(remaining > 0, 'No learning stage budget remains')
        child_started = time.monotonic()
        with (args.output_dir/'learning_child.log').open('x') as log:
            result = process.run_case_process(command, stdout=log, timeout=remaining)
        receipt.update(returncode=result.returncode, normal_exit=result.returncode >= 0, cleanup_completed=True,
            elapsed_seconds=time.monotonic()-child_started, timeout_seconds=remaining,
            log_sha256=sha(args.output_dir/'learning_child.log'))
        write_json(args.output_dir/'child_process.json', receipt)
        summary, complete = child.read_json(inner/'summary.json'), child.read_json(inner/'COMPLETE.json')
        actual_result = result_metadata(args, receipt, summary, complete, sources)
        write_json(index, actual_result)
        index_written = True
        actual_stat = index.stat()
        index_identity = (actual_stat.st_dev, actual_stat.st_ino, sha(index))
        original = checks.verify_diagnostic(child.EVIDENCE_ROOT, child.VARIANT, args.steps, require_learning=args.steps == 500)
        require(original == summary and package.verify_loaded_runtime(runtime, manifest) == manifest
                and software_binding() == sources and not process.torch.cuda.is_initialized(), 'Original learning verification/source/supervisor state differs')
        elapsed = time.monotonic()-started
        require(elapsed <= TIMEOUTS[args.stage], 'Whole learning stage budget exceeded')
        outer = dict(kind='synthetic_learning_transport', source_namespace=str(child.RUNTIME),
            child_inner_result=str(index), evidence_root=str(child.EVIDENCE_ROOT), learning_only=True,
            stage=args.stage, outer_transport=dict(variant=ALIAS, mode=args.mode, steps=args.steps),
            source_evidence=sources, learning_seed=42, formal_source_profile_seed=43,
            elapsed_seconds=elapsed, result_sha256=sha(index), **{name: False for name in child.FALSE_FLAGS})
        write_json(args.output_dir/'summary.json', outer)
        write_json(args.output_dir/'COMPLETE.json', dict(status='completed', kind=outer['kind'], stage=args.stage,
            child_inner_result=str(index), summary_sha256=sha(args.output_dir/'summary.json'),
            learning_only=True, **{name: False for name in child.FALSE_FLAGS}))
        return 0
    except BaseException as error:
        if index_written and index.exists():
            actual_stat = canonical(index).stat()
            if (actual_stat.st_dev, actual_stat.st_ino, sha(index)) == index_identity:
                index.unlink()  # Only this call's unchanged index may be removed.
        write_json(args.output_dir/'FAILED.json', dict(status='failed', stage=args.stage,
            error_type=type(error).__name__, message=str(error), observed_child_process=receipt,
            **{name: False for name in child.FALSE_FLAGS}))
        raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    legacy = legacy_module()
    if argv == ['--help'] or legacy.selected_variant(argv) != ALIAS:
        return legacy.main(argv)  # Original D00 pause and every old identity remain unchanged.
    return run_learning(validate_argv(argv))


if __name__ == '__main__':
    raise SystemExit(main())
