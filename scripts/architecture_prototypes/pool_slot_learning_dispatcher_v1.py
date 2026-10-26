"""Supervise original artificial25/500 learning for seven remaining source slots.

Activate pllm; root alone runs the frozen controls:
  python -B -m pytest -p no:cacheprovider /project_c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/test_slot_learning_binding.py -q
The unchanged worker supplies one fixed d05 alias, profile-synthetic/25 or
train/500 with the existing artificial data directory. Publication and enqueue
are separate root actions. The child is the fixed absolute private source with
-I -B -u --slot ALIAS --stage STAGE --output-dir OWN_FULL_EVIDENCE_RUN.
Actual exit/cleanup and original artifacts precede the inner result index and
original learning verifier. Outer summaries remain transport metadata. Every
other identity delegates to the frozen CE parent; independent CUDA stays paused.
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


ROOT = Path('/project_c20250601/251105016/project/dllm_test')
OWN = ROOT/'output/generic_remaining_slot_learning_integration_20261002'
SELF = 'pool_slot_learning_dispatcher_v1.py'
CHILD = OWN/'cold_slot_learning_child_v1.py'
CHILD_SHA256 = 'e8673c715836cf2c8ac1212f0a87c72a3018a71ceeedd24a32c5b7389e41bdfe'
LEGACY = ROOT/'output/generic_CE_learning_integration_20261002/pool_generic_CE_learning_dispatcher_v1.py'
LEGACY_SHA256 = '36b41524c93a17900c11600cb2bbbf075912242ad838c219b9bb204416ac1ee3'
POOL = ROOT/'output/immune_research_pool_c20250601_20260930'
SLOTS = frozenset(('d05_r20_seed42_learning_v1', 'd05_r20_seed43_learning_v1',
    'd05_r21_seed42_learning_v1', 'd05_r24_seed42_learning_v1', 'd05_r24_seed43_learning_v1',
    'd05_r26_seed42_learning_v1', 'd05_r26_seed43_learning_v1'))
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
    require(type(CHILD_SHA256) is str and re.fullmatch('[0-9a-f]{64}', CHILD_SHA256), 'Actual frozen slot child SHA missing')
    current, source = canonical(__file__), canonical(OWN/SELF)
    require(current in (source, ROOT/'scripts/architecture_prototypes'/SELF) and sha(current) == sha(source),
            'Slot learning dispatcher differs from its frozen private source')
    require(sha(canonical(CHILD)) == CHILD_SHA256, 'Frozen slot learning child changed')
    return SimpleNamespace(**runpy.run_path(str(CHILD), run_name='_slot_learning_fixed_child'))


def legacy_module():
    require(sha(canonical(LEGACY)) == LEGACY_SHA256, 'Frozen CE learning dispatcher changed')
    return SimpleNamespace(**runpy.run_path(str(LEGACY), run_name='_slot_learning_fixed_legacy'))


def slot_binding(alias):
    return child_module().slot_binding(alias)


def validate_argv(argv):
    allowed = {'--variant', '--mode', '--steps', '--profile-dir', '--output-dir'}
    if '--data-dir' in argv or any(value.startswith('--data-dir=') for value in argv):
        allowed.add('--data-dir')
    options = [value.partition('=')[0] for value in argv if value.startswith('--')]
    require(set(options) == allowed and len(options) == len(allowed), 'Only exact learning options allowed; no resume or seed field')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in sorted(allowed):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)
    child = child_module()
    slot = child.slot_binding(args.variant)
    require((args.mode, args.steps) in (('profile-synthetic', '25'), ('train', '500')),
            'Only original artificial25 then artificial500 transport allowed')
    args.stage = 'profile_diagnostic25' if args.steps == '25' else 'synthetic500'
    args.steps = int(args.steps)
    if args.steps == 25:
        require('--data-dir' not in allowed, 'Artificial25 uses only original built-in fixtures')
    else:
        require('--data-dir' in allowed and canonical(args.data_dir) == child.DATA, 'Artificial500 requires the fixed artificial directory; real data is rejected')
    args.output_dir, args.profile_dir = canonical(args.output_dir), canonical(args.profile_dir)
    require(args.output_dir.parent == POOL/'runs' and args.output_dir.parent.is_dir() and not args.output_dir.exists(), 'Fresh fixed pool/runs transport output required')
    require(re.fullmatch(r'[1-9][0-9]*_'+re.escape(args.variant)+r'_[0-9a-f]{8}', args.output_dir.name), 'Transport task identity differs')
    require(args.profile_dir == POOL/'profiles'/args.variant and args.profile_dir.parent.is_dir(), 'Worker profile transport identity differs')
    require(child.read_json(POOL/'pool.json').get('source_root') == str(ROOT), 'Fixed pool source differs')
    child.validate_output(args.variant, Path(slot['evidence_root'])/'runs'/args.output_dir.name, args.stage)
    return args


def inner_job(args):
    return child_module().fixed_stage_job(args.variant, args.stage)


def software_binding(alias):
    child = child_module()
    require(sha(canonical(LEGACY)) == LEGACY_SHA256, 'Frozen CE learning dispatcher changed')
    return dict(child.software_binding(alias), dispatcher_sha256=sha(OWN/SELF), child_sha256=CHILD_SHA256)


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
    slot = child.slot_binding(args.variant)
    run = Path(slot['evidence_root'])/'runs'/args.output_dir.name
    command = [sys.executable, '-I', '-B', '-u', str(CHILD), '--slot', args.variant,
               '--stage', args.stage, '--output-dir', str(run)]
    validate_child_exit(receipt, command, TIMEOUTS[args.stage])
    require(sources == software_binding(args.variant), 'Learning source changed before genuine result index')
    require(summary.get('variant') == slot['variant'] and type(summary.get('steps')) is int and summary['steps'] == args.steps
            and summary.get('status') == 'completed' and type(summary.get('seed')) is int and summary['seed'] == 42
            and summary.get('kind') == 'synthetic_generalization_diagnostic'
            and summary.get('biological_quality_claim') is False and summary.get('validation_uses_test') is False
            and summary.get('source_sha256') == sources['measured_source_sha256']
            and summary.get('micro_batch') == 32 and summary.get('grad_accum') == 8 and summary.get('global_batch') == 256,
            'Actual original artificial learning report/source/seed differs')
    require(complete.get('status') == 'completed' and type(complete.get('global_step')) is int and complete['global_step'] == args.steps
            and complete.get('kind') == summary['kind'] and complete.get('biological_quality_claim') is False
            and complete.get('summary_sha256') == sha(canonical(run/'summary.json'))
            and child.read_json(run/'summary.json') == summary and child.read_json(run/'COMPLETE.json') == complete,
            'Actual original COMPLETE/summary bytes differ')
    execution = child.read_json(run/'learning_execution.json')
    require(execution.get('stage') == args.stage and execution.get('actual_model_job') == inner_job(args)
            and execution.get('source_evidence') == child.software_binding(args.variant)
            and type(execution.get('learning_seed')) is int and execution['learning_seed'] == 42
            and type(execution.get('formal_source_profile_seed')) is int
            and execution['formal_source_profile_seed'] == slot['formal_source_profile_seed']
            and execution.get('summary_sha256') == sha(run/'summary.json')
            and execution.get('complete_sha256') == sha(run/'COMPLETE.json')
            and all(execution.get(name) is False for name in child.FALSE_FLAGS), 'Actual learning child source receipt differs')
    return dict(job_id=args.output_dir.name, job=inner_job(args), state='succeeded', returncode=0,
        actual_child_command=command, observed_child_process=receipt, source_evidence=sources,
        outer_transport=dict(variant=args.variant, mode=args.mode, steps=args.steps), learning_only=True,
        learning_seed=42, formal_source_profile_seed=slot['formal_source_profile_seed'], summary_sha256=sha(run/'summary.json'),
        complete_sha256=sha(run/'COMPLETE.json'), **{name: False for name in child.FALSE_FLAGS})


def write_json(path, value):
    with canonical(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def run_learning(args):
    started = time.monotonic()
    require('torch' not in sys.modules, 'Learning supervisor must enter cold')
    child = child_module()
    slot, sources = child.slot_binding(args.variant), software_binding(args.variant)
    runtime, evidence = Path(slot['runtime_root']), Path(slot['evidence_root'])
    package = SimpleNamespace(**runpy.run_path(str(runtime/child.PACKAGE), run_name='_slot_learning_supervisor_runtime'))
    manifest = package.verify_runtime(runtime)
    package.install_runtime_search(runtime, manifest)
    process = importlib.import_module('scripts.architecture_prototypes.cuda_attention_probe')
    checks = importlib.import_module('scripts.enqueue_immune_refined')
    require(canonical(process.__file__) == runtime/PROCESS_HELPER and sha(process.__file__) == PROCESS_HELPER_SHA256
            and not process.torch.cuda.is_initialized(), 'Original generic supervisor source/CUDA state differs')
    checks.validate_job(evidence, inner_job(args), args.stage)  # No output/index before own original25 gate.
    require(re.fullmatch('[0-7]', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'One claimed pool GPU required')
    inner = child.validate_output(args.variant, evidence/'runs'/args.output_dir.name, args.stage)
    index = canonical(evidence/'results'/(args.output_dir.name+'.json'))
    require(not index.exists(), 'Never overwrite an actual learning result')
    evidence.joinpath('runs').mkdir(exist_ok=True)
    evidence.joinpath('results').mkdir(exist_ok=True)
    args.output_dir.mkdir()
    command = [sys.executable, '-I', '-B', '-u', str(CHILD), '--slot', args.variant,
               '--stage', args.stage, '--output-dir', str(inner)]
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
        original = checks.verify_diagnostic(evidence, slot['variant'], args.steps, require_learning=args.steps == 500)
        require(original == summary and package.verify_loaded_runtime(runtime, manifest) == manifest
                and software_binding(args.variant) == sources and not process.torch.cuda.is_initialized(), 'Original learning verification/source/supervisor state differs')
        elapsed = time.monotonic()-started
        require(elapsed <= TIMEOUTS[args.stage], 'Whole learning stage budget exceeded')
        outer = dict(kind='synthetic_learning_transport', source_namespace=str(runtime),
            child_inner_result=str(index), evidence_root=str(evidence), learning_only=True,
            stage=args.stage, outer_transport=dict(variant=args.variant, mode=args.mode, steps=args.steps),
            source_evidence=sources, learning_seed=42, formal_source_profile_seed=slot['formal_source_profile_seed'],
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
    if argv == ['--help']:
        return legacy.main(argv)
    alias = legacy.legacy_module().selected_variant(argv)  # Preserve original D00 refusal before dispatch.
    if alias not in SLOTS:
        return legacy.main(argv)
    return run_learning(validate_argv(argv))


if __name__ == '__main__':
    raise SystemExit(main())
