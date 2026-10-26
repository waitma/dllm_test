"""Run only the changed capacity-v2 cold controls, once, after conda pllm.

CUDA_VISIBLE_DEVICES='' python -B /project_c20250601/251105016/project/
dllm_test/scripts/research/test_pool_large_batch_capacity_v2_dispatch.py
No model, profiler run, old job main, exec, socket, data or queue is allowed.
All mutation controls use TemporaryDirectory synthetic_fakecontrol metadata.
"""
from contextlib import ExitStack, redirect_stderr
from datetime import datetime, timezone
import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time
import traceback
from unittest.mock import Mock, patch

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
SOURCE = ROOT/'scripts/research/pool_large_batch_capacity_v2_dispatch.py'
OUT = ROOT/'output/v7_capacity_v2_dispatch_controls_20261003'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def rejected(call):
    with redirect_stderr(io.StringIO()):
        try:
            call()
        except (ValueError, KeyError, SystemExit, FileNotFoundError) as error:
            return type(error).__name__
    raise AssertionError('Expected changed-boundary rejection')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    OUT.mkdir(exist_ok=False)
    started, records, status, error = time.monotonic(), [], 'failed', None
    before = {}
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'numpy', 'transformers', 'accelerate', 'dllm'):
            raise AssertionError('No ML imports in cold controls')
        return original_import(name, *args, **kwargs)
    try:
        module = runpy.run_path(str(SOURCE), run_name='_new_capacity_v2_control')
        g = module['prepare'].__globals__
        before = {str(path): sha(path) for path in (SOURCE, Path(__file__), g['LEDGER'], g['FREEZE'], g['PREVIOUS'])}
        ledger = json.loads(g['LEDGER'].read_text())
        argv = ['--variant', g['ALIAS'], '--mode', 'profile-synthetic', '--steps', '36',
                '--profile-dir', str(g['POOL']/'profiles'/g['ALIAS']), '--output-dir', ledger['run_dir']]
        run = Path(ledger['run_dir'])
        assert not run.exists()
        with ExitStack() as forbidden:
            forbidden.enter_context(patch('builtins.__import__', side_effect=guarded_import))
            forbidden.enter_context(patch('subprocess.Popen', side_effect=AssertionError('No subprocess')))
            forbidden.enter_context(patch('socket.socket', side_effect=AssertionError('No socket')))
            execute = forbidden.enter_context(patch('os.execv', side_effect=AssertionError('No actual exec')))
            # One new prepare against the real root-created intent, only cold code.
            args, command = module['prepare'](argv)
            profiler = runpy.run_path(command[4], run_name='_native_parser_only')
            parsed = profiler['parse_args'](command[5:])
            assert (parsed.warmup_steps, parsed.measure_steps, parsed.candidate_seconds, parsed.max_seconds) == (2, 2, 240, 1800)
            assert parsed.variants == ['f000', 'f111', 'cv_segment_global'] and parsed.micro_batches == [4, 16, 32]
            assert parsed.global_batch == 32 and parsed.seed == 42
            assert len(parsed.variants)*len(parsed.micro_batches)*(parsed.warmup_steps+parsed.measure_steps) == 36
            assert profiler['CONFIG']['max_length'] == 2048 and not run.exists()
            execute.assert_not_called()
            records.append({'control': 'actual_v2_intent_prepare_and_native_frozen_parser_once',
                'command': command, 'ledger_sha256': sha(g['LEDGER']), 'planned_updates': 36,
                'cold_legacy_prepare_only': True, 'profiler_executed': False, 'output_created': False})
            print('PASS actual v2 intent + SHA-bound legacy cold prepare + unchanged frozen parser: 9 cells, 36 updates, 240/1800 seconds')

            with tempfile.TemporaryDirectory(prefix='synthetic_fakecontrol_', dir=OUT) as temporary:
                temporary = Path(temporary)
                fake = temporary/'ledger.json'
                write(fake, ledger)
                with patch.dict(g, {'LEDGER': fake}), patch.dict(g, {'previous_module': Mock(side_effect=AssertionError('Unchanged old verifier not repeated'))}):
                    cases = 0
                    for extra in (['--steps', '36'], ['--data-dir', '/unread'], ['--resume-from-checkpoint', '/unread']):
                        rejected(lambda: module['prepare'](argv+extra)); cases += 1
                    for flag, value in (('--steps', '72'), ('--variant', 'p04_v7_capacity'),
                                        ('--mode', 'train'), ('--output-dir', str(g['OLD_RUN']))):
                        changed = argv.copy(); changed[changed.index(flag)+1] = value
                        rejected(lambda: module['prepare'](changed)); cases += 1
                    for key, value in (('automatic_retry', True), ('max_attempts', 2),
                                       ('schema', 'passed'), ('source_freeze', {'path': str(g['FREEZE']), 'sha256': '0'*64})):
                        changed = dict(ledger, **{key: value}); write(fake, changed)
                        with patch.dict(g, {'LEDGER_SHA': sha(fake)}):
                            rejected(lambda: module['prepare'](argv)); cases += 1
                    changed = dict(ledger, protocol=dict(ledger['protocol'], micro_batches=[16]))
                    write(fake, changed)
                    with patch.dict(g, {'LEDGER_SHA': sha(fake)}):
                        rejected(lambda: module['prepare'](argv)); cases += 1
                    fake.unlink(); rejected(lambda: module['prepare'](argv)); cases += 1
                records.append({'control': 'v2_exact_envelope_and_intent', 'rejections': cases,
                    'legacy_verifier_repeated': False})
                print('PASS missing/changed intent, selected-only sweep, duplicate/data/resume options and old output rejected')

                # Semantically fake success cannot be laundered by a matching SHA.
                refs = {}
                for key, reference in g['PRIOR'].items():
                    path = temporary/(key+'.json')
                    value = json.loads(Path(reference['path']).read_text())
                    if key == 'result':
                        value.update(state='succeeded', returncode=0)
                    write(path, value)
                    refs[key] = {'path': str(path), 'sha256': sha(path)}
                with patch.dict(g, {'PRIOR': refs}):
                    rejected(lambda: module['prior_failure'](refs))
                records.append({'control': 'fake_prior_passed_rejected', 'matching_fixture_SHA': True})
                print('PASS fake prior success rejected despite matching artificial metadata SHA')

                pool = temporary/'pool'
                job_id = ledger['job_id']
                fake_run = pool/'runs'/job_id
                fake_run.mkdir(parents=True)
                (fake_run/'summary.json').write_text('{}')
                revised = dict(ledger, pool_dir=str(pool), run_dir=str(fake_run))
                write(fake, revised)
                revised_argv = argv.copy()
                revised_argv[revised_argv.index('--output-dir')+1] = str(fake_run)
                revised_argv[revised_argv.index('--profile-dir')+1] = str(pool/'profiles'/g['ALIAS'])
                with patch.dict(g, {'POOL': pool, 'LEDGER': fake, 'LEDGER_SHA': sha(fake)}):
                    rejected(lambda: module['prepare'](revised_argv))
                    (fake_run/'summary.json').unlink()
                    (pool/'results').mkdir()
                    (pool/'results'/f'{job_id}.json').write_text('{}')
                    rejected(lambda: module['prepare'](revised_argv))
                records.append({'control': 'existing_output_and_terminal_attempt_rejected'})
                print('PASS existing profile output or terminal attempt cannot be reused')

            unknown = ['--variant=p02_v7_f111_seed43', '--mode', 'train', '--steps', '3750', '--opaque', 'unchanged']
            delegated = Mock(return_value='delegated_without_execution')
            with patch.object(g['runpy'], 'run_path', return_value={'main': delegated}) as load:
                assert module['main'](unknown) == 'delegated_without_execution'
                load.assert_called_once_with(str(g['PREVIOUS']))
                delegated.assert_called_once_with(unknown)
            with patch.dict(g, {'PREVIOUS_SHA': '0'*64}), patch.object(g['runpy'], 'run_path') as load:
                rejected(lambda: module['main'](unknown)); load.assert_not_called()
            with patch.dict(g, {'prepare': Mock(return_value=(args, command))}):
                for mask in ('', '0,1', '8'):
                    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': mask}):
                        rejected(lambda: module['main'](argv))
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3'}), patch('os.execv', return_value=None) as captured:
                    module['main'](argv)
                    captured.assert_called_once_with(sys.executable, command)
            records.append({'control': 'SHA_bound_unknown_argv_and_single_mask_exec_contract',
                            'actual_legacy_main_or_exec': False})
            print('PASS unknown argv unchanged through SHA-bound old capacity; single-mask exec mocked')
        assert not run.exists() and not list(OUT.glob('synthetic_fakecontrol_*'))
        assert all(sha(Path(path)) == value for path, value in before.items())
        assert 'torch' not in sys.modules
        status = 'passed'
    except BaseException:
        error = traceback.format_exc()
        print(error)
    finally:
        write(OUT/'receipt.json', {'status': status, 'at_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic()-started, 'execution_count': 1, 'controls': records,
            'source_sha256_before': before, 'source_sha256_after': {path: sha(Path(path)) for path in before},
            'test_path': str(Path(__file__).resolve()), 'python': sys.executable, 'error': error,
            'scope': 'new_stdlib_capacity_v2_dispatch_only', 'model_or_GPU_executed': False,
            'real_data_read': False, 'queue_modified': False, 'legacy_main_executed': False,
            'actual_exec_executed': False, 'temporary_remaining': [str(path) for path in OUT.glob('synthetic_fakecontrol_*')]})
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
