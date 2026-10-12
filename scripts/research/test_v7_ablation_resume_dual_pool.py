"""Run only the new dual-pool/retained-lock control after conda pllm.

CUDA_VISIBLE_DEVICES='' python -B /vepfs-mlp2/c20250601/251105016/project/
dllm_test/scripts/research/test_v7_ablation_resume_dual_pool.py
Reuses fixture constructors, not the previous five-control main. All pools and
checkpoint bytes are temporary; Popen is mocked and no actual process starts.
"""
import builtins
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
SOURCE = ROOT/'scripts/research/pool_v7_ablation_resume.py'
FIXTURE = ROOT/'scripts/research/test_pool_v7_ablation_resume.py'
OUT = ROOT/'output/v7_ablation_resume_dual_pool_controls_20261003'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    OUT.mkdir(exist_ok=False)
    started, status, error, before, observations = time.monotonic(), 'failed', None, {}, {}
    original_import = builtins.__import__
    def cold(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'numpy', 'transformers', 'accelerate', 'dllm'):
            raise AssertionError('No ML imports')
        return original_import(name, *args, **kwargs)
    try:
        with ExitStack() as guards:
            guards.enter_context(patch('builtins.__import__', side_effect=cold))
            guards.enter_context(patch('subprocess.Popen', side_effect=AssertionError('No actual child')))
            guards.enter_context(patch('socket.socket', side_effect=AssertionError('No listener')))
            guards.enter_context(patch('os.execv', side_effect=AssertionError('No exec')))
            module = runpy.run_path(str(SOURCE), run_name='_dual_pool_control')
            fixture_api = runpy.run_path(str(FIXTURE), run_name='_fixture_only_no_old_main')
            g = module['prepare'].__globals__
            api = module['base_api']()
            reject, write = fixture_api['rejected'], fixture_api['write']
            before = {str(path): sha(path) for path in (SOURCE, FIXTURE, Path(__file__), g['BASE'], g['CATALOG'], g['PLAN'], g['FREEZE'])}
            assert not g['TICKETS'].exists()
            with tempfile.TemporaryDirectory(prefix='dual_pool_fakecontrol_', dir=OUT) as temporary:
                root = Path(temporary)
                fixture = fixture_api['Fixture'](root/'origin_fixture', g, api)
                attempt_pool = root/'new_attempt_pool'
                for name in ('runs', 'pending', 'running', 'workers', 'results', 'finished', 'profiles'):
                    (attempt_pool/name).mkdir(parents=True, exist_ok=True)
                fixture.outer = attempt_pool/'runs'/fixture.attempt
                fixture.ticket.update(attempt_pool=str(attempt_pool), attempt_run=str(fixture.outer))
                fixture.argv[fixture.argv.index('--output-dir')+1] = str(fixture.outer)
                fixture.argv[fixture.argv.index('--profile-dir')+1] = str(attempt_pool/'profiles'/fixture.row['alias'])
                fixture.refresh()
                with fixture.context():
                    prepared = fixture.prepare()  # Changed dual-root translation through the real cold base.
                    assert prepared['outer'].is_relative_to(attempt_pool)
                    assert prepared['inner'].is_relative_to(fixture.pool)
                    assert prepared['command'][prepared['command'].index('--output-dir')+1] == str(fixture.inner)
                    assert prepared['command'][-1] == '--resume' and not fixture.outer.exists()
                    command = prepared['command'][:-1]
                    observations.update(origin_pool=str(fixture.pool), attempt_pool=str(attempt_pool),
                                        command=prepared['command'], actual_cold_base_calls=1)
                # No repeated source/capacity checks for the remaining new ownership boundaries.
                with fixture.context(command):
                    old_live = fixture.pool/'running'/f'{fixture.row["job_id"]}.json'
                    write(old_live, fixture.row['job'])
                    reject(fixture.prepare, 'Another pending/running'); old_live.unlink()
                    old_worker = fixture.pool/'workers/old_worker.json'
                    write(old_worker, dict(state='running', job=fixture.ticket['job'], job_id=fixture.attempt, pid=os.getpid()))
                    reject(fixture.prepare, 'Another pending/running'); old_worker.unlink()
                    new_worker = attempt_pool/'workers/new_worker.json'
                    write(new_worker, dict(state='running', job=fixture.ticket['job'], job_id=fixture.attempt, pid=os.getpid()+1000000))
                    reject(fixture.prepare, 'another controller')
                    write(new_worker, dict(state='running', job=fixture.ticket['job'], job_id=fixture.attempt, pid=os.getpid()))
                    write(attempt_pool/'running'/f'{fixture.attempt}.json', fixture.ticket['job'])
                    fixture.prepare()
                    wrong = fixture.argv.copy()
                    wrong[wrong.index('--output-dir')+1] = str(fixture.pool/'runs'/fixture.attempt)
                    reject(lambda: g['prepare'](wrong, ticket_catalog_sha256=sha(fixture.tickets)), 'directories differ')
                    write(attempt_pool/'results'/f'{fixture.attempt}.json', {'artificial_terminal': True})
                    reject(fixture.prepare, 'already terminal')
                    observations['ownership_boundaries'] = ['old_running_denied', 'old_current_ID_worker_denied',
                        'new_wrong_PID_denied', 'new_current_PID_allowed', 'old_outer_denied', 'new_terminal_denied']
                    print('PASS real artificial dual-pool translation keeps frozen inner/origin guards; both pools reject conflicting ownership')

                    inherited = None
                    try:
                        with module['cell_lock'](fixture.row['job_id']) as lock_fd:
                            inherited = os.dup(lock_fd)  # Same open-file description as an inherited child FD.
                            class UnreapedChild:
                                pid, returncode = 987654321, None
                                def wait(self, timeout):
                                    raise subprocess.TimeoutExpired(['mocked_unreaped'], timeout)
                                def poll(self):
                                    return None
                                def terminate(self):
                                    pass
                                def kill(self):
                                    pass
                            with patch('subprocess.Popen', return_value=UnreapedChild()) as spawn, \
                                    patch('os.getpgid', return_value=os.getpgrp()):
                                failed = module['supervise'](['mocked_unreaped'], 1, lock_fd)
                                assert spawn.call_args.kwargs['pass_fds'] == (lock_fd,)
                                assert spawn.call_args.kwargs['start_new_session'] is False
                                assert not failed['child_reaped'] and failed['error']
                        def contender():
                            with module['cell_lock'](fixture.row['job_id']):
                                pass
                        reject(contender, kinds=(BlockingIOError,))
                        reject(lambda: module['completed'](prepared, failed), 'No successful actual child')
                        os.close(inherited); inherited = None
                        with module['cell_lock'](fixture.row['job_id']):
                            pass
                    finally:
                        if inherited is not None:
                            os.close(inherited)
                    observations['retained_lock'] = 'actual dup FD blocks second writer after parent close until final reference closes'
                    observations['actual_Popen'] = False
                    print('PASS inherited flock FD contract keeps exclusion after parent close/uncertain mocked cleanup; no false completion')
            assert not list(OUT.glob('dual_pool_fakecontrol_*')) and 'torch' not in sys.modules
            assert all(sha(Path(path)) == expected for path, expected in before.items())
            status = 'passed'
    except BaseException:
        error = traceback.format_exc()
        print(error)
    finally:
        (OUT/'receipt.json').write_text(json.dumps(dict(status=status, at_utc=datetime.now(timezone.utc).isoformat(),
            execution_count=1, elapsed_seconds=time.monotonic()-started, observations=observations, error=error,
            source_sha256_before=before, source_sha256_after={path: sha(Path(path)) for path in before},
            python=sys.executable, scope='only_new_dual_pool_and_retained_FD_boundary',
            prior_five_controls_rerun=False, actual_model_GPU_or_child_executed=False,
            real_checkpoint_read_or_hashed=False, real_ticket_created=False,
            temporary_remaining=[str(path) for path in OUT.glob('dual_pool_fakecontrol_*')]), indent=2)+'\n')
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
