"""Five new continuation controls; activate conda pllm before running.

CUDA_VISIBLE_DEVICES='' python -B /vepfs-mlp2/c20250601/251105016/project/
dllm_test/scripts/research/test_pool_v7_ablation_resume.py
Only artificial TemporaryDirectory checkpoint bytes are hashed, never loaded.
The actual frozen stdlib prepare runs once with a temporary pool namespace;
child execution is always mocked. No actual tickets, data rows or GPU calls.
"""
import builtins
import copy
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
OUT = ROOT/'output/v7_ablation_resume_controls_20261003'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def rejected(call, contains=None, kinds=(ValueError, FileNotFoundError)):
    try:
        call()
    except kinds as error:
        assert contains is None or contains in str(error), str(error)
        return str(error)
    raise AssertionError('Expected new recovery-boundary rejection')


class Fixture:
    def __init__(self, root, g, api):
        self.root, self.pool, self.g, self.api = root, root/'pool', g, api
        self.tickets = root/'tickets.json'
        self.row = json.loads(g['CATALOG'].read_text())['jobs'][0]
        self.inner = self.pool/'runs'/self.row['job_id']
        self.attempt = '9999999999999999999_'+self.row['alias']+'_c0decafe'
        self.outer = self.pool/'runs'/self.attempt
        for name in ('runs', 'pending', 'running', 'workers', 'results', 'finished', 'profiles'):
            (self.pool/name).mkdir(parents=True, exist_ok=True)
        self.inner.mkdir()
        freeze = json.loads(g['FREEZE'].read_text())
        plan = json.loads(g['PLAN'].read_text())
        sources = {name: freeze['source_sha256'][name] for name in
            ('scripts/research/run_v7_large_batch_ablation.py', 'scripts/research/run_v7_architecture_pilot.py', 'scripts/research/context_pilot_metrics.py')}
        self.provenance = dict(variant=self.row['variant'], seed=self.row['seed'],
            kind='exploratory_architecture_reconstruction', formal_training=False, biological_quality_claim=False,
            generation=False, micro_batch=32, grad_accum=1, global_batch=32, scheduler_horizon=3750,
            evaluation_interval=250, warmup=100, learning_rate=1e-4, source_sha256=sources,
            data={'sha256': api['_runtime']['SAMPLE_SHA'], 'path': str(g['DATA']/'sample_manifest.json')},
            study={'name': g['STUDY'], 'evaluation_micro_batch': 4,
                   'batch_plan': {'path': str(g['PLAN']), 'sha256': g['PLAN_SHA'], 'profile_evidence': plan['profile_evidence']}})
        self.summary = dict(step=250, records=8000, record_exposures=8000, epoch=0, position=8000)
        self.initial = dict(step=0, records=0, record_exposures=0, epoch=0, position=0)
        write(self.inner/'run_provenance.json', self.provenance)
        write(self.inner/'summary.json', self.summary)
        self.curve([self.initial, self.summary])
        (self.inner/'latest.pt').write_bytes(b'ARTIFICIAL_CHECKPOINT_NOT_A_TORCH_FILE')
        result_path = self.pool/'results'/f'{self.row["job_id"]}.json'
        finished_path = self.pool/'finished'/result_path.name
        write(result_path, dict(job_id=self.row['job_id'], job=self.row['job'], state='failed', finished=1.))
        write(finished_path, self.row['job'])
        job = dict(variant=self.row['alias'], mode='train', steps=3500, data_dir=str(g['DATA']),
                   resume_from_checkpoint=str(self.inner/'latest.pt'))
        self.ticket = dict(original_job_id=self.row['job_id'], original_run=str(self.inner), attempt_run=str(self.outer),
            attempt_pool=str(self.pool), prior_step=250, timeout_seconds=60, job=job, evidence={})
        self.paths = dict(result=result_path, finished=finished_path, provenance=self.inner/'run_provenance.json',
                          summary=self.inner/'summary.json', curve=self.inner/'curve.jsonl', checkpoint=self.inner/'latest.pt')
        self.book = dict(schema='v7.ablation.resume_tickets.v1', study=g['STUDY'], automatic_retry=False,
            original_catalog={'path': str(g['CATALOG']), 'sha256': g['CATALOG_SHA']},
            batch_plan={'path': str(g['PLAN']), 'sha256': g['PLAN_SHA']},
            source_freeze={'path': str(g['FREEZE']), 'sha256': g['FREEZE_SHA']},
            dispatcher={'path': str(g['BASE']), 'sha256': g['BASE_SHA']}, tickets={self.attempt: self.ticket})
        self.refresh()
        self.argv = ['--variant', self.row['alias'], '--mode', 'train', '--steps', '3500',
            '--data-dir', str(g['DATA']), '--profile-dir', str(self.pool/'profiles'/self.row['alias']),
            '--output-dir', str(self.outer), '--resume-from-checkpoint', str(self.inner/'latest.pt')]

    def refresh(self):
        self.ticket['evidence'] = {key: {'path': str(path), 'sha256': sha(path)} for key, path in self.paths.items()}
        write(self.tickets, self.book)

    def curve(self, points):
        (self.inner/'curve.jsonl').write_text(''.join(json.dumps(point)+'\n' for point in points))

    def context(self, command=None):
        stack = ExitStack()
        # Real code, same guards; only filesystem pool locations are relocated.
        stack.enter_context(patch.dict(self.api['_runtime'], {'POOL': self.pool}))
        if command is not None:
            modified = command.copy()
            modified[modified.index('--output-dir')+1] = str(self.inner)
            stack.enter_context(patch.dict(self.api, {'prepare': lambda argv: (None, modified)}))
        original_sha = self.g['sha']
        def checkpoint_guard(path):
            if Path(path).name == 'latest.pt':
                assert Path(path).is_relative_to(self.root), 'No real checkpoint hashing permitted'
            return original_sha(path)
        stack.enter_context(patch.dict(self.g, {'POOL': self.pool, 'TICKETS': self.tickets, 'LOCKS': self.root/'locks',
            'base_api': lambda: self.api, 'sha': checkpoint_guard}))
        return stack

    def prepare(self):
        return self.g['prepare'](self.argv, ticket_catalog_sha256=sha(self.tickets))

    def final_inner(self):
        final = dict(step=3750, records=120000, record_exposures=120000, epoch=1, position=0)
        write(self.inner/'summary.json', final)
        self.curve([self.initial, self.summary, final])
        (self.inner/'latest.pt').write_bytes(b'ARTIFICIAL_FINAL_CHECKPOINT_NOT_LOADED')
        write(self.inner/'COMPLETE.json', dict(status='completed', global_step=3750,
            kind='exploratory_architecture_reconstruction', formal_training=False, biological_quality_claim=False,
            summary_sha256=sha(self.inner/'summary.json'), provenance_sha256=sha(self.inner/'run_provenance.json')))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    OUT.mkdir(exist_ok=False)
    started, records, error, status, before = time.monotonic(), [], None, 'failed', {}
    original_import = builtins.__import__
    def cold(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'numpy', 'transformers', 'accelerate', 'dllm'):
            raise AssertionError('No ML imports')
        return original_import(name, *args, **kwargs)
    try:
        with ExitStack() as denied:
            denied.enter_context(patch('builtins.__import__', side_effect=cold))
            denied.enter_context(patch('subprocess.Popen', side_effect=AssertionError('No actual child')))
            denied.enter_context(patch('socket.socket', side_effect=AssertionError('No listener')))
            denied.enter_context(patch('os.execv', side_effect=AssertionError('No exec')))
            module = runpy.run_path(str(SOURCE), run_name='_cold_resume_control')
            g = module['prepare'].__globals__
            assert not g['TICKETS'].exists()
            rejected(lambda: module['main']([]), 'No immutable')
            api = module['base_api']()
            before = {str(path): sha(path) for path in (SOURCE, Path(__file__), g['BASE'], g['CATALOG'], g['PLAN'], g['FREEZE'], api['_runtime']['REGISTRY'])}
            with tempfile.TemporaryDirectory(prefix='resume_synthetic_fakecontrol_', dir=OUT) as directory:
                root = Path(directory)
                first = Fixture(root/'positive', g, api)
                with first.context():
                    prepared = first.prepare()
                    command = prepared['command'][:-1]
                    assert prepared['command'] == command+['--resume']
                    assert command[command.index('--steps')+1] == '3750'
                    assert command[command.index('--micro-batch')+1] == '32'
                    assert command[command.index('--grad-accum')+1] == '1'
                    assert prepared['args'].steps == 3500 and prepared['outer'] != prepared['inner']
                    assert not first.outer.exists()
                records.append({'control': 'missing_real_catalog_and_actual_cold_prepare_translation_once',
                    'temporary_pool_namespace': True, 'remaining_steps': 3500, 'command': prepared['command'],
                    'real_checkpoint_read': False})
                print('PASS missing real tickets rejects; actual SHA-bound stdlib prepare translates only --resume with fixed3750 inner command')

                second = Fixture(root/'negative', g, api)
                failures = []
                with second.context(command):
                    live = second.pool/'running'/f'{second.row["job_id"]}.json'
                    write(live, second.row['job'])
                    failures.append(rejected(second.prepare, 'Another pending/running'))
                    live.unlink()
                    second.ticket['job']['steps'] = 3750; second.refresh()
                    failures.append(rejected(second.prepare, 'Remaining updates'))
                    second.ticket['job']['steps'] = 3500; second.refresh()
                    wrong = dict(second.summary, step=500)
                    write(second.inner/'summary.json', wrong); second.refresh()
                    failures.append(rejected(second.prepare, 'transaction'))
                    write(second.inner/'summary.json', second.summary); second.refresh()
                    previous = copy.deepcopy(second.provenance)
                    previous['study']['name'] = 'v7_context_factorial_120k_v1'
                    write(second.inner/'run_provenance.json', previous); second.refresh()
                    failures.append(rejected(second.prepare, 'old p02'))
                    write(second.inner/'run_provenance.json', second.provenance); second.refresh()
                    (second.inner/'latest.pt').write_bytes(b'changed artificial checkpoint')
                    failures.append(rejected(second.prepare, 'Original bytes changed'))
                records.append({'control': 'live_steps_curve_p02_checkpoint_rejections', 'failures': failures})
                print('PASS live same-cell, wrong remaining budget, curve transaction, old p02 and bound fake checkpoint drift reject')

                with patch.dict(g, {'BASE_SHA': '0'*64}):
                    rejected(module['base_api'], 'Frozen large-v2')
                third = Fixture(root/'source', g, api)
                with third.context(command):
                    previous = copy.deepcopy(third.provenance)
                    previous['source_sha256']['scripts/research/run_v7_architecture_pilot.py'] = '0'*64
                    write(third.inner/'run_provenance.json', previous); third.refresh()
                    rejected(third.prepare, 'Loaded source identity')
                    rejected(lambda: g['prepare'](third.argv, ticket_catalog_sha256='0'*64), 'Original bytes changed')
                records.append({'control': 'dispatcher_loaded_source_catalog_SHA_rejections'})
                print('PASS dispatcher SHA, loaded model source and recovery catalog SHA drift reject')

                fourth = Fixture(root/'failure', g, api)
                with fourth.context(command):
                    with module['cell_lock'](fourth.row['job_id']):
                        def second_lock():
                            with module['cell_lock'](fourth.row['job_id']):
                                pass
                        rejected(second_lock, kinds=(BlockingIOError,))
                    class FakeChild:
                        pid, returncode, calls = 987654321, None, 0
                        def wait(self, timeout):
                            self.calls += 1
                            if self.calls == 1:
                                raise subprocess.TimeoutExpired(['mocked_child'], timeout)
                            self.returncode = -15
                            return -15
                        def poll(self):
                            return self.returncode
                        def terminate(self):
                            pass
                        def kill(self):
                            raise AssertionError('Graceful fake child already reaped')
                    child = FakeChild()
                    with module['cell_lock'](fourth.row['job_id']) as lock_fd, \
                            patch('subprocess.Popen', return_value=child) as spawn, patch('os.getpgid', return_value=os.getpgrp()):
                        failed = module['supervise'](['mocked_child'], 1, lock_fd)
                        assert failed['child_reaped'] and not failed['normal_wait_returned'] and failed['error']
                        assert spawn.call_args.kwargs['start_new_session'] is False
                    with patch.dict(g, {'supervise': lambda *args: failed}), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
                        rejected(lambda: module['main'](fourth.argv, ticket_catalog_sha256=sha(fourth.tickets)), 'No successful actual child')
                    assert (fourth.outer/'FAILURE.json').is_file() and not (fourth.outer/'COMPLETE.json').exists()
                records.append({'control': 'same_cell_lock_timeout_cleanup_and_no_false_completion', 'child_all_mocked': True})
                print('PASS same-cell lock excludes second writer; mocked timeout child reaped; failed child cannot create COMPLETE')

                fifth = Fixture(root/'success', g, api)
                with fifth.context(command):
                    def success(*args):
                        # Actual held lock is checked inside the mocked child lifetime.
                        def conflict():
                            with module['cell_lock'](fifth.row['job_id']):
                                pass
                        rejected(conflict, kinds=(BlockingIOError,))
                        fifth.final_inner()
                        return dict(pid=123456789, parent_pid=os.getpid(), process_group_id=os.getpgrp(), returncode=0,
                            wait_returned=True, normal_wait_returned=True, child_reaped=True, error=None, elapsed_seconds=0.)
                    with patch.dict(g, {'supervise': success}), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
                        assert module['main'](fifth.argv, ticket_catalog_sha256=sha(fifth.tickets)) == 0
                    outer = json.loads((fifth.outer/'COMPLETE.json').read_text())
                    assert outer['global_step'] == 3750 and outer['new_optimizer_updates'] == 3500
                    assert outer['original_run'] == str(fifth.inner)
                    assert outer['inner_evidence']['completion']['sha256'] == sha(fifth.inner/'COMPLETE.json')
                records.append({'control': 'outer_completion_requires_mocked_wait_and_bound_final_inner',
                    'artificial_completion_removed_with_fixture': True, 'actual_GPU_resume_claim': False})
                print('PASS only bound artificial final3750/120000/epoch1 + mocked successful wait permits outer reference receipt')
            assert not list(OUT.glob('resume_synthetic_fakecontrol_*')) and 'torch' not in sys.modules
            assert all(sha(Path(path)) == expected for path, expected in before.items())
            assert not g['TICKETS'].exists()
            status = 'passed'
    except BaseException:
        error = traceback.format_exc()
        print(error)
    finally:
        (OUT/'receipt.json').write_text(json.dumps(dict(status=status, at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-started, execution_count=1, controls=records, error=error,
            source_sha256_before=before, source_sha256_after={path: sha(Path(path)) for path in before},
            python=sys.executable, scope='new_resume_ticket_translation_lock_child_metadata_controls_only',
            real_checkpoint_read_or_hashed=False, real_tickets_created=False, models_or_GPU_executed=False,
            actual_Popen_or_exec=False, public_registry_queue_or_data_modified=False, old_suites_repeated=False,
            temporary_remaining=[str(path) for path in OUT.glob('resume_synthetic_fakecontrol_*')]), indent=2)+'\n')
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
