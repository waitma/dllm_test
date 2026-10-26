"""Check only the new dispatcher with stdlib temporary metadata and mocked exec.

Run once after activating pllm:
  source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh
  conda activate pllm
  CUDA_VISIBLE_DEVICES='' python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_pool_formal_dispatch.py \
    --evidence-dir /project_c20250601/251105016/project/dllm_test/output/v7_formal_dispatch_controls_20261002

Fixtures are explicitly synthetic_fakecontrol dispatcher metadata, not complete
formal plans or admission certificates. No wrapper, legacy main, model, process,
socket, queue, data loader, or GPU is executed. All fixture files are removed.
"""
import argparse
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
SOURCE = ROOT / 'scripts/research/pool_formal_dispatch.py'
WRAPPER = ROOT / 'scripts/research/formal_pool_job.py'
PREVIOUS = ROOT / 'scripts/research/pool_context_ablation_dispatch.py'
EXPECTED_SOURCE = '19f92a73b7c5cd8c2e9448255040d207662bb79317bc26ac352882aeb3bcbea8'
RAW = []
TMP_PARENT = None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class ExecIntercepted(Exception):
    pass


class DispatcherControls(unittest.TestCase):
    def setUp(self):
        self.assertEqual(sha(SOURCE), EXPECTED_SOURCE)
        namespace = runpy.run_path(str(SOURCE), run_name='_synthetic_dispatcher_control')
        self.g = namespace['prepare'].__globals__
        self.assertIs(namespace['main'].__globals__, self.g)
        self.temporary = tempfile.TemporaryDirectory(prefix='synthetic_fakecontrol_', dir=TMP_PARENT)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.own, self.pool, self.data = (self.base / name for name in ('formal_own', 'pool', 'unread_data'))
        self.own.mkdir()
        (self.pool / 'runs').mkdir(parents=True)
        (self.pool / 'profiles').mkdir()
        self.wrapper = self.own / 'formal_pool_job.py'
        self.wrapper.write_bytes(WRAPPER.read_bytes())
        self.plan_path, self.catalog_path = self.own / 'plan.json', self.own / 'catalog.json'
        self.aliases = self.g['ALIASES']
        self.jobs = [f'{9000 + index}_{alias}_1234abcd' for index, alias in enumerate(self.aliases)]
        self.plan = {'schema': 'v7.formal.pool_group_plan.v1', 'synthetic_fakecontrol': True,
                     'scope': 'dispatcher_envelope_only_never_execute', 'pool_dir': str(self.pool),
                     'segment_steps': 1000, 'work': {'data_dir': str(self.data)},
                     'wrapper': {'path': str(self.wrapper), 'sha256': sha(self.wrapper)},
                     'tickets': {job: {'variant': alias, 'mode': 'train', 'steps': 1000,
                                      'data_dir': str(self.data)} for alias, job in zip(self.aliases, self.jobs)}}
        self.reset_plan()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(self.g, {'OWN': self.own, 'POOL': self.pool,
            'DATA': self.data, 'CATALOG': self.catalog_path}))
        self.stack.enter_context(patch('subprocess.Popen', side_effect=AssertionError('No process in dispatcher controls')))
        self.stack.enter_context(patch('socket.socket', side_effect=AssertionError('No socket in dispatcher controls')))

    def reset_plan(self, value=None):
        write(self.plan_path, self.plan if value is None else value)
        ref = {'path': str(self.plan_path), 'sha256': sha(self.plan_path)}
        self.catalog = {'schema': 'v7.formal.pool_job_catalog.v1', 'synthetic_fakecontrol': True,
                        'jobs': {job: dict(ref) for job in self.jobs}}
        write(self.catalog_path, self.catalog)

    def argv(self, index=0):
        return ['--variant', self.aliases[index], '--mode', 'train', '--steps', '1000',
                '--data-dir', str(self.data), '--profile-dir', str(self.pool / 'profiles' / self.aliases[index]),
                '--output-dir', str(self.pool / 'runs' / self.jobs[index])]

    def test_eight_tickets_exec_same_pid_contract_and_mask_not_suffix(self):
        pid, commands = os.getpid(), []
        for index in range(8):
            args, command = self.g['prepare'](self.argv(index))
            self.assertEqual(args.variant, self.aliases[index])
            expected = [sys.executable, '-I', '-B', '-u', str(self.wrapper), '--plan', str(self.plan_path),
                        '--plan-sha', sha(self.plan_path), '--job-id', self.jobs[index]]
            self.assertEqual(command, expected)
            # Opposite ticket suffix and physical mask must still dispatch. The
            # actual slot wrapper, not this ticket, determines the future rank.
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': str(7-index)}), \
                    patch.object(self.g['os'], 'execv', side_effect=ExecIntercepted) as execute:
                with self.assertRaises(ExecIntercepted):
                    self.g['main'](self.argv(index))
                execute.assert_called_once_with(sys.executable, expected)
            self.assertEqual(os.getpid(), pid)
            self.assertNotIn('--rank', command)
            commands.append({'ticket_index': index, 'declared_mask': str(7-index), 'command': command})
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}), patch.object(self.g['os'], 'execv') as execute:
            with self.assertRaises(ValueError):
                self.g['main'](self.argv())
            execute.assert_not_called()
        RAW.append({'control': 'eight_ticket_exec_contract', 'commands': commands,
                    'exec_mocked': True, 'actual_exec_executed': False, 'pid_before_and_after': pid,
                    'plan_sha256': sha(self.plan_path), 'catalog_sha256': sha(self.catalog_path),
                    'wrapper_actual_source_sha256': sha(self.wrapper)})

    def test_envelope_catalog_group_and_actual_hash_rejections(self):
        wrong = copy.deepcopy(self.plan)
        wrong['tickets'][self.jobs[0]]['steps'] = 999
        self.reset_plan(wrong)
        with self.assertRaisesRegex(ValueError, 'envelope'):
            self.g['prepare'](self.argv())
        wrong = copy.deepcopy(self.plan)
        wrong['tickets'].pop(self.jobs[-1])
        self.reset_plan(wrong)
        with self.assertRaises(ValueError):
            self.g['prepare'](self.argv())
        self.reset_plan()
        self.catalog['jobs'][self.jobs[-1]] = {'path': str(self.plan_path), 'sha256': '0'*64}
        write(self.catalog_path, self.catalog)
        with self.assertRaisesRegex(ValueError, 'All eight tickets'):
            self.g['prepare'](self.argv())
        self.reset_plan()
        self.plan_path.write_bytes(self.plan_path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.g['prepare'](self.argv())
        self.reset_plan()
        self.wrapper.write_bytes(self.wrapper.read_bytes() + b'\n# synthetic_fakecontrol mutation\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.g['prepare'](self.argv())
        self.catalog_path.unlink()
        with self.assertRaisesRegex(ValueError, 'canonical regular file'):
            self.g['prepare'](self.argv())
        RAW.append({'control': 'envelope_catalog_source_rejections',
                    'rejected': ['wrong_current_job', 'seven_tickets', 'split_catalog_reference',
                                 'actual_plan_bytes_changed', 'actual_wrapper_bytes_changed', 'missing_catalog']})

    def test_canonical_paths_symlinks_and_output_escape_rejected(self):
        argv = self.argv()
        argv[argv.index('--data-dir')+1] = str(self.base / 'other_unread_data')
        with self.assertRaises(ValueError):
            self.g['prepare'](argv)
        external = self.base / 'outside_pool'
        external.mkdir()
        output = self.pool / 'runs' / self.jobs[0]
        output.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'canonical and nonsymlink'):
            self.g['prepare'](self.argv())
        output.unlink()
        (self.pool / 'runs').rmdir()
        (self.pool / 'runs').symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'canonical and nonsymlink'):
            self.g['prepare'](self.argv())
        (self.pool / 'runs').unlink()
        (self.pool / 'runs').mkdir()
        linked_plan = self.own / 'linked_plan.json'
        linked_plan.symlink_to(self.plan_path)
        bad_ref = {'path': str(linked_plan), 'sha256': sha(self.plan_path)}
        with self.assertRaisesRegex(ValueError, 'canonical regular file'):
            self.g['reference'](bad_ref)
        with self.assertRaisesRegex(ValueError, 'registered output root'):
            self.g['reference']({'path': str(self.base / 'not_in_own.json'), 'sha256': '0'*64})
        RAW.append({'control': 'canonical_path_boundaries', 'rejected':
                    ['wrong_data_path', 'legal_job_symlink_escape', 'runs_parent_symlink', 'plan_symlink', 'outside_formal_root']})

    def test_previous_p02_argv_unchanged_and_source_pin(self):
        argv = ['--variant=p02_v7_f111_seed43', '--mode', 'train', '--steps', '3750',
                '--data-dir', '/synthetic_fakecontrol/not_read', '--profile-dir', '/synthetic_fakecontrol/not_read',
                '--output-dir', '/synthetic_fakecontrol/not_created']
        before = sha(PREVIOUS)
        self.assertEqual(before, self.g['PREVIOUS_SHA'])
        previous_main = Mock(return_value='synthetic_fakecontrol_delegated')
        with patch.object(self.g['runpy'], 'run_path', return_value={'main': previous_main}) as loader, \
                patch.object(self.g['os'], 'execv') as execute:
            self.assertEqual(self.g['main'](argv), 'synthetic_fakecontrol_delegated')
            loader.assert_called_once_with(str(PREVIOUS))
            previous_main.assert_called_once_with(argv)
            execute.assert_not_called()
        self.assertEqual(sha(PREVIOUS), before)
        changed = self.base / 'synthetic_fakecontrol_previous.py'
        changed.write_bytes(PREVIOUS.read_bytes() + b'\n# changed CPU control copy\n')
        with patch.dict(self.g, {'PREVIOUS': changed}), patch.object(self.g['runpy'], 'run_path') as loader:
            with self.assertRaisesRegex(ValueError, 'source changed'):
                self.g['main'](argv)
            loader.assert_not_called()
        RAW.append({'control': 'legacy_delegation', 'argv': argv, 'actual_previous_sha256': before,
                    'legacy_main_mocked': True, 'changed_copy_rejected_before_loading': True})


def main():
    global TMP_PARENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', required=True, type=Path)
    args = parser.parse_args()
    if not args.evidence_dir.is_absolute() or args.evidence_dir != args.evidence_dir.resolve():
        raise ValueError('Use an absolute canonical own evidence directory')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU control caller must explicitly hide CUDA devices')
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    TMP_PARENT = args.evidence_dir / '_cpu_tmp'
    TMP_PARENT.mkdir()
    before = {str(path): sha(path) for path in (SOURCE, WRAPPER, PREVIOUS)}
    started = time.monotonic()
    stream = io.StringIO()
    result = None
    try:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(DispatcherControls))
    finally:
        stdout = stream.getvalue()
        print(stdout, end='')
        (args.evidence_dir / 'stdout.log').write_text(stdout)
        remaining = [str(path) for path in TMP_PARENT.iterdir()]
        after = {str(path): sha(path) for path in (SOURCE, WRAPPER, PREVIOUS)}
        passed = result is not None and result.wasSuccessful() and not remaining and before == after
        record = {'status': 'passed' if passed else 'failed', 'finished_at': datetime.now(timezone.utc).isoformat(),
                  'cases': 0 if result is None else result.testsRun, 'elapsed_seconds': time.monotonic()-started,
                  'python_executable': sys.executable, 'command': [sys.executable, '-B', str(Path(__file__).resolve()),
                      '--evidence-dir', str(args.evidence_dir)], 'source_sha256': before, 'source_unchanged': before == after,
                  'test_source_sha256': sha(Path(__file__)), 'stdout_sha256': sha(args.evidence_dir / 'stdout.log'),
                  'scope': 'synthetic_fakecontrol_dispatcher_only', 'model_executed': False, 'GPU_executed': False,
                  'real_data_read': False, 'real_exec_or_legacy_main_executed': False, 'published': False,
                  'fixture_remaining': remaining, 'controls': RAW}
        (args.evidence_dir / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        if not remaining:
            TMP_PARENT.rmdir()
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
