"""Run one bounded CPU identity/metadata observation check with no pool job.

Activate conda pllm and run:
  CUDA_VISIBLE_DEVICES='' python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_parent_runtime_observation.py

Artificial process records and argv stay in owned synthetic_fakecontrol temp
directories, which are removed. One real V4 process_identity read covers only
this CPU control process and its parent, never the remote GPU container pool.
No exec, socket, subprocess, environment dump, GPU query or dataset is used.
"""
import ast
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
SOURCE = ROOT / 'scripts/research/pool_parent_runtime_observation.py'
FROZEN_PROFILE = ROOT / 'output/v7_large_batch_ablation_20261002/source_snapshot_v1/scripts/research/profile_v7_large_batch.py'
namespace = runpy.run_path(str(SOURCE), run_name='_observer_metadata_control')
g = namespace['observe'].__globals__
API = g['coordinator_api']()
CONTROLS, CPU_IDENTITY = [], {}


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_fakecontrol_pool_parent_')
        self.root = Path(self.temp.name)
        self.pool, self.pool_source = self.root / 'pool', self.root / 'scripts/immune_research_pool.py'
        (self.pool / 'logs').mkdir(parents=True)
        self.pool_source.parent.mkdir()
        self.pool_source.write_text('"""Artificial pool source metadata; never executed."""\n')
        self.run = self.pool / 'runs' / g['JOB_ID']
        self.destination = self.pool / 'logs' / (g['JOB_ID'] + '.pool_parent_runtime_observation.json')
        self.argv = ['--variant', g['ALIAS'], '--mode', 'profile-synthetic', '--steps', '72',
            '--profile-dir', str(self.pool / 'profiles' / g['ALIAS']), '--output-dir', str(self.run)]
        self.pool_argv = ['python', '-u', 'scripts/immune_research_pool.py', 'work', '--pool-dir', str(self.pool),
            '--expected-gpus', '8', '--idle-seconds', '0', '--max-seconds', '258600', '--job-seconds', '86400', '--standby-load']
        self.parent = dict(pid=os.getppid(), parent_pid=1, parent_start_ticks=2, process_group_id=3,
                           start_ticks=4, hostname='synthetic_fakecontrol', boot_id='00000000-0000-0000-0000-000000000001')
        self.controller = {**self.parent, 'pid': os.getpid(), 'parent_pid': self.parent['pid'],
                           'parent_start_ticks': self.parent['start_ticks'], 'process_group_id': os.getpid(), 'start_ticks': 5}
        self.fake_api = {**API, 'process_identity': lambda pid: dict(self.controller if pid == os.getpid() else self.parent)}
        self.stack = ExitStack()
        self.stack.enter_context(patch.dict(g, {'ROOT': self.root, 'POOL': self.pool, 'POOL_SOURCE': self.pool_source}))
        self.stack.enter_context(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3'}))
        self.stack.enter_context(patch.dict(g, {'coordinator_api': lambda: self.fake_api,
            'read_parent_cmdline': lambda pid: list(self.pool_argv), 'read_parent_cwd': lambda pid: str(self.root)}))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()
        self.assertFalse(self.root.exists())

    def test_noncapacity_zero_reads_writes_or_environment_access(self):
        forbidden = Mock(side_effect=AssertionError('Noncapacity must be a pure no-op'))
        candidates = [[], ['--variant', 'p02_v7_f000_seed42'],
            ['--variant', g['ALIAS'], '--output-dir', '/synthetic_fakecontrol/other_job'],
            ['--variant', 'p04_v7_f111_seed42', '--output-dir', str(self.run)]]
        with patch.dict(g, {name: forbidden for name in ('canonical', 'sha', 'coordinator_api',
                        'read_parent_cmdline', 'read_parent_cwd', 'publish')}), \
                patch.object(os.environ, 'get', forbidden), patch('builtins.open', forbidden), \
                patch.object(Path, 'read_bytes', forbidden), patch.object(Path, 'read_text', forbidden), \
                patch.object(Path, 'stat', forbidden):
            for argv in candidates:
                self.assertIs(g['observe'](argv), False)
        forbidden.assert_not_called()
        CONTROLS.append('noncapacity and other job IDs: zero filesystem/environment/observer calls')

    def test_relative_parent_capture_exclusive_and_run_absent(self):
        self.assertTrue(g['observe'](self.argv))
        receipt = json.loads(self.destination.read_text())
        self.assertEqual(receipt['pool_process'], self.parent)
        self.assertEqual(receipt['controller'], self.controller)
        self.assertEqual(receipt['pool_argv'], self.pool_argv)
        self.assertEqual(receipt['pool_cwd'], str(self.root))
        self.assertEqual(receipt['resolved_pool_source'], str(self.pool_source))
        self.assertTrue(receipt['metadata_only'])
        self.assertFalse(receipt['GPU_profile_executed'])
        self.assertFalse(self.run.exists())
        before = self.destination.read_bytes()
        with self.assertRaises(FileExistsError):
            g['observe'](self.argv)
        self.assertEqual(self.destination.read_bytes(), before)
        self.assertEqual(list((self.pool / 'logs').iterdir()), [self.destination])
        CONTROLS.append('relative script resolves through observed cwd; complete fake identity/raw argv captured; exclusive no-overwrite; run absent')

    def test_existing_run_logs_unchanged_and_frozen_profile_compatible(self):
        self.run.mkdir(parents=True)
        (self.run / 'train.log').write_bytes(b'original fakecontrol log\n')
        original = {path.name: path.read_bytes() for path in self.run.iterdir()}
        self.assertTrue(g['observe'](self.argv))
        self.assertEqual({path.name: path.read_bytes() for path in self.run.iterdir()}, original)
        tree = ast.parse(FROZEN_PROFILE.read_text())
        run_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        allowed = ('train.log', 'worker.log', 'stdout.log', 'stderr.log')
        self.assertTrue(any(isinstance(node, ast.Tuple) and all(isinstance(item, ast.Constant) for item in node.elts)
            and tuple(item.value for item in node.elts) == allowed for node in ast.walk(run_node)))
        self.assertTrue(set(original) <= set(allowed))
        CONTROLS.append('existing run_dir bytes unchanged; filenames remain compatible with actual frozen profile run precondition AST')

    def test_identity_argv_cwd_path_and_duplicate_rejections(self):
        parent = copy.deepcopy(self.parent)
        self.parent['start_ticks'] += 1
        with self.assertRaises(ValueError):
            g['observe'](self.argv)
        self.parent = parent
        unstable = Mock(side_effect=[self.controller, self.parent, {**self.controller, 'start_ticks': 999}])
        with patch.dict(g, {'coordinator_api': lambda: {**API, 'process_identity': unstable}}), self.assertRaises(ValueError):
            g['observe'](self.argv)
        for extra in (['--data-dir', '/not_read'], ['--resume-from-checkpoint', '/not_read'], ['--steps', '72']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                g['observe'](self.argv + extra)
        original = self.pool_argv
        for changed in (original + ['--data-dir', '/not_read'], original + ['--expected-gpus', '8'],
                        ['python', '-c', 'not a pool process']):
            self.pool_argv = changed
            with self.assertRaises(ValueError):
                g['observe'](self.argv)
        self.pool_argv = original
        with patch.dict(g, {'read_parent_cwd': lambda pid: str(self.root / 'wrong_cwd')}), self.assertRaises(ValueError):
            g['observe'](self.argv)
        wrong = self.argv.copy()
        wrong[-1] = str(self.root / 'outside' / g['JOB_ID'])
        with self.assertRaises(ValueError):
            g['observe'](wrong)
        self.run.parent.mkdir(parents=True)
        self.run.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            g['observe'](self.argv)
        self.assertFalse(self.destination.exists())
        CONTROLS.append('unstable/wrong-parent identity; unrelated parent options/cwd/source; duplicate or extra capacity flags; path/symlink rejected')

    def test_real_V4_CPU_self_parent_identity_interface_only(self):
        current, parent = API['process_identity'](os.getpid()), API['process_identity'](os.getppid())
        namespace['validate_identities'](current, parent)
        self.assertEqual(API['process_identity'](os.getpid()), current)
        self.assertEqual(API['process_identity'](os.getppid()), parent)
        CPU_IDENTITY.update(scope='actual_local_CPU_control_process_not_GPU_container_pool', controller=current, parent=parent)
        self.assertFalse(self.destination.exists())
        CONTROLS.append('real SHA-bound V4 process_identity read of CPU control self/parent only; no runtime-pool claim')


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run artificial controls with empty CUDA_VISIBLE_DEVICES')
    with patch('subprocess.Popen', side_effect=AssertionError('No subprocess')), \
            patch('socket.socket', side_effect=AssertionError('No socket')), \
            patch('os.execv', side_effect=AssertionError('No exec')):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    assert 'torch' not in sys.modules
    print(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed', cases=result.testsRun,
        controls=CONTROLS, actual_CPU_identity=CPU_IDENTITY,
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (SOURCE, FROZEN_PROFILE, g['COORDINATOR'])},
        GPU_executed=False, actual_pool_observed=False, real_data_read=False, real_queue_changed=False,
        temporary_fixtures_removed=True, actual_entry_deployed=False)))
    sys.exit(0 if result.wasSuccessful() else 1)
