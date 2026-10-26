"""Test the private local guard with temporary artificial metadata and CPU locks.

  source /project_c20250601/251105016/miniforge3/etc/profile.d/conda.sh
  conda activate pllm
  CUDA_VISIBLE_DEVICES='' python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_local_metrics_only_ablation_dispatch.py

The existing artificial pool fixture is constructed, not its tests rerun.
No real prepare/main, sequence reads, model/GPU calls or jobs are executed.
Exec in guard main is mocked; one independent stdlib CPU subprocess execs a
tiny lock probe to check that a shared FD survives exec and stays exclusive.
All fixture files and the pre-exec witness are removed on completion.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path('/project_c20250601/251105016/project/dllm_test')
SCRIPT = ROOT / 'scripts/research/local_metrics_only_ablation_dispatch.py'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


guard = module('private_local_metrics_guard', SCRIPT)
fixtures = module('artificial_pool_metadata_fixture', ROOT / 'scripts/research/test_pool_metrics_only_ablation_dispatch.py')
write = fixtures.write


class GuardCase(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MetadataCase()
        self.fixture.setUp()  # Reuse setup only; never rerun the six pool tests.
        self.addCleanup(self.fixture.doCleanups)
        fixture, pool = self.fixture, fixtures.api.__dict__
        self.pool = pool
        self.runner = fixture.root / 'scripts/run_in_tmux.sh'
        self.runner.parent.mkdir(parents=True)
        self.runner.write_text('# artificial runner source, not executed\n')
        self.runner_sha = hashlib.sha256(self.runner.read_bytes()).hexdigest()
        self.prior_launch_path = pool['LOCAL'] / 'launch_receipt.json'
        managed = json.loads(pool['MANAGED'].read_text())
        managed['runner'] = dict(path=str(self.runner), sha256=self.runner_sha)
        candidate = managed['commands'][0]
        task = 'v7_local_f010_seed42_attempt1'
        self.log = fixture.root / 'logs' / task
        candidate.update(task_name=task, log_dir=str(self.log))
        self.environment = dict(CUDA_VISIBLE_DEVICES='0', **pool['RUN_ENV'])
        cleared = [token for name in ('PYTHONPATH', *pool['DDP_ENV']) for token in ('-u', name)]
        candidate['command'] = ['bash', str(self.runner), task, 'env', *cleared,
            *(k + '=' + v for k, v in self.environment.items()), *candidate['training_argv']]
        self.candidate = candidate
        managed_ref = write(pool['MANAGED'], managed)
        pool['MANAGED_SHA'] = managed_ref['sha256']
        prior_command = list(candidate['training_argv'])
        prior_command[prior_command.index('--variant') + 1] = 'f000'
        prior_dir = pool['LOCAL'] / 'f000_seed42_attempt1'
        prior_command[prior_command.index('--output-dir') + 1] = str(prior_dir)
        prior_launch_ref = write(self.prior_launch_path, dict(kind='independent_fresh_attempt',
            pid=919119, command=prior_command, checkpoint_saving=False,
            source_manifest=str(pool['FREEZE']), source_manifest_sha256=pool['FREEZE_SHA']))
        sources = json.loads(pool['FREEZE'].read_text())['source_sha256']
        summary_ref = write(prior_dir / 'summary.json', dict(step=3750, records=120000,
            subset_train_records=120000, epoch=1, position=0))
        provenance_ref = write(prior_dir / 'run_provenance.json', dict(kind='exploratory_architecture_reconstruction',
            variant='f000', seed=42, checkpoint_saving=False, scheduler_horizon=3750,
            micro_batch=32, grad_accum=1, study=dict(name=pool['STUDY']),
            source_sha256={k: sources[k] for k in pool['MATH_SOURCES']}))
        self.prior_complete_path = prior_dir / 'COMPLETE.json'
        self.prior_complete = dict(status='completed', global_step=3750,
            kind='exploratory_architecture_reconstruction', formal_training=False,
            biological_quality_claim=False, summary_sha256=summary_ref['sha256'],
            provenance_sha256=provenance_ref['sha256'])
        prior_complete_ref = write(self.prior_complete_path, self.prior_complete)
        for cell in fixture.ledger['cells']:
            if (cell['variant'], cell['seed']) == ('f010', 42):
                cell['state'] = 'local_owned'
                lock_ref = cell['attempts'][0]['lock_file']
                cell['attempts'] = [dict(transport='local', variant='f010', seed=42, task_name=task,
                    output_dir=candidate['output_dir'], log_dir=str(self.log), lock_file=lock_ref)]
        self.ledger = fixture.ledger
        ledger_ref = write(fixture.ledger_path, self.ledger)
        device = fixture.ledger_path.parent / 'local_gpu0.lock'
        device.write_bytes(b'')
        self.device_lock = device
        self.admission_path = fixture.ledger_path.parent / 'local_admission.json'
        self.admission = dict(schema='v7.local.metrics_only_admission.v1', study=pool['STUDY'],
            variant='f010', seed=42, task_name=task, output_dir=candidate['output_dir'], log_dir=str(self.log),
            local_hostname=socket.gethostname(), automatic_retry=False, training_argv=candidate['training_argv'],
            managed_local_plan=managed_ref, original_catalog=fixture.catalog['original_catalog'],
            source_freeze=fixture.catalog['source_freeze'], batch_plan=fixture.catalog['batch_plan'],
            guard_source=dict(path=str(SCRIPT), sha256=pool['sha'](SCRIPT)),
            pool_adapter_source=dict(path=str(guard.POOL_ADAPTER), sha256=guard.POOL_ADAPTER_SHA),
            runner=dict(path=str(self.runner), sha256=self.runner_sha), ledger=ledger_ref,
            local_device_lock=dict(path=str(device), sha256=hashlib.sha256(b'').hexdigest()),
            prior_f000=dict(launch_receipt=prior_launch_ref, completion=prior_complete_ref))
        self.rebind_admission()
        self.argv = ['--variant', 'f010', '--seed', '42']
        patches = (patch.object(guard, '_pool_api', return_value=pool),
            patch.object(guard, 'RUNNER', self.runner), patch.object(guard, 'RUNNER_SHA', self.runner_sha),
            patch.object(guard, 'F000_LAUNCH', self.prior_launch_path),
            patch.object(guard, 'F000_LAUNCH_SHA', prior_launch_ref['sha256']),
            patch.object(guard, '_pid_present', return_value=False))
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    def rebind_admission(self):
        ref = write(self.admission_path, self.admission)
        self.kwargs = dict(admission_path=self.admission_path, admission_sha256=ref['sha256'],
            ledger_path=self.fixture.ledger_path, ledger_sha256=self.admission['ledger']['sha256'])

    def rebind_ledger(self):
        self.admission['ledger'] = write(self.fixture.ledger_path, self.ledger)
        self.rebind_admission()

    def prepare(self):
        return guard.prepare(self.argv, **self.kwargs)

    def ready_runtime(self):
        args, _ = self.prepare()
        guard_command = guard.guard_argv(args, **self.kwargs)
        env_prefix = self.candidate['command'][3:-len(args.training_argv)]
        self.log.mkdir(parents=True)
        (self.log / 'started').write_text('2026-10-05T18:00:00Z\n')
        (self.log / 'run.log').write_text('2026-10-05T18:00:00Z START ' + self.candidate['task_name']
            + '\n  cmd:    ' + shlex.join([*env_prefix, *guard_command]) + '\n  TMPDIR: /artificial/tmp\n')
        return dict(pid=os.getpid(), start_ticks=12345,
            boot_id='11111111-2222-3333-4444-555555555555', argv=guard_command)

    def runtime_contexts(self, identity):
        return (patch.dict(os.environ, self.environment, clear=True),
            patch.object(guard.os, 'getcwd', return_value=str(self.fixture.root)),
            patch.object(guard, '_process_snapshot', return_value=identity))

    def test_read_only_prepare_retains_original_argv_and_ignores_other_formal_cells(self):
        formal_ticket = self.pool['POOL'] / 'running/unrelated_formal.json'
        write(formal_ticket, dict(variant='p06_v7_formal_member_7', mode='train'))
        before = {str(p): p.read_bytes() for p in self.fixture.root.rglob('*') if p.is_file()}
        environment = dict(os.environ)
        with patch.dict(self.pool, {'check_formal': lambda *a: self.fail('Local guard must not check formal completion')}):
            _, command = self.prepare()
        self.assertEqual(command, self.candidate['training_argv'])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.fixture.root.rglob('*') if p.is_file()})
        self.assertEqual(environment, dict(os.environ))
        self.assertFalse(self.log.exists())
        self.assertFalse(any(n.split('.')[0] in {'torch', 'transformers', 'accelerate', 'numpy', 'dllm'} for n in sys.modules))

    def test_prior_f000_must_be_gone_and_native_complete(self):
        with patch.object(guard, '_pid_present', return_value=True):
            with self.assertRaisesRegex(ValueError, 'f000 process is still present'):
                self.prepare()
        self.prior_complete['global_step'] = 3500
        self.admission['prior_f000']['completion'] = write(self.prior_complete_path, self.prior_complete)
        self.rebind_admission()
        with self.assertRaisesRegex(ValueError, 'completion/epoch differs'):
            self.prepare()

    def test_source_data_and_task_binding_are_preserved(self):
        original = self.fixture.source_file.read_bytes()
        self.fixture.source_file.write_bytes(original + b'# mutation\n')
        with self.assertRaisesRegex(ValueError, 'Frozen source differs'):
            self.prepare()
        self.fixture.source_file.write_bytes(original)
        self.admission['training_argv'] = list(self.admission['training_argv'])
        self.admission['training_argv'].append('--save-checkpoints')
        self.rebind_admission()
        with self.assertRaisesRegex(ValueError, 'retain existing task/output/log/training argv'):
            self.prepare()

    def test_unique_local_claim_and_pool_duplicate_are_rejected(self):
        other = self.ledger['cells'][0]
        other['state'] = 'local_owned'
        with self.subTest('another local ownership'):
            self.rebind_ledger()
            with self.assertRaisesRegex(ValueError, 'Another local cell is owned'):
                self.prepare()
        other['state'] = 'unclaimed'
        self.rebind_ledger()
        pending = self.pool['POOL'] / 'pending/same_cell.json'
        write(pending, self.fixture.job)
        with self.subTest('same-cell pending'):
            with self.assertRaisesRegex(ValueError, 'active pool attempt'):
                self.prepare()
        pending.unlink()
        write(self.pool['POOL'] / 'results/same_cell.json', dict(state='succeeded',
            returncode=0, job=self.fixture.job))
        with self.subTest('same-cell actual result'):
            with self.assertRaisesRegex(ValueError, 'pool cell already succeeded'):
                self.prepare()

    def test_mocked_exec_records_only_actual_pre_exec_admission(self):
        identity = self.ready_runtime()
        contexts = self.runtime_contexts(identity)
        with contexts[0], contexts[1], contexts[2], patch.object(guard.os, 'execv') as execute:
            guard.main(self.argv, **self.kwargs)
            execute.assert_called_once_with(sys.executable, self.candidate['training_argv'])
        path = self.log / 'guard_execution.json'
        receipt = json.loads(path.read_text())
        self.assertEqual(receipt['schema'], 'v7.local.metrics_only_guard_execution.v1')
        self.assertEqual(receipt['scope'], 'pre_exec_admission_only')
        self.assertEqual(receipt['admission'], dict(path=str(self.admission_path), sha256=self.kwargs['admission_sha256']))
        self.assertEqual(receipt['ledger'], self.admission['ledger'])
        self.assertEqual(receipt['guard_source'], self.admission['guard_source'])
        self.assertEqual(receipt['training_argv'], self.candidate['training_argv'])
        self.assertEqual(receipt['process_identity'], identity)
        self.assertEqual(receipt['CUDA_VISIBLE_DEVICES'], '0')
        for name in ('lock_identity', 'device_lock_identity'):
            self.assertEqual(set(receipt[name]), {'st_dev', 'st_ino', 'inheritable'})
            self.assertIs(receipt[name]['inheritable'], True)
        # Consume the actual writer output through the new independent reader.
        # This adds no model, plotting, queue or real-run activity.
        reporter = module('mixed_guard_writer_reader_check', ROOT / 'scripts/research/plot_v7_mixed_ablation.py')
        guard_command = identity['argv']
        launch = dict(schema='v7.local.tmux_launch.v2',
            training_argv=self.candidate['training_argv'], runner=self.admission['runner'],
            task_name=self.candidate['task_name'], log_dir=str(self.log),
            source_manifest=self.admission['source_freeze']['path'],
            source_manifest_sha256=self.admission['source_freeze']['sha256'], checkpoint_saving=False,
            admission=receipt['admission'], ledger=receipt['ledger'], guard_source=receipt['guard_source'],
            guard_argv=guard_command,
            guard_execution=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            command=self.candidate['command'][:-len(self.candidate['training_argv'])] + guard_command,
            started=dict(path=str(self.log / 'started'),
                sha256=hashlib.sha256((self.log / 'started').read_bytes()).hexdigest()))
        self.assertEqual(reporter.guarded_local_launch(launch, {}), guard_command)
        with patch.object(guard.os, 'execv') as execute:
            with self.assertRaisesRegex(ValueError, 'previous attempt evidence'):
                guard.main(self.argv, **self.kwargs)
            execute.assert_not_called()

    def test_wrong_environment_and_post_lock_local_race_do_not_exec(self):
        identity = self.ready_runtime()
        environment = dict(self.environment, WORLD_SIZE='8')
        with patch.dict(os.environ, environment, clear=True), patch.object(guard.os, 'getcwd', return_value=str(self.fixture.root)), \
                patch.object(guard.os, 'execv') as execute:
            with self.assertRaisesRegex(ValueError, 'environment differs'):
                guard.main(self.argv, **self.kwargs)
            execute.assert_not_called()
        self.assertFalse((self.log / 'guard_execution.json').exists())
        real_prepare, calls = guard.prepare, []
        def raced(*args, **kwargs):
            result = real_prepare(*args, **kwargs)
            calls.append(1)
            if len(calls) == 1:
                Path(self.candidate['output_dir']).mkdir(parents=True)
            return result
        with patch.object(guard, 'prepare', side_effect=raced), patch.object(guard.os, 'execv') as execute:
            with self.assertRaisesRegex(ValueError, 'local output already exists'):
                guard.main(self.argv, **self.kwargs)
            execute.assert_not_called()

    def test_shared_cell_fd_is_actually_inherited_and_exclusive_across_cpu_exec(self):
        args, _ = self.prepare()
        inner = """import fcntl, os, sys
descriptor, path = int(sys.argv[1]), sys.argv[2]
assert os.get_inheritable(descriptor)
assert os.fstat(descriptor).st_ino == os.stat(path).st_ino
other = os.open(path, os.O_RDONLY)
try:
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print('shared_lock_inherited_and_exclusive')
else:
    raise AssertionError('inherited lock did not exclude a second owner')
"""
        outer = """import fcntl, os, sys
descriptor = os.open(sys.argv[1], os.O_RDONLY)
fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.set_inheritable(descriptor, True)
os.execv(sys.executable, [sys.executable, '-I', '-B', '-c', sys.argv[2], str(descriptor), sys.argv[1]])
"""
        result = subprocess.run([sys.executable, '-I', '-B', '-c', outer, str(args.lock_path), inner],
            env={'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True, timeout=15, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'shared_lock_inherited_and_exclusive')
        with args.lock_path.open('r') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


if __name__ == '__main__':
    unittest.main(verbosity=2)
