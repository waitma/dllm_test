"""Check additional-update/resume accounting once with artificial JSON metadata.

Activate conda pllm, then run:
  CUDA_VISIBLE_DEVICES='' python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_formal_segment_dispatch.py

Uses the frozen driver's stdlib parser/digest and the new wrapper's metadata helpers.
All plans, catalogs and checkpoint JSON are synthetic_fakecontrol fixtures in
TemporaryDirectory. No native checkpoint binaries, real data, GPU, actual exec,
listeners, processes, old suites or real plans are used; fixtures are removed.
"""
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
DISPATCH = ROOT / 'scripts/research/pool_formal_dispatch.py'
WRAPPER = ROOT / 'scripts/research/formal_pool_job.py'
DRIVER = ROOT / 'output/v7_formal_eight_gpu_20261002/source_snapshot_v2/scripts/research/run_v7_formal.py'
dispatcher = runpy.run_path(str(DISPATCH), run_name='_synthetic_dispatch_metadata')
g = dispatcher['prepare'].__globals__
wrapper = runpy.run_path(str(WRAPPER), run_name='_synthetic_wrapper_metadata')
driver = runpy.run_path(str(DRIVER), run_name='_frozen_driver_parser_only')
CONTROLS = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    return sha(path)


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


class Fixture:
    def __init__(self, root):
        self.root, self.own, self.pool, self.data = root, root / 'formal', root / 'pool', root / 'unread_data'
        self.source = self.own / 'source_snapshot_v2'
        self.driver = self.source / 'scripts/research/run_v7_formal.py'
        self.driver.parent.mkdir(parents=True)
        self.driver.write_bytes(DRIVER.read_bytes())
        self.wrapper = self.own / 'formal_pool_job.py'
        self.wrapper.write_bytes(WRAPPER.read_bytes())
        self.manifest = self.own / 'source_manifest.json'
        write(self.manifest, {'files': {str(self.driver): sha(self.driver)}, 'synthetic_fakecontrol': True})
        self.shared = self.own / 'shared'
        # The deliberately nonnumeric name proves that step is read from metadata.
        self.checkpoint = self.shared / 'saved_state_without_numeric_step'
        self.state = {'global_step': 6000, 'synthetic_fakecontrol': True}
        self.binding = dict(schema='v7.formal.driver.v1', variant='f000', source_root=str(self.source),
            repo_root=str(root), data_dir=str(self.data), seed=42, micro_batch=2, grad_accum=2,
            global_batch=32, world_size=8, source_manifest_sha256=sha(self.manifest),
            schedule=dict(max_steps=10000, warmup_steps=100, learning_rate=1e-4, scheduler='cosine',
                          eval_every=1000, save_every=1000), synthetic_fakecontrol=True)
        self.save_checkpoint()
        work = {key: self.binding[key] for key in ('variant', 'source_root', 'repo_root', 'data_dir',
                                                  'seed', 'micro_batch', 'grad_accum', 'global_batch')}
        work.update(max_steps=10000, output_dir=str(self.shared), source_manifest=ref(self.manifest),
                    driver=ref(self.driver), selection={'path': str(root / 'unread_selection'), 'sha256': '0' * 64},
                    audit={'path': str(root / 'unread_audit'), 'sha256': '0' * 64})
        argv = []
        for key in ('source_root', 'repo_root', 'data_dir', 'output_dir', 'variant', 'seed', 'micro_batch',
                    'grad_accum', 'global_batch', 'max_steps'):
            argv += ['--' + key.replace('_', '-'), str(work[key])]
        argv += ['--source-manifest', str(self.manifest), '--source-manifest-sha256', sha(self.manifest),
                 '--stop-after', '6500', '--resume-from-checkpoint', str(self.checkpoint)]
        work['driver_argv'] = argv
        self.jobs = [f'{9000 + index}_{alias}_1234abcd' for index, alias in enumerate(g['ALIASES'])]
        tickets = {job: dict(variant=alias, mode='train', steps=500, data_dir=str(self.data),
                            resume_from_checkpoint=str(self.checkpoint)) for job, alias in zip(self.jobs, g['ALIASES'])}
        self.plan = dict(schema='v7.formal.pool_group_plan.v1', pool_dir=str(self.pool), meeting_dir=str(root / 'unused_meeting'),
            group_id='synthetic_fakecontrol_never_publish', pool_process={}, pool_argv=['synthetic_fakecontrol'],
            pool_source={'path': str(root / 'unread_pool.py'), 'sha256': '0' * 64}, wrapper=ref(self.wrapper),
            transport={'path': str(root / 'unread_transport.py'), 'sha256': '0' * 64}, work=work, tickets=tickets,
            join_seconds=60, startup_seconds=60, duration_seconds=300, cleanup_margin_seconds=120, segment_steps=6500)
        self.plan_path, self.catalog = self.own / 'plan.json', self.own / 'catalog.json'
        self.save_plan()

    def save_checkpoint(self):
        files = {'trainer_state.json': write(self.checkpoint / 'trainer_state.json', self.state),
                 'run_provenance.json': write(self.checkpoint / 'run_provenance.json', self.binding)}
        self.sidecar = dict(schema='v7.formal.native_checkpoint.v1', world_size=8, writer_rank=0,
            rank_acknowledgments=list(range(8)), step=self.state['global_step'], files=files,
            binding_sha256=driver['digest'](self.binding), synthetic_fakecontrol=True)
        write(self.checkpoint / 'formal_checkpoint.json', self.sidecar)

    def save_plan(self):
        write(self.plan_path, self.plan)
        write(self.catalog, {'schema': 'v7.formal.pool_job_catalog.v1', 'synthetic_fakecontrol': True,
                            'jobs': {job: ref(self.plan_path) for job in self.jobs}})

    def argv(self, index=0):
        job = self.plan['tickets'][self.jobs[index]]
        argv = ['--variant', job['variant'], '--mode', 'train', '--steps', str(job['steps']), '--data-dir', job['data_dir'],
                '--profile-dir', str(self.pool / 'profiles' / job['variant']), '--output-dir', str(self.pool / 'runs' / self.jobs[index])]
        if 'resume_from_checkpoint' in job:
            argv += ['--resume-from-checkpoint', job['resume_from_checkpoint']]
        return argv

    def check(self):
        return wrapper['validate_segment'](wrapper['prepare'](self.plan),
            driver['parse_args'](self.plan['work']['driver_argv']), binding_digest=driver['digest'])


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic_fakecontrol_formal_segment_')
        self.root = Path(self.tmp.name)
        self.fixture = Fixture(self.root)
        self.mapping = patch.dict(g, {'OWN': self.fixture.own, 'POOL': self.fixture.pool,
                                     'DATA': self.fixture.data, 'CATALOG': self.fixture.catalog})
        self.mapping.start()

    def tearDown(self):
        self.mapping.stop()
        self.tmp.cleanup()
        self.assertFalse(self.root.exists())

    def test_cumulative6500_with500_additional_updates_and_all8_dispatch(self):
        fixture = self.fixture
        receipt = fixture.check()
        self.assertEqual((receipt['prior_step'], receipt['target_step'], receipt['additional_updates']), (6000, 6500, 500))
        self.assertEqual(len(receipt['checkpoint_metadata_sha256']), 3)
        for index in range(8):
            args, command = g['prepare'](fixture.argv(index))
            self.assertEqual(args.steps, 500)
            self.assertEqual(command, [sys.executable, '-I', '-B', '-u', str(fixture.wrapper),
                '--plan', str(fixture.plan_path), '--plan-sha', sha(fixture.plan_path), '--job-id', fixture.jobs[index]])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '7'}), patch.object(g['os'], 'execv') as execute:
            g['main'](fixture.argv())
            execute.assert_called_once()
        self.assertFalse((self.root / 'unused_meeting').exists())
        CONTROLS.append('real frozen parser; cumulative6500-prior6000=500; all8 dispatch; no numeric checkpoint name; mocked exec')

    def test_no_resume_prior_zero(self):
        fixture = self.fixture
        for ticket in fixture.plan['tickets'].values():
            ticket.pop('resume_from_checkpoint')
            ticket['steps'] = 1000
        fixture.plan['segment_steps'] = 1000
        argv = fixture.plan['work']['driver_argv']
        del argv[-2:]
        argv[argv.index('--stop-after') + 1] = '1000'
        fixture.save_plan()
        self.assertEqual(fixture.check()['prior_step'], 0)
        g['prepare'](fixture.argv())
        fixture.plan['segment_steps'] = 1500
        argv[argv.index('--stop-after') + 1] = '1500'
        with self.assertRaisesRegex(ValueError, 'additional updates'):
            fixture.check()
        CONTROLS.append('no resume prior0; new-update difference enforced')

    def test_native_driver_digest_and_compact_plan_digest_differ(self):
        fixture = self.fixture
        self.assertNotEqual(driver['digest'](fixture.binding), wrapper['digest'](fixture.binding))
        self.assertEqual(fixture.sidecar['binding_sha256'], driver['digest'](fixture.binding))
        fixture.check()
        fixture.sidecar['binding_sha256'] = wrapper['digest'](fixture.binding)
        write(fixture.checkpoint / 'formal_checkpoint.json', fixture.sidecar)
        with self.assertRaisesRegex(ValueError, 'Prior checkpoint binding differs'):
            fixture.check()
        fixture.save_plan()
        with self.assertRaisesRegex(ValueError, 'Prior checkpoint binding differs'):
            g['prepare'](fixture.argv())
        CONTROLS.append('actual frozen native driver digest accepted; compact plan digest rejected before exec')

    def test_relative_pool_script_uses_observed_cwd(self):
        cwd = self.root.resolve()
        expected = cwd/'scripts/immune_research_pool.py'
        relative = ['python', '-u', 'scripts/immune_research_pool.py', 'work', '--pool-dir', str(self.fixture.pool)]
        self.assertEqual(wrapper['invocation_source'](relative, cwd), expected)
        absolute = ['python', '-u', str(expected), 'work', '--pool-dir', str(self.fixture.pool)]
        self.assertEqual(wrapper['invocation_source'](absolute, cwd), expected)
        self.assertNotEqual(wrapper['invocation_source'](relative, cwd/'other'), expected)
        for argv, source_cwd in ((['python', '-m', 'pool', 'work'], cwd),
                                 (relative+['work'], cwd), (relative, Path('unobserved_relative_cwd'))):
            with self.subTest(argv=argv, cwd=source_cwd), self.assertRaises(ValueError):
                wrapper['invocation_source'](argv, source_cwd)
        CONTROLS.append('original relative pool script resolved against observed cwd; absolute also accepted; malformed invocation rejected')

    def test_ticket_resume_and_difference_rejections(self):
        fixture = self.fixture
        saved = copy.deepcopy(fixture.plan)
        for field, value in [('steps', 499), ('data_dir', str(self.root / 'other_data')),
                             ('resume_from_checkpoint', str(self.root / 'other_checkpoint'))]:
            fixture.plan = copy.deepcopy(saved)
            fixture.plan['tickets'][fixture.jobs[1]][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                fixture.check()
        fixture.plan = copy.deepcopy(saved)
        for ticket in fixture.plan['tickets'].values():
            ticket['resume_from_checkpoint'] = str(self.root / 'other_checkpoint')
        with self.assertRaisesRegex(ValueError, 'Ticket resume'):
            fixture.check()
        fixture.plan = copy.deepcopy(saved)
        for ticket in fixture.plan['tickets'].values():
            ticket['steps'] = 499
        with self.assertRaisesRegex(ValueError, 'additional updates'):
            fixture.check()
        fixture.save_plan()
        with self.assertRaises(ValueError):
            g['prepare'](fixture.argv())
        CONTROLS.append('eight-ticket steps/data/resume equality; actual driver resume mismatch; wrong delta rejected before exec')

    def test_existing_checkpoint_step_hash_and_binding_rejections(self):
        fixture = self.fixture
        path = fixture.checkpoint / 'trainer_state.json'
        raw = path.read_bytes()
        path.write_bytes(raw + b' ')
        with self.assertRaisesRegex(ValueError, 'Bound file differs'):
            fixture.check()
        path.write_bytes(raw)
        fixture.sidecar['step'] = 5999
        write(fixture.checkpoint / 'formal_checkpoint.json', fixture.sidecar)
        with self.assertRaisesRegex(ValueError, 'step/sidecar'):
            fixture.check()
        fixture.state['global_step'] = 5999
        fixture.save_checkpoint()
        with self.assertRaisesRegex(ValueError, 'additional updates'):
            fixture.check()
        fixture.state['global_step'] = 6500
        fixture.save_checkpoint()
        with self.assertRaisesRegex(ValueError, 'step/sidecar'):
            fixture.check()
        fixture.state['global_step'] = 6000
        fixture.binding['schedule']['max_steps'] = 11000
        fixture.save_checkpoint()
        with self.assertRaisesRegex(ValueError, 'fixed schedule'):
            fixture.check()
        CONTROLS.append('trainer metadata SHA drift; sidecar/global_step mismatch; wrong prior step; no remaining work; prior schedule mismatch')

    def test_frozen_large_batch_fallback_preserves_argv(self):
        self.assertEqual(g['PREVIOUS'], ROOT / 'output/v7_large_batch_ablation_20261002/launcher_source_v1/pool_large_batch_ablation_dispatch.py')
        self.assertEqual(sha(g['PREVIOUS']), g['PREVIOUS_SHA'])
        for argv in (['--variant=p04_v7_capacity', '--mode', 'profile-synthetic'],
                     ['--variant', 'p04_v7_f000_seed42', '--mode', 'train'],
                     ['--variant', 'p02_v7_f111_seed43', '--steps', '3750']):
            previous = Mock(return_value='synthetic_fakecontrol_delegated')
            with patch.object(g['runpy'], 'run_path', return_value={'main': previous}) as loader:
                self.assertEqual(g['main'](argv), 'synthetic_fakecontrol_delegated')
                previous.assert_called_once_with(argv)
                loader.assert_called_once_with(str(g['PREVIOUS']))
        with patch.dict(g, {'PREVIOUS_SHA': '0' * 64}), patch.object(g['runpy'], 'run_path') as loader:
            with self.assertRaises(ValueError):
                g['main'](['--variant', 'p04_v7_capacity'])
            loader.assert_not_called()
        CONTROLS.append('actual frozen fallback SHA; capacity/large-batch/p02 original argv preserved in mocked delegation')


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Artificial metadata controls require empty CUDA_VISIBLE_DEVICES')
    with patch('subprocess.Popen', side_effect=AssertionError('No processes in metadata controls')), \
            patch('socket.socket', side_effect=AssertionError('No sockets in metadata controls')):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    assert 'torch' not in sys.modules
    print(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed', cases=result.testsRun,
        controls=CONTROLS, source_sha256={str(path): sha(path) for path in (DISPATCH, WRAPPER, DRIVER)},
        scope='artificial_JSON_and_frozen_stdlib_parser_only', GPU_executed=False, model_imported=False,
        native_checkpoint_read=False, actual_driver_main_or_transport_executed=False,
        real_plan_queue_or_data_read=False, temporary_fixtures_removed=True)))
    sys.exit(0 if result.wasSuccessful() else 1)
