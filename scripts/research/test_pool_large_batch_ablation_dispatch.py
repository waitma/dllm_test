"""Run one bounded stdlib-only dispatcher check using owned fakecontrol metadata.

Activate conda pllm and run:
  CUDA_VISIBLE_DEVICES='' python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_large_batch_ablation_dispatch.py

Only temporary artificial registries, plans, ledgers, capacity records and
source files are used. The real dispatcher/runner text is read, never an actual
plan, data subset, profile result or queue. Exec, sockets and subprocesses are
intercepted. Temporary fixtures are removed; archive this process stdout.
"""
import copy
from contextlib import redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
ENTRY = ROOT / 'scripts/research/pool_large_batch_ablation_dispatch.py'
module = runpy.run_path(str(ENTRY), run_name='_artificial_dispatch_control')
g = module['prepare'].__globals__
FROZEN_RUNNER = ROOT / 'output/v7_large_batch_ablation_20261002/source_snapshot_v1/scripts/research/run_v7_large_batch_ablation.py'
RUNNER_BYTES = FROZEN_RUNNER.read_bytes()
CHECKS = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    return sha(path)


class FakeControl:
    def __init__(self, root):
        self.root, self.own, self.pool, self.data = root, root / 'own', root / 'pool', root / 'metadata_only_data'
        self.snapshot = self.own / 'source_snapshot_v1'
        self.registry, self.plan_path, self.catalog_path = root / 'registry.py', self.own / 'batch_selection_v1.json', self.own / 'jobs.json'
        self.sample_sha = write(self.data / 'sample_manifest.json', {'synthetic_fakecontrol': True, 'no_sequences': True})
        self.sources = {}
        for index in range(568):
            name = ('scripts/research/run_v7_large_batch_ablation.py' if index == 0 else
                    'scripts/research/profile_v7_large_batch.py' if index == 1 else
                    'dllm/pipelines/immune_llada/research_refined/factorial.py' if index == 2 else f'fixture/source_{index}.py')
            path = self.snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = RUNNER_BYTES.replace(g['SAMPLE_SHA'].encode(), self.sample_sha.encode()) if index == 0 else b'synthetic_fakecontrol = True\n'
            path.write_bytes(content)
            self.sources[name] = sha(path)
        evidence = {}
        for index in range(2):
            path = self.own / f'artificial_cpu_evidence_{index}.json'
            evidence[str(path)] = write(path, {'synthetic_fakecontrol': True, 'GPU_executed': False})
        self.freeze = self.own / 'source_snapshot_v1.json'
        self.freeze_sha = write(self.freeze, dict(snapshot_root=str(self.snapshot), study=g['STUDY'], horizon=3750,
            global_batch=32, evaluation_micro_batch=4, sample_manifest_sha256=self.sample_sha,
            source_sha256=self.sources, evidence_sha256=evidence, synthetic_fakecontrol=True))
        self.capacity_run = self.pool / 'runs' / g['CAPACITY_JOB']
        self.paths = {'ledger': self.own / 'capacity_job.json', 'summary': self.capacity_run / 'summary.json',
            'completion': self.capacity_run / 'COMPLETE.json', 'result': self.pool / 'results' / (g['CAPACITY_JOB'] + '.json')}
        envelope = dict(variant='p04_v7_capacity', mode='profile-synthetic', steps=72)
        self.ledger = dict(schema='v7.capacity.enqueue_intent.v1', job_id=g['CAPACITY_JOB'], job=envelope,
            pool_dir=str(self.pool), run_dir=str(self.capacity_run), state='pending', synthetic_fakecontrol=True)
        self.result = dict(job_id=g['CAPACITY_JOB'], job=envelope, state='succeeded', returncode=0,
            argv=[sys.executable, '-u', str(root / 'scripts/run_immune_research.py'), '--variant', 'p04_v7_capacity',
                  '--mode', 'profile-synthetic', '--steps', '72', '--profile-dir', str(self.pool / 'profiles/p04_v7_capacity'),
                  '--output-dir', str(self.capacity_run)], synthetic_fakecontrol=True)
        measured_sources = {str(self.snapshot / name): self.sources[name] for name in
            ('scripts/research/profile_v7_large_batch.py', 'dllm/pipelines/immune_llada/research_refined/factorial.py')}
        self.summary = dict(kind='synthetic_whole_model_capacity_sweep', status='completed',
            biological_quality_claim=False, planned_optimizer_updates=72, optimizer_update_hard_cap=100,
            global_batch=32, candidate_seconds=180, max_seconds=1800, source_root=str(self.snapshot),
            profile_script_sha256=self.sources['scripts/research/profile_v7_large_batch.py'], config=g['CONFIG'],
            measurements=[], synthetic_fakecontrol=True)
        for variant in g['PROFILE_VARIANTS']:
            for micro in (4, 16, 32):
                self.summary['measurements'].append(dict(kind='synthetic_whole_model_capacity', variant=variant,
                    micro_batch=micro, grad_accum=32 // micro, global_batch=32, status='passed', shape=[micro, 2048],
                    attended_positions_per_record=2048, config=g['CONFIG'], gpu='ARTIFICIAL A100 80GB', total_gib=80.,
                    precision='fp32_parameters_adam_bf16_autocast', activation_checkpointing='whole_layer', seed=42,
                    warmup_steps=2, measure_steps=6, optimizer='AdamW', learning_rate=1e-4, max_grad_norm=1.,
                    biological_quality_claim=False, actual_corpus_test=False, source_sha256=measured_sources,
                    peak_allocated_gib=30., peak_reserved_gib=40., completed_optimizer_steps=8,
                    measured_losses=[1.] * 6, optimizer_state_tensor_bytes=128, measured_seconds=1.,
                    input_tokens_per_second=32 * 2048 * 6, synthetic_fakecontrol=True))
        self.plan = dict(schema='v7.large_batch_selection.v1', study=g['STUDY'], status='selected',
            micro_batch=16, grad_accum=2, global_batch=32, evaluation_micro_batch=4,
            sample_manifest_sha256=self.sample_sha, variants=list(g['METHODS']), seeds=[42, 43],
            profile_evidence=[], synthetic_fakecontrol=True)
        self.catalog = dict(study=g['STUDY'], source_freeze={'path': str(self.freeze), 'sha256': self.freeze_sha},
                            jobs=[], synthetic_fakecontrol=True)
        for index, (alias, (variant, seed)) in enumerate(g['ALIASES'].items()):
            self.catalog['jobs'].append(dict(alias=alias, variant=variant, seed=seed,
                job_id=f'{9999990000 + index}_{alias}_c0decafe',
                job=dict(variant=alias, mode='train', steps=3750, data_dir=str(self.data))))
        self.publish()

    def publish(self):
        summary_sha = write(self.paths['summary'], self.summary)
        write(self.paths['completion'], dict(kind='synthetic_capacity_profile_only', status='completed',
            summary_sha256=summary_sha, training_completed=False, actual_corpus_test=False, checkpoint_created=False))
        write(self.paths['ledger'], self.ledger)
        write(self.paths['result'], self.result)
        self.plan['profile_evidence'] = [{'path': str(path), 'sha256': sha(path)} for path in self.paths.values()]
        self.plan_sha = write(self.plan_path, self.plan)
        self.catalog['batch_plan'] = {'path': str(self.plan_path), 'sha256': self.plan_sha}
        self.catalog_sha = write(self.catalog_path, self.catalog)
        self.entries = {alias: dict(launcher_only=True, model_variant=variant, seed=seed, study=g['STUDY'],
            jobs_catalog_sha256=self.catalog_sha, batch_plan_sha256=self.plan_sha, source_freeze_sha256=self.freeze_sha)
            for alias, (variant, seed) in g['ALIASES'].items()}
        self.registry.write_text('"""Artificial metadata only; never imported."""\nVARIANTS = ' + repr(self.entries) + '\n')

    def globals(self):
        return dict(ROOT=self.root, OWN=self.own, POOL=self.pool, DATA=self.data, REGISTRY=self.registry,
                    FREEZE=self.freeze, FREEZE_SHA=self.freeze_sha, PLAN=self.plan_path, CATALOG=self.catalog_path,
                    SAMPLE_SHA=self.sample_sha)

    def argv(self, index=0):
        row = self.catalog['jobs'][index]
        return ['--variant', row['alias'], '--mode', 'train', '--steps', '3750', '--data-dir', str(self.data),
                '--profile-dir', str(self.pool / 'profiles' / row['alias']),
                '--output-dir', str(self.pool / 'runs' / row['job_id'])]


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic_fakecontrol_large_batch_dispatch_')
        self.root = Path(self.tmp.name)
        self.fixture = FakeControl(self.root)
        self.patch = patch.dict(g, self.fixture.globals())
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()
        self.assertFalse(self.root.exists())

    def rejected(self, call):
        with redirect_stderr(io.StringIO()), self.assertRaises((ValueError, SystemExit, FileNotFoundError)):
            call()

    def test_all22_binding_two_selected_batches_and_exec(self):
        fixture = self.fixture
        for index, row in enumerate(fixture.catalog['jobs']):
            args, command = g['prepare'](fixture.argv(index))
            self.assertEqual(command[:5], [sys.executable, '-I', '-B', '-u', str(fixture.snapshot / 'scripts/research/run_v7_large_batch_ablation.py')])
            self.assertEqual(command[command.index('--variant') + 1], row['variant'])
            self.assertEqual(command[command.index('--seed') + 1], str(row['seed']))
            self.assertEqual(command[command.index('--micro-batch') + 1], '16')
            self.assertEqual(command[command.index('--batch-plan-sha256') + 1], fixture.plan_sha)
            self.assertFalse(args.output_dir.exists())
        fixture.plan.update(micro_batch=32, grad_accum=1)
        fixture.publish()
        self.assertEqual(g['prepare'](fixture.argv())[1][-7:-5], ['32', '--grad-accum'])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '6'}), patch.object(g['os'], 'execv') as execute:
            g['main'](fixture.argv())
            execute.assert_called_once()
            self.assertEqual(execute.call_args.args[0], sys.executable)
        CHECKS.append('all22 metadata bindings; 16x2 and32x1; mocked execv; no output creation')

    def test_missing_plan_profile_selected_oom_and_memory(self):
        fixture = self.fixture
        raw = fixture.plan_path.read_bytes()
        fixture.plan_path.unlink()
        self.rejected(lambda: g['prepare'](fixture.argv()))
        fixture.plan_path.write_bytes(raw)
        raw = fixture.paths['summary'].read_bytes()
        fixture.paths['summary'].unlink()
        self.rejected(lambda: g['prepare'](fixture.argv()))
        fixture.paths['summary'].write_bytes(raw)
        item = fixture.summary['measurements'][1]
        saved = copy.deepcopy(item)
        for updates in (dict(status='oom', expected_capacity_failure=True, error='ARTIFICIAL OOM'),
                        dict(peak_reserved_gib=69.), dict(completed_optimizer_steps=7),
                        dict(measured_losses=[1.] * 5), dict(gpu='ARTIFICIAL OTHER GPU')):
            item.clear(); item.update(saved); item.update(updates)
            fixture.publish()
            self.rejected(lambda: g['prepare'](fixture.argv()))
        item.clear(); item.update(saved)
        fixture.summary['measurements'][2].update(status='oom', expected_capacity_failure=True, error='ARTIFICIAL OOM')
        fixture.publish()
        g['prepare'](fixture.argv())  # Unselected capacity OOM is permissible.
        fixture.summary['measurements'][-1] = fixture.summary['measurements'][0]
        fixture.publish()
        self.rejected(lambda: g['prepare'](fixture.argv()))
        CHECKS.append('missing plan/profile; selected OOM/85percent/steps/loss/hardware; unselected OOM allowed; duplicate9cell rejected')

    def test_source_path_and_cross_group_rejections(self):
        fixture = self.fixture
        path = fixture.snapshot / 'fixture/source_3.py'
        original = path.read_bytes()
        path.write_bytes(original + b'# drift\n')
        self.rejected(lambda: g['prepare'](fixture.argv()))
        path.write_bytes(original)
        fixture.entries[next(iter(g['ALIASES']))]['jobs_catalog_sha256'] = '0' * 64
        fixture.registry.write_text('VARIANTS = ' + repr(fixture.entries))
        self.rejected(lambda: g['prepare'](fixture.argv()))
        fixture.publish()
        fixture.catalog['jobs'][1]['seed'] = 42
        fixture.publish()
        self.rejected(lambda: g['prepare'](fixture.argv()))
        fixture.catalog['jobs'][1]['seed'] = 43
        fixture.publish()
        fake_output = fixture.pool / 'runs' / fixture.catalog['jobs'][0]['job_id']
        fake_output.parent.mkdir(parents=True, exist_ok=True)
        fake_output.symlink_to(self.root, target_is_directory=True)
        self.rejected(lambda: g['prepare'](fixture.argv()))
        fake_output.unlink()
        fixture.result['returncode'] = 1
        fixture.publish()
        self.rejected(lambda: g['prepare'](fixture.argv()))
        fixture.result['returncode'] = 0
        fixture.result['argv'] += ['--data-dir', '/synthetic_fakecontrol/forbidden']
        fixture.publish()
        self.rejected(lambda: g['prepare'](fixture.argv()))
        CHECKS.append('source drift; cross catalog SHA/seed; output symlink; pool nonzero and envelope changes rejected')

    def test_exact_external_envelope_and_masks(self):
        fixture = self.fixture
        for extras in (['--resume'], ['--micro-batch', '32'], ['--grad-accum', '1'], ['--batch-plan', '/fake'],
                       ['--variant', fixture.catalog['jobs'][0]['alias']]):
            self.rejected(lambda: g['prepare'](fixture.argv() + extras))
        for flag, value in (('--steps', '100'), ('--mode', 'eval'), ('--data-dir', '/synthetic_fakecontrol/wrong')):
            changed = fixture.argv()
            changed[changed.index(flag) + 1] = value
            self.rejected(lambda: g['prepare'](changed))
        for mask in ('', '0,1', '8', 'GPU-abc'):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': mask}), patch.object(g['os'], 'execv') as execute:
                self.rejected(lambda: g['main'](fixture.argv()))
                execute.assert_not_called()
        CHECKS.append('micro/resume/plan overrides and duplicate arguments rejected; masks0..7 only')

    def test_unknown_alias_delegation_and_sha(self):
        argv = ['--variant=p04_v7_capacity', '--mode', 'profile-synthetic', '--steps', '72']
        legacy = Mock(return_value='synthetic_fakecontrol_delegated')
        with patch.object(g['runpy'], 'run_path', return_value={'main': legacy}) as loader:
            self.assertEqual(g['main'](argv), 'synthetic_fakecontrol_delegated')
            loader.assert_called_once_with(str(g['PREVIOUS']))
            legacy.assert_called_once_with(argv)
        with patch.dict(g, {'PREVIOUS_SHA': '0' * 64}), patch.object(g['runpy'], 'run_path') as loader:
            self.rejected(lambda: g['main'](argv))
            loader.assert_not_called()
        CHECKS.append('unknown alias argv unchanged; SHA-bound capacity delegation mocked')


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Require empty CUDA_VISIBLE_DEVICES for artificial controls')
    with patch('subprocess.Popen', side_effect=AssertionError('No processes in metadata controls')), \
            patch('socket.socket', side_effect=AssertionError('No sockets in metadata controls')):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    assert 'torch' not in sys.modules
    print(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed', cases=result.testsRun,
        controls=CHECKS, source_sha256=sha(ENTRY), frozen_runner_text_sha256=hashlib.sha256(RUNNER_BYTES).hexdigest(),
        scope='owned_artificial_stdlib_metadata_only_not_GPU_acceptance', models_imported=False,
        GPU_executed=False, real_profile_or_plan_read=False, real_data_read=False, real_queue_changed=False,
        exec_or_previous_main_executed=False, temporary_fixtures_removed=True)))
    sys.exit(0 if result.wasSuccessful() else 1)
