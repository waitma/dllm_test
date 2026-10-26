"""Check the mixed reporter with disposable artificial metadata, not training.

Activate conda pllm, then run:
  CUDA_VISIBLE_DEVICES='' python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_plot_v7_mixed_ablation.py

Reuses ArtificialStudy; all sources, receipts and figures are artificial and
live in TemporaryDirectory. No real sequence rows, models, checkpoints or
GPU operations are used, and original plot tests are not rerun.
"""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mixed_plot', ROOT / 'plot_v7_mixed_ablation.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
spec = importlib.util.spec_from_file_location('artificial_large_study', ROOT / 'test_plot_v7_large_batch_ablation.py')
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
write = fixture_module.write


def reference(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def text_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return reference(path)


class MixedFixture:
    def __init__(self, root):
        self.root = root
        self.study = fixture_module.ArtificialStudy(root / 'study',
            [(v, s) for v in plot.base.VARIANTS for s in plot.base.SEEDS], (0, 500, 3750))
        names = list(self.study.provenance('f000', 42)['source_sha256'])
        self.sources = {}
        self.manifests = {}
        for revision in ('original', 'metrics_only'):
            directory = root / revision
            sources = {name: text_file(directory / name, '# artificial source ' +
                (revision if 'architecture_pilot' in name else 'shared') + '\n')['sha256'] for name in names}
            manifest = root / (revision + '_source.json')
            write(manifest, dict(snapshot_root=str(directory), source_sha256=sources))
            self.sources[revision], self.manifests[revision] = sources, reference(manifest)
        for entry in self.study.jobs['jobs']:
            directory = self.study.pool / 'runs' / entry['job_id']
            provenance = json.loads((directory / 'run_provenance.json').read_text())
            provenance.update(source_sha256=self.sources['original'], checkpoint_saving=True)
            digest = write(directory / 'run_provenance.json', provenance)
            complete = json.loads((directory / 'COMPLETE.json').read_text())
            complete['provenance_sha256'] = digest
            write(directory / 'COMPLETE.json', complete)
        self.cells = [dict(variant=x['variant'], seed=x['seed'], transport='pool',
            attempt_id=x['job_id'], job_id=x['job_id'], source_cohort='original',
            pool_dir=str(self.study.pool), catalog=reference(self.study.jobs_path)) for x in self.study.jobs['jobs']]
        self.selection_path = root / 'mixed_selection.json'

    def local(self, variant='f000', seed=42):
        old = next(x for x in self.cells if (x['variant'], x['seed']) == (variant, seed))
        directory = self.root / f'{variant}_seed{seed}_attempt1'
        directory.mkdir()
        old_dir = self.study.pool / 'runs' / old['job_id']
        for name in ('curve.jsonl', 'summary.json', 'progress.json'):
            (directory / name).write_bytes((old_dir / name).read_bytes())
        provenance = json.loads((old_dir / 'run_provenance.json').read_text())
        provenance.update(source_sha256=self.sources['metrics_only'], checkpoint_saving=False)
        prov_sha = write(directory / 'run_provenance.json', provenance)
        complete = json.loads((old_dir / 'COMPLETE.json').read_text())
        complete['provenance_sha256'] = prov_sha
        write(directory / 'COMPLETE.json', complete)
        entry = dict(variant=variant, seed=seed, transport='local',
            attempt_id=directory.name, source_cohort='metrics_only', run_dir=str(directory), exit_evidence=None)
        argv = ['/artificial/bin/python', '-I', '-B', '-u', str(self.root / 'metrics_only/scripts/research/run_v7_large_batch_ablation.py'),
            '--variant', variant, '--seed', str(seed), '--data-dir', str(self.study.root / 'unread_data'),
            '--output-dir', str(directory), '--steps', '3750', '--eval-every', '250',
            '--micro-batch', '16', '--grad-accum', '2', '--batch-plan', str(self.study.plan_path),
            '--batch-plan-sha256', self.study.plan_sha]
        launch = dict(kind='independent_fresh_attempt', pid=1001, command=argv,
            source_manifest=self.manifests['metrics_only']['path'],
            source_manifest_sha256=self.manifests['metrics_only']['sha256'], checkpoint_saving=False,
            versions={'torch': provenance['torch_version']})
        launch_path = self.root / (directory.name + '_launch.json')
        write(launch_path, launch); entry['launch_receipt'] = reference(launch_path)
        self.cells[self.cells.index(old)] = entry
        return entry

    def managed_exit(self, entry, code):
        launch_path = Path(entry['launch_receipt']['path'])
        launch = json.loads(launch_path.read_text())
        task = 'artificial_' + entry['attempt_id']
        runner = self.root / 'scripts/run_in_tmux.sh'
        text_file(runner, '# artificial runner; never executed\n')
        log = self.root / 'logs' / task
        started = text_file(log / 'started', '2026-10-05T12:00:00Z\n')
        argv = launch['command']
        envelope = ['bash', str(runner), task, 'env']
        for name in sorted(plot.CLEAR_ENV):
            envelope += ['-u', name]
        envelope += [name + '=' + value for name, value in plot.LOCAL_ENV.items()] + argv
        text_file(log / 'run.log', '2026-10-05T12:00:00Z START ' + task + '\n  cmd:    ' +
                  shlex.join(envelope[3:]) + '\n  TMPDIR: /artificial/tmp\n')
        launch.update(schema='v7.local.tmux_launch.v1', training_argv=argv, command=envelope,
            runner=reference(runner), task_name=task, log_dir=str(log), started=started)
        write(launch_path, launch); entry['launch_receipt'] = reference(launch_path)
        sentinel = 'done' if code == 0 else 'failed'
        terminal = text_file(log / sentinel, '2026-10-05T13:00:00Z\n')
        code_ref = text_file(log / 'exit_code', str(code) + '\n')
        exit_receipt = dict(schema='v7.local.tmux_exit.v1', launch_receipt=entry['launch_receipt'],
            runner=launch['runner'], task_name=task, command=envelope, started=started,
            exit_code=code_ref, done=terminal if code == 0 else None,
            failed=terminal if code != 0 else None, pane_returncode=0, launcher_returncode=0)
        path = self.root / (entry['attempt_id'] + '_exit.json')
        write(path, exit_receipt); entry['exit_evidence'] = reference(path)
        return path

    def collect(self, endpoint=3750):
        selection = plot.make_selection(dict(path=str(self.study.plan_path), sha256=self.study.plan_sha),
            self.manifests, self.cells)
        write(self.selection_path, selection)
        return plot.collect(self.selection_path, endpoint)

    def guarded_exit(self, entry, code):
        """Create artificial v2 records; neither tmux nor a guard is executed."""
        exit_path = self.managed_exit(entry, code)
        launch_path = Path(entry['launch_receipt']['path'])
        launch = json.loads(launch_path.read_text())
        argv = launch['training_argv']
        variant, seed = entry['variant'], entry['seed']
        guard = text_file(self.root / 'scripts/research/local_metrics_only_ablation_dispatch.py',
            '# artificial guard; never executed\n')
        adapter = text_file(self.root / 'scripts/research/pool_metrics_only_ablation_dispatch.py',
            '# artificial pool adapter; never executed\n')
        managed_path = self.root / 'managed.json'
        candidate = dict(variant=variant, seed=seed, training_argv=argv,
            task_name=launch['task_name'], output_dir=entry['run_dir'], log_dir=launch['log_dir'])
        write(managed_path, dict(runner=launch['runner'], commands=[candidate]))
        ledger_path = self.root / 'claims/ledger.json'
        cell_lock = text_file(ledger_path.parent / f'cell_locks/{variant}_seed{seed}.lock', '')
        device_lock = text_file(ledger_path.parent / 'local_gpu0.lock', '')
        cells = [dict(variant=v, seed=s, state='unclaimed', attempts=[])
            for v in plot.base.VARIANTS for s in plot.base.SEEDS]
        claim = next(x for x in cells if (x['variant'], x['seed']) == (variant, seed))
        claim.update(state='local_owned', attempts=[dict(transport='local', variant=variant, seed=seed,
            task_name=launch['task_name'], output_dir=entry['run_dir'], log_dir=launch['log_dir'], lock_file=cell_lock)])
        write(ledger_path, dict(schema='v7.metrics_only.ablation_claims.v1', study=plot.large.STUDY, cells=cells))
        admission_path = self.root / 'admission.json'
        admission = dict(schema='v7.local.metrics_only_admission.v1', study=plot.large.STUDY,
            automatic_retry=False, variant=variant, seed=seed, task_name=launch['task_name'],
            output_dir=entry['run_dir'], log_dir=launch['log_dir'], training_argv=argv,
            managed_local_plan=reference(managed_path), original_catalog=reference(self.study.jobs_path),
            source_freeze=self.manifests['metrics_only'], batch_plan=reference(self.study.plan_path),
            guard_source=guard, pool_adapter_source=adapter, runner=launch['runner'],
            ledger=reference(ledger_path), local_device_lock=device_lock)
        write(admission_path, admission)
        admission_ref = reference(admission_path)
        guard_argv = [argv[0], '-I', '-B', '-u', guard['path'],
            '--admission-path', admission_ref['path'], '--admission-sha256', admission_ref['sha256'],
            '--ledger-path', admission['ledger']['path'], '--ledger-sha256', admission['ledger']['sha256'],
            '--variant', variant, '--seed', str(seed)]
        execution_path = Path(launch['log_dir']) / 'guard_execution.json'
        write(execution_path, dict(schema='v7.local.metrics_only_guard_execution.v1',
            scope='pre_exec_admission_only', admission=admission_ref, ledger=admission['ledger'],
            guard_source=guard, variant=variant, seed=seed, task_name=launch['task_name'],
            training_argv=argv, CUDA_VISIBLE_DEVICES='0',
            process_identity=dict(pid=1001, start_ticks=100, boot_id='artificial', argv=guard_argv),
            lock_identity=dict(st_dev=1, st_ino=2, inheritable=True),
            device_lock_identity=dict(st_dev=1, st_ino=3, inheritable=True)))
        launch.update(schema='v7.local.tmux_launch.v2', admission=admission_ref,
            ledger=admission['ledger'], guard_source=guard, guard_argv=guard_argv,
            guard_execution=reference(execution_path), command=launch['command'][:-len(argv)] + guard_argv)
        text_file(Path(launch['log_dir']) / 'run.log', '2026-10-05T12:00:00Z START ' + launch['task_name'] +
            '\n  cmd:    ' + shlex.join(launch['command'][3:]) + '\n  TMPDIR: /artificial/tmp\n')
        write(launch_path, launch); entry['launch_receipt'] = reference(launch_path)
        receipt = json.loads(exit_path.read_text())
        receipt.update(schema='v7.local.tmux_exit.v2', launch_receipt=entry['launch_receipt'])
        receipt.update({k: launch[k] for k in ('command', 'admission', 'ledger', 'guard_source',
            'guard_argv', 'guard_execution')})
        write(exit_path, receipt); entry['exit_evidence'] = reference(exit_path)
        return exit_path


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v7_mixed_plot_artificial_')
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()
        self.assertFalse(self.root.exists())

    def test_fixed_endpoint_keeps_shorter_and_pending_cells_missing(self):
        fixture = MixedFixture(self.root)
        entry = fixture.local()
        directory = Path(entry['run_dir'])
        points = [fixture.study.point(x) for x in (0, 250)]
        (directory / 'curve.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in points))
        (directory / 'COMPLETE.json').unlink()
        fixture.cells = [fixture.cells[0], fixture.cells[2]]
        report = fixture.collect(500)
        self.assertEqual(report['endpoint_step'], 500)
        self.assertNotIn('f000/s42', report['endpoints'])
        self.assertIn('f001/s42', report['endpoints'])
        self.assertEqual(len(report['missing_endpoint_cells']), 21)
        self.assertEqual(len(report['runs']), 22)
        self.assertFalse(report['coverage_complete'])
        self.assertFalse(report['final'])

    def test_complete_local_artifacts_do_not_infer_exit_or_source_equality(self):
        fixture = MixedFixture(self.root)
        fixture.local()
        report = fixture.collect()
        run = next(x for x in report['runs'] if x['transport'] == 'local')
        self.assertTrue(report['coverage_complete'])
        self.assertFalse(report['all_exit_verified'])
        self.assertFalse(report['source_homogeneous'])
        self.assertFalse(report['final'])
        self.assertEqual(run['status'], 'completed_exit_unconfirmed')
        self.assertIsNone(run['actual_returncode'])
        self.assertIsNone(run['job_id'])
        self.assertTrue(any(not x['source_homogeneous_pair'] for x in report['paired_deltas']))
        prefix = self.root / 'figures/mixed'
        plot.export(prefix, report)
        with prefix.with_suffix('.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 44)
        self.assertEqual({x['mask'] for x in rows}, {'mask_mean', 'mask100'})
        self.assertEqual({x['variant'] for x in rows}, set(plot.base.VARIANTS))
        self.assertEqual(len(report['figures']), 3)
        svg = Path(report['figures'][0]).read_text()
        self.assertIn('mask_mean', svg)
        self.assertIn('mask100', svg)
        self.assertIn('metrics_only', svg)
        self.assertNotIn('FINAL', svg)

    def test_source_bytes_and_cross_revision_protocol_are_checked(self):
        fixture = MixedFixture(self.root / 'source')
        fixture.local()
        path = self.root / 'source/metrics_only/scripts/research/run_v7_architecture_pilot.py'
        path.write_text('# changed after freeze\n')
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            fixture.collect()
        fixture = MixedFixture(self.root / 'protocol')
        entry = fixture.local()
        path = Path(entry['run_dir']) / 'run_provenance.json'
        provenance = json.loads(path.read_text()); provenance['torch_version'] = 'DIFFERENT'
        digest = write(path, provenance)
        complete_path = Path(entry['run_dir']) / 'COMPLETE.json'
        complete = json.loads(complete_path.read_text()); complete['provenance_sha256'] = digest
        write(complete_path, complete)
        launch_path = Path(entry['launch_receipt']['path'])
        launch = json.loads(launch_path.read_text()); launch['versions']['torch'] = 'DIFFERENT'
        write(launch_path, launch); entry['launch_receipt'] = reference(launch_path)
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            fixture.collect()

    def test_native_completion_sha_and_failure_are_not_bypassed(self):
        fixture = MixedFixture(self.root / 'sha')
        entry = fixture.local(); path = Path(entry['run_dir']) / 'COMPLETE.json'
        complete = json.loads(path.read_text()); complete['summary_sha256'] = '0' * 64
        write(path, complete)
        with self.assertRaisesRegex(ValueError, 'artifact SHA differs'):
            fixture.collect()
        fixture = MixedFixture(self.root / 'failure')
        entry = fixture.local(); write(Path(entry['run_dir']) / 'FAILURE.json', {})
        report = fixture.collect()
        run = next(x for x in report['runs'] if x['transport'] == 'local')
        self.assertEqual(run['status'], 'failed')
        self.assertFalse(run['artifact_complete'])
        self.assertFalse(report['coverage_complete'])

    def test_tmux_uses_actual_training_exit_and_start_binding(self):
        for code in (0, 1):
            fixture = MixedFixture(self.root / str(code))
            entry = fixture.local(); exit_path = fixture.managed_exit(entry, code)
            report = fixture.collect()
            run = next(x for x in report['runs'] if x['transport'] == 'local')
            self.assertTrue(report['all_exit_verified'])
            self.assertEqual(run['actual_returncode'], code)
            self.assertEqual(run['terminal_verified'], code == 0)
            self.assertEqual(report['coverage_complete'], code == 0)
            receipt = json.loads(exit_path.read_text()); receipt['task_name'] += '_other'
            write(exit_path, receipt); entry['exit_evidence'] = reference(exit_path)
            with self.assertRaisesRegex(ValueError, 'Exit/start identity differs'):
                fixture.collect()

    def test_guarded_tmux_keeps_native_completion_and_actual_exit_requirements(self):
        for code in (0, 9):
            fixture = MixedFixture(self.root / str(code))
            entry = fixture.local(); fixture.guarded_exit(entry, code)
            report = fixture.collect()
            run = next(x for x in report['runs'] if x['transport'] == 'local')
            self.assertEqual(run['actual_returncode'], code)
            self.assertEqual(run['terminal_verified'], code == 0)
            self.assertEqual(report['coverage_complete'], code == 0)
            self.assertFalse(report['final'])
            if code == 0:
                entry['exit_evidence'] = None
                run = next(x for x in fixture.collect()['runs'] if x['transport'] == 'local')
                self.assertEqual(run['status'], 'completed_exit_unconfirmed')
                self.assertIsNone(run['actual_returncode'])

    def test_guarded_tmux_rejects_substituted_execution_and_claim_evidence(self):
        fixture = MixedFixture(self.root / 'execution')
        entry = fixture.local(); exit_path = fixture.guarded_exit(entry, 0)
        launch_path = Path(entry['launch_receipt']['path']); launch = json.loads(launch_path.read_text())
        path = Path(launch['guard_execution']['path']); receipt = json.loads(path.read_text())
        receipt['scope'] = 'training_succeeded'; write(path, receipt)
        launch['guard_execution'] = reference(path); write(launch_path, launch)
        entry['launch_receipt'] = reference(launch_path)
        receipt = json.loads(exit_path.read_text()); receipt.update(launch_receipt=entry['launch_receipt'],
            guard_execution=launch['guard_execution']); write(exit_path, receipt)
        entry['exit_evidence'] = reference(exit_path)
        with self.assertRaisesRegex(ValueError, 'Actual guard pre-exec receipt differs'):
            fixture.collect()
        fixture = MixedFixture(self.root / 'ledger')
        entry = fixture.local(); fixture.guarded_exit(entry, 0)
        path = self.root / 'ledger/claims/ledger.json'; ledger = json.loads(path.read_text())
        ledger['cells'][0]['state'] = 'reserved_pool'; write(path, ledger)
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            fixture.collect()


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    print(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed', cases=result.testsRun,
        scope='artificial_metadata_and_matplotlib_only', GPU_executed=False, models_imported=False,
        real_data_read=False, checkpoints_created=0, temporary_artifacts_removed=True,
        source_sha256=hashlib.sha256((ROOT / 'plot_v7_mixed_ablation.py').read_bytes()).hexdigest())))
    sys.exit(0 if result.wasSuccessful() else 1)
