"""Run bounded artificial metadata/figure checks without models or real data.

Activate conda pllm, then run:
  CUDA_VISIBLE_DEVICES='' python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_plot_v7_large_batch_ablation.py

All fixture files and rendered images live in TemporaryDirectory and are removed.
There are no binary checkpoints, real sequence rows, ML imports or GPU calls.
"""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ENTRY = Path(__file__).with_name('plot_v7_large_batch_ablation.py').resolve()
spec = importlib.util.spec_from_file_location('large_batch_plot', ENTRY)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
ARTIFACT_CHECKS = []


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ArtificialStudy:
    def __init__(self, root, cells=(), steps=(0,)):
        self.root, self.pool = root, root / 'pool'
        self.plan_path, self.jobs_path = root / 'selection.json', root / 'jobs.json'
        self.profile = root / 'artificial_profile_reference.json'
        profile_sha = write(self.profile, {'fixture': True, 'GPU_executed': False})
        self.plan = dict(schema=plot.PLAN_SCHEMA, study=plot.STUDY, status='selected',
            micro_batch=16, grad_accum=2, global_batch=32, evaluation_micro_batch=4,
            sample_manifest_sha256=plot.base.SAMPLE_SHA, variants=list(plot.base.VARIANTS),
            seeds=list(plot.base.SEEDS), profile_evidence=[{'path': str(self.profile), 'sha256': profile_sha}])
        self.plan_sha = write(self.plan_path, self.plan)
        self.plan_ref = dict(path=str(self.plan_path), sha256=self.plan_sha,
                             profile_evidence=self.plan['profile_evidence'])
        self.jobs = dict(study=plot.STUDY, batch_plan={'path': str(self.plan_path), 'sha256': self.plan_sha}, jobs=[])
        for index, (variant, seed) in enumerate(cells):
            alias = f'artificial_v7_{variant}_seed{seed}'
            job_id = f'artificial_{index}_{alias}'
            entry = dict(variant=variant, seed=seed, job_id=job_id, alias=alias,
                job=dict(variant=alias, mode='train', steps=3750, data_dir=str(root / 'unread_data')))
            self.jobs['jobs'].append(entry)
            directory = self.pool / 'runs' / job_id
            provenance = self.provenance(variant, seed)
            prov_sha = write(directory / 'run_provenance.json', provenance)
            points = [self.point(step) for step in steps]
            (directory / 'curve.jsonl').write_text(''.join(json.dumps(point) + '\n' for point in points))
            write(directory / 'progress.json', {'step': steps[-1], 'records': steps[-1] * 32})
            if steps[-1] == 3750:
                summary_sha = write(directory / 'summary.json', points[-1])
                write(directory / 'COMPLETE.json', dict(status='completed', global_step=3750,
                    kind=provenance['kind'], formal_training=False, biological_quality_claim=False,
                    provenance_sha256=prov_sha, summary_sha256=summary_sha))
                write(self.pool / 'results' / f'{job_id}.json', dict(job_id=job_id, state='succeeded', returncode=0))
        write(self.jobs_path, self.jobs)

    def provenance(self, variant, seed):
        return dict(kind='exploratory_architecture_reconstruction', formal_training=False,
            biological_quality_claim=False, generation=False, variant=variant, seed=seed,
            config=dict(d_model=768, n_layers=8, n_heads=12, mlp_hidden_size=3072, max_length=2048),
            source_sha256={name: '1' * 64 for name in ('scripts/research/run_v7_large_batch_ablation.py',
                'scripts/research/run_v7_architecture_pilot.py', 'scripts/research/context_pilot_metrics.py')},
            micro_batch=16, grad_accum=2, global_batch=32, scheduler_horizon=3750, warmup=100,
            learning_rate=1e-4, loss='existing_micro_mean', mask_protocol='independent_seed_per_global_microstep',
            evaluation_interval=250, torch_version='ARTIFICIAL_METADATA_ONLY', common_initialization_sha256=f'{seed:064x}',
            study=dict(name=plot.STUDY, factor_order=['G', 'S', 'L'],
                relation_objective='existing_binding_positives_only', context_sequence_loss=False,
                negative_pair_construction=False, expected_train_records=120000, complete_epoch_updates=3750,
                evaluation='source_and_context_strata_with_relation_token_counts_v1', evaluation_micro_batch=4,
                loss='existing_micro_mean', training_batch_protocol='micro16_ga2_global32',
                mask_stream='seed*1000003+(step-1)*grad_accum+physical_micro_index',
                p02_mask_or_gradient_equivalence_claim=False, profile_status='selected',
                batch_plan=self.plan_ref, input_preflight_sha256='2' * 64),
            data=dict(path=str(self.root / 'unread_data/sample_manifest.json'), sha256=plot.base.SAMPLE_SHA,
                manifest={'files': {'train': {'records': 120000},
                    'valid': {'records': 1024, 'source_counts': {'ARTIFICIAL': 1024}}}}))

    @staticmethod
    def point(step):
        aggregate = {'mask_levels': {}, 'mask_mean': {}}
        for name in plot.base.MASKS:
            level = dict(aa_count=1024, terminal_count=1024, relation_count=1024, selected_count=3072)
            for metric, count in plot.base.COUNT_FOR.items():
                value = .5 if metric.endswith('accuracy') else 1.
                level[metric], level[metric + '_sum'] = value, value * level[count]
                aggregate['mask_mean'][metric] = value
            aggregate['mask_levels'][name] = level
        stratum = {**aggregate, 'records': 1024}
        return dict(step=step, records=step * 32, input_tokens=step * 128, eligible_tokens=step * 64,
            selected_aa_terminal_tokens=step * 32, train_seconds=float(step), epoch=int(step == 3750),
            position=(step * 32) % 120000, subset_train_records=120000, peak_allocated_gib=1.,
            validation=dict(records=1024, token_pooled=aggregate, sources={'ARTIFICIAL': stratum},
                            context_strata={'with_context': stratum}))

    def collect(self):
        return plot.collect(self.jobs_path, self.pool, self.plan_path, self.plan_sha)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v7_large_batch_plot_metadata_')
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()
        self.assertFalse(self.root.exists())

    def test_empty_and_step_zero_export_no_figures(self):
        for name, cells in [('empty', []), ('initial', [('f000', 42), ('f001', 42)])]:
            fixture = ArtificialStudy(self.root / name, cells)
            report = fixture.collect()
            self.assertEqual(report['status'], 'awaiting_shared_evaluation')
            self.assertIsNone(report['all_22_common_step'])
            self.assertEqual(len(report['missing_endpoint_cells']), 22)
            prefix = self.root / name / 'figures/result'
            plot.export(prefix, report)
            self.assertEqual(report['figures'], [])
            self.assertFalse(list(prefix.parent.glob('*.png')))
            with prefix.with_suffix('.csv').open() as stream:
                self.assertTrue(set(plot.PROTOCOL_COLUMNS) <= set(csv.DictReader(stream).fieldnames))
            self.assertEqual(report['profile_evidence_interpretation'], 'byte_identity_only_not_GPU_pass_certification')

    def test_partial_common_step_and_missing_factorial_cells(self):
        report = ArtificialStudy(self.root, [('f000', 42), ('f001', 42)], (0, 250)).collect()
        self.assertEqual((report['status'], report['endpoint_step'], report['endpoint_scope']), ('partial', 250, 'partial'))
        self.assertIsNone(report['all_22_common_step'])
        self.assertEqual(len(report['missing_endpoint_cells']), 20)
        self.assertFalse(report['factorial_contrasts'])
        self.assertTrue(report['paired_deltas'])

    def test_full_22_terminal_figures_and_exports(self):
        cells = [(variant, seed) for variant in plot.base.VARIANTS for seed in plot.base.SEEDS]
        fixture = ArtificialStudy(self.root, cells, (0, 250, 3750))
        report = fixture.collect()
        self.assertEqual((report['status'], report['all_22_common_step']), ('final', 3750))
        self.assertEqual(len(report['factorial_contrasts']), 14)
        prefix = self.root / 'figures/result'
        plot.export(prefix, report)
        self.assertEqual(len(report['figures']), 12)
        for name in report['figures']:
            path = Path(name)
            raw = path.read_bytes()
            self.assertGreater(len(raw), 100)
            if path.suffix == '.svg':
                self.assertIn(b'Large-batch context ablation', raw)
                self.assertNotIn(b'p02 context ablation', raw)
            elif path.suffix == '.pdf':
                self.assertTrue(raw.startswith(b'%PDF'))
            else:
                self.assertTrue(raw.startswith(b'\x89PNG\r\n\x1a\n'))
            ARTIFACT_CHECKS.append(dict(name=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                        scope='artificial_fixture_removed'))
        for suffix in ('', '_paired', '_interactions', '_removal', '_runs'):
            with prefix.with_name(prefix.name + suffix).with_suffix('.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertTrue(all(row['study'] == plot.STUDY and row['evaluation_micro_batch'] == '4'
                and row['micro_batch'] == '16' and row['batch_plan_sha256'] == fixture.plan_sha for row in rows))
        result_path = fixture.pool / 'results' / (fixture.jobs['jobs'][0]['job_id'] + '.json')
        result_path.unlink()
        self.assertEqual(fixture.collect()['status'], 'partial')

    def test_identity_and_cohort_mismatch_rejected(self):
        fixture = ArtificialStudy(self.root, [('f000', 42), ('f001', 42)], (0, 250))
        path = fixture.pool / 'runs' / fixture.jobs['jobs'][1]['job_id'] / 'run_provenance.json'
        original = json.loads(path.read_text())
        changes = [('study', 'name', 'v7_context_factorial_120k_v1'),
                   ('study', 'evaluation_micro_batch', 16), ('study', 'profile_status', 'pending'),
                   ('study', 'batch_plan', {}), ('config', 'max_length', 128),
                   ('source_sha256', 'scripts/research/context_pilot_metrics.py', '3' * 64)]
        for parent, key, value in changes:
            changed = copy.deepcopy(original)
            changed[parent][key] = value
            write(path, changed)
            with self.subTest(field=(parent, key)), self.assertRaises(ValueError):
                fixture.collect()
        changed = copy.deepcopy(original)
        changed.update(micro_batch=32, grad_accum=1)
        write(path, changed)
        with self.assertRaises(ValueError):
            fixture.collect()

    def test_evidence_and_ledger_binding_rejected(self):
        fixture = ArtificialStudy(self.root)
        raw_profile, raw_plan = fixture.profile.read_bytes(), fixture.plan_path.read_bytes()
        fixture.profile.write_bytes(raw_profile + b' ')
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            fixture.collect()
        fixture.profile.write_bytes(raw_profile)
        fixture.plan_path.write_bytes(raw_plan + b' ')
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            fixture.collect()
        fixture.plan_path.write_bytes(raw_plan)
        fixture.jobs['study'] = 'v7_context_factorial_120k_v1'
        write(fixture.jobs_path, fixture.jobs)
        with self.assertRaisesRegex(ValueError, 'ledger study'):
            fixture.collect()
        fixture = ArtificialStudy(self.root / 'alias', [('f000', 42)])
        fixture.jobs['jobs'][0]['alias'] = 'p02_v7_f000_seed42'
        write(fixture.jobs_path, fixture.jobs)
        with self.assertRaisesRegex(ValueError, 'old p02'):
            fixture.collect()


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    print(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed', cases=result.testsRun,
        scope='artificial_metadata_and_matplotlib_only', GPU_executed=False, models_imported=False,
        real_data_read=False, checkpoints_created=0, temporary_artifacts_removed=True,
        source_sha256=hashlib.sha256(ENTRY.read_bytes()).hexdigest(), artifact_checks=ARTIFACT_CHECKS)))
    sys.exit(0 if result.wasSuccessful() else 1)
