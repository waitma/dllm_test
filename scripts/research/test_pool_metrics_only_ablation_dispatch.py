"""Run artificial metadata checks for the private post-formal pool adapter.

  source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
  conda activate pllm
  CUDA_VISIBLE_DEVICES='' python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_metrics_only_ablation_dispatch.py

Fixtures contain no sequences, models, GPU calls, jobs or real completion data.
They are created in one temporary directory and removed by unittest cleanup.
The original capacity/native-resume validators are already independently
verified; these tests stub their delegation and test the new metadata boundary.
Even the exec boundary is mocked: no actual main() training is executed.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
SCRIPT = ROOT / 'scripts/research/pool_metrics_only_ablation_dispatch.py'
FROZEN_RUNNER = ROOT / 'output/v7_local_ablation_20261005/source_snapshot_v1/scripts/research/run_v7_large_batch_ablation.py'
spec = importlib.util.spec_from_file_location('private_metrics_pool_adapter', SCRIPT)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class MetadataCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='metadata_metrics_adapter_', dir=ROOT / 'output')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        r = self.root
        old, local, pool, old_pool = (r / name for name in ('old', 'local', 'pool', 'old_pool'))
        data, sup, formal, formal_run = (r / name for name in ('sample', 'supervisor', 'formal', 'formal_run'))
        self.values = dict(ROOT=r, OLD=old, LOCAL=local, POOL=pool, OLD_POOL=old_pool,
            DATA=data, PLAN=old / 'batch_selection_v1.json', ORIGINAL=old / 'jobs.json',
            FREEZE=local / 'source_snapshot_v1.json', MANAGED=local / 'managed_plan.json',
            SUP=sup, FORMAL=formal, FORMAL_RUN=formal_run, POOL_SOURCE=r / 'pool_source.py',
            RESUME_API=formal / 'resume_api.py', PARENT_API=r / 'parent_api.py')
        for p in (pool / 'runs', pool / 'workers', pool / 'pending', pool / 'running', pool / 'results',
                  old_pool / 'running', old_pool / 'pending', old_pool / 'results'):
            p.mkdir(parents=True)
        parent_ref = write(r / 'parent/dataset_manifest.json', {'training_ready': True})
        self.parent_path = Path(parent_ref['path'])
        parent_ref['training_ready'] = True
        certificates = {}
        for name, key in (('audit_certificate', 'manifest_sha256'),
                          ('downstream_scope_certificate', 'dataset_manifest_sha256')):
            entry = write(r / 'parent' / (name + '.json'), dict(status='passed', pending=[],
                forbidden_overlap_rows=0, family_split_overlap=0, **{key: parent_ref['sha256']}))
            entry.update(status='passed', pending=[])
            certificates[name] = entry
        sample_ref = write(data / 'sample_manifest.json', dict(schema='v7.exploratory_uncertified.v1',
            status='exploratory_uncertified', formal_training_ready=False,
            parent=dict(manifest=parent_ref, certificates=certificates),
            files=dict(train=dict(records=120000), valid=dict(records=1024))))
        self.values['SAMPLE_SHA'] = sample_ref['sha256']
        snapshot = local / 'source_snapshot_v1'
        sources = {}
        for name in (*api.MATH_SOURCES, *(f'fixture/{i}.py' for i in range(559))):
            p = snapshot / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('# artificial metadata-only frozen source: ' + name + '\n')
            sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        runner = snapshot / 'scripts/research/run_v7_large_batch_ablation.py'
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text(FROZEN_RUNNER.read_text().replace(api.SAMPLE_SHA, sample_ref['sha256']))
        sources[str(runner.relative_to(snapshot))] = hashlib.sha256(runner.read_bytes()).hexdigest()
        self.source_file = snapshot / api.MATH_SOURCES[0]
        old_freeze_ref = write(old / 'source_snapshot_v1.json', {'fixture': 'old capacity source'})
        freeze_ref = write(self.values['FREEZE'], dict(schema='v7.metrics_only_ablation.source.v1',
            snapshot_root=str(snapshot), base_snapshot=old_freeze_ref['path'],
            base_snapshot_sha256=old_freeze_ref['sha256'], source_sha256=sources))
        self.values['FREEZE_SHA'] = freeze_ref['sha256']
        preflight_ref = write(data.parent / 'context_augmentation_preflight.json', dict(status='passed',
            sample_manifest_sha256=sample_ref['sha256'],
            layout_sha256=sources['dllm/pipelines/immune_llada/v3/layout.py']))
        self.values['PREFLIGHT_SHA'] = preflight_ref['sha256']
        evidence = write(old_pool / 'capacity.json', {'fixture': 'original old-pool capacity evidence'})
        plan_ref = write(self.values['PLAN'], dict(schema='v7.large_batch_selection.v1', study=api.STUDY,
            status='selected', micro_batch=32, grad_accum=1, global_batch=32,
            evaluation_micro_batch=4, sample_manifest_sha256=sample_ref['sha256'],
            variants=list(api.METHODS), seeds=[42, 43], profile_evidence=[evidence]))
        self.values['PLAN_SHA'] = plan_ref['sha256']
        self.alias = 'p04_v7_f010_seed42'
        self.job_id = '200_' + self.alias + '_1234abcd'
        self.job = dict(variant=self.alias, mode='train', steps=3750, data_dir=str(data))
        original_rows = [dict(variant=v, seed=s, alias=a, job_id='100_' + a + '_00000001',
            job=dict(variant=a, mode='train', steps=3750, data_dir=str(data))) for a, (v, s) in api.ALIASES.items()]
        original_ref = write(self.values['ORIGINAL'], dict(study=api.STUDY, jobs=original_rows))
        self.values['ORIGINAL_SHA'] = original_ref['sha256']
        self.old_job_id = next(row['job_id'] for row in original_rows if row['alias'] == self.alias)
        self.local_output, self.local_log = r / 'unused_local/f010_seed42_attempt1', r / 'unused_logs/f010_seed42_attempt1'
        command = [sys.executable, '-I', '-B', '-u', str(runner), '--variant', 'f010', '--seed', '42',
            '--data-dir', str(data), '--output-dir', str(self.local_output), '--steps', '3750',
            '--eval-every', '250', '--micro-batch', '32', '--grad-accum', '1', '--batch-plan',
            str(self.values['PLAN']), '--batch-plan-sha256', plan_ref['sha256']]
        managed_ref = write(self.values['MANAGED'], dict(commands=[dict(variant='f010', seed=42,
            output_dir=str(self.local_output), log_dir=str(self.local_log), training_argv=command)]))
        self.values['MANAGED_SHA'] = managed_ref['sha256']
        stage = formal / 'launch_20261005/segment_25042'
        formal_ids = [f'300_p06_v7_formal_member_{i}_00000002' for i in range(8)]
        formal_results = {j: write(pool / 'results' / (j + '.json'), dict(job_id=j, state='succeeded',
            returncode=0)) for j in formal_ids}
        write(stage / 'enqueue_receipt.json', dict(job_ids=formal_ids))
        self.state_path = stage / 'meeting/coordinator/state.json'
        write(self.state_path, dict(terminal='succeeded', abort=None))
        self.receipt_path = stage / 'meeting/coordinator/rank3_parent_receipt.json'
        for rank in range(8):
            write(stage / f'meeting/coordinator/rank{rank}_parent_receipt.json', dict(rank=rank, error=None,
                cleanup=dict(returncode=0, wait_observed=True, process_absent=True,
                    process_group_gone=True, cleanup_completed=True)))
        binding = 'a' * 64
        marker_ref = write(formal_run / 'COMPLETE.json', dict(status='completed', step=25042,
            world_size=8, binding_sha256=binding))
        prior_ref = write(sup / 'checkpoint_evidence/checkpoint-25016/formal_checkpoint.json',
            dict(saved_parameter_digest='b' * 64))
        sidecar_ref = write(sup / 'checkpoint_evidence/checkpoint-25042/formal_checkpoint.json',
            dict(step=25042, world_size=8, binding_sha256=binding,
                rank_acknowledgments=list(range(8)), saved_parameter_digest='c' * 64))
        self.resume = dict(prior_step=25016, final_step=25042, binding_sha256=binding,
            sidecar_sha256=prior_ref['sha256'], ranks=list(range(8)))
        resume_ref = write(formal_run / 'RESUME_25016_TO_25042.json', self.resume)
        completion_ref = write(sup / 'COMPLETE.json', dict(status='actual_segment_passed',
            step=25042, prior_step=25016, checkpoint=dict(step=25042, sidecar=sidecar_ref),
            marker=marker_ref, resume=resume_ref, pool_results=formal_results))
        self.values['POOL_SOURCE'].write_text('# artificial pool source\n')
        self.values['POOL_SOURCE_SHA'] = hashlib.sha256(self.values['POOL_SOURCE'].read_bytes()).hexdigest()
        self.parent = dict(schema='v7.pool_parent.runtime_observation.v1', pool_dir=str(pool),
            resolved_pool_source=str(self.values['POOL_SOURCE']),
            source_sha256={str(self.values['POOL_SOURCE']): self.values['POOL_SOURCE_SHA']},
            pool_process={'fixture_pid': os.getppid()}, pool_argv=['fixture pool'], pool_cwd=str(r))
        parent_observation = write(pool / 'logs/observed_parent.json', self.parent)
        self.catalog_path, self.ledger_path = r / 'admission/catalog.json', r / 'admission/ledger.json'
        self.lock = self.ledger_path.parent / 'cell_locks/f010_seed42.lock'
        self.lock.parent.mkdir(parents=True)
        self.lock.write_bytes(b'')
        lock_ref = dict(path=str(self.lock), sha256=hashlib.sha256(b'').hexdigest())
        self.ledger = dict(schema='v7.metrics_only.ablation_claims.v1', study=api.STUDY,
            pool_dir=str(pool), cells=[dict(variant=v, seed=s, state='reserved_pool' if (v, s) == ('f010', 42)
                else 'unclaimed', attempts=[dict(transport='pool', pool_dir=str(pool), job_id=self.job_id,
                    lock_file=lock_ref)] if (v, s) == ('f010', 42) else []) for v, s in api.ALIASES.values()])
        ledger_ref = write(self.ledger_path, self.ledger)
        self.catalog = dict(schema='v7.metrics_only.pool_catalog.v1', study=api.STUDY, pool_dir=str(pool),
            batch_plan=plan_ref, source_freeze=freeze_ref, original_catalog=original_ref,
            managed_local_plan=managed_ref, adapter_source=dict(path=str(SCRIPT), sha256=api.sha(SCRIPT)),
            formal_completion=completion_ref, formal_stage=str(stage), ledger=ledger_ref,
            pool_parent_observation=parent_observation, automatic_retry=False,
            jobs=[dict(variant='f010', seed=42, alias=self.alias, job_id=self.job_id, job=self.job)])
        self.rebind_catalog()
        self.argv = ['--variant', self.alias, '--mode', 'train', '--steps', '3750', '--data-dir', str(data),
            '--profile-dir', str(pool / 'profiles' / self.alias), '--output-dir', str(pool / 'runs' / self.job_id)]
        self.capacity_calls = []
        self.legacy = dict(POOL=old_pool, PLAN=self.values['PLAN'], FREEZE=old / 'source_snapshot_v1.json',
            FREEZE_SHA=old_freeze_ref['sha256'], check_freeze=lambda canonical: (old / 'source_snapshot_v1',
                {n: sources[n] for n in api.MATH_SOURCES}))
        def capacity(plan, old_snapshot, old_sources, canonical):
            self.assertEqual(self.legacy['POOL'], old_pool)
            self.assertEqual(old_snapshot, old / 'source_snapshot_v1')
            self.assertEqual(len(old_sources), len(api.MATH_SOURCES))
            self.capacity_calls.append('original capacity namespace')
        self.legacy['check_capacity'] = capacity
        old_api = api._api
        def fixture_api(path, expected):
            if path == self.values['RESUME_API']:
                return {'complete_receipt': lambda ranks: self.resume if ranks == list(range(8)) else None}
            if path == self.values['PARENT_API']:
                return {'coordinator_api': lambda: {'process_identity': lambda pid: self.parent['pool_process']},
                    'read_parent_cmdline': lambda pid: self.parent['pool_argv'],
                    'read_parent_cwd': lambda pid: self.parent['pool_cwd']}
            return old_api(path, expected)
        for context in (patch.dict(api.__dict__, self.values), patch.object(api, '_legacy_runtime', return_value=self.legacy),
                        patch.object(api, '_api', side_effect=fixture_api)):
            context.start()
            self.addCleanup(context.stop)

    def rebind_catalog(self):
        ref = write(self.catalog_path, self.catalog)
        self.kwargs = dict(catalog_path=self.catalog_path, catalog_sha256=ref['sha256'],
            ledger_path=self.ledger_path, ledger_sha256=self.catalog['ledger']['sha256'])

    def rebind_ledger(self):
        self.catalog['ledger'] = write(self.ledger_path, self.ledger)
        self.rebind_catalog()

    def prepare(self):
        return api.prepare(self.argv, **self.kwargs)

    def own_ticket(self, device='4'):
        write(self.values['POOL'] / 'running' / (self.job_id + '.json'), self.job)
        expected = [sys.executable, '-u', str(self.root / 'scripts/run_immune_research.py')] + self.argv[:6]
        expected += ['--profile-dir', self.argv[self.argv.index('--profile-dir') + 1],
            '--output-dir', self.argv[self.argv.index('--output-dir') + 1], '--data-dir', str(self.values['DATA'])]
        write(self.values['POOL'] / 'workers/fixture_4.json', dict(job_id=self.job_id, state='running',
            device=device, job=self.job, argv=expected, pid=os.getpid()))

    def test_prepare_is_read_only_and_keeps_protocol_and_original_capacity(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        environment = dict(os.environ)
        args, command = self.prepare()
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertEqual(environment, dict(os.environ))
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[:4], [sys.executable, '-I', '-B', '-u'])
        self.assertNotIn('--resume', command)
        self.assertNotIn('--save-checkpoints', command)
        self.assertFalse(args.output_dir.exists())
        self.assertEqual(self.capacity_calls, ['original capacity namespace'])
        self.assertFalse(any(n.split('.')[0] in {'torch', 'transformers', 'accelerate', 'numpy', 'dllm'} for n in sys.modules))

    def test_formal_terminal_result_and_cleanup_are_required(self):
        with self.subTest('no completed group'):
            write(self.state_path, dict(terminal=None, abort=None))
            with self.assertRaisesRegex(ValueError, 'terminal succeeded'):
                self.prepare()
        write(self.state_path, dict(terminal='succeeded', abort=None))
        with self.subTest('cleanup wait must be observed'):
            receipt = json.loads(self.receipt_path.read_text())
            receipt['cleanup']['wait_observed'] = False
            write(self.receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, 'cleanup'):
                self.prepare()
        receipt['cleanup']['wait_observed'] = True
        write(self.receipt_path, receipt)
        with self.subTest('actual result is not an inferred zero exit'):
            ref = self.catalog['formal_completion']
            complete = json.loads(Path(ref['path']).read_text())
            first_id, result_ref = next(iter(complete['pool_results'].items()))
            result = json.loads(Path(result_ref['path']).read_text())
            result['returncode'] = None
            complete['pool_results'][first_id] = write(Path(result_ref['path']), result)
            self.catalog['formal_completion'] = write(Path(ref['path']), complete)
            self.rebind_catalog()
            with self.assertRaisesRegex(ValueError, 'exit was not verified'):
                self.prepare()

    def test_source_parent_ready_and_unique_claim_fail_closed(self):
        original = self.source_file.read_bytes()
        self.source_file.write_bytes(original + b'# changed\n')
        with self.assertRaisesRegex(ValueError, 'Frozen source differs'):
            self.prepare()
        self.source_file.write_bytes(original)
        parent = json.loads(self.parent_path.read_text())
        parent['training_ready'] = False
        write(self.parent_path, parent)
        with self.assertRaisesRegex(ValueError, 'Metadata SHA differs'):
            self.prepare()
        parent['training_ready'] = True
        write(self.parent_path, parent)
        self.ledger['cells'][0] = dict(self.ledger['cells'][1])
        self.rebind_ledger()
        with self.assertRaisesRegex(ValueError, 'Duplicate/unknown ledger cell'):
            self.prepare()

    def test_existing_local_original_success_and_formal_member_block_claim(self):
        self.local_log.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'local attempt already exists'):
            self.prepare()
        self.local_log.rmdir()
        old_result = self.values['OLD_POOL'] / 'results' / (self.old_job_id + '.json')
        write(old_result, dict(state='succeeded', returncode=0))
        with self.assertRaisesRegex(ValueError, 'Original cell already succeeded'):
            self.prepare()
        old_result.unlink()
        formal_ticket = self.values['POOL'] / 'pending/late_formal.json'
        write(formal_ticket, dict(variant='p06_v7_formal_member_7'))
        with self.assertRaisesRegex(ValueError, 'Formal ticket remains active'):
            self.prepare()

    def test_mocked_exec_preserves_worker_string_device_and_cuda_mask(self):
        self.own_ticket()
        environment = {'CUDA_VISIBLE_DEVICES': '4'}
        with patch.dict(os.environ, environment, clear=True), patch.object(api.os, 'execv') as execute:
            api.main(self.argv, **self.kwargs)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '4')
            self.assertEqual(execute.call_count, 1)
            executable, command = execute.call_args.args
            self.assertEqual(executable, sys.executable)
            self.assertEqual(command[0], sys.executable)
        self.own_ticket(device=4)
        with patch.dict(os.environ, environment, clear=True), patch.object(api.os, 'execv') as execute:
            with self.assertRaisesRegex(ValueError, 'worker ownership'):
                api.main(self.argv, **self.kwargs)
            execute.assert_not_called()

    def test_post_lock_recheck_blocks_local_race_and_replaced_lock_inode(self):
        self.own_ticket()
        real_prepare = api.prepare
        def raced(*args, **kwargs):
            result = real_prepare(*args, **kwargs)
            self.local_output.mkdir(parents=True)
            return result
        with patch.object(api, 'prepare', side_effect=raced), patch.object(api.os, 'execv') as execute:
            with self.assertRaisesRegex(ValueError, 'local attempt already exists'):
                api.main(self.argv, **self.kwargs)
            execute.assert_not_called()
        self.local_output.rmdir()
        def replaced(*args, **kwargs):
            result = real_prepare(*args, **kwargs)
            alternative = self.lock.with_suffix('.replacement')
            alternative.write_bytes(b'')
            alternative.replace(self.lock)
            return result
        with patch.object(api, 'prepare', side_effect=replaced), patch.object(api.os, 'execv') as execute:
            with self.assertRaisesRegex(ValueError, 'lock inode changed'):
                api.main(self.argv, **self.kwargs)
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
