"""Check only the new post-generation scheduling boundary with artificial files.

Activate conda pllm, then run:
python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_post_generation_acceptance.py
Temporary fixtures contain no sequences, weights, real audit certificates or
jobs. They are removed after each test. No training entry or GPU is invoked.
Existing model/source/data gates are not retested or replaced in production.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts/research'/filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


gate = module('post_generation_gate', 'v7_post_generation_acceptance.py')
adapter = module('post_generation_adapter', 'pool_metrics_only_ablation_v2_dispatch.py')


def reference(path):
    return {'path': str(path), 'sha256': gate.sha(path)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')
    return reference(path)


class CompletionBoundary(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='metadata_post_generation_', dir=ROOT/'output')
        self.addCleanup(tmp.cleanup)
        r = self.r = Path(tmp.name)
        self.epoch, self.gen, self.pool = r/'epoch2', r/'generation', r/'pool'
        epoch, gen, pool = self.epoch, self.gen, self.pool
        old = r/'old_supervisor'
        old.mkdir(); (old/'STOP').touch()
        stage = epoch/'launches_v1/segment_50082'
        self.state = stage/'meeting/coordinator/state.json'
        write(self.state, {'terminal': 'succeeded', 'abort': None})
        self.training = {'status': 'actual_world8_epoch2_continuation_passed', 'step': 50082,
            'prior_step': 50050, 'epoch': 2, 'eight_native_restores_and_owned_exits': True,
            'checkpoint': write(epoch/'checkpoint.json', {'fixture_only': True}), 'actual_pool_results': []}
        jobs = []
        for rank in range(8):
            job = 'fixture_training_'+str(rank); jobs.append(job)
            self.training['actual_pool_results'].append(write(pool/'results'/(job+'.json'),
                {'job_id': job, 'state': 'succeeded', 'returncode': 0}))
            write(stage/f'meeting/coordinator/rank{rank}_parent_receipt.json', {'rank': rank, 'error': None,
                'cleanup': dict(returncode=0, wait_observed=True, process_absent=True,
                    process_group_gone=True, cleanup_completed=True)})
        write(stage/'enqueue_receipt.json', {'job_ids': jobs})
        self.catalog = {'formal_stage': str(stage), 'formal_completion': write(epoch/'COMPLETE.json', self.training)}
        dispatcher = gen/'dispatcher.py'
        dispatcher.parent.mkdir(parents=True, exist_ok=True)
        dispatcher.write_text('"""Artificial delegation probe; no model or data."""\ndef completion(release):\n    assert release["fixture_original_gate"] is True\n')
        public, registry = gen/'public_after_candidate.py', gen/'registry_after_candidate.py'
        public.write_text('# fixture public\n'); registry.write_text('# fixture registry\n')
        files = {str(p): gate.sha(p) for p in (dispatcher, public, registry)}
        package = gen/'package_v1.json'; self.package_ref = write(package, {'files': files})
        self.release = {'schema': 'v7.cdr.pool_release.v1', 'work_type': 'generation_evaluation',
            'optimizer_updates': 0, 'automatic_retry': False, 'training_acceptance': self.catalog['formal_completion'],
            'files': {**files, str(r/'scripts/run_immune_research.py'): gate.sha(public),
                str(r/'dllm/pipelines/immune_llada/research/model.py'): gate.sha(registry)},
            'fixture_original_gate': True}
        tasks, rows = [], []
        for dataset, mode in gate.CELLS:
            job = f'fixture_{dataset}_{mode}'; output = r/'outputs'/job; run = pool/'runs'/job
            cell = [dataset, mode]; count = 3127 if dataset == 'kong' else 60
            result = write(pool/'results'/(job+'.json'), dict(job_id=job, state='succeeded', returncode=0))
            exited = write(run/'EVALUATION_EXIT.json', dict(returncode=0))
            metrics = write(output/mode/'metrics.json', dict(status='complete', n_sequences=count, n_expected=count, n_missing=0))
            summary = write(output/'summary.json', dict(status='complete', step=50082, dataset=dataset,
                coverage='assigned_mode', assigned_mode=mode))
            audit = output/mode/'input_audit.jsonl'; audit.write_text('{"fixture_only":true}\n')
            complete = write(run/'COMPLETE.json', dict(status='generation_evaluation_complete',
                work_type='generation_evaluation', optimizer_updates=0, checkpoint=self.training['checkpoint'],
                actual_evaluator_exit=exited, metrics=metrics, summary=summary))
            tasks.append(dict(cell=cell, n_sequences=count, actual_pool_result=result, actual_completion=complete,
                metrics=metrics, input_audit=reference(audit)))
            rows.append(dict(job_id=job, cell=cell, evaluation_output=str(output)))
        self.release['catalog'] = write(gen/'catalog.json', {'jobs': rows})
        self.done = dict(status='nine_actual_cdr_tasks_passed', checkpoint=self.training['checkpoint'],
            total_cdr_cases=9741, all_actual_pool_and_evaluator_exits_zero=True, tasks=tasks)
        self.catalog.update(generation_release=write(gen/'release.json', self.release),
            generation_completion=write(gen/'COMPLETE.json', self.done), generation_package=self.package_ref)
        context = patch.dict(gate.__dict__, dict(ROOT=r, EPOCH=epoch, PARENT=r/'parent', GENERATION=gen,
            POOL=pool, OLD_SUPERVISOR=old, PACKAGE=package, PACKAGE_SHA=self.package_ref['sha256'], DISPATCHER=dispatcher))
        context.start(); self.addCleanup(context.stop)

    def refresh_done(self):
        self.catalog['generation_completion'] = write(self.gen/'COMPLETE.json', self.done)

    def test_complete_metadata_is_readonly_and_preserves_legacy_stop(self):
        before = {p: p.read_bytes() for p in self.r.rglob('*') if p.is_file()}
        result = gate.check(self.catalog)
        self.assertEqual(result['generation_tasks'], 9)
        self.assertEqual(before, {p: p.read_bytes() for p in self.r.rglob('*') if p.is_file()})
        self.assertTrue((self.r/'old_supervisor/STOP').exists())
        self.assertFalse(any(n.split('.')[0] in {'torch', 'transformers', 'numpy', 'dllm'} for n in sys.modules))

    def test_one_epoch_does_not_admit_ablations(self):
        self.training.update(step=25042, epoch=1)
        self.catalog['formal_completion'] = write(self.epoch/'COMPLETE.json', self.training)
        with self.assertRaisesRegex(ValueError, 'two-epoch endpoint'):
            gate.check(self.catalog)

    def test_rank_cleanup_and_failed_actual_exit(self):
        path = Path(self.catalog['formal_stage'])/'meeting/coordinator/rank6_parent_receipt.json'
        receipt = gate.read(path); receipt['cleanup']['process_group_gone'] = False; write(path, receipt)
        with self.assertRaisesRegex(ValueError, 'rank cleanup'):
            gate.check(self.catalog)
        receipt['cleanup']['process_group_gone'] = True; write(path, receipt)
        first = self.training['actual_pool_results'][0]; result = gate.read(first['path']); result['returncode'] = 1
        self.training['actual_pool_results'][0] = write(Path(first['path']), result)
        self.catalog['formal_completion'] = write(self.epoch/'COMPLETE.json', self.training)
        with self.assertRaisesRegex(ValueError, 'pool exit'):
            gate.check(self.catalog)

    def test_missing_or_duplicate_generation_cell(self):
        original = list(self.done['tasks'])
        for tasks in (original[:-1], original[:-1]+[original[0]]):
            self.done['tasks'] = tasks; self.refresh_done()
            with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
                gate.check(self.catalog)

    def test_evaluator_failure_not_hidden_by_pool_success(self):
        task = self.done['tasks'][0]; cp = Path(task['actual_completion']['path']); complete = gate.read(cp)
        ep = Path(complete['actual_evaluator_exit']['path'])
        complete['actual_evaluator_exit'] = write(ep, {'returncode': 1})
        task['actual_completion'] = write(cp, complete); self.refresh_done()
        with self.assertRaisesRegex(ValueError, 'evaluator exit failed'):
            gate.check(self.catalog)

    def test_changed_audit_or_published_source_fails(self):
        audit = Path(self.done['tasks'][0]['input_audit']['path']); old = audit.read_bytes(); audit.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'input audit changed'):
            gate.check(self.catalog)
        audit.write_bytes(old)
        gate.DISPATCHER.write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'dependency changed'):
            gate.check(self.catalog)

    def test_active_generation_ticket_and_explicit_stop(self):
        job = 'fixture_kong_cdrh1'; ticket = self.pool/'running'/(job+'.json'); write(ticket, {})
        with self.assertRaisesRegex(ValueError, 'ticket still active'):
            gate.check(self.catalog)
        ticket.unlink(); (self.gen/'STOP').touch()
        with self.assertRaisesRegex(ValueError, 'stage has stopped'):
            gate.check(self.catalog)


class ClaimBoundary(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='metadata_ablation_claim_', dir=ROOT/'output')
        self.addCleanup(tmp.cleanup); r = self.r = Path(tmp.name)
        local, pool, old = r/'local', r/'pool', r/'oldpool'; self.pool, self.old = pool, old
        self.ledger_path = local/'guarded_dispatch_v1/claims.json'
        lock = self.ledger_path.parent/'cell_locks/f010_seed42.lock'; lock.parent.mkdir(parents=True); lock.touch()
        alias = 'p04_v7_f010_seed42'; self.jobid = '123_'+alias+'_abcd1234'
        job = dict(variant=alias, mode='train', steps=3750, data_dir=str(adapter.DATA))
        self.catalog = dict(automatic_retry=False, ledger={'path':str(self.ledger_path), 'sha256':'fixture'},
            jobs=[dict(variant='f010', seed=42, alias=alias, job_id=self.jobid, job=job)])
        self.original = {'jobs':[dict(variant=v,seed=s,job_id=f'old_{v}_{s}') for v,s in adapter.ALIASES.values()]}
        cells = [dict(variant=v,seed=s,state='unclaimed',attempts=[]) for v,s in adapter.ALIASES.values()]
        selected = next(c for c in cells if (c['variant'],c['seed'])==('f010',42))
        selected.update(state='reserved_pool',attempts=[dict(transport='pool',pool_dir=str(pool),job_id=self.jobid,lock_file=reference(lock))])
        self.ledger = dict(schema='v7.metrics_only.ablation_claims.v1',study=adapter.STUDY,pool_dir=str(pool),cells=cells)
        self.managed = {'commands':[dict(variant='f010',seed=42,output_dir=str(local/'candidate'),log_dir=str(r/'logs'))]}
        self.args = SimpleNamespace(variant=alias,output_dir=pool/'runs'/self.jobid)
        context = patch.dict(adapter.__dict__, dict(LOCAL=local,POOL=pool,OLD_POOL=old))
        context.start(); self.addCleanup(context.stop)

    def check(self):
        return adapter.check_claim(self.catalog,self.ledger,self.args,self.managed,self.original)

    def test_common_lock_directory_is_required(self):
        self.assertEqual(self.check()[1].parent,self.ledger_path.parent/'cell_locks')
        self.catalog['ledger']['path']=str(self.r/'disjoint/claims.json')
        with self.assertRaisesRegex(ValueError,'same original lock directory'):
            self.check()

    def test_original_completion_prevents_duplicate_even_without_exit_result(self):
        write(self.old/'runs/old_f010_42/COMPLETE.json',{'fixture':True})
        with self.assertRaisesRegex(ValueError,'already has completion'):
            self.check()

    def test_failed_new_attempt_is_not_automatically_retried(self):
        write(self.pool/'results/previous.json',{'job':{'variant':self.args.variant},'state':'failed','returncode':1})
        with self.assertRaisesRegex(ValueError,'no automatic retry'):
            self.check()

    def test_generation_still_running_prevents_ablation(self):
        write(self.pool/'running/generation.json',{'variant':'p07_v7_cdr_kong_cdrh1'})
        with self.assertRaisesRegex(ValueError,'generation ticket remains active'):
            self.check()


class SourceAndLiveBoundary(unittest.TestCase):
    def test_original_data_model_worker_and_exec_code_are_unchanged(self):
        functions=[]
        for name in ('pool_metrics_only_ablation_dispatch.py','pool_metrics_only_ablation_v2_dispatch.py'):
            tree=ast.parse((ROOT/'scripts/research'/name).read_text())
            functions.append({n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)})
        for name in ('check_data','check_sources_and_batch','check_worker','main'):
            self.assertEqual(functions[0][name],functions[1][name])

    def test_actual_unfinished_training_is_rejected_without_writes(self):
        # There is no fabricated receipt or fallback in the real namespace.
        path=gate.EPOCH/'COMPLETE.json'
        if path.exists():
            self.skipTest('Actual endpoint now exists; this pre-completion check is no longer applicable')
        catalog={'formal_completion':{'path':str(path),'sha256':'0'*64}}
        with self.assertRaisesRegex(ValueError,'evidence missing'):
            gate.check(catalog)
        self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
