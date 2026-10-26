"""Check only the new recovery lineage boundary with temporary metadata.

Activate conda pllm, then run:
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_v7_post_generation_recovery_lineage.py
Reuse the original fixture builder without rerunning its accepted tests. No
sequence, checkpoint, real audit certificate, GPU or queue entry is created.
"""
import ast
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path('/project_c20250601/251105016/project/dllm_test')
LOCAL = ROOT/'output/v7_local_ablation_20261005'
OLD = LOCAL/'post_generation_pool_v1/source_v1'
NEW = LOCAL/'post_generation_pool_v2/source_v1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


gate = module('recovery_post_generation_gate', NEW/'v7_post_generation_acceptance.py')
fixtures = module('original_post_generation_fixtures',
                  ROOT/'scripts/research/test_v7_post_generation_acceptance.py')


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


class RecoveryLineage(unittest.TestCase):
    def setUp(self):
        context = patch.object(fixtures, 'gate', gate)
        context.start(); self.addCleanup(context.stop)
        self.fixture = f = fixtures.CompletionBoundary(
            'test_complete_metadata_is_readonly_and_preserves_legacy_stop')
        f.setUp(); self.addCleanup(f.doCleanups)
        self.legacy = f.r/'legacy_epoch'
        proof = self.proof = self.legacy/'SUPERVISION_STOPPED.json'
        fixtures.write(proof, {'fixture_only': True, 'original_failure': 'preserved'})
        helper = f.epoch/'common.py'
        helper.write_text('"""Artificial file-only delegation probe, never a launcher."""\n'
            'def fixed():\n    raise AssertionError("Never probe root PIDs on workers")\n'
            'def failed_group_checked():\n'
            '    return dict(failure_preserved=True, failed_ranks_cleaned=8, training_restarted=False)\n')
        package = f.epoch/'recovery_package.json'
        reference = fixtures.write(package, dict(schema='v7.distinct_recovery_package.v1',
            goal_thread_id='01a0f245-4559-7b62-9dc2-0917a8ed2267',
            edges=[[45050, 45058], [45058, 50058], [50058, 50082]],
            files={str(path): gate.sha(path) for path in (proof, helper)}))
        context = patch.dict(gate.__dict__, dict(LEGACY_EPOCH=self.legacy,
            RECOVERY_PACKAGE=package, RECOVERY_PACKAGE_SHA=reference['sha256'],
            RECOVERY_COMMON=helper))
        context.start(); self.addCleanup(context.stop)
        f.training['prior_step'] = 50058
        f.catalog['formal_completion'] = fixtures.write(f.epoch/'COMPLETE.json', f.training)
        f.release['training_acceptance'] = f.catalog['formal_completion']
        f.catalog['generation_release'] = fixtures.write(f.gen/'release.json', f.release)

    def test_new_lineage_accepts_preserved_failure_without_root_pid_probe(self):
        before = {p: p.read_bytes() for p in self.fixture.r.rglob('*') if p.is_file()}
        result = gate.check(self.fixture.catalog)
        self.assertEqual(result['status'], 'actual_training_and_generation_evidence_passed')
        self.assertEqual(before, {p: p.read_bytes() for p in self.fixture.r.rglob('*') if p.is_file()})
        self.assertTrue(self.proof.exists())

    def test_changed_original_failure_is_rejected(self):
        self.proof.write_text('{"fixture_only":true,"original_failure":"changed"}\n')
        with self.assertRaisesRegex(ValueError, 'Preserved original failure'):
            gate.check(self.fixture.catalog)

    def test_old_endpoint_cannot_substitute_for_new_recovery(self):
        f = self.fixture
        fixtures.write(self.legacy/'COMPLETE.json', f.training)
        (f.epoch/'COMPLETE.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Required canonical evidence missing'):
            gate.check(f.catalog)

    def test_wrong_predecessor_step_is_rejected(self):
        f = self.fixture
        f.training['prior_step'] = 50050
        f.catalog['formal_completion'] = fixtures.write(f.epoch/'COMPLETE.json', f.training)
        with self.assertRaisesRegex(ValueError, 'two-epoch endpoint'):
            gate.check(f.catalog)

    def test_explicit_stops_still_block_both_lineages(self):
        for directory in (self.legacy, self.legacy/'generation_pool_v1',
                          self.fixture.epoch, self.fixture.gen):
            for name in ('STOP', 'STOP_NOW'):
                with self.subTest(directory=str(directory), name=name):
                    directory.mkdir(parents=True, exist_ok=True)
                    marker = directory/name; marker.touch()
                    try:
                        with self.assertRaisesRegex(ValueError, 'stop|stopped'):
                            gate.check(self.fixture.catalog)
                    finally:
                        marker.unlink()


class SourceScope(unittest.TestCase):
    def test_adapter_and_existing_publication_body_are_unchanged(self):
        name = 'pool_metrics_only_ablation_v2_dispatch.py'
        self.assertEqual(functions(OLD/name), functions(NEW/name))
        name = 'publish_v7_post_generation_ablation.py'
        previous, current = functions(OLD/name), functions(NEW/name)
        changed = {k for k in previous if previous[k] != current[k]}
        self.assertEqual(changed, {'stopped', 'check_predecessors'})
        self.assertEqual(set(previous), set(current))
        name = 'v7_post_generation_acceptance.py'
        previous, current = functions(OLD/name), functions(NEW/name)
        self.assertEqual({k for k in previous if previous[k] != current[k]}, {'check'})
        self.assertEqual(set(current)-set(previous), {'check_preserved_failure'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
