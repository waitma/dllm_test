"""CPU-only protocol checks; artificial Linear only, no records/groups/jobs.

Activate conda pllm, then run python -B /vepfs-mlp2/c20250601/251105016/
project/dllm_test/scripts/research/test_v7_formal_protocol.py.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('run_v7_formal.py')
SPEC = importlib.util.spec_from_file_location('v7_formal_protocol_tested', PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class ProtocolTests(unittest.TestCase):
    def arguments(self, *extra):
        return ['--source-root', '/tmp/frozen', '--repo-root', '/tmp', '--data-dir', '/tmp/data', '--output-dir', '/tmp/run',
                '--source-manifest', '/tmp/sources.json', '--source-manifest-sha256', 'a'*64,
                '--variant', 'r20', '--seed', '43', '--micro-batch', '4', '--grad-accum', '8',
                '--global-batch', '256', '--max-steps', '10000', *extra]

    def test_explicit_world_budget_and_fixed_horizon(self):
        args = driver.parse_args(self.arguments('--stop-after', '20'))
        self.assertEqual((args.global_batch, args.max_steps, args.stop_after), (256, 10000, 20))
        for extra in [('--global-batch', '32'), ('--grad-accum', '0'),
                      ('--stop-after', '10001'), ('--learning-rate', 'nan')]:
            with self.assertRaises(ValueError):
                driver.parse_args(self.arguments(*extra))

    def test_external_launcher_required(self):
        distributed = SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 8,
            get_rank=lambda: 3, get_backend=lambda: 'nccl')
        cuda = SimpleNamespace(device_count=lambda: 1, current_device=lambda: 0)
        env = {'RANK': '3', 'WORLD_SIZE': '8', 'LOCAL_RANK': '0'}
        self.assertEqual(driver.check_launcher(distributed, cuda, env), 3)
        for key, value in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '3')]:
            with self.assertRaises(ValueError):
                driver.check_launcher(distributed, cuda, {**env, key: value})
        distributed.is_initialized = lambda: False
        with self.assertRaises(ValueError):
            driver.check_launcher(distributed, cuda, env)

    def test_sources_reject_omission_and_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            files = {str(PATH.resolve()): driver.sha(PATH)}
            for name in (*driver.ADAPTERS, driver.CONTRACT, driver.GATE_HELPER):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# artificial source fixture\n')
                files[str(path)] = driver.sha(path)
            manifest = root/'sources.json'
            manifest.write_text(json.dumps({'files': files}))
            args = SimpleNamespace(repo_root=root, source_root=root, source_manifest=manifest,
                                   source_manifest_sha256=driver.sha(manifest))
            self.assertEqual(driver.verify_sources(args), files)
            (root/driver.CONTRACT).write_text('# changed\n')
            with self.assertRaises(ValueError):
                driver.verify_sources(args)
            files.pop(str(PATH.resolve()))
            manifest.write_text(json.dumps({'files': files}))
            args.source_manifest_sha256 = driver.sha(manifest)
            with self.assertRaises(ValueError):
                driver.verify_sources(args)

    def test_factorial_names_and_distinct_roots(self):
        for name in (*driver.VARIANTS, *driver.FACTORIAL):
            args = driver.parse_args(self.arguments('--variant', name))
            self.assertEqual(args.variant, name)
            self.assertNotEqual(args.source_root, args.repo_root)
        self.assertEqual(len(driver.FACTORIAL), 11)

    def test_mixed_training_cache_rejected_before_import(self):
        root = PATH.resolve().parents[2]
        args = SimpleNamespace(source_root=root, repo_root=root)
        with patch.dict(driver.sys.modules, {'dllm._foreign_fixture': SimpleNamespace(__file__='/tmp/foreign/model.py')}):
            with self.assertRaisesRegex(ValueError, 'Mixed training'):
                driver.isolate_training_sources(args)

    def test_helper_subprocess_is_isolated_and_metadata_only(self):
        args = SimpleNamespace(repo_root=Path('/tmp/real'), source_root=Path('/tmp/frozen'),
            data_dir=Path('/tmp/data'), source_manifest=Path('/tmp/sources.json'), source_manifest_sha256='a'*64)
        helper = str(args.repo_root/driver.GATE_HELPER)
        receipt = {'schema': 'v7.formal.read_only_parent_check.v1', 'status': 'passed',
            'repo_root': str(args.repo_root), 'data_dir': str(args.data_dir),
            'metadata': {'dataset_manifest.json': driver.FINAL_MANIFEST},
            'source_manifest_sha256': args.source_manifest_sha256, 'executed_sources': {helper: 'b'*64}}
        with patch.object(driver, 'verify_sources', return_value={helper: 'b'*64}), \
                patch.object(driver, 'sha', return_value='b'*64), \
                patch.object(driver.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(receipt), returncode=0, stderr='')) as run:
            self.assertEqual(driver.check_parent(args), receipt)
            command = run.call_args.args[0]
            self.assertEqual(command[1:3], ['-I', '-B'])
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertNotIn('RANK', run.call_args.kwargs['env'])
            self.assertEqual(run.call_args.kwargs['cwd'], args.repo_root)

    def test_actual_parameter_digest_detects_rank_or_file_difference(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temporary:
            model = torch.nn.Linear(2, 1)
            first = driver.parameter_digest(model)
            save_file(model.state_dict(), str(Path(temporary)/'model.safetensors'))
            self.assertEqual(first, driver.parameter_digest(model, Path(temporary)))
            with torch.no_grad():
                model.weight[0, 0] += 1
            self.assertNotEqual(first, driver.parameter_digest(model))
            self.assertEqual(first, driver.parameter_digest(model, Path(temporary)))

    def test_real_gate_source_closure_rejects_unbound_or_changed_code(self):
        spec = importlib.util.spec_from_file_location('formal_gate_protocol_fixture', PATH.with_name('check_v7_formal_parent.py'))
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            files = {}
            for name in helper.GATE_FILES:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# synthetic gate source fixture\n')
                files[str(path)] = helper.sha(path)
            manifest = root/'sources.json'
            manifest.write_text(json.dumps({'files': files}))
            args = SimpleNamespace(repo_root=root, source_manifest=manifest, source_manifest_sha256=helper.sha(manifest))
            self.assertEqual(helper.check_sources(args), files)
            with patch.dict(helper.sys.modules, {'scripts._unexpected_gate': SimpleNamespace(__file__=str(root/'unbound.py'))}):
                with self.assertRaises(ValueError):
                    helper.check_sources(args, loaded=True)
            (root/helper.GATE_FILES[0]).write_text('# drift\n')
            with self.assertRaises(ValueError):
                helper.check_sources(args)

    def test_collective_failure_propagates(self):
        def gather(target, packet, group):
            target[:] = [packet.copy() for _ in range(8)]
        dist = SimpleNamespace(all_gather_object=gather)
        self.assertEqual(driver.collective(dist, None, lambda: 12), [12]*8)
        with self.assertRaises(ValueError):
            driver.collective(dist, None, lambda: 1/0)


if __name__ == '__main__':
    unittest.main()
