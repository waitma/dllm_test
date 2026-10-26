"""Check best3+latest ownership and failure handling using temporary fake states.

After conda activate pllm:
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_native_best3_latest.py
No real checkpoint, model, corpus, GPU or process is touched.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from native_best3_latest import binding_digest, plan, retain, sha


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='v7_best3_latest_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.owners = {}
        for name in ('old', 'new'):
            path = self.root/name
            path.mkdir()
            provenance = {'identity': name}
            (path/'run_provenance.json').write_text(json.dumps(provenance, indent=2))
            self.owners[str(path)] = binding_digest(provenance)

    def checkpoint(self, root, step, loss):
        path = self.root/root/f'checkpoint-{step}'
        path.mkdir()
        provenance = json.loads((path.parent/'run_provenance.json').read_text())
        # Native root and checkpoint serialization can differ.
        (path/'run_provenance.json').write_text(json.dumps(provenance, separators=(',', ':')))
        (path/'trainer_state.json').write_text(json.dumps(dict(global_step=step,
            log_history=[dict(step=step, eval_loss=loss)])))
        for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state_0.pth'):
            (path/name).write_bytes(b'fake-owned-state')
        files = {p.name: sha(p) for p in path.iterdir()}
        (path/'formal_checkpoint.json').write_text(json.dumps(dict(step=step, world_size=8,
            writer_rank=0, rank_acknowledgments=list(range(8)), files=files)))
        return path

    def test_best_three_plus_worse_latest_across_explicit_lineage(self):
        for step, value in [(1, .1), (2, .2), (3, .3)]:
            self.checkpoint('old', step, value)
        discarded = self.checkpoint('new', 4, .9)
        latest = self.checkpoint('new', 5, .8)
        decision = retain(self.owners, latest)
        self.assertEqual([x['step'] for x in decision['keep']], [1, 2, 3, 5])
        self.assertFalse(discarded.exists())
        self.assertTrue((latest/'optimizer.pt').exists())
        self.assertTrue((self.root/'old/checkpoint-1/optimizer.pt').exists())
        self.assertEqual(len(list((latest.parent/'checkpoint_retention_evidence').glob('checkpoint-4_*/formal_checkpoint.json'))), 1)

    def test_latest_overlaps_best_and_ties_prefer_earlier_step(self):
        for step, value in [(1, .5), (2, .5), (3, .5), (4, .1)]:
            latest = self.checkpoint('new', step, value)
        result = retain(self.owners, latest)
        self.assertEqual([Path(p).name for p in result['best']], ['checkpoint-4', 'checkpoint-1', 'checkpoint-2'])
        self.assertEqual(len(result['keep']), 3)

    def test_restart_reconstructs_from_native_files(self):
        for step, value in [(1, .1), (2, .2), (3, .3), (4, .4)]:
            latest = self.checkpoint('new', step, value)
        retain(self.owners, latest)
        newer = self.checkpoint('new', 5, .01)
        result = retain(self.owners, newer)
        self.assertEqual([x['step'] for x in result['keep']], [1, 2, 5])

    def test_nan_prevents_all_pruning(self):
        for step in range(1, 6):
            latest = self.checkpoint('new', step, float('nan') if step == 2 else step)
        with self.assertRaisesRegex(ValueError, 'Finite exact-step'):
            retain(self.owners, latest)
        self.assertEqual(len(list(latest.parent.glob('checkpoint-*'))), 5)

    def test_checkpoint_symlink_rejected(self):
        latest = self.checkpoint('new', 2, .1)
        (latest.parent/'checkpoint-1').symlink_to(latest, target_is_directory=True)
        with self.assertRaises(ValueError):
            retain(self.owners, latest)
        self.assertTrue(latest.exists())

    def test_metadata_tamper_rejected_before_delete(self):
        earlier = self.checkpoint('new', 1, 9)
        latest = self.checkpoint('new', 2, .1)
        (earlier/'trainer_state.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            retain(self.owners, latest)
        self.assertTrue(earlier.exists())

    def test_nonlatest_trigger_and_duplicate_steps_rejected(self):
        earlier = self.checkpoint('new', 1, .2)
        latest = self.checkpoint('new', 2, .1)
        with self.assertRaisesRegex(ValueError, 'newest'):
            plan(self.owners, earlier)
        self.checkpoint('old', 2, .1)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            plan(self.owners, latest)

    def test_fsdp_full_inventory_via_actual_entry(self):
        import importlib.util
        entry = Path('/project_c20250601/251105016/project/dllm_test/scripts/research/run_v7_fsdp_best3.py')
        spec = importlib.util.spec_from_file_location('_test_fsdp_best3_entry', entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for step, loss in [(1, .1), (2, .2), (3, .3), (4, .9), (5, .8)]:
            latest = self.checkpoint('new', step, loss)
            old = latest/'formal_checkpoint.json'
            sidecar = json.loads(old.read_text())
            sidecar['schema'] = 'v7.fsdp.checkpoint.v1'
            sidecar['files'] = {name: {'bytes': (latest/name).stat().st_size, 'sha256': value}
                                for name, value in sidecar['files'].items()}
            (latest/'fsdp_checkpoint.json').write_text(json.dumps(sidecar))
            old.unlink()
        binding = json.loads((latest.parent/'run_provenance.json').read_text())
        result = module.retain_checkpoint(latest, binding)
        self.assertEqual([x['step'] for x in result['keep']], [1, 2, 3, 5])
        self.assertFalse((latest.parent/'checkpoint-4').exists())
        self.assertTrue((latest/'optimizer.pt').exists())

    def test_fsdp_wrong_metadata_size_rejected_before_pruning(self):
        latest = self.checkpoint('new', 1, .2)
        p = latest/'formal_checkpoint.json'
        sidecar = json.loads(p.read_text())
        value = sidecar['files']['trainer_state.json']
        sidecar['files']['trainer_state.json'] = {'bytes': 0, 'sha256': value}
        p.write_text(json.dumps(sidecar))
        with self.assertRaisesRegex(ValueError, 'inventory size'):
            retain(self.owners, latest)
        self.assertTrue(latest.exists())

    def test_unknown_owner_rejected(self):
        latest = self.checkpoint('new', 1, .1)
        invalid = dict(self.owners)
        invalid[str(latest.parent)] = '0'*64
        with self.assertRaisesRegex(ValueError, 'owner'):
            retain(invalid, latest)
        self.assertTrue(latest.exists())


if __name__ == '__main__':
    unittest.main()
