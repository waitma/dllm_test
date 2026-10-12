"""Run metadata-only retention checks after activating conda pllm.

Run this absolute file with python -B. Uses artificial temp files only; no
models, real datasets, CUDA, platform operations, or persistent checkpoints.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.cleanup_test_artifacts import clean_once
from scripts.research.result_retention import record_observation


class RetentionChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='v7_retention_control_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pool = self.root / 'pool'
        for name in ('runs', 'running', 'results', 'monitor'):
            (self.pool / name).mkdir(parents=True, exist_ok=True)

    def run_dir(self, name='123_p04_v7_f001_seed42_1234abcd'):
        path = self.pool / 'runs' / name
        path.mkdir()
        return path

    def test_default_records_every_point_without_serializing(self):
        directory = self.root / 'metrics'
        directory.mkdir()
        def forbidden_writer():
            self.fail('Metrics-only recording invoked a model writer')
        for step in (0, 250):
            record_observation(directory, {'step': step, 'loss': 1.0}, checkpoint_writer=forbidden_writer)
        points = [json.loads(line) for line in (directory / 'curve.jsonl').read_text().splitlines()]
        self.assertEqual([p['step'] for p in points], [0, 250])
        self.assertEqual(json.loads((directory / 'summary.json').read_text()), points[-1])
        self.assertEqual({p.name for p in directory.iterdir()}, {'curve.jsonl', 'summary.json'})

    def test_explicit_writer_only_runs_when_enabled(self):
        directory = self.root / 'explicit'
        directory.mkdir()
        calls = []
        record_observation(directory, {'step': 1}, save_checkpoint=True,
                           checkpoint_writer=lambda: calls.append('saved'))
        self.assertEqual(calls, ['saved'])

    def test_active_model_removed_but_writer_temp_and_metrics_survive(self):
        run = self.run_dir()
        (self.pool / 'running' / (run.name + '.json')).write_text('{}')
        (run / 'latest.pt').write_bytes(b'artificial-model')
        (run / 'latest.pt.tmp').write_bytes(b'active-writer')
        (run / 'summary.json').write_text('{"step":250}')
        before = (run / 'summary.json').read_bytes()
        events = clean_once(self.pool)
        self.assertEqual(len(events), 1)
        self.assertFalse((run / 'latest.pt').exists())
        self.assertTrue((run / 'latest.pt.tmp').exists())
        self.assertEqual((run / 'summary.json').read_bytes(), before)

    def test_terminal_failed_temp_removed(self):
        run = self.run_dir()
        (self.pool / 'results' / (run.name + '.json')).write_text('{"state":"failed"}')
        (run / 'latest.pt.tmp').write_bytes(b'incomplete-write')
        self.assertEqual(len(clean_once(self.pool)), 1)
        self.assertFalse((run / 'latest.pt.tmp').exists())

    def test_other_runs_and_symlinks_are_preserved(self):
        other = self.run_dir('123_formal_training_1234abcd')
        (other / 'model.safetensors').write_bytes(b'outside-test-scope')
        run = self.run_dir()
        outside = self.root / 'outside.pt'
        outside.write_bytes(b'protected-target')
        (run / 'latest.pt').symlink_to(outside)
        self.assertEqual(clean_once(self.pool), [])
        self.assertTrue(outside.exists())
        self.assertTrue((other / 'model.safetensors').exists())

    def test_stale_monitor_temps_removed_recent_writes_survive(self):
        stale = self.pool / 'monitor' / 'state.json.old.tmp'
        fresh = self.pool / 'monitor' / 'state.json.new.tmp'
        stale.write_text('incomplete-old-state')
        fresh.write_text('current-write')
        os.utime(stale, (time.time() - 1800, time.time() - 1800))
        self.assertEqual(len(clean_once(self.pool)), 1)
        self.assertFalse(stale.exists())
        self.assertTrue(fresh.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
