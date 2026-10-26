"""CPU checks for interrupted checkpoints and stale monitoring.

Activate pllm; python -B /ABS/scripts/tests/immune_llada/test_diffusion_spot_control.py.
"""
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'research'))
from run_diffusion_spot import latest_checkpoint
import run_diffusion_spot
from monitor_diffusion_spot import observe


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class SpotControlTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def checkpoint(self, step, complete=True):
        path = self.root / 'train' / f'checkpoint-{step}'
        path.mkdir(parents=True)
        if complete:
            save(path / 'checkpoint_receipt.json', dict(status='complete', step=step))
        return path

    def test_skip_interrupted_save_and_resume_newest(self):
        first = self.checkpoint(28000)
        newest = self.checkpoint(28500)
        self.checkpoint(29000, complete=False)
        self.assertEqual(latest_checkpoint(self.root / 'train', first, 28000), newest)

    def test_refuse_complete_receipt_with_wrong_step(self):
        path = self.checkpoint(28000)
        save(path / 'checkpoint_receipt.json', dict(status='complete', step=27500))
        with self.assertRaisesRegex(ValueError, 'step mismatch'):
            latest_checkpoint(self.root / 'train', path, 28000)

    def test_never_resume_below_authorized_minimum(self):
        path = self.checkpoint(27500)
        with self.assertRaisesRegex(ValueError, 'No complete'):
            latest_checkpoint(self.root / 'train', path, 28000)

    def monitoring_fixture(self, age=7200):
        control = self.root / 'control'
        output = self.root / 'train'
        save(control / 'spec.json', dict(output_dir=str(output)))
        save(control / 'submission.json', dict(task_id='t-test'))
        save(output / 'status.json', dict(status='training', step=28225, target_steps=200000,
             time_unix=time.time()-age, loss=.5))
        save(output / 'latest.json', dict(step=28000))
        return control

    def test_stale_training_field_is_not_running(self):
        control = self.monitoring_fixture()
        with patch('monitor_diffusion_spot.platform', return_value=dict(Status='Running')):
            result = observe(control)
        self.assertEqual(result['training_status'], 'stalled_or_starting')
        self.assertTrue(result['alerts'])

    def test_fresh_training_reports_running(self):
        control = self.monitoring_fixture(age=2)
        with patch('monitor_diffusion_spot.platform', return_value=dict(Status='Running')):
            self.assertEqual(observe(control)['training_status'], 'training')

    def test_query_error_does_not_claim_training(self):
        control = self.monitoring_fixture(age=2)
        with patch('monitor_diffusion_spot.platform', side_effect=RuntimeError('credentials missing')):
            self.assertEqual(observe(control)['training_status'], 'unknown')

    def test_platform_success_requires_actual_target_completion(self):
        control = self.monitoring_fixture()
        with patch('monitor_diffusion_spot.platform', return_value=dict(Status='Success')):
            self.assertEqual(observe(control)['training_status'], 'stopped_or_failed')

    def test_queue_does_not_report_old_training_as_live(self):
        control = self.monitoring_fixture(age=2)
        with patch('monitor_diffusion_spot.platform', return_value=dict(Status='Queue')):
            self.assertEqual(observe(control)['training_status'], 'queue')

    def test_child_failure_is_visible_in_console_and_attempt_log(self):
        control = self.root / 'control'
        output = self.root / 'train'
        original = self.root / 'original'
        attempt = self.root / 'attempt'
        attempt.mkdir()
        save(control / 'spec.json', dict(original_bundle=str(original)))
        save(original / 'plan.json', dict(output_dir=str(output)))
        save(output / 'status.json', dict(status='stopped'))
        result = dict(status='verified', bundle=str(attempt), starting_step=28000,
                      target_steps=200000)
        real_popen = subprocess.Popen

        def child(arguments, **kwargs):
            return real_popen([sys.executable, '-c',
                "import sys; print('data gate failed', file=sys.stderr); sys.exit(1)"], **kwargs)

        console = io.StringIO()
        with patch.object(sys, 'argv', ['runner', '--control', str(control)]), \
                patch.object(run_diffusion_spot, 'prepare', return_value=(attempt, output, result)), \
                patch.object(run_diffusion_spot.subprocess, 'Popen', side_effect=child), \
                patch.object(run_diffusion_spot.signal, 'signal'), redirect_stdout(console):
            with self.assertRaises(SystemExit) as error:
                run_diffusion_spot.main()
        self.assertEqual(error.exception.code, 1)
        self.assertIn('data gate failed', console.getvalue())
        self.assertIn('data gate failed', (attempt / 'torchrun.log').read_text())
        self.assertEqual(json.loads((control / 'runtime.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
