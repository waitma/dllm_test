"""Check new segment-boundary accounting with in-memory artificial metadata only.

Activate conda pllm, then run:
  CUDA_VISIBLE_DEVICES='' python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_observe_v7_logged_steps.py

No training data, model, GPU, subprocess, checkpoint weights or fixtures are used.
"""
import runpy
import unittest


MODULE = '/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/observe_v7_logged_steps.py'
reader = runpy.run_path(MODULE, run_name='_progress_reader_control')
infer = reader['infer_logged_steps']


def row(step):
    return dict(loss=1 / (step + 1), epoch=step / 25042, learning_rate=reader['learning_rate'](step))


def log(steps):
    return ''.join(repr(row(step)) + '\n' for step in steps).encode()


def native(saved, steps):
    return dict(global_step=saved, max_steps=25042, logging_steps=25,
                log_history=[dict(row(step), step=step) for step in steps])


class SegmentProgressTests(unittest.TestCase):
    def test_regular_segment_and_incomplete_tail(self):
        value = infer(log([25, 50]), native(25, [25]), 16, 5016)
        self.assertEqual(value['latest_logged_training_step'], 50)
        self.assertEqual(value['native_anchor_steps_matched'], [25])
        value = infer(log([25]) + repr(row(50)).encode(), native(25, [25]), 16, 5016)
        self.assertEqual(value['latest_logged_training_step'], 25)

    def test_saved_5016_is_not_logged_5016(self):
        value = infer(log(range(25, 5001, 25)), native(5016, [4975, 5000]), 16, 5016)
        self.assertEqual(value['latest_logged_training_step'], 5000)
        self.assertEqual(value['latest_completed_checkpoint_step'], 5016)

    def test_resumed_segment_uses_next_global_logging_multiple(self):
        value = infer(log([5025, 5050]), native(5016, [5000]), 5016, 10016)
        self.assertEqual(value['latest_logged_training_step'], 5050)
        self.assertEqual(value['native_anchor_steps_matched'], [])

    def test_empty_new_segment_reports_only_saved_training_history(self):
        value = infer(b'', native(5016, [5000]), 5016, 10016)
        self.assertEqual(value['latest_logged_training_step'], 5000)
        self.assertEqual(value['latest_log_source'], 'saved_native_history')

    def test_missing_duplicate_and_inconsistent_native_logs_are_rejected(self):
        for steps in ([5025, 5075], [5025, 5025]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                infer(log(steps), native(5016, [5000]), 5016, 10016)
        state = native(5025, [5000, 5025])
        state['log_history'][-1]['loss'] += 1
        with self.assertRaises(ValueError):
            infer(log([5025]), state, 5016, 10016)
        with self.assertRaises(ValueError):
            infer(b'', native(5025, [5000, 5025]), 5016, 10016)


if __name__ == '__main__':
    unittest.main(verbosity=2)
