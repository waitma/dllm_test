"""CPU replay guards: python -B -m unittest test_arch70m_local_recheck."""
import unittest
from arch70m_local_recheck import ROOT, ORIGINAL, JOBS, comparison, safe_output


class RecheckTests(unittest.TestCase):
    def test_output_cannot_overwrite_original_or_parent(self):
        for path in (ORIGINAL, ORIGINAL / 'replay', ORIGINAL.parent,
                     ROOT / 'output/arch70m_v8_20261028/replay'):
            with self.assertRaises(ValueError): safe_output(path, ORIGINAL)
        self.assertEqual(safe_output(ROOT / 'output/recheck_test', ORIGINAL), ROOT / 'output/recheck_test')

    def test_small_float_drift_is_recorded_without_widening_thresholds(self):
        result = comparison({'score': .5}, {'score': .5000001})
        self.assertFalse(result['exact_equal']); self.assertTrue(result['strict_close'])
        self.assertGreater(result['max_absolute_difference'], 0)
        self.assertFalse(comparison({'score': .5}, {'score': .50001})['strict_close'])

    def test_identity_order_and_discrete_counts_are_not_approximate(self):
        a = [{'key': 'r1', 'count': 10000000}, {'key': 'r2', 'count': 10000000}]
        self.assertFalse(comparison(a, a[::-1])['strict_close'])
        self.assertFalse(comparison(a, [{'key': 'r1', 'count': 10000001}, a[1]])['strict_close'])
        self.assertFalse(comparison({'x': None}, {'x': 0})['strict_close'])
        self.assertFalse(comparison({'label': True}, {'label': 1})['strict_close'])

    def test_missing_observations_and_nonfinite_values_reject(self):
        self.assertFalse(comparison({'a': 1}, {})['strict_close'])
        self.assertFalse(comparison([1, 2], [1])['strict_close'])
        self.assertFalse(comparison(float('nan'), float('nan'))['strict_close'])
        self.assertFalse(comparison(float('inf'), float('inf'))['strict_close'])

    def test_replay_covers_all_nine_final_checkpoints(self):
        self.assertEqual(len(set(JOBS)), 9)
        self.assertEqual(set(JOBS), {(v, s) for v in ('A0', 'B0', 'B1') for s in (42, 43, 44)})
        self.assertEqual([s for _, s in JOBS[:3]], [44, 44, 44])


if __name__ == '__main__': unittest.main()
