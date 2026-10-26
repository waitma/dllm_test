"""Activate pllm/protenix_abtcr; python -B scripts/tests/test_cdr_inference_sampling.py."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT/'scripts/downstream'))
from cdr_inference_sampling import adaptive_positions, sample, transition_rates


class SamplingTests(unittest.TestCase):
    def test_transition_constraints_and_marginal(self):
        for steps in (1, 2, 4, 8, 64):
            for eta in (0., .1, .3, 1.):
                for step in range(steps):
                    sigma, unmask = transition_rates(step, steps, eta)
                    self.assertTrue(0 <= sigma <= 1 and 0 <= unmask <= 1)
                    alpha = step/steps
                    self.assertAlmostEqual(alpha*(1-sigma)+(1-alpha)*unmask, (step+1)/steps)
                    if step == steps-1:
                        self.assertEqual(sigma, 0.)
                        self.assertEqual(unmask, 1.)

    def test_entropy_uses_sum_minus_max_and_progress(self):
        probs = torch.tensor([[[.5, .5], [.5, .5], [.5, .5]]])
        pending = torch.ones((1, 3), dtype=torch.bool)
        self.assertEqual(adaptive_positions(pending, probs, 'entropy', 0.).sum(), 1)
        self.assertEqual(adaptive_positions(pending, probs, 'entropy', .7).sum(), 2)
        self.assertEqual(adaptive_positions(pending, probs, 'entropy', 2.).sum(), 3)
        self.assertEqual(adaptive_positions(pending, probs, 'threshold', .99).sum(), 1)
        pending[0, 0] = False
        got = adaptive_positions(pending, probs, 'threshold', .99)
        self.assertFalse(got[0, 0])
        self.assertEqual(got.sum(), 1)

    def test_real_airgen_methods_on_context_dependent_toy(self):
        evaluator_path = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/airgen_evaluate.py'
        spec = importlib.util.spec_from_file_location('airgen_test', evaluator_path)
        evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
        bridge = type('Bridge', (), evaluator.methods())()
        bridge.mask_id, bridge.bos_id, bridge.pad_id, bridge.eos_id = 0, 3, 4, 5
        bridge.special_token_list = [0, 3, 4]
        bridge.forward_encoder = lambda batch: None
        calls = []
        def net(input_ids, chain_ids):
            calls.append(input_ids.clone())
            logits = torch.zeros((*input_ids.shape, 6))
            logits[..., 1] = torch.arange(input_ids.shape[1]).float()+1
            logits[..., 2] = input_ids.eq(1).sum(-1)[:, None] * 3.
            logits[..., 5] = -20.
            return dict(logits=logits, last_hidden_state=None)
        bridge.net = net
        batch = dict(input_ids=torch.tensor([[3, 0, 0, 0, 0, 5], [3, 0, 0, 5, 4, 4]]),
                     chain_ids=torch.zeros((2, 6), dtype=torch.long))
        target = batch['input_ids'].eq(0)
        for steps in (1, 2, 4, 8):
            baseline, trace = sample(bridge, batch, target, dict(kind='scheduled', steps=steps))
            expected, _ = bridge.generate(batch, max_iter=steps, temperature=1.,
                partial_masks=~target & batch['input_ids'].ne(bridge.bos_id),
                sampling_strategy='argmax', cfg_scale=0.)
            self.assertTrue(torch.equal(baseline, expected))
            for prev, cur in zip(trace, trace[1:]):
                self.assertTrue(torch.equal(prev['after'], cur['before']))
            if steps == 4:
                gold = torch.full_like(baseline, 2)
                oracle, replayed = sample(bridge, batch, target, dict(kind='oracle', steps=steps),
                                           gold=gold, replay=trace)
                for x, y in zip(trace, replayed): self.assertTrue(torch.equal(x['commit'], y['commit']))
        for config in [dict(kind='repair', fraction=.5, steps=4),
                       dict(kind='random_repair', fraction=.5, steps=4),
                       dict(kind='remdm', eta=.3, steps=8),
                       dict(kind='threshold', value=.8, steps=8),
                       dict(kind='entropy', value=.5, steps=8)]:
            a, tr = sample(bridge, batch, target, config)
            b, _ = sample(bridge, batch, target, config)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(a[~target], batch['input_ids'][~target]))
            self.assertFalse(a[target].eq(0).any())
            for prev, cur in zip(tr, tr[1:]): self.assertTrue(torch.equal(prev['after'], cur['before']))
        with self.assertRaises(ValueError):
            sample(bridge, batch, target, dict(kind='repair', fraction=.5, steps=4), gold=gold)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
