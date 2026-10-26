"""Activate pllm/protenix_abtcr; python -B scripts/tests/test_cdr_marginal_inference.py."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'downstream'))
from cdr_marginal_inference import aggregate, mix_prediction, row_generators, random_rows


def predictor(tokens, corrupt_visible=False):
    # Independent rows; context residues affect probabilities at hidden sites.
    context = tokens.eq(1).sum(-1).float()[:, None]
    position = torch.arange(tokens.shape[1]).float()[None, :]
    p1 = (context*.4-position*.2+.35).sigmoid()
    p = torch.zeros((*tokens.shape, 4)); p[..., 1] = p1; p[..., 2] = 1-p1
    if corrupt_visible:
        p[tokens.ne(0)] = torch.tensor([0., .001, .999, 0.])
    return p


class MarginalTests(unittest.TestCase):
    def setUp(self):
        self.tokens = torch.tensor([[3, 0, 0, 1, 3], [3, 0, 2, 3, 3]])
        self.batch = {'input_ids': self.tokens}
        self.target = self.tokens.eq(0)
        self.prior = predictor(self.tokens)

    def test_visible_scores_do_not_change_aggregates(self):
        configs = [dict(name='mc', kind='mc_context', reveal=.5, temperature=1., draws=[4, 8]),
                   dict(name='loo', kind='leave_one_out'),
                   dict(name='branch', kind='branch_context', anchor_rule='entropy', anchors=3, topk=2)]
        for config in configs:
            a = list(aggregate(predictor, self.batch, self.target, 0, self.prior, config))
            b = list(aggregate(lambda x: predictor(x, True), self.batch, self.target, 0, self.prior, config))
            for x, y in zip(a, b):
                self.assertTrue(torch.allclose(x[1], y[1]))
                pred, dist = mix_prediction(self.tokens, self.target, self.prior, x[1], .5)
                self.assertTrue(torch.equal(pred[~self.target], self.tokens[~self.target]))
                self.assertFalse(pred[self.target].eq(0).any())

    def test_row_random_streams_and_mc_are_batch_invariant(self):
        config = dict(name='mc', kind='mc_context', reveal=.5, temperature=1., draws=[8])
        full = list(aggregate(predictor, self.batch, self.target, 0, self.prior, config))[0][1]
        for i in range(2):
            tok = self.tokens[i:i+1]
            one = list(aggregate(predictor, {'input_ids': tok}, tok.eq(0), 0, self.prior[i:i+1], config))[0][1]
            self.assertTrue(torch.equal(full[i:i+1], one))
        gs = row_generators(self.tokens, 42)
        values = random_rows(gs, (20,), 'cpu')
        self.assertFalse(torch.equal(values[0], values[1]))

    def test_branch_matches_explicit_two_token_expectation(self):
        config = dict(name='branch', kind='branch_context', anchor_rule='confidence', anchors=1, topk=2)
        result = list(aggregate(predictor, self.batch, self.target, 0, self.prior, config))[0]
        posterior, counts, nfe = result[1:4]
        expected = self.prior.clone()
        for i in range(2):
            pos = self.target[i].nonzero().flatten().tolist()
            anchor = min(pos, key=lambda j: (-float(self.prior[i, j].max()), j))
            others = [j for j in pos if j != anchor]
            for j in others:
                expected[i, j] = 0
                for token in (1, 2):
                    x = self.tokens.clone(); x[i, anchor] = token
                    expected[i, j] += self.prior[i, anchor, token]*predictor(x)[i, j]
            self.assertEqual(counts[i, anchor], 0)
        self.assertTrue(torch.allclose(posterior, expected))
        self.assertEqual(nfe, 3)

    def test_loo_queries_hide_every_target_and_prior_anchor_recovers_base(self):
        seen = []
        def observe(tag, inputs, scored, probs):
            self.assertTrue(inputs[scored].eq(0).all())
            seen.append(scored)
        config = dict(name='loo', kind='leave_one_out')
        result = list(aggregate(predictor, self.batch, self.target, 0, self.prior, config, observe))[0]
        self.assertTrue(torch.stack(seen).sum(0)[self.target].eq(1).all())
        self.assertEqual(result[3], 3)
        pred, _ = mix_prediction(self.tokens, self.target, self.prior, result[1], 1.)
        self.assertTrue(torch.equal(pred[self.target], self.prior.argmax(-1)[self.target]))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
