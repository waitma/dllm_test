"""Check focal math, EOS isolation and accumulation on masked vocabularies.

Activate pllm; python -B -m pytest
scripts/tests/immune_llada/test_diffusion_continuation_losses.py -q.
"""
import torch
from torch.nn import functional as F

from scripts.research.diffusion_continuation_losses import weighted_token_loss


def test_zero_gamma_is_weighted_ce_with_ignored_and_forbidden_tokens():
    logits = torch.tensor([[[1., 2., float("-inf")], [3., -1., float("-inf")],
                            [0., 0., float("-inf")]]], requires_grad=True)
    labels = torch.tensor([[1, 0, -100]])
    weights = torch.tensor([[.5, 3., 0.]])
    result = weighted_token_loss(logits, labels, weights, eos_id=1)
    expected = (F.cross_entropy(logits[0, :2], labels[0, :2], reduction="none") * weights[0, :2]).sum()
    torch.testing.assert_close(result["loss_sum"], expected)
    result["loss_sum"].backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2].count_nonzero() == 0


def test_eos_focal_matches_formula_and_leaves_other_tokens_and_raw_ce_intact():
    logits = torch.tensor([[[0., 4.], [4., 0.], [2., 1.]]], requires_grad=True)
    labels = torch.tensor([[1, 1, 0]])
    weights = torch.ones_like(labels, dtype=torch.float)
    result = weighted_token_loss(logits, labels, weights, eos_id=1, eos_gamma=2)
    ce = F.cross_entropy(logits[0], labels[0], reduction="none")
    expected = (ce[:2] * (1-ce[:2].neg().exp()).square()).sum() + ce[2]
    torch.testing.assert_close(result["loss_sum"], expected)
    torch.testing.assert_close(result["ce_sum"], ce.sum())
    torch.testing.assert_close(result["eos_ce_sum"], ce[:2].sum())
    gradient = torch.autograd.grad(result["loss_sum"], logits, retain_graph=True)[0]
    plain = torch.autograd.grad(ce.sum(), logits)[0]
    torch.testing.assert_close(gradient[0, 2], plain[0, 2])
    assert gradient[0, 0].abs().sum() < plain[0, 0].abs().sum()


def test_chunk_sums_and_gradients_match_full_batch_with_empty_chunk():
    logits = torch.randn(3, 4, 5, generator=torch.Generator().manual_seed(4), requires_grad=True)
    labels = torch.tensor([[1, 4, 2, -100], [-100, -100, -100, -100], [4, 0, 1, 3]])
    weights = torch.ones_like(labels, dtype=torch.float) * .25
    full = weighted_token_loss(logits, labels, weights, eos_id=4, eos_gamma=1)["loss_sum"]
    chunks = sum(weighted_token_loss(logits[i:i+1], labels[i:i+1], weights[i:i+1],
                                    eos_id=4, eos_gamma=1)["loss_sum"] for i in range(3))
    torch.testing.assert_close(chunks, full)
    a = torch.autograd.grad(full, logits, retain_graph=True)[0]
    b = torch.autograd.grad(chunks, logits)[0]
    torch.testing.assert_close(a, b)
