"""Activate pllm; CUDA_VISIBLE_DEVICES='' python -B -m pytest THIS_FILE -q.

Check candidate Pair casts against the original outputs and gradients on CPU.
Full-size BF16 CUDA parity is recorded by the separate diagnostic entrypoint.
"""
from copy import deepcopy

import pytest
import torch

from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import ResiduePairEncoder, PairConfig
from scripts.research.stage1_pair_speed_candidates import autocast_pairs, symmetric_attention_bias


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("dense", [False, True])
def test_precat_cast_preserves_pair_output_and_gradients(bf16, dense):
    torch.set_num_threads(1)
    torch.manual_seed(174)
    first = ResiduePairEncoder(32, PairConfig(width=8))
    second = deepcopy(first)
    shape_left, shape_right = ((2, 3, 1, 8), (2, 1, 7, 8)) if dense else ((13, 8), (13, 8))
    dtype = torch.bfloat16 if bf16 else torch.float32
    left = torch.randn(shape_left).to(dtype).requires_grad_()
    right = torch.randn(shape_right).to(dtype).requires_grad_()
    other_left, other_right = left.detach().clone().requires_grad_(), right.detach().clone().requires_grad_()
    pl, pr = torch.randint(0, 80, shape_left[:-1]), torch.randint(0, 80, shape_right[:-1])
    cl, cr = torch.randint(-1, 3, shape_left[:-1]), torch.randint(-1, 3, shape_right[:-1])
    rl, rr = torch.randint(0, first.num_roles, shape_left[:-1]), torch.randint(0, first.num_roles, shape_right[:-1])
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        a = first.pairs(left, right, pl, pr, cl, cr, rl, rr)
        b = autocast_pairs(second, other_left, other_right, pl, pr, cl, cr, rl, rr)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    upstream = torch.randn_like(a)
    (a*upstream).sum().backward()
    (b*upstream).sum().backward()
    torch.testing.assert_close(left.grad, other_left.grad, atol=0, rtol=0)
    torch.testing.assert_close(right.grad, other_right.grad, atol=0, rtol=0)
    assert list(first.state_dict()) == list(second.state_dict())
    for (name, p), (_, q) in zip(first.named_parameters(), second.named_parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("length", [9, 97])
def test_upper_triangle_reconstruction_and_backward(checkpointed, length):
    torch.set_num_threads(1)
    torch.manual_seed(175)
    first = ResiduePairEncoder(16, PairConfig(width=8, query_chunk=16,
        checkpoint_pair_chunks=checkpointed)).double().train()
    second = deepcopy(first)
    out1 = torch.nn.Linear(8, 4, bias=False).double()
    out2 = deepcopy(out1)
    h1 = torch.randn(2, length, 16, dtype=torch.float64, requires_grad=True)
    h2 = h1.detach().clone().requires_grad_()
    positions = torch.randint(0, 80, (2, length))
    chains = torch.randint(-1, 3, (2, length))
    roles = torch.randint(0, first.num_roles, (2, length))
    valid = torch.rand(2, length) > .2
    a = first.attention_bias(h1, positions, chains, roles, valid, out1)
    b = symmetric_attention_bias(second, h2, positions, chains, roles, valid, out2)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(b, b.transpose(-1, -2), atol=0, rtol=0)
    upstream = torch.randn_like(a)
    (a*upstream).sum().backward()
    (b*upstream).sum().backward()
    torch.testing.assert_close(h1.grad, h2.grad, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(out1.weight.grad, out2.weight.grad, atol=1e-10, rtol=1e-10)
    for (name, p), (_, q) in zip(first.named_parameters(), second.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-10, msg=name)
