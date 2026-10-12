"""Run in pllm: pytest -q scripts/tests/immune_llada/test_attention_backends_cpu.py.

Pure random CPU tensors; no project model, tokenizer, dataset or GPU access.
"""
import pytest
import torch

from scripts.architecture_prototypes.attention_backends import allowed_mask, compare


@pytest.mark.parametrize("method", ["chunked", "checkpoint_chunked", "flex_eager"])
@pytest.mark.parametrize("kind", ["dense", "table"])
@pytest.mark.parametrize("mode", ["global", "chain", "local"])
@pytest.mark.parametrize("padding", [False, True])
def test_outputs_and_all_gradients(method, kind, mode, padding):
    torch.set_num_threads(2)
    evidence = compare(method, kind, mode, padding)
    assert evidence["passed"]
    if kind == "dense" or mode == "global":
        assert evidence["bias_gradient_l1"] > 0


def test_padding_queries_keep_fallback_keys_and_valid_queries_cannot_use_padding():
    groups = torch.tensor([[-1, 0, 0, 1, 1]])
    valid = torch.tensor([[True, True, True, False, False]])
    allowed = allowed_mask(groups, valid, "chain")
    assert not allowed[:, :, 3:].any()
    assert torch.equal(allowed[0, 3:], valid.expand(2, -1))
    assert not allowed[0, 0, 1]  # prefix cannot relay to another group
    assert allowed[0, 1, 2]


def test_empty_records_rejected_before_softmax():
    with pytest.raises(ValueError, match="allowed key"):
        allowed_mask(torch.tensor([[0, 0]]), torch.tensor([[False, False]]), "global")
