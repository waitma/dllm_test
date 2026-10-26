"""Compare the readable model with the original research composition on CPU.

Activate pllm, then run:
    CUDA_VISIBLE_DEVICES='' USE_TF=0 python -B -m pytest scripts/tests/immune_llada/test_fixed_chain_readability.py -q
No training jobs or persistent checkpoints are created.
"""

from contextlib import nullcontext

import pytest
import torch

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain import (
    FixedChainCollator, FixedChainConfig, FixedChainModel,
)
from dllm.pipelines.immune_llada.fixed_chain.generation import generate
from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy


class OriginalFixedChainModel(FactorialModel):
    """Original composition; forward/loss/blocks use untouched research modules."""

    def __init__(self, config):
        super().__init__(config, "f010")
        self.variant = "fixed_chain_full_global"
        self.options["segment_global"] = True
        self.terminal_id = self.tokenizer.eos_token_id
        self.allowed_target.zero_()
        self.allowed_target[self.tokenizer.encode_residues("LAGVSERTIDPKQNFYMHWCXBUZO")] = True
        self.allowed_target[self.terminal_id] = True


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return FixedChainConfig(
        d_model=32, n_heads=4, n_layers=4, mlp_hidden_size=64,
        max_length=256, first_chain_capacity=40, second_chain_capacity=36,
        full_capacity=78,
    )


def records():
    return [
        BioSeqRecord([
            BioSeqChain("ACDEFG", "antigen"),
            BioSeqChain("CASSF", "antibody_heavy"),
            BioSeqChain("CAVF", "antibody_light"),
        ], "antibody_antigen", "fixture"),
        BioSeqRecord([
            BioSeqChain("ACD", "peptide"),
            BioSeqChain("XXCASSF", "tcr_beta", metadata={
                "synthetic_residue_mask": [1, 1, 0, 0, 0, 0, 0],
            }),
        ], "tcr_epitope", "fixture"),
    ]


def models():
    torch.manual_seed(91)
    old = OriginalFixedChainModel(config())
    torch.manual_seed(91)
    new = FixedChainModel(config())
    assert list(old.state_dict()) == list(new.state_dict())
    assert [n for n, _ in old.named_parameters()] == [n for n, _ in new.named_parameters()]
    for name, expected in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], expected, rtol=0, atol=0)
    new.load_state_dict(old.state_dict(), strict=True)
    return old, new


def precision(dtype):
    return nullcontext() if dtype == torch.float32 else torch.autocast("cpu", dtype=dtype)


def assert_gradients_equal(old, new):
    for (name, expected), (other_name, actual) in zip(old.named_parameters(), new.named_parameters()):
        assert name == other_name
        assert (expected.grad is None) == (actual.grad is None), name
        if actual.grad is not None:
            assert torch.isfinite(actual.grad).all(), name
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=2e-6, msg=name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_original_weights_logits_loss_gradients_and_checkpointing(dtype, padded, checkpointed):
    old, new = models()
    batch = FixedChainCollator(config())(records() if padded else records()[:1])
    assert bool((~batch["attention_mask"]).any()) == padded
    if checkpointed:
        new.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    results = []
    for model in (old, new):
        torch.manual_seed(120)
        with precision(dtype):
            result = model(**batch)
            # Two live graphs expose accidental bias-cache reuse across forwards.
            second = model(**batch, token_t=torch.ones_like(batch["input_ids"], dtype=torch.float))
            loss = result["loss"] + 0.37 * second["loss"]
        loss.backward()
        results.append((result, second))
    for old_result, new_result in zip(*results):
        assert old_result.keys() == new_result.keys()
        for key in old_result:
            torch.testing.assert_close(new_result[key], old_result[key], rtol=0, atol=0)
    assert_gradients_equal(old, new)
    for model in (old, new):
        model.eval()
    with precision(dtype), torch.no_grad():
        before = old.denoise(**old.feature_args(batch), output_hidden_states=True)
        after = new.denoise(**new.feature_args(batch), output_hidden_states=True)
        for expected, actual in zip(before, after):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(new(**batch)["loss"], old(**batch)["loss"], rtol=0, atol=0)


def test_corruption_masks_and_representation_match_original():
    old, new = models()
    batch = FixedChainCollator(config())(records())
    outputs = []
    for model in (old, new):
        torch.manual_seed(13)
        outputs.append(model.corrupt(batch))
    for before, after in zip(*outputs):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    batch = FixedChainCollator(config(), mode="representation")(records())
    before, after = old.represent(batch), new.represent(batch)
    for key in ("residue_embeddings", "residue_mask", "pooled"):
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    for expected, actual in zip(before["chain_embeddings"], after["chain_embeddings"]):
        assert expected.keys() == actual.keys()
        for uid in expected:
            torch.testing.assert_close(actual[uid], expected[uid], rtol=0, atol=0)


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_generation_matches_original_for_both_completion_directions(temperature):
    old, new = models()
    for target in ("antibody_heavy", "antibody_light"):
        batch = FixedChainCollator(config(), mode="generation", target_roles=[target])(records()[:1])
        before = generate(old, batch, steps=4, temperature=temperature, seed=73)
        after = new.generate(batch, steps=4, temperature=temperature, seed=73)
        assert before == after


def test_adamw_state_can_resume_with_same_parameter_order():
    old, new = models()
    batch = FixedChainCollator(config())(records())
    token_t = torch.ones_like(batch["input_ids"], dtype=torch.float)
    old_optimizer = torch.optim.AdamW(old.parameters(), lr=1e-4)
    old(**batch, token_t=token_t)["loss"].backward()
    old_optimizer.step()
    new.load_state_dict(old.state_dict(), strict=True)
    new_optimizer = torch.optim.AdamW(new.parameters(), lr=1e-4)
    # Clone optimizer tensors: load_state_dict can otherwise alias CPU states.
    from copy import deepcopy
    new_optimizer.load_state_dict(deepcopy(old_optimizer.state_dict()))
    for model, optimizer in ((old, old_optimizer), (new, new_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        model(**batch, token_t=token_t)["loss"].backward()
        optimizer.step()
    for name, expected in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("backend", ["auto", "math", "efficient"])
def test_cpu_sdpa_policy_preserves_caller_flags(monkeypatch, backend):
    monkeypatch.setenv("IMMUNE_SDPA_BACKEND", backend)
    model = FixedChainModel(config())
    batch = FixedChainCollator(config())(records())
    def flags():
        return tuple(getattr(torch.backends.cuda, name + "_sdp_enabled")()
                     for name in ("flash", "mem_efficient", "math", "cudnn"))
    before = flags()
    model(**batch)["loss"].backward()
    assert flags() == before
    assert all(block.sdpa_backend == backend for block in model.backbone.transformer.blocks)


def test_invalid_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("IMMUNE_SDPA_BACKEND", "typo")
    with pytest.raises(ValueError, match="IMMUNE_SDPA_BACKEND"):
        FixedChainModel(config())
