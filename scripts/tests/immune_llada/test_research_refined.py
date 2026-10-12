"""Production bias-cache equivalence; activate pllm, then run:

USE_TF=0 python -m pytest -q scripts/tests/immune_llada/test_research_refined.py
CPU only, artificial inputs, FP32/BF16. Temporary checkpoints are deleted eagerly.
"""
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from dllm.pipelines.immune_llada.research_next.model import ArchitectureModel
from dllm.pipelines.immune_llada.research_refined.model import BASE_VARIANTS, RefinedModel
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from scripts.tests.immune_llada.test_research_next import rows, deterministic_mask


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    # Four layers exercise repeated isolated/global masks; 48 slots exceed radius32.
    return ImmuneConfig(ablation='A4', d_model=32, n_layers=4, n_heads=4,
                        mlp_hidden_size=64, max_length=256, full_capacity=48, fragment_capacity=40)


def collate(records):
    settings = config()
    return ImmuneCollator('A4', max_length=settings.max_length,
                          full_capacity=settings.full_capacity,
                          fragment_capacity=settings.fragment_capacity)(records)


def precision(dtype):
    return torch.autocast('cpu', dtype=dtype) if dtype != torch.float32 else nullcontext()


def models(variant):
    torch.manual_seed(91)
    old = ArchitectureModel(config(), BASE_VARIANTS[variant])
    torch.manual_seed(91)
    new = RefinedModel(config(), variant)
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())
    new.load_state_dict(old.state_dict(), strict=True)
    return old, new


def compare_gradients(old, new, dtype):
    for (name, expected), (other_name, actual) in zip(old.named_parameters(), new.named_parameters()):
        assert name == other_name
        assert (expected.grad is None) == (actual.grad is None), name
        if expected.grad is not None:
            assert torch.isfinite(actual.grad).all(), name
            torch.testing.assert_close(actual.grad, expected.grad,
                                       atol=2e-6 if dtype == torch.float32 else 3e-5,
                                       rtol=1e-5, msg=name)


@pytest.mark.parametrize('variant', sorted(BASE_VARIANTS))
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('padded', [False, True])
def test_identical_weights_logits_loss_and_all_gradients(variant, dtype, padded):
    old, new = models(variant)
    batch = collate(rows() if padded else rows()[:1])
    assert bool((~batch['attention_mask']).any()) == padded
    with sdpa_kernel(SDPBackend.MATH):
        losses = []
        for model in (old, new):
            torch.manual_seed(120)
            with precision(dtype):
                result = model(**batch, token_t=deterministic_mask(batch))
            assert torch.isfinite(result['loss'])
            result['loss'].backward()
            losses.append(result['loss'].detach())
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        compare_gradients(old, new, dtype)
        old.eval(); new.eval()
        with precision(dtype), torch.no_grad():
            before = old.denoise(**old.feature_args(batch), output_hidden_states=True)
            after = new.denoise(**new.feature_args(batch), output_hidden_states=True)
        for expected, actual in zip(before, after):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('variant', sorted(BASE_VARIANTS))
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_checkpoint_two_live_forward_graphs_match_uncached_parent(variant, dtype):
    old, new = models(variant)
    new.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    first, second = collate(rows()), collate(rows()[2:])
    losses = []
    with sdpa_kernel(SDPBackend.MATH):
        for model in (old, new):
            torch.manual_seed(33)
            with precision(dtype):
                a = model(**first, token_t=deterministic_mask(first))['loss']
                b = model(**second, token_t=deterministic_mask(second, 1))['loss']
                combined = a+.37*b
            combined.backward()
            losses.append(combined.detach())
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        compare_gradients(old, new, dtype)


@pytest.mark.parametrize('variant', sorted(BASE_VARIANTS))
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_bias_storage_reused_within_forward_and_never_across_calls(variant, dtype):
    old, new = models(variant)
    old.eval(); new.eval()
    batch = collate(rows())
    changed = {name: value.clone() for name, value in batch.items()}
    changed['chain_ids'][:, 8:16] += 10
    changed['attention_mask'][:, 3] = False
    captured = []
    hooks = [block.register_forward_pre_hook(lambda _module, args: captured.append(args[1]))
             for block in new.backbone.transformer.blocks]
    try:
        with sdpa_kernel(SDPBackend.MATH), precision(dtype), torch.no_grad():
            a = new.denoise(**new.feature_args(batch))[0]
            first_biases = list(captured)
            captured.clear()
            b = new.denoise(**new.feature_args(changed))[0]
            expected_b = old.denoise(**old.feature_args(changed))[0]
            second_biases = list(captured)
            captured.clear()
            repeated = new.denoise(**new.feature_args(batch))[0]
        assert torch.equal(a, repeated)
        torch.testing.assert_close(b, expected_b, rtol=0, atol=0)
        expected_unique = 2 if variant in {'r25_refined_late_fusion', 'r26_refined_local_global'} else 1
        assert len({value.data_ptr() for value in first_biases}) == expected_unique
        assert not {id(value) for value in first_biases} & {id(value) for value in second_biases}
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize('variant', sorted(BASE_VARIANTS))
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_save_reload_and_invalid_schema_rejected_with_eager_cleanup(variant, dtype, tmp_path):
    _, model = models(variant)
    model.eval()
    batch = collate(rows())
    with TemporaryDirectory(prefix='refined-', dir=tmp_path) as temporary:
        directory = Path(temporary)
        model.save_pretrained(directory)
        # Existing LLaDAModel construction enables global Flash SDP. Load before
        # entering the common math backend context, otherwise only the second
        # forward may use CPU Flash SDP despite an outer sdpa_kernel(MATH).
        restored = RefinedModel.from_pretrained(directory).eval()
        assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
        with sdpa_kernel(SDPBackend.MATH), precision(dtype), torch.no_grad():
            before = model.denoise(**model.feature_args(batch), output_hidden_states=True)
            after = restored.denoise(**restored.feature_args(batch), output_hidden_states=True)
        assert restored.variant == variant and restored.options == model.options
        for expected, actual in zip(before, after):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        settings_path = directory/'research_config.json'
        settings = json.loads(settings_path.read_text())
        settings['schema'] = 'unknown.schema'
        settings_path.write_text(json.dumps(settings))
        with pytest.raises(ValueError, match='schema/options'):
            RefinedModel.from_pretrained(directory)
        settings['schema'] = 'immune_architecture.refined.v1'
        settings['parent_variant'] = 'different_parent'
        settings_path.write_text(json.dumps(settings))
        with pytest.raises(ValueError, match='schema/options'):
            RefinedModel.from_pretrained(directory)
    assert not directory.exists()
