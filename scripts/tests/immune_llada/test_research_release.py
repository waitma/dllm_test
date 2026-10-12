"""Factory integration; run only after the research-family bridge is deployed.

After activating pllm:
USE_TF=0 python -m pytest scripts/tests/immune_llada/test_research_release.py -q
These tests intentionally require R12–R19 in the legacy public registry.
"""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from dllm.pipelines.immune_llada.research.model import ResearchModel, VARIANTS as PUBLIC_VARIANTS
from dllm.pipelines.immune_llada.research_next.model import ArchitectureModel, VARIANTS
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.immune_llada.v3.layout import GenerationRequest, ImmuneCollator
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny():
    return ImmuneConfig(ablation='A4', d_model=32, n_layers=2, n_heads=4,
                        mlp_hidden_size=64, max_length=256, full_capacity=24,
                        fragment_capacity=16)


def generation_batch():
    c = ImmuneCollator('A4', mode='generation', max_length=256,
                       full_capacity=24, fragment_capacity=16)
    return c([GenerationRequest(conditions=(), targets=(('tcr_beta', 'junction'),))])


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_legacy_factory_constructs_architecture_subtype_with_registered_options(variant):
    registered = dict(PUBLIC_VARIANTS[variant])
    assert registered.pop('architecture_v2') is True
    assert registered == VARIANTS[variant]
    torch.manual_seed(12)
    direct = ArchitectureModel(tiny(), variant)
    torch.manual_seed(12)
    constructed = ResearchModel(tiny(), variant)
    assert type(constructed) is ArchitectureModel
    assert constructed.variant == variant and constructed.options == VARIANTS[variant]
    # Identical initialization rules out duplicate subclass initialization.
    assert direct.state_dict().keys() == constructed.state_dict().keys()
    for name, expected in direct.state_dict().items():
        assert torch.equal(constructed.state_dict()[name], expected), name


def test_legacy_loader_delegates_v2_checkpoint_without_dropping_new_parameters(tmp_path):
    torch.manual_seed(31)
    model = ResearchModel(tiny(), 'r13_head_output_gate').eval()
    model.save_pretrained(tmp_path)
    assert json.loads((tmp_path/'research_config.json').read_text())['schema'] == 'immune_architecture.v2'
    restored = ResearchModel.from_pretrained(tmp_path).eval()
    assert type(restored) is ArchitectureModel
    assert restored.variant == model.variant and restored.options == model.options
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, expected in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], expected), name
    batch = generation_batch()
    with torch.no_grad():
        expected = model.denoise(**model.feature_args(batch))[0]
        actual = restored.denoise(**restored.feature_args(batch))[0]
    assert torch.equal(expected, actual)


def test_factory_architecture_deepcopy_bypasses_argument_requiring_initialization():
    model = ResearchModel(tiny(), 'r19_value_residual').eval()
    cloned = copy.deepcopy(model)
    assert type(cloned) is ArchitectureModel
    assert cloned.variant == model.variant and cloned.options == model.options
    for (name, expected), (copy_name, actual) in zip(model.named_parameters(), cloned.named_parameters()):
        assert name == copy_name and torch.equal(actual, expected)
        assert actual.data_ptr() != expected.data_ptr(), name
    batch = generation_batch()
    with torch.no_grad():
        a = model.denoise(**model.feature_args(batch))[0]
        b = cloned.denoise(**cloned.feature_args(batch))[0]
    assert torch.equal(a, b)


def test_legacy_runner_make_model_routes_new_variant_and_keeps_checkpoint_strategy():
    from scripts import run_immune_research as runner
    model = runner.make_model(SimpleNamespace(cpu_test=True, variant='r19_value_residual'))
    assert type(model) is ArchitectureModel
    assert model.variant == 'r19_value_residual'
    assert model.options == VARIANTS[model.variant]
    assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
    assert model.backbone.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer
    assert model.config.n_layers == 2
