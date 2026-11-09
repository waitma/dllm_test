"""Activate pllm; USE_TF=0 python -B -m pytest THIS_ABSOLUTE_FILE -q.

Check that naming today's architecture V12 preserves its computation, dispatches
through the trainer, and keeps independently versioned checkpoint contracts.
"""
from dataclasses import asdict
import json

import pytest
import torch

from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.v10_airgen_qk_swiglu_bias import V10QKSwiGLUBiasModel
from dllm.pipelines.immune_llada.fixed_chain.v12_airgen import (
    ARCHITECTURE, SCHEMA, INPUT_FEATURES, OBJECTIVE, RECIPE, V12Collator, V12Model, corrupt,
    parameter_count, swiglu_hidden_size,
)
from scripts.research.bert_mlm_objectives import components, for_model
from scripts.research.run_bert_mlm import data_model_for_plan, expected_parameter_count
from scripts.tests.immune_llada.test_v10_airgen import records
from scripts.tests.immune_llada.test_v10_airgen_qk_swiglu_bias import config


def test_v12_numerics_and_dispatch_preserve_today_architecture():
    torch.set_num_threads(2)
    torch.manual_seed(93)
    previous = V10QKSwiGLUBiasModel(config())
    current = V12Model(config())
    current.load_state_dict(previous.state_dict(), strict=True)
    c = V12Collator(config())
    batch = corrupt(c(records()), c.tokenizer, torch.Generator().manual_seed(95))
    assert current.variant == ARCHITECTURE and current.options['model_version'] == 'V12'
    assert set(current.feature_args(batch)) == set(INPUT_FEATURES)
    before, after = previous(**batch), current(**batch)
    assert torch.equal(before['logits'], after['logits'])
    assert torch.equal(before['loss_sum'], after['loss_sum'])
    assert for_model(current)[1] is V12Model
    assert components(OBJECTIVE, None, ARCHITECTURE)[1] is V12Model
    assert sum(p.numel() for p in current.parameters()) == parameter_count(config())
    with pytest.raises(ValueError):
        components(OBJECTIVE, {}, ARCHITECTURE)


def test_v12_checkpoint_identifies_model_and_rejects_historical_schema(tmp_path):
    model = V12Model(config())
    directory = tmp_path/'v12'
    model.save_pretrained(directory)
    settings = json.loads((directory/'research_config.json').read_text())
    assert settings['schema'] == SCHEMA and settings['options']['model_version'] == 'V12'
    restored = V12Model.from_pretrained(directory)
    assert restored.variant == ARCHITECTURE
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
    with pytest.raises(ValueError):
        V10QKSwiGLUBiasModel.from_pretrained(directory)
    previous = tmp_path/'historical_v10'
    V10QKSwiGLUBiasModel(config()).save_pretrained(previous)
    with pytest.raises(ValueError):
        V12Model.from_pretrained(previous)


def test_trainer_validates_v12_dimensions_and_parameter_budget():
    reference = FixedChainConfig()
    legacy = dict(asdict(reference), d_model=768, n_layers=8, n_heads=12, mlp_hidden_size=3072)
    current = dict(legacy, d_model=1280, n_layers=3, n_heads=20, mlp_hidden_size=swiglu_hidden_size(1280))
    plan = dict(architecture=ARCHITECTURE, model_version='V12', objective=OBJECTIVE,
                recipe=RECIPE, model=current, data_model=legacy, input_features=list(INPUT_FEATURES),
                parameter_budget=dict(parameters=77476073, reference_parameters=75755753))
    assert data_model_for_plan(plan) == legacy
    assert expected_parameter_count(plan) == 77476073
    with pytest.raises(ValueError):
        data_model_for_plan(dict(plan, input_features=[*INPUT_FEATURES, 'scope_ids']))
