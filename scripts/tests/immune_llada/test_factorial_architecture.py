"""CPU-only contracts using integer tensors, with eager temporary checkpoint cleanup.

Run:
  source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
  conda activate pllm
  CUDA_VISIBLE_DEVICES='' USE_TF=0 python -B -m pytest -q /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_factorial_architecture.py

No dataset, sequence record, Trainer, GPU operation or production gate is used.
"""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from dllm.pipelines.immune_llada.research_refined.factorial import (
    FACTOR_ORDER, LEGACY_EQUIVALENTS, SCHEMA, VARIANTS, FactorialModel)
from dllm.pipelines.immune_llada.research_refined.model import RefinedModel
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return ImmuneConfig(ablation='A4', d_model=16, n_layers=4, n_heads=2,
                        mlp_hidden_size=32, max_length=96,
                        full_capacity=40, fragment_capacity=40)


def integer_batch(model, short=False):
    chain = torch.tensor([[-1, -1] + [0] * 35 + [1] * 35 + [-1, -1]]).repeat(2, 1)
    positions = torch.tensor([[0, 0] + list(range(35)) * 2 + [0, 0]]).repeat(2, 1)
    valid = torch.ones_like(chain, dtype=torch.bool)
    valid[:, -2:] = False
    valid[1, 52:] = False
    allowed = model.allowed_target.nonzero().flatten()
    ids = allowed[torch.arange(chain.numel()).reshape(chain.shape) % len(allowed)]
    slots = (chain >= 0) & valid
    batch = dict(input_ids=ids, labels=ids.clone(), attention_mask=valid,
                 role_ids=torch.zeros_like(chain), scope_ids=torch.zeros_like(chain),
                 chain_ids=chain, position_ids=positions, target_slot_mask=slots,
                 eligible_mask=slots.clone(), residue_mask=slots.clone(),
                 terminal_mask=torch.zeros_like(valid))
    if short:
        batch = {name: value[:1, :20].clone() for name, value in batch.items()}
    return batch


def token_mask(batch):
    return (batch['eligible_mask'] & (torch.arange(batch['input_ids'].shape[1])[None] % 2 == 0)).float()


def gradients_equal(first, second):
    for (name, a), (other, b) in zip(first.named_parameters(), second.named_parameters()):
        assert name == other and (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            assert torch.isfinite(b.grad).all(), name
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=1e-5, msg=name)


def test_explicit_factorial_options_and_shared_parameters():
    assert FACTOR_ORDER == ('G', 'S', 'L')
    assert {name for name in VARIANTS if name.startswith('f') and name != 'fixed_gate_control'} == {
        'f000', 'f001', 'f010', 'f011', 'f100', 'f101', 'f110', 'f111'}
    torch.manual_seed(11)
    reference = RefinedModel(config(), 'r20_refined_control')
    for variant, options in VARIANTS.items():
        model = FactorialModel(config(), variant)
        gates_before = {name: value.clone() for name, value in model.state_dict().items() if '.gate.' in name}
        report = model.load_common_parameters(reference)
        assert model.options == options
        assert report['ignored_source_only'] == []
        for name in report['copied']:
            assert torch.equal(model.state_dict()[name], reference.state_dict()[name]), name
        for name, value in gates_before.items():
            assert torch.equal(model.state_dict()[name], value)
        for block in model.backbone.transformer.blocks:
            assert (block.gate is not None) == bool(options.get('head_gate'))
            assert torch.all(block.attention_scale == options.get('residual_scale', 1.))
            assert torch.all(block.ffn_scale == options.get('residual_scale', 1.))
    with pytest.raises(ValueError, match='Unknown factorial'):
        FactorialModel(config(), 'f112')
    incompatible = copy.deepcopy(reference)
    incompatible.backbone.transformer.blocks[0].base.q_proj.weight = torch.nn.Parameter(torch.zeros(1, 1))
    with pytest.raises(ValueError, match='shape/dtype'):
        FactorialModel(config()).load_common_parameters(incompatible)


@pytest.mark.parametrize('variant', list(LEGACY_EQUIVALENTS))
def test_legacy_single_factor_and_control_exact_numerics(variant):
    torch.manual_seed(21)
    old = RefinedModel(config(), LEGACY_EQUIVALENTS[variant])
    new = FactorialModel(config(), variant)
    new.load_state_dict(old.state_dict(), strict=True)
    batch = integer_batch(new)
    with sdpa_kernel(SDPBackend.MATH):
        losses = [model(**batch, token_t=token_mask(batch))['loss'] for model in (old, new)]
        for loss in losses:
            loss.backward()
        torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
        gradients_equal(old, new)
        with torch.no_grad():
            expected = old.denoise(**old.feature_args(batch), output_hidden_states=True)
            actual = new.denoise(**new.feature_args(batch), output_hidden_states=True)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_segment_global_isolation_radius_padding_and_gradient_paths():
    model = FactorialModel(config(), 'cv_segment_global').eval()
    local_model = FactorialModel(config(), 'f001').eval()
    batch = integer_batch(model)
    args = [batch[key] for key in ('chain_ids', 'position_ids', 'attention_mask')]
    segment = model.layer_allowed(0, *args)
    global_mask = model.layer_allowed(1, *args)
    local = local_model.layer_allowed(0, *args)
    assert segment[0, 2, 36] and not local[0, 2, 36]  # Distance 34 > radius 32.
    assert local[0, 2, 34] and not local[0, 2, 35]
    assert not segment[:, :2, 2:72].any() and not segment[:, 2:72, :2][batch['attention_mask'][:, 2:72]].any()
    assert not segment[0, 2:37, 37:72].any()
    assert global_mask[0, 2:37, 37:72].all()
    assert not (segment & ~batch['attention_mask'][:, None, :]).any()
    assert segment.any(-1).all()
    assert not (segment[0].float() @ segment[0].float())[2:37, 37:72].any()
    assert [model.mask_key(i) for i in range(4)] == [('segment',), ('global',)] * 2
    x = torch.randn(2, 74, 16, requires_grad=True)
    bias = torch.zeros(2, 2, 74, 74).masked_fill(~segment[:, None], torch.finfo(torch.float32).min)
    with sdpa_kernel(SDPBackend.MATH):
        output, _ = model.backbone.transformer.blocks[0](x, bias, batch['position_ids'])
        grad = torch.autograd.grad(output[0, 37:72].square().sum(), x)[0]
        assert grad[0, :37].eq(0).all() and grad[0, 72:].eq(0).all()
        altered = x.detach().clone()
        altered[0, :37] += 5
        altered[0, 72:] += 7
        with torch.no_grad():
            changed, _ = model.backbone.transformer.blocks[0](altered, bias, batch['position_ids'])
        torch.testing.assert_close(changed[0, 37:72], output[0, 37:72], atol=0, rtol=0)
    empty = {**batch, 'attention_mask': torch.zeros_like(batch['attention_mask'])}
    with pytest.raises(ValueError, match='at least one'):
        model.denoise(**model.feature_args(empty))


@pytest.mark.parametrize('variant', ['f011', 'f101', 'f110', 'f111', 'cv_segment_global'])
def test_new_combination_checkpoint_two_live_graphs_and_gradients(variant):
    torch.manual_seed(31)
    ordinary = FactorialModel(config(), variant)
    checkpointed = copy.deepcopy(ordinary)
    checkpointed.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    first, second = integer_batch(ordinary), integer_batch(ordinary, short=True)
    values = []
    with sdpa_kernel(SDPBackend.MATH):
        for model in (ordinary, checkpointed):
            a = model(**first, token_t=token_mask(first))['loss']
            b = model(**second, token_t=token_mask(second))['loss']
            combined = a + .37 * b
            combined.backward()
            assert torch.isfinite(combined)
            values.append(combined.detach())
        torch.testing.assert_close(values[0], values[1], atol=0, rtol=0)
        gradients_equal(ordinary, checkpointed)
    for block in ordinary.backbone.transformer.blocks:
        if block.gate is not None:
            assert block.gate.weight.grad is not None and block.gate.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('variant', list(VARIANTS))
def test_save_restore_and_explicit_schema_options_vocabulary(variant, tmp_path):
    model = FactorialModel(config(), variant).eval()
    batch = integer_batch(model, short=True)
    with TemporaryDirectory(prefix='factorial-cpu-', dir=tmp_path) as temporary:
        directory = Path(temporary)
        model.save_pretrained(directory)
        restored = FactorialModel.from_pretrained(directory).eval()
        assert restored.options == model.options and restored.variant == variant
        assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
        with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
            before = model.denoise(**model.feature_args(batch), output_hidden_states=True)
            after = restored.denoise(**restored.feature_args(batch), output_hidden_states=True)
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        path = directory / 'research_config.json'
        original = json.loads(path.read_text())
        assert original['schema'] == SCHEMA
        for field, bad in [('schema', 'unknown'), ('options', {'residual_scale': .9}),
                           ('factor_order', ['L', 'S', 'G'])]:
            path.write_text(json.dumps({**original, field: bad}))
            with pytest.raises(ValueError, match='schema/options'):
                FactorialModel.from_pretrained(directory)
        path.write_text(json.dumps(original))
        (directory / 'vocabulary.json').write_text('[]')
        with pytest.raises(ValueError, match='Vocabulary'):
            FactorialModel.from_pretrained(directory)
    assert not directory.exists()
