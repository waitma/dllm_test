"""Architecture contracts; activate pllm, then run:

USE_TF=0 python -m pytest scripts/tests/immune_llada/test_research_next.py -q
All inputs are artificial strings; tests run on CPU and use temporary checkpoints.
"""
from contextlib import nullcontext
import copy
import re
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.research.model import ResearchModel
from dllm.pipelines.immune_llada.research_next.model import ArchitectureBlock, ArchitectureModel, VARIANTS
from dllm.pipelines.immune_llada.v3.layout import GenerationRequest, ImmuneCollator
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
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


def collator(mode='train'):
    config = tiny()
    return ImmuneCollator('A4', mode=mode, max_length=config.max_length,
                          full_capacity=config.full_capacity,
                          fragment_capacity=config.fragment_capacity)


def rows():
    return [
        BioSeqRecord([BioSeqChain('CASGGF', 'tcr_beta'),
                      BioSeqChain('CAVW', 'tcr_alpha')], 'tcr', 'artificial_test'),
        BioSeqRecord([BioSeqChain('AAGGSS', 'antigen'),
                      BioSeqChain('CAASW', 'antibody_heavy'),
                      BioSeqChain('CGAAF', 'antibody_light')],
                     'antibody_antigen', 'artificial_test'),
        BioSeqRecord([BioSeqChain('CSSGGAAW', 'nanobody_vhh')],
                     'nanobody', 'artificial_test'),
    ]


def deterministic_mask(batch, offset=0):
    index = torch.arange(batch['input_ids'].shape[1])[None]
    return (batch['eligible_mask'] & ((index+offset) % 2 == 0)).float()


def load_old_control(new, old):
    mapped = {}
    for name, value in old.state_dict().items():
        match = re.match(r'(backbone\.transformer\.blocks\.\d+)\.(.*)', name)
        if match:
            name = match[1]+'.base.'+match[2]
        mapped[name] = value
    new.load_state_dict(mapped, strict=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_control_matches_r00_same_weights_rng_and_sdpa_with_padding(dtype):
    torch.manual_seed(7)
    old = ResearchModel(tiny())
    new = ArchitectureModel(tiny())
    load_old_control(new, old)
    batch = collator()(rows())
    assert (~batch['attention_mask']).any()
    precision = lambda: torch.autocast('cpu', dtype=dtype) if dtype != torch.float32 else nullcontext()
    with sdpa_kernel(SDPBackend.MATH):
        torch.manual_seed(84)
        with precision():
            expected = old(**batch)['loss']
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(84)
        with precision():
            actual = new(**batch)['loss']
        assert torch.equal(torch.get_rng_state(), expected_rng)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        old.eval(); new.eval()
        with precision(), torch.no_grad():
            old_logits, old_hidden = old.denoise(**old.feature_args(batch), output_hidden_states=True)
            new_logits, new_hidden = new.denoise(**new.feature_args(batch), output_hidden_states=True)
        torch.testing.assert_close(new_logits, old_logits, rtol=0, atol=0)
        torch.testing.assert_close(new_hidden, old_hidden, rtol=0, atol=0)


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_finite_training_added_parameter_gradients_and_exact_reload(variant, tmp_path):
    torch.manual_seed(42)
    model = ArchitectureModel(tiny(), variant)
    batch = collator()(rows())
    result = model(**batch, token_t=deterministic_mask(batch))
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for block in model.backbone.transformer.blocks:
        if block.gate is not None:
            assert block.gate.weight.grad is not None
            assert block.gate.weight.grad.abs().sum() > 0
        for name in ('attention_scale', 'ffn_scale'):
            value = getattr(block, name)
            if isinstance(value, nn.Parameter):
                assert value.grad is not None and value.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        before = model.denoise(**model.feature_args(batch), output_hidden_states=True)
    model.save_pretrained(tmp_path)
    restored = ArchitectureModel.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        after = restored.denoise(**restored.feature_args(batch), output_hidden_states=True)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert model.variant == restored.variant and model.options == restored.options


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_whole_layer_checkpoint_two_live_views_match_all_parameter_gradients(variant):
    torch.manual_seed(101)
    ordinary = ArchitectureModel(tiny(), variant)
    checkpointed = copy.deepcopy(ordinary)
    checkpointed.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    batch_a = collator()(rows())
    # Different length and batch size expose accidental forward-global state.
    batch_b = collator()(rows()[2:])
    losses = []
    for model in (ordinary, checkpointed):
        first = model(**batch_a, token_t=deterministic_mask(batch_a))['loss']
        second = model(**batch_b, token_t=deterministic_mask(batch_b, 1))['loss']
        combined = first+.37*second
        losses.append(combined.detach())
        combined.backward()
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for (name, expected), (other_name, actual) in zip(ordinary.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        assert (expected.grad is None) == (actual.grad is None), name
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=1e-5, msg=name)


def test_value_residual_keeps_gradient_to_first_values_without_cross_call_cache():
    model = ArchitectureModel(tiny(), 'r19_value_residual')
    block = model.backbone.transformer.blocks[1]
    x = torch.randn(1, 9, 32)
    first = torch.randn_like(x, requires_grad=True)
    bias = torch.zeros(1, 4, 9, 9)
    positions = torch.arange(9)[None]
    out, _ = block(x, bias, positions, first)
    out.square().sum().backward()
    assert first.grad is not None and first.grad.abs().sum() > 0
    model.eval()
    batch_a, batch_b = collator()(rows()), collator()(rows()[2:])
    with torch.no_grad():
        a = model.denoise(**model.feature_args(batch_a))[0]
        model.denoise(**model.feature_args(batch_b))
        repeated = model.denoise(**model.feature_args(batch_a))[0]
    assert torch.equal(a, repeated)
    assert not any('first_value' in name for name in vars(model))


@pytest.mark.parametrize('variant', ['r17_chain_late_fusion', 'r18_chain_local_global'])
def test_chain_isolated_layers_no_prefix_relay_or_padding_keys(variant):
    model = ArchitectureModel(tiny(), variant).eval()
    chain = torch.tensor([[-1, -1, 0, 0, 1, 1, -1]])
    pos = torch.tensor([[0, 0, 0, 1, 0, 1, 0]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    local = model.layer_allowed(0, chain, pos, valid)[0]
    global_layer = model.layer_allowed(1, chain, pos, valid)[0]
    assert not local[:, -1].any()
    assert not local[2:4, 4:6].any() and not local[4:6, 2:4].any()
    assert not local[:2, 2:6].any() and not local[2:6, :2].any()
    assert local.any(-1).all()
    # Two successive isolated operations must not relay through metadata.
    assert not (local.float() @ local.float())[2:4, 4:6].any()
    assert global_layer[2:4, 4:6].all()
    x = torch.randn(1, 7, 32, requires_grad=True)
    bias = torch.zeros(1, 4, 7, 7).masked_fill(~local[None, None], torch.finfo(torch.float32).min)
    h, _ = model.backbone.transformer.blocks[0](x, bias, pos)
    grad = torch.autograd.grad(h[:, 4:6].square().sum(), x)[0]
    assert grad[:, 2:4].eq(0).all() and grad[:, :2].eq(0).all()
    changed = x.detach().clone()
    changed[:, :4] += torch.randn_like(changed[:, :4])*10
    changed[:, -1] += 100
    with torch.no_grad():
        h_changed, _ = model.backbone.transformer.blocks[0](changed, bias, pos)
        assert torch.equal(h[:, 4:6], h_changed[:, 4:6])
        second, _ = model.backbone.transformer.blocks[1](h, bias, pos)
        second_changed, _ = model.backbone.transformer.blocks[1](h_changed, bias, pos)
        assert torch.equal(second[:, 4:6], second_changed[:, 4:6])
        global_bias = torch.zeros_like(bias).masked_fill(~global_layer[None, None], torch.finfo(torch.float32).min)
        fused, _ = model.backbone.transformer.blocks[1](h, global_bias, pos)
        fused_changed, _ = model.backbone.transformer.blocks[1](h_changed, global_bias, pos)
        assert not torch.allclose(fused[:, 4:6], fused_changed[:, 4:6])


def test_local_window_radius_32_does_not_cross_chain_or_permutation():
    model = ArchitectureModel(tiny(), 'r18_chain_local_global')
    chain = torch.tensor([[0]*35+[1]*35])
    pos = torch.tensor([list(range(35))*2])
    valid = torch.ones_like(chain, dtype=torch.bool)
    allowed = model.layer_allowed(0, chain, pos, valid)[0]
    assert allowed[0, 32] and not allowed[0, 33]
    assert not allowed[34, 35]  # Adjacent in storage, different chains.
    assert model.layer_allowed(1, chain, pos, valid)[0, 0, 69]


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_whole_chain_permutation_equivariance_with_padding(variant):
    torch.manual_seed(17)
    model = ArchitectureModel(tiny(), variant).eval()
    batch = collator('representation')([rows()[0], rows()[1]])
    length = batch['input_ids'].shape[1]
    permutations = []
    for chain_ids, valid in zip(batch['chain_ids'], batch['attention_mask']):
        groups = [((chain_ids == uid) & valid).nonzero().flatten()
                  for uid in chain_ids[valid & (chain_ids >= 0)].unique(sorted=True).flip(0)]
        prefix = ((chain_ids < 0) & valid).nonzero().flatten()
        padding = (~valid).nonzero().flatten()
        permutations.append(torch.cat((prefix, *groups, padding)))
    permutation = torch.stack(permutations)
    assert permutation.shape == (2, length)
    changed = {key: value.gather(1, permutation) for key, value in batch.items()}
    with torch.no_grad():
        before = model.denoise(**model.feature_args(batch), output_hidden_states=True)
        after = model.denoise(**model.feature_args(changed), output_hidden_states=True)
    undo = permutation.argsort(1)
    for expected, actual in zip(before, after):
        restored = actual.gather(1, undo[..., None].expand(-1, -1, actual.shape[-1]))
        torch.testing.assert_close(restored, expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_padding_cannot_change_attended_logits_or_hidden_states(variant):
    torch.manual_seed(15)
    model = ArchitectureModel(tiny(), variant).eval()
    alone = collator('representation')([rows()[2]])
    padded = collator('representation')([rows()[2], rows()[1]])
    length = alone['input_ids'].shape[1]
    with torch.no_grad():
        a = model.denoise(**model.feature_args(alone), output_hidden_states=True)
        b = model.denoise(**model.feature_args(padded), output_hidden_states=True)
    for expected, actual in zip(a, b):
        torch.testing.assert_close(actual[:1, :length], expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_reference_free_generate_preserves_conditions_and_feature_whitelist(variant):
    model = ArchitectureModel(tiny(), variant).eval()
    request = GenerationRequest(conditions=(BioSeqChain('AGSGA', 'antigen'),),
                                targets=(('antibody_heavy', 'junction'), ('antibody_light', 'junction')))
    batch = collator('generation')([request])
    assert batch['labels'].eq(-100).all()
    assert not {'labels', 'residue_mask', 'terminal_mask', 'eligible_mask'} & set(model.feature_args(batch))
    generated = model.generate(batch, steps=2)
    slots = batch['target_slot_mask']
    assert torch.equal(generated['input_ids'][~slots], batch['input_ids'][~slots])
    assert len(generated['chains'][0]) == 2
    assert all(len(chain['sequence']) <= chain['capacity'] for chain in generated['chains'][0])


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_partial_noise_label_masks_cannot_leak_into_denoising(variant):
    model = ArchitectureModel(tiny(), variant).eval()
    batch = collator()(rows())
    noised, _, selected = model.corrupt(batch, token_t=deterministic_mask(batch))
    assert selected.any() and (batch['eligible_mask'] & ~selected).any()
    assert torch.equal(noised[~batch['eligible_mask']], batch['input_ids'][~batch['eligible_mask']])
    batch['input_ids'] = noised
    with torch.no_grad():
        before = model.denoise(**model.feature_args(batch))[0]
        batch['labels'].fill_(100000)
        for key in ('residue_mask', 'terminal_mask', 'eligible_mask'):
            batch[key] = ~batch[key]
        after = model.denoise(**model.feature_args(batch))[0]
    assert torch.equal(before, after)
    with pytest.raises(TypeError):
        model.denoise(**model.feature_args(batch), labels=batch['labels'])


def test_exact_parameter_overhead_and_fixed_controls_have_no_new_parameters():
    baseline = sum(p.numel() for p in ArchitectureModel(tiny()).parameters())
    for variant, options in VARIANTS.items():
        model = ArchitectureModel(tiny(), variant)
        expected = 2*32*4 if options.get('head_gate') else 2*2*32 if options.get('learn_scale') else 0
        assert sum(p.numel() for p in model.parameters())-baseline == expected
    # Exercise the actual new block constructor at production width without
    # allocating eight complete 76M-parameter models merely to count gates.
    for variant, expected in [('r13_head_output_gate', 73728), ('r15_layerscale', 12288),
                              ('r14_fixed_half_attention', 0), ('r16_fixed_residual_scale', 0)]:
        dummy = nn.Identity()
        dummy.config = SimpleNamespace(d_model=768, n_heads=12)
        block = ArchitectureBlock(dummy, VARIANTS[variant])
        assert 8*sum(p.numel() for p in block.parameters()) == expected
