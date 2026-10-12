"""CPU equivalence checks for an undeployed per-forward attention-bias cache.

After activating pllm:
USE_TF=0 python -m pytest scripts/tests/immune_llada/test_architecture_bias_cache.py -q
No GPU jobs, checkpoint files or production source edits are performed.
"""
from contextlib import nullcontext
import copy

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.architecture_prototypes.bias_cache import bind_bias_cache, count_bias_allocations
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.research_next.model import ArchitectureModel, VARIANTS
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(layers=4):
    return ImmuneConfig(ablation='A4', d_model=32, n_layers=layers, n_heads=4,
                        mlp_hidden_size=64, max_length=256, full_capacity=24, fragment_capacity=16)


def collator(mode='train'):
    c = config()
    return ImmuneCollator('A4', mode=mode, max_length=c.max_length,
                          full_capacity=c.full_capacity, fragment_capacity=c.fragment_capacity)


def rows():
    return [
        BioSeqRecord([BioSeqChain('CASGGF', 'tcr_beta'), BioSeqChain('CAVW', 'tcr_alpha')],
                     'tcr', 'artificial_cache_test'),
        # >32 residues exercise an actual local-window boundary, not just chain isolation.
        BioSeqRecord([BioSeqChain('AG'*25, 'antigen'), BioSeqChain('CAASW', 'antibody_heavy'),
                      BioSeqChain('CGAAF', 'antibody_light')], 'antibody_antigen', 'artificial_cache_test'),
        BioSeqRecord([BioSeqChain('CSSGGAAW', 'nanobody_vhh')], 'nanobody', 'artificial_cache_test'),
    ]


def mask(batch, offset=0):
    index = torch.arange(batch['input_ids'].shape[1])[None]
    return (batch['eligible_mask'] & ((index+offset) % 2 == 0)).float()


def models(variant, checkpoint=False, layers=4):
    torch.manual_seed(101)
    ordinary = ArchitectureModel(config(layers), variant)
    with torch.no_grad():
        ordinary.chain_bias.copy_(torch.linspace(-.4, .6, ordinary.chain_bias.numel()).view_as(ordinary.chain_bias))
    if checkpoint:
        ordinary.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    return ordinary, bind_bias_cache(copy.deepcopy(ordinary))


def precision(dtype):
    return torch.autocast('cpu', dtype=dtype) if dtype == torch.bfloat16 else nullcontext()


def assert_gradients(ordinary, cached):
    expected, actual = dict(ordinary.named_parameters()), dict(cached.named_parameters())
    assert expected.keys() == actual.keys()
    for name, parameter in expected.items():
        other = actual[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all() and torch.isfinite(other.grad).all()
            torch.testing.assert_close(other.grad, parameter.grad, atol=2e-6, rtol=1e-5, msg=name)
    assert ordinary.chain_bias.grad is not None and ordinary.chain_bias.grad.abs().sum() > 0


@pytest.mark.parametrize('variant', sorted(VARIANTS))
@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_logits_hidden_loss_and_every_parameter_gradient_match(variant, padded, dtype):
    ordinary, cached = models(variant)
    batch = collator()(rows() if padded else [rows()[1], rows()[1]])
    assert bool((~batch['attention_mask']).any()) == padded
    records = []
    with sdpa_kernel(SDPBackend.MATH):
        for model in (ordinary, cached):
            model.train()
            features = model.feature_args(batch)
            features['input_ids'] = features['input_ids'].masked_fill(mask(batch).bool(), model.tokenizer.mask_token_id)
            with precision(dtype):
                logits, hidden = model.denoise(**features, output_hidden_states=True)
                result = model(**batch, token_t=mask(batch))
            assert torch.isfinite(result['loss'])
            records.append((logits.detach(), hidden.detach(), result['loss'].detach()))
            result['loss'].backward()
    for before, after in zip(*records):
        torch.testing.assert_close(after, before, atol=0, rtol=0)
    assert_gradients(ordinary, cached)


@pytest.mark.parametrize('variant', sorted(VARIANTS))
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_checkpoint_two_live_views_preserve_all_gradients(variant, dtype):
    ordinary, cached = models(variant, checkpoint=True)
    a, b = collator()(rows()), collator()(rows()[2:])
    losses = []
    with sdpa_kernel(SDPBackend.MATH):
        for model in (ordinary, cached):
            with precision(dtype):
                first = model(**a, token_t=mask(a))['loss']
                second = model(**b, token_t=mask(b, 1))['loss']
                combined = first+.37*second
            losses.append(combined.detach())
            combined.backward()
    torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
    assert_gradients(ordinary, cached)


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_cache_is_not_shared_across_calls_shapes_or_parameter_updates(variant):
    ordinary, cached = models(variant)
    ordinary.eval(); cached.eval()
    a, b = collator()(rows()), collator()(rows()[2:])
    state_keys = set(vars(cached))
    with torch.no_grad():
        first = cached.denoise(**cached.feature_args(a), output_hidden_states=True)
        cached.denoise(**cached.feature_args(b))
        repeated = cached.denoise(**cached.feature_args(a), output_hidden_states=True)
        assert all(torch.equal(x, y) for x, y in zip(first, repeated))
        for model in (ordinary, cached):
            model.chain_bias[0].add_(.1)
        expected = ordinary.denoise(**ordinary.feature_args(a), output_hidden_states=True)
        actual = cached.denoise(**cached.feature_args(a), output_hidden_states=True)
    assert all(torch.equal(x, y) for x, y in zip(expected, actual))
    assert set(vars(cached)) == state_keys
    assert cached.state_dict().keys() == ordinary.state_dict().keys()


@pytest.mark.parametrize('variant', sorted(VARIANTS))
def test_padding_is_invisible_after_cache_and_local_schedule(variant):
    _, cached = models(variant)
    cached.eval()
    alone = collator('representation')([rows()[2]])
    padded = collator('representation')([rows()[2], rows()[1]])
    length = alone['input_ids'].shape[1]
    with torch.no_grad():
        a = cached.denoise(**cached.feature_args(alone), output_hidden_states=True)
        b = cached.denoise(**cached.feature_args(padded), output_hidden_states=True)
    for expected, actual in zip(a, b):
        torch.testing.assert_close(actual[:1, :length], expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('variant', sorted(VARIANTS))
@pytest.mark.parametrize('padded', [False, True])
def test_measured_bias_operation_counts_and_storage_reuse(variant, padded):
    ordinary, cached = models(variant, checkpoint=True, layers=8)
    batch = collator()(rows() if padded else [rows()[1], rows()[1]])
    original, logits, hidden = count_bias_allocations(ordinary, batch)
    optimized, new_logits, new_hidden = count_bias_allocations(cached, batch)
    assert torch.equal(logits, new_logits) and torch.equal(hidden, new_hidden)
    different_masks = variant in ('r17_chain_late_fusion', 'r18_chain_local_global')
    expected_fills = int(padded)+int(different_masks)
    assert original['masked_fill_count'] == 8
    assert original['unique_block_bias_storages'] == 8
    assert optimized['masked_fill_count'] == expected_fills
    assert optimized['unique_block_bias_storages'] == 1+int(different_masks)
    assert original['block_invocations'] == optimized['block_invocations'] == 8
    elements = torch.tensor(optimized['bias_shape']).prod().item()
    assert optimized['masked_fill_output_bytes_total'] == elements*4*expected_fills


def test_invalid_attention_and_label_feature_contracts_are_preserved():
    ordinary, cached = models('r18_chain_local_global')
    batch = collator()(rows())
    features = cached.feature_args(batch)
    with pytest.raises(TypeError):
        cached.denoise(**features, labels=batch['labels'])
    features['attention_mask'] = features['attention_mask'].clone()
    features['attention_mask'][0] = False
    for model in (ordinary, cached):
        with pytest.raises(ValueError, match='at least one attended token'):
            model.denoise(**features)


def test_prototype_is_outside_live_fingerprint_and_does_not_patch_model_class():
    from scripts.run_immune_architecture import fingerprints
    before = fingerprints()
    ordinary, cached = models('r12_architecture_control')
    assert ordinary.denoise.__func__ is ArchitectureModel.denoise
    assert cached.denoise.__func__ is not ArchitectureModel.denoise
    assert not any('architecture_prototypes' in path or 'test_architecture_bias_cache' in path for path in before)
    assert fingerprints() == before
