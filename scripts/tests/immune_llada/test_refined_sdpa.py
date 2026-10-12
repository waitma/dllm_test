"""After activating pllm, run USE_TF=0 python -m pytest -q this_file.

CPU policy checks and bounded CUDA BF16 backend/gradient checks; no checkpoints.
"""
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from dllm.pipelines.immune_llada.research_refined.model import RefinedModel
from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from scripts.tests.immune_llada.test_research_next import rows, deterministic_mask


def settings():
    return ImmuneConfig(ablation='A4', d_model=128, n_layers=2, n_heads=2,
                        mlp_hidden_size=256, max_length=256,
                        full_capacity=48, fragment_capacity=40)


def flags():
    return tuple(getattr(torch.backends.cuda, name+'_sdp_enabled')()
                 for name in ('flash', 'mem_efficient', 'math', 'cudnn'))


@pytest.mark.parametrize('policy', ['auto', 'math', 'efficient'])
def test_cpu_fallback_and_flag_restore(monkeypatch, policy):
    monkeypatch.setenv('IMMUNE_SDPA_BACKEND', policy)
    model = RefinedModel(settings())
    batch = ImmuneCollator('A4', max_length=256, full_capacity=48, fragment_capacity=40)(rows())
    with sdpa_kernel(SDPBackend.MATH):
        before = flags()
        result = model(**batch, token_t=deterministic_mask(batch))
        result['loss'].backward()
        assert torch.isfinite(result['loss'])
        assert flags() == before
    assert all(block.sdpa_backend == policy for block in model.backbone.transformer.blocks)


def test_invalid_policy_rejected(monkeypatch):
    monkeypatch.setenv('IMMUNE_SDPA_BACKEND', 'typo')
    with pytest.raises(ValueError, match='IMMUNE_SDPA_BACKEND'):
        RefinedModel(settings())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('variant', ['r20_refined_control', 'r26_refined_local_global', 'f011'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cuda_bf16_checkpoint_backend_and_gradients(monkeypatch, variant, dtype):
    if not torch.cuda.is_bf16_supported():
        pytest.skip('Native BF16 required')
    batch = ImmuneCollator('A4', max_length=256, full_capacity=48, fragment_capacity=40)(rows())
    mask = deterministic_mask(batch).cuda()
    batch = {k: v.cuda() for k, v in batch.items()}
    reference = None
    for policy in ('math', 'auto', 'efficient'):
        monkeypatch.setenv('IMMUNE_SDPA_BACKEND', policy)
        torch.manual_seed(42)
        factory = FactorialModel if variant == 'f011' else RefinedModel
        model = factory(settings(), variant).cuda().train()
        model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
        # Caller MATH flags must survive forward and backward; inner policy must
        # still select efficient during non-reentrant checkpoint recomputation.
        with sdpa_kernel(SDPBackend.MATH):
            before = flags()
            torch.manual_seed(84)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                    loss = model(**batch, token_t=mask)['loss']
                loss.backward()
            assert flags() == before
        names = {event.key for event in prof.key_averages()}
        if policy != 'math':
            assert 'aten::_scaled_dot_product_efficient_attention' in names
            assert 'aten::_scaled_dot_product_efficient_attention_backward' in names
        gradients = {name: p.grad.detach().float().cpu() for name, p in model.named_parameters()
                     if p.grad is not None}
        assert gradients['chain_bias'].abs().sum() > 0
        assert all(torch.isfinite(g).all() for g in gradients.values())
        if reference is None:
            reference = (loss.detach().cpu(), gradients)
        else:
            tolerance = 2e-3 if dtype == torch.bfloat16 else 2e-6
            torch.testing.assert_close(loss.detach().cpu(), reference[0], atol=tolerance, rtol=tolerance)
            assert gradients.keys() == reference[1].keys()
            for name, grad in gradients.items():
                expected = reference[1][name]
                # BF16 fused and math kernels have different summation orders.
                relative = (grad-expected).norm() / expected.norm().clamp_min(1e-8)
                # The small chain-bias reduction is cancellation sensitive in
                # BF16 (3.27% measured for this control). FP32 is independently
                # checked tightly to distinguish rounding from semantic errors.
                rtol, atol = (.05, 2e-5) if dtype == torch.bfloat16 else (1e-4, 2e-7)
                assert relative < rtol or (grad-expected).abs().max() < atol, (name, relative)
        del model, loss, gradients
