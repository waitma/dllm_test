"""Activate pllm/protenix_abtcr; python -B -m pytest THIS_FILE -q."""
from pathlib import Path
import pytest
import torch

from downstream.fixed_chain_pairing import decode, protocol, validate_plan
from dllm.pipelines.immune_llada.fixed_chain import FixedChainConfig, FixedChainModel
from scripts.downstream.run_fixed_chain_pairing_reference import reference_input


def small_model():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    return FixedChainModel(FixedChainConfig(
        d_model=24, n_heads=3, n_layers=2, mlp_hidden_size=48,
        max_length=256, first_chain_capacity=40, second_chain_capacity=36,
        full_capacity=78)).eval()


def test_matches_validated_sampler_and_restores_rng():
    from scripts.downstream.debug_pairing_sampler import sample
    from scripts.downstream.run_post_training_pairing_reference import decode as post_decode
    assert post_decode is decode
    m = small_model()
    features, target, positions = reference_input(m, 'ACDEFG', 12)
    original = {k: v.clone() for k, v in features.items()}
    state = torch.random.get_rng_state().clone()
    actual = decode(m, features, target, positions, seed=42)
    assert torch.equal(state, torch.random.get_rng_state())
    expected = sample(m, features, target, 'legacy', 42)
    assert actual['raw_light_token_ids'][:12] == expected['raw_light_token_ids']
    assert (actual.get('sequence') or '') == expected['sequence']
    assert all(torch.equal(v, features[k]) for k, v in original.items())
    assert actual == decode(m, features, target, positions, seed=42)


def test_early_eos_does_not_leave_unfilled_target_positions(monkeypatch):
    from dllm.pipelines.qwen3_vl_arch import sampling_bioseq
    m = small_model()
    f, target, pos = reference_input(m, 'ACDEFG', 4)
    a = m.tokenizer.encode_residues('A')[0]
    proposed = f['input_ids'].clone()
    proposed[target] = a
    proposed[0, pos[1]] = m.tokenizer.eos_token_id
    confidence = torch.zeros_like(proposed, dtype=torch.float32)
    confidence[0, pos[:4]] = torch.tensor([.8, .9, .7, .6])
    monkeypatch.setattr(sampling_bioseq, '_sample_tokens', lambda *args: (proposed, confidence))
    result = decode(m, f, target, pos, seed=42, steps=4)
    assert result['steps'] == 4 and result['sequence'] == 'A' and result['early_eos']
    assert result['raw_light_token_ids'][:4] == [a, m.tokenizer.eos_token_id, a, a]
    assert result['raw_light_token_ids'][4] == m.tokenizer.eos_token_id
    assert set(result['raw_light_token_ids'][5:]) == {m.tokenizer.mask_token_id}


def test_old_or_unbound_plans_are_rejected():
    import downstream.fixed_chain_pairing as shared
    source = str(Path(shared.__file__).resolve())
    validate_plan({'protocol': protocol(), 'source_sha256': {source: 'checked-by-runner'}})
    for bad in [{}, {'protocol': {'name': 'fixed_chain_pairing_reference_length_v1'}},
                {'protocol': protocol(), 'source_sha256': {}}]:
        with pytest.raises(ValueError):
            validate_plan(bad)
    changed = protocol(); changed['confidence'] = 'raw_probability'
    with pytest.raises(ValueError):
        validate_plan({'protocol': changed, 'source_sha256': {source: 'x'}})
