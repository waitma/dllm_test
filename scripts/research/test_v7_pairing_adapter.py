"""Check new pairing decoding invariants on a tiny frozen model and artificial text.

Activate conda pllm; run with the worker-compatible protenix_abtcr Python and
CUDA_VISIBLE_DEVICES='': python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_v7_pairing_adapter.py
No real sequences, production weights, annotation tools, GPU, or checkpoints.
"""
from pathlib import Path
import json
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
SOURCE = ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/source_snapshot_v1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))
from downstream.v7_pairing import pairing_input, pairing_decode, light_result
from downstream.v7_native import input_fingerprint
from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
import torch


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Invalid generation request was accepted')


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = FactorialModel(ImmuneConfig(d_model=24, n_layers=2, n_heads=3,
        mlp_hidden_size=48, max_length=128, full_capacity=24, fragment_capacity=12), 'f011').eval()
    assert Path(sys.modules[type(model).__module__].__file__).resolve().is_relative_to(SOURCE)
    # Only the declared length/prefix enter the light input, never reference suffixes.
    prefix_cases = 0
    for prefix in ('', 'ACD'):
        inputs, target, light = pairing_input(model, 'ACDEFGHIK', 11, prefix=prefix)
        assert int(target.sum()) == 11-len(prefix) and int(light.sum()) == 11
        assert torch.equal(inputs['input_ids'][light & ~target],
                           torch.tensor(model.tokenizer.encode_residues(prefix)))
        before = input_fingerprint(inputs)
        calls = []
        original = model.denoise
        def capture(**actual):
            calls.append({k: v.clone() for k, v in actual.items()})
            return original(**actual)
        model.denoise = capture
        rng_before = torch.random.get_rng_state().clone()
        try:
            output, audit = pairing_decode(model, inputs, target, max_iter=16, seed=17)
        finally:
            model.denoise = original
        assert torch.equal(rng_before, torch.random.get_rng_state())
        assert input_fingerprint(inputs) == before and torch.equal(output[~target], inputs['input_ids'][~target])
        assert 1 <= len(calls) == audit['forward_calls'] <= 16
        committed = torch.zeros_like(target)
        for call, observation in zip(calls, audit['trace']):
            assert set(call) == set(inputs) and 'labels' not in call
            assert torch.equal(call['input_ids'][~target], inputs['input_ids'][~target])
            newly_visible = target & call['input_ids'].ne(model.tokenizer.mask_token_id)
            assert (committed <= newly_visible).all()
            assert torch.equal(call['input_ids'][newly_visible], output[newly_visible])
            assert int((target & ~newly_visible).sum()) == observation['pending_before']
            assert observation['pending_after'] <= observation['pending_before']
            assert input_fingerprint(call) == observation['actual_forward_input_sha256']
            committed = newly_visible
        again, _ = pairing_decode(model, inputs, target, max_iter=16, seed=17)
        assert torch.equal(output, again)
        result = light_result(model, output, light)
        assert result['known_light_length'] == 11 and result['repaired'] is False
        prefix_cases += 1
    # The one-step argmax path must produce the exact native logits' choice.
    inputs, target, light = pairing_input(model, 'ACDEFGHIK', 11)
    output, audit = pairing_decode(model, inputs, target, max_iter=1, sampling_strategy='argmax')
    with torch.inference_mode():
        expected, _ = model.denoise(**inputs)
    assert audit['forward_calls'] == 1 and torch.equal(output[target], expected.argmax(-1)[target])
    # Early delimiters remain in raw output and shorten the interpreted result.
    original = model.denoise
    def terminals(**features):
        logits = torch.full((*features['input_ids'].shape, model.tokenizer.vocab_size), -float('inf'))
        logits[..., model.terminal_id] = 0
        return logits, None
    model.denoise = terminals
    try:
        output, _ = pairing_decode(model, inputs, target, max_iter=4, sampling_strategy='argmax')
    finally:
        model.denoise = original
    result = light_result(model, output, light)
    assert result['sequence'] == '' and result['early_terminal_count'] == 11
    assert result['raw_light_token_ids'] == [model.terminal_id]*11 and result['repaired'] is False
    # Visible heavy conditions are retained and remain distinguishable.
    other, _, _ = pairing_input(model, 'WCDEFGHIK', 11)
    assert input_fingerprint(other) != input_fingerprint(inputs)
    bad = {**inputs, 'labels': inputs['input_ids'].clone()}
    revealed = {**inputs, 'input_ids': inputs['input_ids'].masked_fill(target, model.tokenizer.encode_residues('A')[0])}
    cases = [lambda: pairing_input(model, 'ACDEFGHIK', 24),
             lambda: pairing_input(model, 'ACDEFGHIK', 3, prefix='ACD'),
             lambda: pairing_input(model, 'ACDEFGHIK', 11, prefix='AC'),
             lambda: pairing_decode(model, bad, target),
             lambda: pairing_decode(model, revealed, target),
             lambda: pairing_decode(model, inputs, target, max_iter=0)]
    for case in cases:
        rejected(case)
    print(json.dumps(dict(status='passed',synthetic_p0_p3_cases=prefix_cases,
        actual_frozen_tiny_native_forward=True,committed_tokens_visible_pending_masked=True,
        visible_conditions_unchanged=True,caller_rng_restored=True,seed_reproducible=True,
        native_argmax_equivalence=True,early_terminal_retained_without_repair=True,
        rejected_invalid_requests=len(cases),corpus_read=False,production_weights_loaded=False,
        annotation_run=False,gpu_used=False,checkpoint_files_created=0)),flush=True)


if __name__ == '__main__':
    main()
