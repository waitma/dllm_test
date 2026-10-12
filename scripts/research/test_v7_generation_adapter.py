"""CPU checks of the new V7 CDR adapter using tiny synthetic chains only.

Activate conda pllm; run with the protenix_abtcr python and CUDA_VISIBLE_DEVICES=''.
No benchmark/training data or production weights are read, and no GPU is used.
The tiny save/reload checkpoint lives in TemporaryDirectory and is removed.
"""
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
SOURCE = ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/source_snapshot_v1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))
from downstream.v7_native import cdr_input, cdr_argmax, input_fingerprint
from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig
import torch


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = FactorialModel(ImmuneConfig(d_model=24, n_layers=2, n_heads=3, mlp_hidden_size=48,
        max_length=128, full_capacity=24, fragment_capacity=12), 'f011').eval()
    assert Path(sys.modules[type(model).__module__].__file__).resolve().is_relative_to(SOURCE)
    counts = 0
    for heavy, light, role, start, end in [('ACDEFGHIK', 'LMNPQRSTV', 'antibody_heavy', 2, 5),
            ('ACDEFGHIK', 'LMNPQRSTV', 'antibody_light', 3, 6),
            ('ACDEFGHIK', '', 'antibody_heavy', 1, 4)]:
        features, mask = cdr_input(model, heavy, light, role, start, end)
        reference = heavy if role == 'antibody_heavy' else light
        altered = reference[:start]+'G'*(end-start)+reference[end:]
        h, l = (altered, light) if role == 'antibody_heavy' else (heavy, altered)
        other, other_mask = cdr_input(model, h, l, role, start, end)
        assert all(torch.equal(features[k], other[k]) for k in features) and torch.equal(mask, other_mask)
        assert int(mask.sum()) == end-start
        calls = []
        original = model.denoise
        def capture(**actual):
            calls.append({key: value.detach().clone() for key, value in actual.items()})
            return original(**actual)
        model.denoise = capture
        try:
            generated, audit = cdr_argmax(model, features, mask)
        finally:
            model.denoise = original
        assert len(calls) == 1 and set(calls[0]) == set(features)
        assert all(torch.equal(features[k], calls[0][k]) for k in features)
        assert torch.equal(generated[~mask], features['input_ids'][~mask])
        assert audit['actual_forward_input_sha256'] == input_fingerprint(calls[0])
        with torch.no_grad():
            expected, _ = model.denoise(**features)
        assert torch.equal(generated[mask], expected.argmax(-1)[mask])
        repeated, _ = cdr_argmax(model, other, other_mask)
        assert torch.equal(generated, repeated)
        assert model.allowed_target[generated[mask]].all()
        assert 'labels' not in calls[0] and 'residue_mask' not in calls[0] and 'terminal_mask' not in calls[0]
        counts += 1
    # Visible context is still a model input and is not accidentally all masked.
    a, mask = cdr_input(model, 'ACDEFGHIK', 'LMNPQRSTV', 'antibody_heavy', 2, 5)
    b, _ = cdr_input(model, 'ACDEFGHIK', 'AMNPQRSTV', 'antibody_heavy', 2, 5)
    assert not torch.equal(a['input_ids'], b['input_ids'])
    original = model.denoise
    def delimiter_proposal(**features):
        logits = torch.zeros((*features['input_ids'].shape, model.tokenizer.vocab_size))
        logits[:, :, model.terminal_id] = 100
        return logits, None
    model.denoise = delimiter_proposal
    try:
        output, _ = cdr_argmax(model, a, mask)
        assert (output[mask] == model.terminal_id).all() and output[mask].numel() == 3
    finally:
        model.denoise = original
    rejected = 0
    for start, end in [(-1, 3), (2, 2), (0, 20)]:
        try:
            cdr_input(model, 'ACDEFGHIK', '', 'antibody_heavy', start, end)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Invalid mask span accepted')
    with TemporaryDirectory(prefix='v7_generation_adapter_') as directory:
        model.save_pretrained(directory)
        restored = FactorialModel.from_pretrained(directory).eval()
        actual, _ = cdr_argmax(model, a, mask)
        reloaded, _ = cdr_argmax(restored, a, mask)
        assert torch.equal(actual, reloaded)
    assert not Path(directory).exists()
    print(json.dumps(dict(status='passed', synthetic_native_forward_cases=counts,
        actual_frozen_f011_forward=True, hidden_reference_perturbation_equal=True,
        actual_forward_whitelist_captured=True, output_equals_native_argmax=True,
        visible_context_preserved=True, predicted_delimiter_not_repaired_or_dropped=True,
        rejected_invalid_spans=rejected, tiny_save_reload_equal=True, temporary_checkpoints_removed=True,
        corpus_read=False, production_weights_loaded=False, gpu_used=False)), flush=True)


if __name__ == '__main__':
    main()
