"""CPU regressions for reference-length pairing and committed diffusion states.

Run: python -m pytest scripts/tests/bioseq/test_pairing_generation_protocol.py -q
No pretrained weights or GPU are required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarTokenizer
from dllm.pipelines.qwen3_vl_arch import sampling_bioseq as sampling
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    BioSeqDiffusionTransformerConfig,
    BioSeqEncoderDiffusionModel,
    BioSeqNoEncoderDiffusionModel,
)
from downstream.grammar.common import antibody_pair_record
from downstream.grammar.light_chain_pairing import generate_for_batch
from downstream.grammar.masks import light_chain_generation_partial_mask
from examples.llada.protein_fusion_model import RemapCollator, decoder_mask_token_id


@pytest.mark.parametrize("max_iter", [1, 4, 16])
def test_next_forward_only_reads_committed_tokens(monkeypatch, max_iter):
    tokenizer = GrammarTokenizer()
    batch = GrammarBioSeqCollator(tokenizer)([
        antibody_pair_record("ACDEFG", "DIQMTQSPSS"),
        antibody_pair_record("ACDE", "DIQMTQ"),
    ])
    model = BioSeqNoEncoderDiffusionModel(BioSeqDiffusionTransformerConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
        mask_token_id=tokenizer.mask_token_id,
    )).eval()
    partial = light_chain_generation_partial_mask(batch, tokenizer, prompt_residues=3)
    generation = sampling.build_generation_mask(batch, partial)
    inputs = []

    def fake_logits(**kwargs):
        tokens = kwargs["output_tokens"]
        inputs.append(tokens.clone())
        logits = torch.full((*tokens.shape, tokenizer.vocab_size), -100.0)
        # Distinct confidence per position; a different proposal each iteration
        # also checks that previously committed residues are never rewritten.
        residue_id = tokenizer.encode_residues("ACDEFGHIKLMNPQRSTVWY")[len(inputs) % 19]
        logits[..., residue_id] = torch.arange(tokens.size(1)).float() / tokens.size(1)
        return logits

    monkeypatch.setattr(sampling, "_model_logits", fake_logits)
    output, _, history = sampling.generate_bioseq(
        model, batch, partial_mask=partial,
        config=sampling.BioSeqGenerateConfig(max_iter=max_iter, sampling_strategy="argmax"),
        return_history=True,
    )
    assert len(history) == len(inputs) + 1
    for step, state in enumerate(history):
        pending = state.eq(tokenizer.mask_token_id) & generation
        expected = (generation.sum(dim=1).float() * (1.0 - step / max_iter)).long()
        assert torch.equal(pending.sum(dim=1), expected)
        assert torch.equal(state[~generation], batch["input_ids"][~generation])
        if step < len(inputs):
            assert torch.equal(inputs[step], state)
        if step:
            committed = generation & history[step - 1].ne(tokenizer.mask_token_id)
            assert torch.equal(state[committed], history[step - 1][committed])
    assert torch.equal(history[-1], output)
    assert not output[generation].eq(tokenizer.mask_token_id).any()


@pytest.mark.parametrize("prompt", [0, 3])
def test_reference_suffix_cannot_change_model_inputs(monkeypatch, prompt):
    import downstream.grammar.light_chain_pairing as pairing

    tokenizer = GrammarTokenizer()
    captured = []

    class RecordingFusion(BioSeqEncoderDiffusionModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(
                encoder_mask_token_id=tokenizer.mask_token_id,
                forbidden_target_token_ids=(),
            )

        def _denoise(self, **kwargs):
            # Capture the actual model boundary, including ESMC-vs-LLaDA ID
            # spaces, rather than only reconstructing the intended tensors.
            captured.append({key: value.clone() for key, value in kwargs.items()
                             if torch.is_tensor(value)})
            return SimpleNamespace(logits=torch.zeros((*kwargs["input_ids"].shape, 256)))

    model = RecordingFusion()

    def capture_generate(model, batch, *, partial_mask, **kwargs):
        generation = sampling.build_generation_mask(batch, partial_mask)
        decoder_mask_id = 200
        decoder = batch["input_ids"].masked_fill(generation, decoder_mask_id)
        sampling._model_logits(
            model=model, batch=batch, output_tokens=decoder,
            generation_mask=generation, mask_token_id=decoder_mask_id,
            timesteps=torch.ones(1), cfg_scale=0.0, partial_mask=partial_mask,
        )
        assert captured[-1]["input_ids"][generation].eq(decoder_mask_id).all()
        assert captured[-1]["encoder_input_ids"][:, 1, 1 + prompt:11].eq(tokenizer.mask_token_id).all()
        assert generation.sum(dim=1).tolist() == [10 - prompt, 10 - prompt]
        # A real generation would fill the seven masked sites. This fixture
        # only exercises input construction and parsing, not model quality.
        return batch["input_ids"] - 100, torch.zeros_like(batch["input_ids"], dtype=torch.float)

    monkeypatch.setattr(pairing, "run_grammar_generate", capture_generate)
    for light in ("DIQMTQSPSS", "DIQAAAAAAA" if prompt else "AAAAAAAAAA"):
        rows = generate_for_batch(
            [("ACDEFG", light, {})], model=model,
            collator=RemapCollator(GrammarBioSeqCollator(tokenizer), torch.arange(tokenizer.vocab_size) + 100),
            tokenizer=tokenizer,
            device=torch.device("cpu"), num_seqs=2, max_iter=4,
            sampling_strategy="argmax", temperature=1.0, light_prompt_tokens=prompt,
            light_length_mode="reference", length_prior=None,
        )
        assert len(rows) == 2
        assert all(row["target_light_length"] == len(light) for row in rows)
        assert all(row["raw_l_sequence"] == light for row in rows)
    for key in captured[0]:
        assert torch.equal(captured[0][key], captured[1][key]), key


@pytest.mark.parametrize("light", ["", "D", "DIQ"])
def test_reference_requires_a_generated_suffix(light):
    with pytest.raises(ValueError, match="beyond the declared prompt"):
        generate_for_batch(
            [("ACDEFG", light, {})], model=None, collator=None, tokenizer=None,
            device=torch.device("cpu"), num_seqs=1, max_iter=4,
            sampling_strategy="argmax", temperature=1.0, light_prompt_tokens=3,
            light_length_mode="reference", length_prior=None,
        )


@pytest.mark.parametrize("lookup,expected", [(None, 99), (0, 99), (32, 32)])
def test_shared_decoder_mask_id_preserves_fallback(lookup, expected):
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: lookup, unk_token_id=0, mask_token_id=99)
    assert decoder_mask_token_id(tokenizer) == expected


@pytest.mark.parametrize("prompt", [0, 3])
@pytest.mark.parametrize("scale", [0.0, 1.0, 1.5])
def test_cfg_preserves_structure_and_masks_both_condition_streams(prompt, scale):
    tokenizer = GrammarTokenizer()
    batch = GrammarBioSeqCollator(tokenizer)([
        antibody_pair_record("ACDEFG", "DIQMTQSPSS"),
        antibody_pair_record("ACDE", "DIQMTQ"),
    ])
    partial = light_chain_generation_partial_mask(batch, tokenizer, prompt_residues=prompt)
    generation = sampling.build_generation_mask(batch, partial)
    decoder_mask = 200
    tokens = batch["input_ids"].masked_fill(generation, decoder_mask)
    # Previously committed target tokens must also survive the unconditioned pass.
    tokens[0, generation[0].nonzero()[0, 0]] = tokenizer.encode_residues("W")[0]
    calls = []

    class RecordingFusion(BioSeqEncoderDiffusionModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(encoder_mask_token_id=tokenizer.mask_token_id,
                                          forbidden_target_token_ids=())

        def _denoise(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(logits=torch.full((*tokens.shape, 256), 2.0 if len(calls) == 1 else 1.0))

    logits = sampling._model_logits(
        model=RecordingFusion(), batch=batch, output_tokens=tokens,
        generation_mask=generation, mask_token_id=decoder_mask,
        timesteps=torch.ones(1), cfg_scale=scale, partial_mask=partial,
    )
    assert len(calls) == (2 if scale else 1)
    assert torch.allclose(logits, torch.full_like(logits, 2.0 + scale))
    assert torch.equal(calls[0]["input_ids"], tokens)
    committed_encoder_slot = (0, 1, 1 + prompt)
    assert calls[0]["encoder_input_ids"][committed_encoder_slot] == tokenizer.encode_residues("W")[0]
    if scale:
        condition = partial & batch["residue_mask"].bool()
        assert calls[1]["input_ids"][condition].eq(decoder_mask).all()
        assert torch.equal(calls[1]["input_ids"][~condition], tokens[~condition])
        emask = batch["encoder_residue_mask"].bool()
        expected = calls[0]["encoder_input_ids"].masked_fill(emask, tokenizer.mask_token_id)
        expected[committed_encoder_slot] = tokenizer.encode_residues("W")[0]
        assert torch.equal(calls[1]["encoder_input_ids"], expected)
        assert torch.equal(calls[1]["encoder_input_ids"][~emask], calls[0]["encoder_input_ids"][~emask])
        for key in ("attention_mask", "position_ids_inner", "position_ids_chain", "encoder_attention_mask"):
            assert torch.equal(calls[0][key], calls[1][key])


@pytest.mark.parametrize("prompt", [0, 3])
@pytest.mark.parametrize("scale", [0.0, 1.5])
def test_fusion_encoder_tracks_generated_state_without_reference_leakage(prompt, scale):
    """Exercise actual sampler/forward boundaries, remapped IDs and padded batches."""
    tokenizer = GrammarTokenizer()
    decoder_mask = 200
    residue_grammar_ids = torch.tensor(tokenizer.encode_residues("ACDEFGHIKLMNPQRSTVWY"))
    lookup = torch.arange(tokenizer.vocab_size) + 100
    lookup[tokenizer.mask_token_id] = decoder_mask
    collator = RemapCollator(GrammarBioSeqCollator(tokenizer), lookup)

    class RecordingFusion(BioSeqEncoderDiffusionModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(
                encoder_mask_token_id=tokenizer.mask_token_id,
                mask_token_id=decoder_mask, time_epsilon=1e-3,
                forbidden_target_token_ids=(),
            )
            inverse = torch.full((256,), -1, dtype=torch.long)
            inverse[lookup] = torch.arange(tokenizer.vocab_size)
            self.register_buffer("llada_to_grammar_ids", inverse)
            self.register_buffer("_residue_token_ids", lookup[residue_grammar_ids])
            self.calls = []

        def _denoise(self, **kwargs):
            self.calls.append({k: v.clone() for k, v in kwargs.items() if torch.is_tensor(v)})
            logits = torch.full((*kwargs["input_ids"].shape, 256), -100.0)
            # Change proposals between steps; committed residues must survive.
            step = (len(self.calls) - 1) // (2 if scale else 1)
            logits[..., self._residue_token_ids[step % len(residue_grammar_ids)]] = torch.arange(
                kwargs["input_ids"].size(1), dtype=torch.float32
            ) / kwargs["input_ids"].size(1)
            return SimpleNamespace(logits=logits)

    captured_runs = []
    for lights in (("DIQMTQSPSS", "DIQMTQ"),
                   ("DIQAAAAAAA", "DIQAAA") if prompt else ("AAAAAAAAAA", "AAAAAA")):
        # Intentionally leave ground truth in encoder_input_ids to ensure the
        # sampler itself never restores it, even beyond pairing's placeholders.
        batch = collator([antibody_pair_record(h, l) for h, l in zip(("ACDEFG", "ACDE"), lights)])
        original_encoder = batch["encoder_input_ids"].clone()
        partial = light_chain_generation_partial_mask(batch, tokenizer, prompt_residues=prompt)
        generation = sampling.build_generation_mask(batch, partial)
        model = RecordingFusion()
        output, _, history = sampling.generate_bioseq(
            model, batch, partial_mask=partial,
            config=sampling.BioSeqGenerateConfig(max_iter=4, sampling_strategy="argmax", cfg_scale=scale),
            return_history=True,
        )
        cond_calls = model.calls[::2] if scale else model.calls
        assert len(cond_calls) == 4
        for step, call in enumerate(cond_calls):
            assert torch.equal(call["input_ids"], history[step])
            for row, light in enumerate(lights):
                target_dec = history[step][row, generation[row]]
                expected = torch.where(target_dec.eq(decoder_mask), tokenizer.mask_token_id,
                                       model.llada_to_grammar_ids[target_dec])
                actual = call["encoder_input_ids"][row, 1, 1 + prompt:1 + len(light)]
                assert torch.equal(actual, expected)
                if scale:
                    assert torch.equal(model.calls[2 * step + 1]["encoder_input_ids"][
                        row, 1, 1 + prompt:1 + len(light)], expected)
                assert torch.equal(call["encoder_input_ids"][row, 0], original_encoder[row, 0])
                assert torch.equal(call["encoder_input_ids"][row, 1, :1 + prompt],
                                   original_encoder[row, 1, :1 + prompt])
                assert torch.equal(call["encoder_input_ids"][row, 1, 1 + len(light):],
                                   original_encoder[row, 1, 1 + len(light):])
        assert torch.equal(batch["encoder_input_ids"], original_encoder)
        assert not output[generation].eq(decoder_mask).any()
        captured_runs.append(model.calls)
    for first, changed_reference in zip(*captured_runs):
        for key in first:
            assert torch.equal(first[key], changed_reference[key]), key


def test_encoder_feedback_proxy_mapping_and_unsupported_tokens():
    """Never send LLaDA IDs or grammar delimiters into ESMC residue slots."""
    inverse = torch.full((256,), -1, dtype=torch.long)
    inverse[105] = 5
    inverse[150] = 50  # A grammar delimiter, not an ESMC residue.
    model = SimpleNamespace(llada_to_grammar_ids=inverse, _residue_token_ids=torch.tensor([105]))
    batch = {
        "encoder_input_ids": torch.tensor([[[0, 7, 8, 9, 10, 11, 12, 2, 1]]]),
        "encoder_position_ids": torch.tensor([[-1, 1, 2, 3, 4, 5, 6, -1]]),
        "residue_mask": torch.tensor([[False, True, True, True, True, True, True, False]]),
    }
    original = batch["encoder_input_ids"].clone()
    # Known residue, pending mask, unmapped ID, out-of-range ID, delimiter;
    # the final residue is visible context and must not be overwritten.
    tokens = torch.tensor([[150, 105, 200, 222, 999, 150, 105, 150]])
    generation = torch.tensor([[False, True, True, True, True, True, False, False]])
    actual = sampling._encoder_input_from_output_tokens(model, batch, tokens, generation, 200, 32)
    assert actual.tolist() == [[[0, 5, 32, 32, 32, 32, 12, 2, 1]]]
    assert torch.equal(batch["encoder_input_ids"], original)
