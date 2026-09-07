"""Smoke tests for grammar BioSeq iterative denoising.

Run with::

    pytest scripts/tests/bioseq/test_sampling_bioseq.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from dllm.pipelines.qwen3_vl_arch.data import (
    BioSeqChain,
    BioSeqRecord,
    GrammarBioSeqCollator,
    GrammarTokenizer,
    Esm2SequenceTokenizer,
)
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    BioSeqDiffusionTransformerConfig,
    BioSeqEncoderDiffusionModel,
    BioSeqNoEncoderDiffusionModel,
    apply_decoder_corruption_to_encoder,
)
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import (
    BioSeqGenerateConfig,
    build_generation_mask,
    generate_bioseq,
    resolve_partial_mask,
)
from downstream.grammar.masks import cdr_generation_partial_mask, light_chain_generation_partial_mask


def _tiny_config(tokenizer: GrammarTokenizer) -> BioSeqDiffusionTransformerConfig:
    return BioSeqDiffusionTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        dropout=0.0,
        max_position_embeddings=256,
        mask_token_id=tokenizer.mask_token_id,
        time_epsilon=0.75,
    )


class TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask.to(hidden.dtype).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden)


def _antibody_batch(tokenizer: GrammarTokenizer) -> dict[str, torch.Tensor]:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("QVQLVQSGAE", "antibody_heavy"),
            BioSeqChain("DIQMTQSPSS", "antibody_light"),
        ],
        task_type="antibody",
        source="unit",
    )
    return GrammarBioSeqCollator(tokenizer)([record])


def test_partial_mask_protects_structure_and_fixed_context() -> None:
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    batch = _antibody_batch(tokenizer)
    partial = resolve_partial_mask(batch, None)
    assert not (batch["structure_token_mask"] & ~partial).any()
    assert not (batch["relation_token_mask"] & ~partial).any()
    assert not (batch["fixed_context_mask"] & ~partial).any()


def test_light_chain_mask_only_targets_light_residues() -> None:
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    batch = _antibody_batch(tokenizer)
    partial = light_chain_generation_partial_mask(batch, tokenizer)
    generation = build_generation_mask(batch, partial)
    tokens = tokenizer.decode_tokens(batch["input_ids"][0].tolist())
    generated = [tokens[index] for index, active in enumerate(generation[0].tolist()) if active]
    assert generated
    assert all(token in {"D", "I", "Q", "M", "T", "S", "P"} for token in generated)


def test_no_encoder_generate_runs_and_unmasks() -> None:
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    batch = _antibody_batch(tokenizer)
    model = BioSeqNoEncoderDiffusionModel(_tiny_config(tokenizer)).eval()
    partial = light_chain_generation_partial_mask(batch, tokenizer)
    config = BioSeqGenerateConfig(max_iter=6, sampling_strategy="argmax")
    output_tokens, _, history = generate_bioseq(
        model,
        batch,
        partial_mask=partial,
        config=config,
        return_history=True,
    )
    generation = build_generation_mask(batch, partial)
    assert output_tokens.shape == batch["input_ids"].shape
    assert history[-1].eq(tokenizer.mask_token_id).sum() <= history[0].eq(tokenizer.mask_token_id).sum()
    assert output_tokens[generation].ne(tokenizer.mask_token_id).all()


def test_encoder_generate_runs_with_per_chain_inputs() -> None:
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    batch = _antibody_batch(tokenizer)
    config = _tiny_config(tokenizer)
    model = BioSeqEncoderDiffusionModel(
        config,
        encoder=TinyEncoder(tokenizer.vocab_size, hidden_size=config.hidden_size),
    ).eval()
    partial = light_chain_generation_partial_mask(batch, tokenizer)
    output_tokens, _ = generate_bioseq(
        model,
        batch,
        partial_mask=partial,
        config=BioSeqGenerateConfig(max_iter=4, sampling_strategy="argmax"),
    )
    assert output_tokens.shape == batch["input_ids"].shape


def test_encoder_never_sees_target_residues_during_decoding() -> None:
    """Regression: the ESMC conditioning stream must not leak the reference.

    ``apply_decoder_corruption_to_encoder`` re-exposes ``batch["encoder_input_ids"]``
    (built by the collator from the *clean* record) wherever the mirrored mask is
    unset. Mirroring only the still-masked positions therefore handed the decoder the
    ground-truth residue at every position iterative decoding had already committed,
    which turned mask-fill into copying. Inference must mirror the whole generation
    mask for the entire trajectory.
    """

    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    batch = _antibody_batch(tokenizer)
    partial = resolve_partial_mask(batch, light_chain_generation_partial_mask(batch, tokenizer))
    generation = build_generation_mask(batch, partial)
    mask_id = tokenizer.mask_token_id

    encoder_residue = batch["encoder_residue_mask"]
    light_row = encoder_residue[0, 1]
    reference_light = batch["encoder_input_ids"][0, 1][light_row]
    target_positions = generation[0].nonzero(as_tuple=True)[0]
    assert target_positions.numel() > 0

    output_tokens = batch["input_ids"].clone().masked_fill(generation, mask_id)
    filler = int(batch["input_ids"][0, target_positions[0]])

    for committed in (0, target_positions.numel() // 2, target_positions.numel()):
        tokens = output_tokens.clone()
        if committed:
            tokens[0, target_positions[:committed]] = filler
        noised = apply_decoder_corruption_to_encoder(
            batch=dict(batch, input_ids=tokens),
            corruption_mask=generation,
            mask_token_id=mask_id,
        )
        visible = noised[0, 1][light_row]
        # Every generated light residue must read <mask> on the encoder side.
        assert int((visible == mask_id).sum()) >= target_positions.numel(), (
            f"encoder exposes reference residues at {committed} committed positions"
        )
        leaked = int(((visible == reference_light) & (visible != mask_id)).sum())
        assert leaked <= reference_light.numel() - target_positions.numel(), (
            f"encoder leaked {leaked} reference residues at {committed} committed positions"
        )


def test_cdr_partial_mask_targets_span() -> None:
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    heavy = BioSeqChain(
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR",
        "antibody_heavy",
        regions={
            "FR1": "EVQLVESGGGLVQPGGSLRLSCAAS",
            "CDR1": "GFTFSSYA",
            "FR2": "MSWVRQAPGKGLEWVSA",
            "CDR2": "ISGSGGST",
            "FR3": "YYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
            "CDR3": "AR",
            "FR4": "",
        },
    )
    record = BioSeqRecord(chains=[heavy, BioSeqChain("DIQMTQSPSS", "antibody_light")], task_type="antibody", source="unit")
    batch = GrammarBioSeqCollator(tokenizer)([record])
    partial = cdr_generation_partial_mask(batch, tokenizer, heavy, "heavy", "CDR1")
    generation = build_generation_mask(batch, partial)
    assert generation.sum().item() == len(heavy.regions["CDR1"])
