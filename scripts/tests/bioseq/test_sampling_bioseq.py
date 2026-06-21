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


def test_encoder_generate_runs_with_proxy_stream() -> None:
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    batch = _antibody_batch(tokenizer)
    config = _tiny_config(tokenizer)
    model = BioSeqEncoderDiffusionModel(config, encoder=TinyEncoder(tokenizer.vocab_size)).eval()
    partial = light_chain_generation_partial_mask(batch, tokenizer)
    output_tokens, _ = generate_bioseq(
        model,
        batch,
        partial_mask=partial,
        config=BioSeqGenerateConfig(max_iter=4, sampling_strategy="argmax"),
    )
    assert output_tokens.shape == batch["input_ids"].shape


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
