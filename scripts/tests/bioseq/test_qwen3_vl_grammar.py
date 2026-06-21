"""Tests for the grammar-v1 BioSeq training representation.

Run with::

    pytest scripts/tests/bioseq/test_qwen3_vl_grammar.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from dllm.pipelines.qwen3_vl_arch.data import (
    BioSeqChain,
    BioSeqRecord,
    Esm2SequenceTokenizer,
    GrammarBioSeqCollator,
    GrammarRenderer,
    GrammarTokenizer,
)
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    BioSeqDiffusionTransformerConfig,
    BioSeqEncoderDiffusionModel,
    BioSeqNoEncoderDiffusionModel,
    apply_decoder_corruption_to_encoder,
    sample_bioseq_diffusion_noise,
)


def rendered_tokens(record: BioSeqRecord) -> tuple[list[str], dict]:
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    row = GrammarRenderer(tokenizer).encode(record)
    return tokenizer.decode_tokens(row["input_ids"]), row


def test_grammar_renders_oas_and_ots_in_canonical_chain_order() -> None:
    antibody = BioSeqRecord(
        chains=[
            BioSeqChain("HHH", "antibody_heavy"),
            BioSeqChain("LLL", "antibody_light"),
        ],
        task_type="antibody",
        source="oas_paired",
    )
    tcr = BioSeqRecord(
        chains=[
            BioSeqChain("BBB", "tcr_beta"),
            BioSeqChain("AAA", "tcr_alpha"),
        ],
        task_type="tcr",
        source="ots_paired",
    )

    antibody_tokens, _ = rendered_tokens(antibody)
    tcr_tokens, _ = rendered_tokens(tcr)

    assert antibody_tokens == [
        "<generate>",
        "<proas>",
        "H",
        "H",
        "H",
        "<proae>",
        "<binding>",
        "<probs>",
        "L",
        "L",
        "L",
        "<probd>",
    ]
    assert tcr_tokens == [
        "<generate>",
        "<proas>",
        "A",
        "A",
        "A",
        "<proae>",
        "<binding>",
        "<probs>",
        "B",
        "B",
        "B",
        "<probd>",
    ]


def test_antibody_pair_keeps_both_chains_when_oas_labels_both_as_heavy() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("AAA", "antibody_heavy"),
            BioSeqChain("CCC", "antibody_heavy"),
        ],
        task_type="antibody",
        source="oas_paired",
    )

    tokens, _ = rendered_tokens(record)

    assert tokens == [
        "<generate>",
        "<proas>",
        "A",
        "A",
        "A",
        "<proae>",
        "<binding>",
        "<probs>",
        "C",
        "C",
        "C",
        "<probd>",
    ]


def test_tcr_pair_keeps_both_chains_when_ots_labels_both_as_beta() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("BBB", "tcr_beta"),
            BioSeqChain("CCC", "tcr_beta"),
        ],
        task_type="tcr",
        source="ots_paired",
    )

    tokens, _ = rendered_tokens(record)

    assert tokens == [
        "<generate>",
        "<proas>",
        "C",
        "C",
        "C",
        "<proae>",
        "<binding>",
        "<probs>",
        "B",
        "B",
        "B",
        "<probd>",
    ]


def test_grammar_renders_tcr_peptide_and_ppi() -> None:
    tcr = BioSeqRecord(
        chains=[
            BioSeqChain("AAA", "tcr_alpha"),
            BioSeqChain("BBB", "tcr_beta"),
            BioSeqChain("PEP", "antigen"),
        ],
        task_type="tcr_epitope",
        source="vdjdb",
    )
    ppi = BioSeqRecord(
        chains=[
            BioSeqChain("AAAA", "protein_a"),
            BioSeqChain("CCCC", "protein_b"),
        ],
        task_type="ppi",
        source="string_ppi",
    )

    tcr_tokens, _ = rendered_tokens(tcr)
    ppi_tokens, _ = rendered_tokens(ppi)

    assert tcr_tokens[:5] == ["<peptides>", "P", "E", "P", "<peptided>"]
    assert tcr_tokens[5:] == [
        "<generate>",
        "<proas>",
        "A",
        "A",
        "A",
        "<proae>",
        "<binding>",
        "<probs>",
        "B",
        "B",
        "B",
        "<probd>",
    ]
    assert ppi_tokens == [
        "<protas>",
        "A",
        "A",
        "A",
        "A",
        "<protad>",
        "<binding>",
        "<protbs>",
        "C",
        "C",
        "C",
        "C",
        "<protbd>",
    ]


def test_only_fix_span_is_protected_from_diffusion() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("ANT", "antigen"),
            BioSeqChain("HHH", "antibody_heavy"),
            BioSeqChain("LLL", "antibody_light"),
        ],
        task_type="antibody_antigen",
        source="unit",
    )
    tokens, row = rendered_tokens(record)
    fixed_positions = [index for index, value in enumerate(row["fixed_context_mask"]) if value]

    assert [tokens[index] for index in fixed_positions] == ["<fixs>", "A", "N", "T", "<fixd>"]
    assert all(
        bool(fixed) != bool(eligible)
        for fixed, eligible in zip(row["fixed_context_mask"], row["diffusion_eligible_mask"])
    )

    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    batch = GrammarBioSeqCollator(tokenizer)([record])
    torch.manual_seed(0)
    _, labels, corruption, _ = sample_bioseq_diffusion_noise(
        batch,
        mask_token_id=tokenizer.mask_token_id,
        time_epsilon=0.999,
    )
    assert not (corruption & batch["fixed_context_mask"]).any()
    assert (labels.ne(-100) == corruption).all()
    assert (corruption & batch["structure_token_mask"]).any()
    assert (corruption & batch["relation_token_mask"]).any()


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(33, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=hidden)


def test_encoder_proxy_uses_current_noisy_grammar_stream() -> None:
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    record = BioSeqRecord(
        chains=[
            BioSeqChain("HHH", "antibody_heavy"),
            BioSeqChain("LLL", "antibody_light"),
        ],
        task_type="antibody",
        source="unit",
    )
    batch = GrammarBioSeqCollator(tokenizer)([record])
    corruption = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
    corruption[0, 0] = True
    corruption[0, 2] = True
    noised_encoder = apply_decoder_corruption_to_encoder(
        batch,
        corruption_mask=corruption,
        mask_token_id=tokenizer.mask_token_id,
    )

    assert noised_encoder[0, 0, 0].item() == tokenizer.mask_token_id
    assert noised_encoder[0, 0, 2].item() == tokenizer.mask_token_id
    assert batch["encoder_input_ids"][0, 0, 0].item() == 31

    config = BioSeqDiffusionTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        dropout=0.0,
        max_position_embeddings=128,
        mask_token_id=tokenizer.mask_token_id,
    )
    model = BioSeqEncoderDiffusionModel(config, encoder=TinyEncoder())
    output = model.compute_loss(batch)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.encoder_condition is not None
    assert output.encoder_condition.shape[:2] == batch["input_ids"].shape


def test_grammar_batch_uses_generic_embeddings_without_chain_boundary_leakage() -> None:
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    record = BioSeqRecord(
        chains=[
            BioSeqChain("HHH", "antibody_heavy"),
            BioSeqChain("LLL", "antibody_light"),
        ],
        task_type="antibody",
        source="unit",
    )
    batch = GrammarBioSeqCollator(tokenizer)([record])
    active = batch["attention_mask"]

    assert batch["chain_role_ids"][active].unique().tolist() == [0]
    assert batch["position_ids_chain"][active].unique().tolist() == [0]
    assert batch["position_ids_chain"][~active].unique().tolist() in ([], [-1])

    config = BioSeqDiffusionTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        dropout=0.0,
        max_position_embeddings=128,
        mask_token_id=tokenizer.mask_token_id,
    )
    model = BioSeqNoEncoderDiffusionModel(config)
    output = model.compute_loss(batch)
    assert output.loss is not None
    output.loss.backward()

    assert model.decoder.chain_role_embeddings.weight.grad is not None
    assert model.decoder.chain_position_embeddings.weight.grad is not None
