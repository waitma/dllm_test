"""Tests for the grammar-v2 BioSeq training representation.

Run with::

    pytest scripts/tests/bioseq/test_qwen3_vl_grammar.py -q
"""

from __future__ import annotations

import random
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
from dllm.pipelines.qwen3_vl_arch.data.esm_encoding import HuggingFaceEsmTokenizerAdapter
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    BioSeqDiffusionTransformerConfig,
    BioSeqEncoderDiffusionModel,
    BioSeqNoEncoderDiffusionModel,
    apply_decoder_corruption_to_encoder,
    sample_bioseq_diffusion_noise,
)


def rendered_tokens(record: BioSeqRecord, *, rng: random.Random | None = None) -> tuple[list[str], dict]:
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    row = GrammarRenderer(tokenizer, rng=rng).encode(record)
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

    antibody_tokens, antibody_row = rendered_tokens(antibody)
    tcr_tokens, _ = rendered_tokens(tcr)

    assert antibody_tokens == [
        "<prots>",
        "<ab>",
        "H",
        "H",
        "H",
        ".",
        "L",
        "L",
        "L",
        "<protd>",
    ]
    assert antibody_row["fixed_context_mask"][1] == 1
    assert antibody_row["fixed_context_mask"][0] == 0
    assert all(not value for value in antibody_row["fixed_context_mask"][2:])

    assert tcr_tokens == [
        "<prots>",
        "<tcr>",
        "A",
        "A",
        "A",
        ".",
        "B",
        "B",
        "B",
        "<protd>",
    ]


def test_nanobody_uses_nb_marker_inside_prots() -> None:
    nanobody = BioSeqRecord(
        chains=[BioSeqChain("VHHVHH", "nanobody_vhh")],
        task_type="nanobody",
        source="unit",
    )

    tokens, row = rendered_tokens(nanobody)

    assert tokens == ["<prots>", "<nb>", "V", "H", "H", "V", "H", "H", "<protd>"]
    assert row["fixed_context_mask"][1] == 1
    assert row["fixed_context_mask"][0] == 0


def test_antibody_pair_keeps_both_chains_in_one_prots_block() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("AAA", "antibody_heavy"),
            BioSeqChain("CCC", "antibody_light"),
        ],
        task_type="antibody",
        source="oas_paired",
    )

    tokens, _ = rendered_tokens(record)

    assert tokens == [
        "<prots>",
        "<ab>",
        "A",
        "A",
        "A",
        ".",
        "C",
        "C",
        "C",
        "<protd>",
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
        labels={"relation": "binding"},
    )

    tcr_tokens, tcr_row = rendered_tokens(tcr)
    ppi_tokens, ppi_row = rendered_tokens(ppi)

    assert tcr_tokens == [
        "<prots>",
        "<pep>",
        "P",
        "E",
        "P",
        "<protd>",
        "<binding>",
        "<prots>",
        "<tcr>",
        "A",
        "A",
        "A",
        ".",
        "B",
        "B",
        "B",
        "<protd>",
    ]
    assert all(tcr_row["fixed_context_mask"][:7])
    assert tcr_row["fixed_context_mask"][7] == 0
    assert tcr_row["fixed_context_mask"][8] == 1
    assert not any(tcr_row["fixed_context_mask"][9:])

    assert ppi_tokens == [
        "<prots>",
        "A",
        "A",
        "A",
        "A",
        "<protd>",
        "<binding>",
        "<prots>",
        "C",
        "C",
        "C",
        "C",
        "<protd>",
    ]
    assert ppi_row["grammar_name"] == "ppi_conditional"
    assert all(ppi_row["fixed_context_mask"][:7])
    assert not any(ppi_row["fixed_context_mask"][7:])


def test_antigen_antibody_fixes_antigen_and_binding() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("ANT", "antigen"),
            BioSeqChain("HHH", "antibody_heavy"),
            BioSeqChain("LLL", "antibody_light"),
        ],
        task_type="antibody_antigen",
        source="unit",
        labels={"relation": "binding"},
    )

    tokens, row = rendered_tokens(record)

    assert tokens == [
        "<prots>",
        "A",
        "N",
        "T",
        "<protd>",
        "<binding>",
        "<prots>",
        "<ab>",
        "H",
        "H",
        "H",
        ".",
        "L",
        "L",
        "L",
        "<protd>",
    ]
    assert all(row["fixed_context_mask"][:6])
    assert row["fixed_context_mask"][6] == 0
    assert row["fixed_context_mask"][7] == 1
    assert not any(row["fixed_context_mask"][8:])


def test_nanobody_antigen_uses_nb_marker_in_receptor_block() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("ANT", "antigen"),
            BioSeqChain("VHHVHH", "nanobody_vhh"),
        ],
        task_type="nanobody_antigen",
        source="unit",
        labels={"relation": "binding"},
    )

    tokens, row = rendered_tokens(record)

    assert tokens == [
        "<prots>",
        "A",
        "N",
        "T",
        "<protd>",
        "<binding>",
        "<prots>",
        "<nb>",
        "V",
        "H",
        "H",
        "V",
        "H",
        "H",
        "<protd>",
    ]
    assert row["grammar_name"] == "antigen_nanobody"
    assert row["fixed_context_mask"][7] == 1


def test_esmc_hf_tokenizer_supports_chain_separator() -> None:
    adapter = HuggingFaceEsmTokenizerAdapter.from_pretrained(
        "/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-300M"
    )
    tokenizer = GrammarTokenizer(adapter)
    assert tokenizer.chain_separator_id() == adapter.token_id(".")
    record = BioSeqRecord(
        chains=[
            BioSeqChain("HHH", "antibody_heavy"),
            BioSeqChain("LLL", "antibody_light"),
        ],
        task_type="antibody",
        source="unit",
    )
    batch = GrammarBioSeqCollator(tokenizer)([record])
    assert tokenizer.chain_separator_id() in batch["input_ids"][0].tolist()


def test_multi_chain_positions_use_distinct_chain_indices() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("HHH", "antibody_heavy"),
            BioSeqChain("LLL", "antibody_light"),
        ],
        task_type="antibody",
        source="unit",
    )
    _, row = rendered_tokens(record)
    chain_ids = row["position_ids_chain"]
    residue_chain_ids = [chain_ids[index] for index, class_id in enumerate(row["token_class_ids"]) if class_id == 1]
    assert residue_chain_ids == [0, 0, 0, 1, 1, 1]


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(33, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=hidden)


def test_encoder_uses_per_chain_inputs() -> None:
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
    assert batch["encoder_input_ids"].shape[1] == 2
    assert batch["chain_ids"] is not None
    assert "encoder_position_ids" not in batch

    corruption = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
    heavy_start = batch["position_ids_chain"][0].tolist().index(0)
    corruption[0, heavy_start] = True
    noised_encoder = apply_decoder_corruption_to_encoder(
        batch,
        corruption_mask=corruption,
        mask_token_id=tokenizer.mask_token_id,
    )
    assert noised_encoder[0, 0, 1].item() == tokenizer.mask_token_id

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
    model = BioSeqEncoderDiffusionModel(config, encoder=TinyEncoder(hidden_size=16))
    output = model.compute_loss(batch)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.encoder_condition is not None


def test_no_encoder_grammar_batch_runs() -> None:
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
    assert model.decoder.chain_position_embeddings.weight.grad is not None


def test_diffusion_respects_fixed_context() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("ANT", "antigen"),
            BioSeqChain("HHH", "antibody_heavy"),
            BioSeqChain("LLL", "antibody_light"),
        ],
        task_type="antibody_antigen",
        source="unit",
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
    assert (corruption & batch["residue_mask"]).any()
