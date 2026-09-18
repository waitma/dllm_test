"""Tests for the grammar-v2 BioSeq training representation.

Run with::

    pytest scripts/tests/bioseq/test_qwen3_vl_grammar.py -q
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dllm.pipelines.immune_llada.data import (
    BioSeqChain,
    BioSeqRecord,
    Esm2SequenceTokenizer,
    GrammarBioSeqCollator,
    GrammarRenderer,
    GrammarTokenizer,
)
from dllm.pipelines.immune_llada.data.esm_encoding import HuggingFaceEsmTokenizerAdapter
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


# Unconditional records open with a fixed, residue-free no-context block so their
# layout matches conditioned ones slot-for-slot. All four tokens are fixed
# context: attention only, no diffusion loss, no corruption.
_NULL_PREFIX = ["<prots>", "<null>", "<protd>", "<unknown>"]


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

    assert antibody_tokens == _NULL_PREFIX + [
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
    # The no-context prefix is fixed; everything the model must generate is not.
    assert antibody_row["fixed_context_mask"][: len(_NULL_PREFIX)] == [1, 1, 1, 1]
    assert all(not value for value in antibody_row["fixed_context_mask"][len(_NULL_PREFIX) :])

    # The receptor block encodes beta before alpha (heavy-analog first), matching
    # the antibody heavy-before-light layout.
    assert tcr_tokens == _NULL_PREFIX + [
        "<prots>",
        "<tcr>",
        "B",
        "B",
        "B",
        ".",
        "A",
        "A",
        "A",
        "<protd>",
    ]


def test_nanobody_uses_nb_marker_inside_prots() -> None:
    nanobody = BioSeqRecord(
        chains=[BioSeqChain("VHHVHH", "nanobody_vhh")],
        task_type="nanobody",
        source="unit",
    )

    tokens, row = rendered_tokens(nanobody)

    assert tokens == _NULL_PREFIX + ["<prots>", "<nb>", "V", "H", "H", "V", "H", "H", "<protd>"]
    assert row["fixed_context_mask"][: len(_NULL_PREFIX)] == [1, 1, 1, 1]
    assert all(not value for value in row["fixed_context_mask"][len(_NULL_PREFIX) :])


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

    assert tokens == _NULL_PREFIX + [
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
        "B",
        "B",
        "B",
        ".",
        "A",
        "A",
        "A",
        "<protd>",
    ]
    assert all(tcr_row["fixed_context_mask"][:7])
    assert tcr_row["fixed_context_mask"][7] == 0
    assert tcr_row["fixed_context_mask"][8] == 0
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


def test_catalysis_renders_target_fixed_actor_generated() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("TARGETSEQ", "protein_a"),
            BioSeqChain("ACTORSEQ", "protein_b"),
        ],
        task_type="ppi",
        source="unit",
        split="train",
        labels={"relation": "catalysis"},
    )
    tokens, row = rendered_tokens(record)
    assert tokens[:4] == ["<prots>", "T", "A", "R"]
    assert "<catalysis>" in tokens
    assert tokens.index("<catalysis>") < tokens.index("A", tokens.index("<catalysis>"))


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
    assert row["fixed_context_mask"][7] == 0
    assert not any(row["fixed_context_mask"][8:])


def test_tcr_pmhc_layout_and_fixed_masks() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("GSHSMRY", "mhc"),
            BioSeqChain("IQRTP", "hla"),
            BioSeqChain("SIIN", "peptide"),
            BioSeqChain("CAV", "tcr_alpha"),
            BioSeqChain("CASS", "tcr_beta"),
        ],
        task_type="tcr_pmhc",
        source="unit",
    )
    tokens, row = rendered_tokens(record)
    assert row["grammar_name"] == "tcr_pmhc"
    assert tokens == [
        "<prots>",
        "G",
        "S",
        "H",
        "S",
        "M",
        "R",
        "Y",
        ".",
        "I",
        "Q",
        "R",
        "T",
        "P",
        "<protd>",
        "<binding>",
        "<prots>",
        "<pep>",
        "S",
        "I",
        "I",
        "N",
        "<protd>",
        "<binding>",
        "<prots>",
        "<tcr>",
        "C",
        "A",
        "S",
        "S",
        ".",
        "C",
        "A",
        "V",
        "<protd>",
    ]
    # MHC block + first binding + peptide block + second binding are fixed.
    assert all(row["fixed_context_mask"][:24])
    assert row["fixed_context_mask"][24] == 0
    assert not any(row["fixed_context_mask"][25:])

    collator = GrammarBioSeqCollator(GrammarTokenizer(Esm2SequenceTokenizer()))
    batch = collator([record])
    assert batch["encoder_input_ids"].shape[1] == 5


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
    assert all(row["fixed_context_mask"][:6])
    assert row["fixed_context_mask"][6] == 0
    assert row["fixed_context_mask"][7] == 0
    assert not any(row["fixed_context_mask"][8:])


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


class NaNOnEmptyEncoder(nn.Module):
    """Mimics a real attention encoder: an all-masked row softmaxes to NaN."""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(33, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            denom = attention_mask.to(hidden.dtype).sum(dim=-1, keepdim=True).unsqueeze(-1)
            hidden = hidden / denom  # all-zero attention -> divide by zero -> NaN
        return SimpleNamespace(last_hidden_state=hidden)


def test_encoder_ragged_chain_counts_stay_finite() -> None:
    """Mixed chain counts pad empty chain rows; encoder must not leak NaN."""

    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    three_chain = BioSeqRecord(
        chains=[
            BioSeqChain("AAA", "tcr_alpha"),
            BioSeqChain("BBB", "tcr_beta"),
            BioSeqChain("PEP", "antigen"),
        ],
        task_type="tcr_epitope",
        source="unit",
    )
    two_chain = BioSeqRecord(
        chains=[
            BioSeqChain("AAA", "tcr_alpha"),
            BioSeqChain("BBB", "tcr_beta"),
        ],
        task_type="tcr",
        source="unit",
    )
    batch = GrammarBioSeqCollator(tokenizer)([three_chain, two_chain])
    assert batch["encoder_input_ids"].shape[1] >= 3
    # the 2-chain record has at least one padding chain row
    assert batch["encoder_chain_mask"].sum().item() < batch["encoder_chain_mask"].numel()
    # Root-cause invariant: NO encoder chain row may be fully masked. An all-zero
    # attention row would make the encoder softmax over -inf (NaN) and break ESM2
    # token-dropout (divide by attention_mask.sum()==0). Padding rows must still be
    # minimally valid (<cls><eos>), even though chain_mask zeroes their condition.
    per_row_attention = batch["encoder_attention_mask"].reshape(-1, batch["encoder_attention_mask"].shape[-1])
    assert per_row_attention.sum(dim=-1).min().item() >= 1
    padded_rows = ~batch["encoder_chain_mask"].reshape(-1)
    assert padded_rows.any()  # this batch does contain padding rows
    assert per_row_attention[padded_rows].sum(dim=-1).min().item() >= 1

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
    model = BioSeqEncoderDiffusionModel(config, encoder=NaNOnEmptyEncoder(hidden_size=16))
    output = model.compute_loss(batch)
    assert output.encoder_condition is not None
    assert torch.isfinite(output.encoder_condition).all()
    assert output.loss is not None and torch.isfinite(output.loss)


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


def _assert_decoder_encoder_residue_ids_match(batch: dict[str, torch.Tensor]) -> None:
    for batch_index in range(batch["input_ids"].shape[0]):
        for seq_index in range(batch["input_ids"].shape[1]):
            if not batch["residue_mask"][batch_index, seq_index]:
                continue
            chain_index = int(batch["chain_ids"][batch_index, seq_index].item())
            inner_index = int(batch["position_ids_inner"][batch_index, seq_index].item())
            if chain_index < 0 or inner_index < 0:
                continue
            encoder_residue_positions = torch.nonzero(
                batch["encoder_residue_mask"][batch_index, chain_index],
                as_tuple=False,
            ).flatten()
            assert inner_index < encoder_residue_positions.numel()
            encoder_position = int(encoder_residue_positions[inner_index].item())
            decoder_id = int(batch["input_ids"][batch_index, seq_index].item())
            encoder_id = int(batch["encoder_input_ids"][batch_index, chain_index, encoder_position].item())
            assert decoder_id == encoder_id


def test_per_chain_encoder_ids_match_decoder_for_esmc_adapter() -> None:
    esmc_dir = (
        "/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-300M"
    )
    if not __import__("pathlib").Path(esmc_dir, "tokenizer.json").is_file():
        import pytest

        pytest.skip("local ESMC-300M snapshot not available")

    tokenizer = GrammarTokenizer(HuggingFaceEsmTokenizerAdapter.from_pretrained(esmc_dir))
    record = BioSeqRecord(
        chains=[
            BioSeqChain("ACDEFGHIK", "antigen"),
            BioSeqChain("QVQLV", "antibody_heavy"),
            BioSeqChain("DIQMT", "antibody_light"),
        ],
        task_type="antibody_antigen",
        source="unit",
        labels={"relation": "binding"},
    )
    batch = GrammarBioSeqCollator(tokenizer)([record])
    assert batch["encoder_input_ids"].shape[1] == 3
    _assert_decoder_encoder_residue_ids_match(batch)


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


def test_record_within_max_protein_length() -> None:
    from dllm.pipelines.immune_llada.data.records import record_within_max_protein_length

    ok = BioSeqRecord(
        chains=[BioSeqChain("A" * 1024, "protein_a"), BioSeqChain("B" * 10, "protein_b")],
        task_type="ppi",
        source="unit",
    )
    long = BioSeqRecord(
        chains=[BioSeqChain("A" * 1025, "protein_a"), BioSeqChain("B" * 10, "protein_b")],
        task_type="ppi",
        source="unit",
    )
    assert record_within_max_protein_length(ok, 1024)
    assert not record_within_max_protein_length(long, 1024)



def test_grammar_renderer_rejects_long_ppi_without_loader_filter() -> None:
    record = BioSeqRecord(
        chains=[BioSeqChain("A" * 1025, "protein_a"), BioSeqChain("B" * 10, "protein_b")],
        task_type="ppi",
        source="unit",
        labels={"relation": "binding"},
    )
    tokenizer = GrammarTokenizer(Esm2SequenceTokenizer())
    with pytest.raises(ValueError, match="filter at loader"):
        GrammarRenderer(tokenizer, ppi_max_protein_length=1024).encode(record)


def _shuffle_probe_records(count: int) -> list[BioSeqRecord]:
    return [
        BioSeqRecord(
            chains=[BioSeqChain("A" * (index + 1), "protein_a")],
            task_type="generic",
            source="unit",
        )
        for index in range(count)
    ]


def test_streaming_shuffle_is_count_preserving_and_reproducible() -> None:
    from dllm.pipelines.immune_llada.data.grammar import _streaming_shuffle

    records = _shuffle_probe_records(200)
    original = [record.sequences[0] for record in records]

    first = [
        record.sequences[0]
        for record in _streaming_shuffle(iter(records), 32, random.Random(123))
    ]
    second = [
        record.sequences[0]
        for record in _streaming_shuffle(iter(records), 32, random.Random(123))
    ]

    # Count preserving: same multiset, same length (keeps DDP batch counts intact).
    assert sorted(first) == sorted(original)
    assert len(first) == len(original)
    # Reproducible for a fixed seed; actually reorders for buffer_size > 1.
    assert first == second
    assert first != original


def test_streaming_shuffle_window_of_one_is_identity() -> None:
    from dllm.pipelines.immune_llada.data.grammar import _streaming_shuffle

    records = _shuffle_probe_records(50)
    passthrough = list(_streaming_shuffle(iter(records), 1, random.Random(0)))
    assert [record.sequences[0] for record in passthrough] == [
        record.sequences[0] for record in records
    ]


def test_grammar_arrow_source_shuffle_pass_seed_differs_per_pass() -> None:
    """Two passes over the same shard should reshuffle (no repeated order)."""

    from dllm.pipelines.immune_llada.data.grammar import (
        GrammarArrowSource,
        GrammarArrowSourceConfig,
    )

    source = GrammarArrowSource.__new__(GrammarArrowSource)
    source.config = GrammarArrowSourceConfig(
        name="unit", split="train", shuffle_buffer_size=16, shuffle_seed=7
    )
    source._shuffle_pass = 0
    records = _shuffle_probe_records(100)

    source._raw_records = lambda shard_index, num_shards: iter(records)  # type: ignore[method-assign]
    first = [r.sequences[0] for r in source.iter_records(shard_index=0, num_shards=1)]
    source._raw_records = lambda shard_index, num_shards: iter(records)  # type: ignore[method-assign]
    second = [r.sequences[0] for r in source.iter_records(shard_index=0, num_shards=1)]

    assert sorted(first) == sorted(second)
    assert first != second
