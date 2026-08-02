"""Regression tests for role-first TCR grammar rendering.

Run:
    conda run -n pllm python -m pytest \
        /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_grammar_tcr_role_resolution.py -q
"""

from __future__ import annotations

from dllm.pipelines.qwen3_vl_arch.data.grammar import (
    TOKEN_CLASS_RESIDUE,
    GrammarRenderer,
    GrammarTokenizer,
)
from dllm.pipelines.qwen3_vl_arch.data.records import BioSeqChain, BioSeqRecord


TOKENIZER = GrammarTokenizer()
RENDERER = GrammarRenderer(TOKENIZER)


def _render(record: BioSeqRecord) -> tuple[list[str], dict]:
    row = RENDERER.encode(record)
    return TOKENIZER.decode_tokens(row["input_ids"]), row


def test_beta_peptide_record_does_not_duplicate_peptide_in_tcr_block() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("PEPTIDE", "peptide"),
            BioSeqChain("CASSQYF", "tcr_beta"),
        ],
        task_type="tcr",
        source="grammar_embed",
    )

    tokens, row = _render(record)

    assert tokens == [
        "<prots>",
        "<pep>",
        "P",
        "E",
        "P",
        "T",
        "I",
        "D",
        "E",
        "<protd>",
        "<binding>",
        "<prots>",
        "<tcr>",
        "C",
        "A",
        "S",
        "S",
        "Q",
        "Y",
        "F",
        "<protd>",
    ]
    assert row["grammar_name"] == "tcr_peptide"
    residue_chain_ids = {
        chain_id
        for chain_id, class_id in zip(row["position_ids_chain"], row["token_class_ids"])
        if class_id == TOKEN_CLASS_RESIDUE
    }
    assert residue_chain_ids == {0, 1}


def test_context_free_untyped_tcr_pair_keeps_legacy_positional_fallback() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("BBB", "other"),
            BioSeqChain("AAA", "other"),
        ],
        task_type="tcr",
        source="legacy_tcr_pair",
    )

    tokens, row = _render(record)

    assert tokens == [
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
    assert row["grammar_name"] == "tcr_pair"
