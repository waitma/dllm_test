"""Tests for grammar unknown relation token."""

from __future__ import annotations

from dllm.pipelines.qwen3_vl_arch.data.grammar import GRAMMAR_RELATIONS, _relation_token
from dllm.pipelines.qwen3_vl_arch.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.qwen3_vl_arch.data.grammar import GrammarRenderer, GrammarTokenizer


def test_unknown_in_relation_vocabulary() -> None:
    assert "unknown" in GRAMMAR_RELATIONS
    assert "unknown_relation" not in GRAMMAR_RELATIONS


def test_relation_token_defaults_to_unknown() -> None:
    assert _relation_token(None) == "<unknown>"
    assert _relation_token("") == "<unknown>"
    assert _relation_token("unknown_relation") == "<unknown>"


def test_renderer_uses_unknown_between_chains() -> None:
    record = BioSeqRecord(
        chains=[
            BioSeqChain("ACDEFG", "protein_a"),
            BioSeqChain("GHIKLM", "protein_b"),
        ],
        task_type="ppi",
        source="generic_pair",
        labels={},
    )
    encoded = GrammarRenderer(GrammarTokenizer()).encode(record)
    tokens = GrammarTokenizer().decode_tokens(encoded["input_ids"])
    assert "<unknown>" in tokens
