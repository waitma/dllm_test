"""Tests for canonical PPI split policies."""

from __future__ import annotations

import pytest

from dllm.pipelines.qwen3_vl_arch.data.ppi_splits import (
    BERNETT_STRING_90_90,
    MINT_STRING_PRETRAIN,
    normalize_split_name,
    splits_allowed_for_grammar_mix,
    validate_split,
)


def test_normalize_split_aliases() -> None:
    assert normalize_split_name("validation") == "valid"
    assert normalize_split_name("training") == "train"


def test_mint_pretrain_splits() -> None:
    assert validate_split("stringdb_mint", "train") == "pretrain_train"
    assert validate_split("stringdb_mint", "valid") == "pretrain_valid"
    with pytest.raises(ValueError, match="not allowed"):
        validate_split("stringdb_mint", "test")


def test_bernett_90_90_grammar_mix_excludes_test() -> None:
    allowed = splits_allowed_for_grammar_mix("string_model_org_90_90_split")
    assert allowed == ("train", "valid")
    assert validate_split("string_model_org_90_90_split", "test") == "supervised_test"


def test_humanppi_cross_species_is_eval_only() -> None:
    assert validate_split("saprot_humanppi", "cross_species_test") == "eval_only"


def test_policy_ids_are_stable() -> None:
    assert MINT_STRING_PRETRAIN.policy_id == "mint_string_pretrain_v1"
    assert BERNETT_STRING_90_90.policy_id == "bernett_string_90_90_hf"
