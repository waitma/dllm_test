"""Tests for STRING detailed-link channel parsing."""

from __future__ import annotations

from dllm.pipelines.qwen3_vl_arch.data.string_channels import parse_string_detailed_link


def test_parse_detailed_physical_dominant() -> None:
    line = "9606.ENSP000001 9606.ENSP000002 900 10 20 800 50 100 200 30"
    parsed = parse_string_detailed_link(line)
    assert parsed is not None
    assert parsed[0] == "9606.ENSP000001"
    assert parsed[1] == "9606.ENSP000002"
    assert parsed[2] == "binding"
    assert parsed[3] == "p"


def test_parse_detailed_expression_dominant() -> None:
    line = "9606.ENSP000001 9606.ENSP000002 500 10 20 30 900 100 200 30"
    parsed = parse_string_detailed_link(line)
    assert parsed is not None
    assert parsed[2] == "expression"
    assert parsed[3] == "a"
