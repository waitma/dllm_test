"""Tests for PPI relation inference."""

from __future__ import annotations

from dllm.pipelines.qwen3_vl_arch.data.ppi_relations import (
    infer_grammar_relation,
    infer_relation_from_unified_row,
)


def test_task_family_defaults_to_binding() -> None:
    assert infer_grammar_relation(task_family="ppi_binary", source_id="figshare_gold_standard") == "binding"


def test_neutralization_source() -> None:
    assert (
        infer_grammar_relation(
            task_family="antibody_neutralization",
            source_id="covabdab_neutralization",
        )
        == "neutralization"
    )


def test_negative_label_maps_to_nonbinding() -> None:
    assert infer_grammar_relation(task_family="ppi_binary", label="0") == "nonbinding"


def test_string_channel_activation() -> None:
    assert infer_grammar_relation(string_channel="activation") == "activation"


def test_unmapped_task_family_uses_unknown() -> None:
    assert infer_grammar_relation(task_family="antibody_binding", source_id="flab") == "unknown"
    assert infer_grammar_relation(task_family="ppi_mutation_affinity", source_id="skempi") == "unknown"


def test_unified_row_inference() -> None:
    row = {
        "task_family": "antibody_neutralization",
        "source_id": "covabdab_neutralization",
        "label": "1",
    }
    assert infer_relation_from_unified_row(row) == "neutralization"
