#!/usr/bin/env python
"""Regression tests for the complete paper-faithful TCRT5 evaluator.

Run with::

    /vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python -m pytest -q \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/test_tcrt5_full_eval.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from tcr_generation_bench.tcrt5_full_eval import (  # noqa: E402
    AUTHOR_MODEL_LABELS,
    EvaluationBlock,
    ReferenceIndex,
    corpus_char_bleu,
    exact_metrics,
    js_divergence_nats,
    load_author_main_blocks,
    olga_protocol_audit_rows,
    positional_entropy_rows,
    ranked_exact_metrics,
    score_block,
    sentence_char_bleu,
)


def test_exact_metrics_preserve_official_duplicate_semantics() -> None:
    result = exact_metrics(
        ["CASSA", "CASSA", "CASSX", "CASSB"],
        ["CASSA", "CASSB", "CASSC"],
        k=4,
    )
    assert result["precision"] == pytest.approx(3 / 4)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(12 / 17)


def test_ranked_prefix_ap_is_mean_of_every_precision_prefix() -> None:
    result = ranked_exact_metrics(
        ["HIT1", "MISS", "HIT2", "MISS"],
        ["HIT1", "HIT2"],
        k=4,
        cutoffs=(1, 2, 4),
    )
    expected = (1.0 + 0.5 + 2 / 3 + 0.5) / 4
    assert result["average_precision_prefix"] == pytest.approx(expected)
    assert result["exact_hit_ranks"] == "1;3"
    assert result["hit_at_1"] == 1
    assert result["exact_unique_count_at_4"] == 2


def test_native_recovery_uses_levenshtein_selection_then_hamming_identity() -> None:
    # AB... -> BA... can be represented by two edits, but all four positions
    # differ.  This catches the tempting, non-paper 1-Levenshtein/length port.
    index = ReferenceIndex(["BADC"])
    assert index.native_recovery("ABCD") == 0.0

    # If no same-length reference exists, the upstream fallback does use
    # reference-length-normalised Levenshtein identity.
    fallback = ReferenceIndex(["ABCDE"])
    assert fallback.native_recovery("ABCD") == pytest.approx(0.8)


def test_sentence_char_bleu_matches_nltk_gold_values() -> None:
    assert sentence_char_bleu(
        "CASSLGQETQYF",
        ["CASSLGQETQYF", "CASSPGQETQYF"],
    ) == 1.0
    assert sentence_char_bleu(
        "CASSLGQETQYF",
        ["CASSPGGNTIYF", "CASSQETQYF"],
    ) == pytest.approx(0.6493358309501979, abs=1e-15)
    assert sentence_char_bleu("AAAA", ["ABCD"]) == pytest.approx(
        1.2882297539194154e-231,
        rel=1e-12,
    )
    assert corpus_char_bleu(
        ["CASSLGQETQYF", "CASSLGQETQYF", "AAAA", "CAS"],
        [
            ["CASSLGQETQYF", "CASSPGQETQYF"],
            ["CASSPGGNTIYF", "CASSQETQYF"],
            ["ABCD"],
            ["CASR", "CATS"],
        ],
    ) == pytest.approx(0.7313302058787068, abs=1e-15)


def test_kmer_js_uses_natural_log_units() -> None:
    assert js_divergence_nats({"AA": 2}, {"BB": 3}) == pytest.approx(
        math.log(2.0)
    )
    assert js_divergence_nats({"AA": 2}, {"AA": 3}) == 0.0


def test_positional_entropy_direction_is_generated_minus_reference() -> None:
    block = EvaluationBlock(
        model="toy",
        target_id="toy_target",
        generated=("AAAA", "AAAA"),
        references=("AAAA", "ABAA"),
        rank_basis="toy",
    )
    rows = positional_entropy_rows([block])
    position_two = rows.loc[rows["position_1based"] == 2].iloc[0]
    assert position_two["generated_entropy_gap_inclusive_nats"] == 0.0
    assert position_two["reference_entropy_gap_inclusive_nats"] == pytest.approx(
        math.log(2.0)
    )
    assert position_two["delta_entropy_gap_inclusive_nats"] == pytest.approx(
        -math.log(2.0)
    )


def test_one_real_author_block_per_model_matches_released_atomic_metrics() -> None:
    blocks, author = load_author_main_blocks()
    first_by_model = {}
    for block in blocks:
        first_by_model.setdefault(block.model, block)
    assert len(first_by_model) == 12

    author = author.copy()
    author["model_label"] = author["model"].map(AUTHOR_MODEL_LABELS)
    author = author.set_index(["model_label", "Unnamed: 0"])
    for model, block in first_by_model.items():
        reconstructed = score_block(block, k=100)
        official = author.loc[(model, block.target_id)]
        assert reconstructed["precision_at_k"] == pytest.approx(
            official["precision"], abs=1e-15
        )
        assert reconstructed["recall_at_k"] == pytest.approx(
            official["recall"], abs=1e-15
        )
        assert reconstructed["f1_at_k"] == pytest.approx(
            official["f1"], abs=1e-15
        )
        assert reconstructed["mean_edit_distance"] == pytest.approx(
            official["d_edit"], abs=1e-15
        )
        assert reconstructed["native_recovery_mean"] == pytest.approx(
            official["seq_recovery"], abs=1e-15
        )


def test_olga_protocol_inputs_match_the_paper_pinned_setup() -> None:
    audit = olga_protocol_audit_rows()
    assert len(audit) == 8
    assert audit["agreement"].all()
