"""CPU regressions for fixed-canvas overflow handling.

Run with::

    python -m pytest scripts/tests/immune_llada/test_v2_canvas_overflow.py -q

The fixed canvas makes two row classes unrenderable: receptor chains longer
than their canvas, and rows whose padded grammar length exceeds
``max_sequence_length``. Prepared shards were budgeted under the old
variable-length grammar, so both classes exist in the corpus and cannot be
re-filtered without rebuilding it. ``drop_overflow_records=True`` drops and
counts them; the default still raises.
"""

from __future__ import annotations

import pytest

from dllm.pipelines.immune_llada.data import (
    DROP_BUDGET_MAX_LENGTH,
    DROP_RECEPTOR_CANVAS_OVERFLOW,
    FIXED_HEAVY_DECODER_SLOTS,
    FIXED_LIGHT_DECODER_SLOTS,
    GrammarBioSeqCollator,
    GrammarTokenizer,
    ReceptorCanvasOverflow,
)
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from downstream.grammar.common import antibody_pair_record


def _collator(*, drop: bool, max_sequence_length: int = 2112) -> GrammarBioSeqCollator:
    return GrammarBioSeqCollator(
        GrammarTokenizer(),
        max_sequence_length=max_sequence_length,
        fixed_receptor_lengths=True,
        drop_overflow_records=drop,
    )


def _oversize_heavy_record() -> BioSeqRecord:
    """A heavy chain one residue past the canvas; light stays well inside."""

    return antibody_pair_record("A" * FIXED_HEAVY_DECODER_SLOTS, "C" * 100)


def _oversize_light_record() -> BioSeqRecord:
    return antibody_pair_record("A" * 100, "C" * FIXED_LIGHT_DECODER_SLOTS)


def _long_antigen_record(antigen_length: int) -> BioSeqRecord:
    return BioSeqRecord(
        chains=[
            BioSeqChain(sequence="G" * antigen_length, role="antigen"),
            BioSeqChain(sequence="A" * 118, role="antibody_heavy"),
            BioSeqChain(sequence="C" * 107, role="antibody_light"),
        ],
        task_type="antibody_antigen",
        source="test_asd_antibody",
        labels={"relation": "binding"},
    )


# --- receptor chains longer than the canvas ---------------------------------


@pytest.mark.parametrize(
    "record_factory", [_oversize_heavy_record, _oversize_light_record]
)
def test_oversize_receptor_raises_by_default(record_factory) -> None:
    with pytest.raises(ReceptorCanvasOverflow):
        _collator(drop=False)([record_factory()])


def test_oversize_receptor_is_dropped_and_counted() -> None:
    collator = _collator(drop=True)
    batch = collator([_oversize_heavy_record(), antibody_pair_record("A" * 120, "C" * 105)])

    assert batch["input_ids"].shape[0] == 1
    assert collator.drop_counts[DROP_RECEPTOR_CANVAS_OVERFLOW] == 1
    assert collator.drop_counts[DROP_BUDGET_MAX_LENGTH] == 0
    by_source = collator.drop_counts_by_source[DROP_RECEPTOR_CANVAS_OVERFLOW]
    assert sum(by_source.values()) == 1


def test_receptor_at_the_canvas_limit_still_renders() -> None:
    """The reserved EOS slot is the only thing separating fit from overflow."""

    collator = _collator(drop=False)
    batch = collator(
        [
            antibody_pair_record(
                "A" * (FIXED_HEAVY_DECODER_SLOTS - 1),
                "C" * (FIXED_LIGHT_DECODER_SLOTS - 1),
            )
        ]
    )

    assert int(batch["chain_eos_mask"].sum()) == 2
    assert int(batch["chain_slot_mask"].sum()) == (
        FIXED_HEAVY_DECODER_SLOTS + FIXED_LIGHT_DECODER_SLOTS
    )


# --- grammar length over max_sequence_length --------------------------------


def _length_over_budget(budget: int) -> int:
    """Smallest antigen length whose fixed-canvas row exceeds ``budget``."""

    probe = _collator(drop=False, max_sequence_length=10**6)
    for antigen_length in range(600, 1024):
        rendered = probe.renderer.encode(_long_antigen_record(antigen_length))
        if len(rendered["input_ids"]) > budget:
            return antigen_length
    raise AssertionError("no antigen length exceeded the budget")


def test_grammar_over_budget_raises_by_default() -> None:
    antigen_length = _length_over_budget(1024)
    with pytest.raises(ValueError, match="exceeds max_sequence_length"):
        _collator(drop=False, max_sequence_length=1024)([_long_antigen_record(antigen_length)])


def test_grammar_over_budget_is_dropped_and_counted() -> None:
    antigen_length = _length_over_budget(1024)
    collator = _collator(drop=True, max_sequence_length=1024)
    batch = collator(
        [_long_antigen_record(antigen_length), antibody_pair_record("A" * 120, "C" * 105)]
    )

    assert batch["input_ids"].shape[0] == 1
    assert collator.drop_counts[DROP_BUDGET_MAX_LENGTH] == 1
    assert collator.drop_counts[DROP_RECEPTOR_CANVAS_OVERFLOW] == 0
    assert collator.drop_counts_by_source[DROP_BUDGET_MAX_LENGTH]["test_asd_antibody"] == 1


# --- ledger and failure modes -----------------------------------------------


def test_drop_ledger_accumulates_both_reasons_across_batches() -> None:
    antigen_length = _length_over_budget(1024)
    collator = _collator(drop=True, max_sequence_length=1024)
    keep = antibody_pair_record("A" * 120, "C" * 105)

    collator([_oversize_heavy_record(), keep])
    collator([_long_antigen_record(antigen_length), keep])

    assert dict(collator.drop_counts) == {
        DROP_RECEPTOR_CANVAS_OVERFLOW: 1,
        DROP_BUDGET_MAX_LENGTH: 1,
    }


def test_fully_dropped_batch_fails_loudly() -> None:
    """An empty batch must not reach the model as a silent no-op."""

    collator = _collator(drop=True)
    with pytest.raises(ValueError, match="overflowed the fixed canvas"):
        collator([_oversize_heavy_record(), _oversize_light_record()])


def test_legacy_collator_is_unaffected_by_the_canvas_limits() -> None:
    """A chain that overflows the v2 canvas is still ordinary legacy input."""

    legacy = GrammarBioSeqCollator(GrammarTokenizer())
    batch = legacy([_oversize_heavy_record()])

    assert batch["input_ids"].shape[0] == 1
    # No EOS canvas, so there is nothing for the v2 code paths to key off:
    # chain_slot_mask degenerates to residue_mask and encoder_slot_mask, which
    # every v2 branch tests for, is absent.
    assert not bool(batch["chain_eos_mask"].any())
    assert bool(batch["chain_slot_mask"].eq(batch["residue_mask"]).all())
    assert "encoder_slot_mask" not in batch
    assert int(batch["residue_mask"].sum()) == FIXED_HEAVY_DECODER_SLOTS + 100
