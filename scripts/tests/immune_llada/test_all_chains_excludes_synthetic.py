"""All-chains eligibility: exclude synthetic X; include relation targets.

The renderer already drops ``synthetic_residue_mask`` from
``diffusion_eligible_mask`` and already puts trainable
``<binding>`` / ``<nonbinding>`` in that mask (``is_fixed=False``). The BERT
all-chains helper rebuilds eligibility from ``residue_mask`` and must therefore
(1) re-exclude synthetic placeholders and (2) OR in ``relation_target_mask``
(not ``relation_token_mask``). This suite calls the helpers on a hand-built
CPU batch — no ESMC or LLaDA weights.

Run with ``pytest scripts/tests/immune_llada/test_all_chains_excludes_synthetic.py``.
"""

from __future__ import annotations

import pytest
import torch

from examples.llada.protein_fusion_model import (
    all_residue_eligible_mask,
    sample_bioseq_bert_noise,
)

_MASK_ID = 99
_N_TRIALS = 200


def _layout_batch(*, include_synthetic_key: bool = True) -> dict[str, torch.Tensor]:
    """One row: grammar / antigen / generated / synthetic-X / pad.

    Positions (S=16)::

        0      pad
        1-2    grammar (not residue)
        3-6    antigen / MHC — real residues, fixed context
        7-10   generated-chain real residues
        11-14  synthetic completion ``X``
        15     pad
    """

    seq_len = 16
    input_ids = torch.arange(100, 100 + seq_len, dtype=torch.long).unsqueeze(0)
    attention = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    residue = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    synthetic = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]], dtype=torch.bool)
    generated = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention,
        "residue_mask": residue,
        "diffusion_eligible_mask": generated,
        "diffusion_loss_mask": generated,
    }
    if include_synthetic_key:
        batch["synthetic_residue_mask"] = synthetic
    return batch


def _antigen_positions(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["residue_mask"] & batch["attention_mask"] & ~batch["diffusion_eligible_mask"] & ~batch.get(
        "synthetic_residue_mask", torch.zeros_like(batch["residue_mask"])
    )


def _legacy_generated_only_eligible(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pre-fix ``all_chain_targets=False`` eligible set (does not read synthetic)."""

    residue_mask = batch.get("residue_mask")
    loss_mask = batch.get("diffusion_loss_mask", batch.get("diffusion_target_mask"))
    explicit = batch.get("diffusion_eligible_mask")
    eligible = explicit.bool() if explicit is not None else loss_mask.bool()
    if explicit is None and residue_mask is not None:
        eligible = eligible & residue_mask.bool()
    attention = batch.get("attention_mask")
    if attention is not None:
        eligible = eligible & attention.bool()
    return eligible


def _legacy_all_residue_eligible(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pre-fix all-chains eligible set: residue ∩ attention, synthetic ignored."""

    eligible = batch["residue_mask"].bool()
    attention = batch.get("attention_mask")
    if attention is not None:
        eligible = eligible & attention.bool()
    return eligible


def _replay_bert_selection(
    batch: dict[str, torch.Tensor],
    eligible: torch.Tensor,
    *,
    seed: int,
    mask_ratio: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the helper's RNG so we can assert bit-identical selection."""

    input_ids = batch["input_ids"]
    torch.manual_seed(seed)
    batch_size, seq_len = input_ids.shape
    selection = (torch.rand(batch_size, seq_len) < mask_ratio) & eligible
    for row in range(batch_size):
        if eligible[row].any() and not selection[row].any():
            valid = torch.nonzero(eligible[row], as_tuple=False).flatten()
            choice = valid[torch.randint(valid.numel(), (1,))]
            selection[row, choice] = True
    labels = input_ids.masked_fill(~selection, -100)
    return labels, selection


def test_all_chains_never_selects_synthetic_over_many_trials() -> None:
    batch = _layout_batch()
    synthetic = batch["synthetic_residue_mask"]
    for trial in range(_N_TRIALS):
        torch.manual_seed(trial)
        _, labels, selected = sample_bioseq_bert_noise(
            batch, _MASK_ID, all_chain_targets=True
        )
        assert not (selected & synthetic).any()
        assert not ((labels != -100) & synthetic).any()
        eligible = all_residue_eligible_mask(batch)
        assert not (eligible & synthetic).any()


def test_all_chains_still_selects_fixed_context_residues() -> None:
    batch = _layout_batch()
    antigen = _antigen_positions(batch)
    assert antigen.any()
    eligible = all_residue_eligible_mask(batch)
    assert torch.equal(eligible[antigen], torch.ones(int(antigen.sum()), dtype=torch.bool))

    seen_antigen = torch.zeros_like(antigen)
    for trial in range(_N_TRIALS):
        torch.manual_seed(1000 + trial)
        _, _, selected = sample_bioseq_bert_noise(
            batch, _MASK_ID, mask_ratio=0.4, all_chain_targets=True
        )
        seen_antigen |= selected & antigen
    assert seen_antigen.any(), "all-chains widening must still hit antigen/MHC residues"


def test_all_chain_targets_false_is_bit_identical_to_legacy_eligible() -> None:
    batch = _layout_batch()
    legacy = _legacy_generated_only_eligible(batch)
    seed = 20260912
    torch.manual_seed(seed)
    _, labels, selected = sample_bioseq_bert_noise(
        batch, _MASK_ID, all_chain_targets=False
    )
    expect_labels, expect_selected = _replay_bert_selection(batch, legacy, seed=seed)
    assert torch.equal(selected, expect_selected)
    assert torch.equal(labels, expect_labels)
    assert torch.equal(legacy, batch["diffusion_eligible_mask"] & batch["attention_mask"])


def test_all_synthetic_residue_batch_raises() -> None:
    seq_len = 8
    residue = torch.ones(1, seq_len, dtype=torch.bool)
    synthetic = torch.ones(1, seq_len, dtype=torch.bool)
    batch = {
        "input_ids": torch.arange(seq_len).unsqueeze(0),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "residue_mask": residue,
        "synthetic_residue_mask": synthetic,
        "diffusion_eligible_mask": torch.zeros(1, seq_len, dtype=torch.bool),
        "diffusion_loss_mask": torch.zeros(1, seq_len, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="no eligible target tokens"):
        sample_bioseq_bert_noise(batch, _MASK_ID, all_chain_targets=True)
    with pytest.raises(ValueError, match="no eligible target tokens"):
        all_residue_eligible_mask(batch)


def test_missing_synthetic_key_matches_legacy_all_chains() -> None:
    batch = _layout_batch(include_synthetic_key=False)
    assert "synthetic_residue_mask" not in batch
    legacy = _legacy_all_residue_eligible(batch)
    assert torch.equal(all_residue_eligible_mask(batch), legacy)

    seed = 7
    torch.manual_seed(seed)
    _, labels, selected = sample_bioseq_bert_noise(
        batch, _MASK_ID, all_chain_targets=True
    )
    expect_labels, expect_selected = _replay_bert_selection(batch, legacy, seed=seed)
    assert torch.equal(selected, expect_selected)
    assert torch.equal(labels, expect_labels)
    # The positions that *would* be synthetic are still eligible — v3 behaviour.
    would_be_synthetic = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]], dtype=torch.bool
    )
    assert (legacy & would_be_synthetic).any()


def test_all_zero_synthetic_mask_is_a_noop() -> None:
    batch = _layout_batch()
    batch["synthetic_residue_mask"] = torch.zeros_like(batch["residue_mask"])
    legacy = _legacy_all_residue_eligible(batch)
    assert torch.equal(all_residue_eligible_mask(batch), legacy)


def test_bert_fallback_never_picks_synthetic() -> None:
    """A near-zero mask_ratio forces the >=1-token fallback on almost every row."""

    batch = _layout_batch()
    synthetic = batch["synthetic_residue_mask"]
    real_eligible = (
        batch["residue_mask"] & batch["attention_mask"] & ~synthetic
    )
    for trial in range(_N_TRIALS):
        torch.manual_seed(trial + 50_000)
        _, labels, selected = sample_bioseq_bert_noise(
            batch, _MASK_ID, mask_ratio=1e-6, all_chain_targets=True
        )
        assert int(selected.sum()) >= 1
        assert not (selected & synthetic).any()
        assert (selected & real_eligible).equal(selected)
        assert not ((labels != -100) & synthetic).any()


_BINDING_ID = 200
_NONBINDING_ID = 201
_UNKNOWN_ID = 202
_RESIDUE_POOL = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)


def _relation_layout_batch() -> dict[str, torch.Tensor]:
    """One row mixing residues, synthetic X, and both relation-mask kinds.

    Positions (S=16)::

        0      pad
        1      ``<unknown>`` null-prefix — relation token, not a target
        2      grammar (structure)
        3-5    antigen / MHC residues (fixed context)
        6      MHC→peptide presentation ``<binding>`` — relation token, not a target
        7-10   generated-chain real residues
        11     trainable ``<nonbinding>`` — ``relation_target_mask``
        12-14  synthetic completion ``X``
        15     pad
    """

    input_ids = torch.tensor(
        [[
            0,
            _UNKNOWN_ID,
            50,
            10, 11, 12,
            _BINDING_ID,
            13, 14, 15, 16,
            _NONBINDING_ID,
            17, 18, 19,
            0,
        ]],
        dtype=torch.long,
    )
    attention = torch.tensor(
        [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool
    )
    residue = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0]], dtype=torch.bool
    )
    synthetic = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]], dtype=torch.bool
    )
    # generated residues only — relation target is added by the renderer to
    # diffusion_eligible_mask in real data, but kept out here so the False
    # branch cannot pick it up from this key.
    generated = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool
    )
    relation_token = torch.tensor(
        [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=torch.bool
    )
    relation_target = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=torch.bool
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "residue_mask": residue,
        "synthetic_residue_mask": synthetic,
        "diffusion_eligible_mask": generated,
        "diffusion_loss_mask": generated,
        "relation_token_mask": relation_token,
        "relation_target_mask": relation_target,
    }


def test_all_chains_selects_relation_targets_over_many_trials() -> None:
    batch = _relation_layout_batch()
    target = batch["relation_target_mask"]
    assert int(target.sum()) == 1
    seen = torch.zeros_like(target)
    labelled = 0
    for trial in range(_N_TRIALS):
        torch.manual_seed(2_000 + trial)
        _, labels, selected = sample_bioseq_bert_noise(
            batch, _MASK_ID, mask_ratio=0.4, all_chain_targets=True
        )
        hit = selected & target
        seen |= hit
        if hit.any():
            labelled += 1
            assert torch.equal(labels[hit], batch["input_ids"][hit])
        assert not ((labels != -100) & target & ~selected).any()
    assert seen.any(), "all-chains BERT must select relation_target_mask positions"
    assert labelled > 0


def test_fixed_relation_tokens_never_selected() -> None:
    """§5.6 trap: relation_token_mask \\ relation_target_mask is never eligible."""

    batch = _relation_layout_batch()
    fixed_relation = batch["relation_token_mask"] & ~batch["relation_target_mask"]
    assert int(fixed_relation.sum()) == 2  # <unknown> prefix + presentation <binding>
    for trial in range(_N_TRIALS):
        torch.manual_seed(3_000 + trial)
        noised, labels, selected = sample_bioseq_bert_noise(
            batch,
            _MASK_ID,
            mask_ratio=0.4,
            residue_token_ids=_RESIDUE_POOL,
            all_chain_targets=True,
        )
        assert not (selected & fixed_relation).any()
        assert not ((labels != -100) & fixed_relation).any()
        assert torch.equal(noised[fixed_relation], batch["input_ids"][fixed_relation])


def test_relation_target_never_rewritten_to_residue() -> None:
    batch = _relation_layout_batch()
    target = batch["relation_target_mask"]
    original = batch["input_ids"]
    pool = set(_RESIDUE_POOL.tolist())
    saw_selected = False
    saw_forced_random = False
    for trial in range(_N_TRIALS):
        torch.manual_seed(4_000 + trial)
        noised, labels, selected = sample_bioseq_bert_noise(
            batch,
            _MASK_ID,
            mask_ratio=0.4,
            residue_token_ids=_RESIDUE_POOL,
            all_chain_targets=True,
        )
        hit = selected & target
        if hit.any():
            saw_selected = True
            values = noised[hit]
            assert ((values == _MASK_ID) | (values == original[hit])).all()
            assert not any(int(v) in pool for v in values.tolist())
            assert torch.equal(labels[hit], original[hit])
        torch.manual_seed(5_000 + trial)
        noised_rand, _, selected_rand = sample_bioseq_bert_noise(
            batch,
            _MASK_ID,
            mask_ratio=0.4,
            mask_prob=0.0,
            random_prob=1.0,
            residue_token_ids=_RESIDUE_POOL,
            all_chain_targets=True,
        )
        hit_rand = selected_rand & target
        if hit_rand.any():
            saw_forced_random = True
            assert (noised_rand[hit_rand] == _MASK_ID).all()
    assert saw_selected
    assert saw_forced_random


def test_relation_layout_still_never_selects_synthetic() -> None:
    batch = _relation_layout_batch()
    synthetic = batch["synthetic_residue_mask"]
    for trial in range(_N_TRIALS):
        torch.manual_seed(6_000 + trial)
        _, labels, selected = sample_bioseq_bert_noise(
            batch, _MASK_ID, mask_ratio=0.4, all_chain_targets=True
        )
        assert not (selected & synthetic).any()
        assert not ((labels != -100) & synthetic).any()


def test_all_chain_targets_false_ignores_relation_target_key() -> None:
    """False branch stays generated-only; it must not OR in relation_target_mask."""

    batch = _relation_layout_batch()
    target = batch["relation_target_mask"]
    seed = 20260912
    torch.manual_seed(seed)
    _, labels, selected = sample_bioseq_bert_noise(
        batch, _MASK_ID, all_chain_targets=False
    )
    legacy = _legacy_generated_only_eligible(batch)
    expect_labels, expect_selected = _replay_bert_selection(batch, legacy, seed=seed)
    assert torch.equal(selected, expect_selected)
    assert torch.equal(labels, expect_labels)
    assert not (selected & target).any()
    for trial in range(_N_TRIALS):
        torch.manual_seed(7_000 + trial)
        _, _, selected_i = sample_bioseq_bert_noise(
            batch, _MASK_ID, mask_ratio=0.4, all_chain_targets=False
        )
        assert not (selected_i & target).any()


def test_missing_relation_target_key_matches_pre_change() -> None:
    batch = _layout_batch()
    assert "relation_target_mask" not in batch
    expected = (
        batch["residue_mask"].bool()
        & batch["attention_mask"].bool()
        & ~batch["synthetic_residue_mask"].bool()
    )
    seed = 11
    torch.manual_seed(seed)
    _, labels, selected = sample_bioseq_bert_noise(
        batch, _MASK_ID, all_chain_targets=True
    )
    expect_labels, expect_selected = _replay_bert_selection(batch, expected, seed=seed)
    assert torch.equal(selected, expect_selected)
    assert torch.equal(labels, expect_labels)
