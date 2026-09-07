"""GIDD noise, Ophiuchus ratio buckets, focal/reciprocal loss, LLaDA2 edit remask.

Run with::

    pytest scripts/tests/bioseq/test_gidd_ophiuchus_edit.py -q
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    compute_masked_cross_entropy,
    sample_bioseq_diffusion_noise,
    sample_chain_conditioned_timesteps,
)
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import (
    BioSeqGenerateConfig,
    apply_gidd_self_correct,
    apply_llada2_edits,
    gidd_self_correct_positions,
    llada2_edit_positions,
)


MASK_ID = 32
RESIDUE_IDS = torch.tensor([10, 11, 12, 13], dtype=torch.long)


def _noise_batch(batch_size: int = 4, seq_len: int = 10) -> dict[str, torch.Tensor]:
    """Grammar at 0:1, residue targets at 2:7, pad at 7:, distinct residue ids."""

    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    input_ids[:, 0] = 1
    input_ids[:, 1] = 2
    # Eligible residues sit outside the uniform pool so replacements are obvious.
    residue_values = torch.tensor([14, 15, 16, 17, 18], dtype=torch.long)
    input_ids[:, 2:7] = residue_values
    input_ids[:, 7:] = 0

    attention = torch.ones(batch_size, seq_len, dtype=torch.bool)
    attention[:, 7:] = False
    residue = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    residue[:, 2:7] = True
    eligible = residue.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "diffusion_loss_mask": eligible,
        "diffusion_eligible_mask": eligible,
        "residue_mask": residue,
    }


def _two_chain_batch(batch_size: int = 32, seq_len: int = 16) -> dict[str, torch.Tensor]:
    residue = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    residue[:, 2:6] = True
    residue[:, 8:12] = True
    chain_ids = torch.full((batch_size, seq_len), -1, dtype=torch.long)
    chain_ids[:, 2:6] = 0
    chain_ids[:, 8:12] = 1
    attention = torch.ones(batch_size, seq_len, dtype=torch.bool)
    input_ids = torch.randint(10, 20, (batch_size, seq_len))
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "diffusion_loss_mask": residue,
        "diffusion_eligible_mask": residue,
        "residue_mask": residue,
        "chain_ids": chain_ids,
    }


def _classify_ratio_buckets(result) -> dict[str, int]:
    heavy_t = (result.token_t * result.heavy_mask.float()).sum(dim=1) / result.heavy_mask.sum(
        dim=1
    ).clamp_min(1)
    light_t = (result.token_t * result.light_mask.float()).sum(dim=1) / result.light_mask.sum(
        dim=1
    ).clamp_min(1)
    h2l = heavy_t.eq(0.0)
    l2h = light_t.eq(0.0) & ~h2l
    single = (heavy_t.eq(1.0) | light_t.eq(1.0)) & ~h2l & ~l2h
    remaining = ~(h2l | l2h | single)
    joint = remaining & heavy_t.eq(light_t)
    independent = remaining & ~joint
    return {
        "single": int(single.sum()),
        "h2l": int(h2l.sum()),
        "l2h": int(l2h.sum()),
        "independent": int(independent.sum()),
        "joint": int(joint.sum()),
        "heavy_t": heavy_t,
        "light_t": light_t,
    }


def test_uniform_ratio_zero_matches_implicit_default() -> None:
    batch = _noise_batch()
    kwargs = {"batch": batch, "mask_token_id": MASK_ID, "time_epsilon": 0.1}

    torch.manual_seed(7)
    default = sample_bioseq_diffusion_noise(**kwargs)
    torch.manual_seed(7)
    explicit = sample_bioseq_diffusion_noise(**kwargs, uniform_ratio=0.0)

    for left, right in zip(default, explicit, strict=True):
        assert torch.equal(left, right)


def test_uniform_ratio_positive_only_replaces_residues() -> None:
    batch = _noise_batch(batch_size=8, seq_len=10)
    token_t = torch.full(batch["input_ids"].shape, 0.5)
    torch.manual_seed(3)
    noised, labels, corruption, _ = sample_bioseq_diffusion_noise(
        batch,
        mask_token_id=MASK_ID,
        time_epsilon=0.1,
        uniform_ratio=0.5,
        gidd_gamma=1.0,
        residue_token_ids=RESIDUE_IDS,
        token_t=token_t,
    )

    original = batch["input_ids"]
    grammar = ~batch["diffusion_eligible_mask"]
    residue = batch["residue_mask"]
    assert torch.equal(noised[grammar], original[grammar])

    mask_sites = residue & noised.eq(MASK_ID)
    uniform_sites = residue & noised.ne(original) & noised.ne(MASK_ID)
    keep_sites = residue & noised.eq(original)
    assert uniform_sites.any(), "expected at least one GIDD uniform replacement"
    assert mask_sites.any(), "expected at least one absorbing-mask site"
    assert torch.isin(noised[uniform_sites], RESIDUE_IDS).all()
    assert labels[mask_sites].ne(-100).all()
    assert labels[uniform_sites].ne(-100).all()
    assert labels[keep_sites].eq(-100).all()
    assert labels[grammar].eq(-100).all()
    assert torch.equal(corruption, mask_sites | uniform_sites)


def test_ophiuchus_ratio_buckets_and_val_independent() -> None:
    ratios = {
        "independent_loss_ratio": 0.625,
        "single_chain_ratio": 0.125,
        "heavy2light_loss_ratio": 0.125,
        "light2heavy_loss_ratio": 0.125,
        "joint_loss_ratio": 0.0,
    }
    batch = _two_chain_batch(batch_size=32)
    torch.manual_seed(11)
    result = sample_chain_conditioned_timesteps(batch, ratios=ratios, time_epsilon=1e-3)
    counts = _classify_ratio_buckets(result)

    bsz = 32
    expected_single = int(bsz * ratios["single_chain_ratio"] / 2) * 2
    expected_h2l = int(bsz * ratios["heavy2light_loss_ratio"])
    expected_l2h = int(bsz * ratios["light2heavy_loss_ratio"])
    expected_independent = int(bsz * ratios["independent_loss_ratio"])
    assigned = expected_single + expected_h2l + expected_l2h + expected_independent
    expected_joint = bsz - assigned

    assert counts["single"] == expected_single
    assert counts["h2l"] == expected_h2l
    assert counts["l2h"] == expected_l2h
    assert counts["independent"] == expected_independent
    assert counts["joint"] == expected_joint
    assert counts["single"] + counts["h2l"] + counts["l2h"] + counts["independent"] + counts["joint"] == bsz

    torch.manual_seed(11)
    val = sample_chain_conditioned_timesteps(
        batch, ratios=ratios, time_epsilon=1e-3, stage="val"
    )
    val_counts = _classify_ratio_buckets(val)
    assert val_counts["h2l"] == 0
    assert val_counts["l2h"] == 0
    assert val_counts["single"] == 0
    assert (val_counts["heavy_t"] != val_counts["light_t"]).any()
    assert val_counts["heavy_t"].gt(0).all() and val_counts["light_t"].gt(0).all()
    assert val_counts["heavy_t"].lt(1).any() and val_counts["light_t"].lt(1).any()


def test_focal_matches_manual_one_minus_p_true_nll() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.4, -1.0],
                [0.1, 1.5, 0.2],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1, -100]])
    got = compute_masked_cross_entropy(logits, labels, focal=True, focal_gamma=1.0)

    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    loss_mask = labels.ne(-100)
    p_true = torch.exp(-nll)
    manual = ((1.0 - p_true) * nll)[loss_mask].sum() / loss_mask.sum().clamp_min(1)
    assert torch.allclose(got, manual, atol=1e-6)


def test_reciprocal_token_weights_scale_nll() -> None:
    logits = torch.tensor(
        [[[0.0, 2.0, -0.5], [1.2, 0.3, 0.1]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[1, 0]])
    timesteps = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    token_weights = 1.0 / (timesteps + 1.0 / 20.0)

    got = compute_masked_cross_entropy(logits, labels, token_weights=token_weights)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    expected = (nll * token_weights).sum() / labels.ne(-100).sum().clamp_min(1)
    assert torch.allclose(got, expected, atol=1e-6)

    unweighted = compute_masked_cross_entropy(logits, labels)
    assert not torch.allclose(got, unweighted, atol=1e-5)


def test_chain_split_loss_does_not_leak_other_chain() -> None:
    logits = torch.zeros(1, 2, 3)
    logits[0, 0, 0] = 5.0
    logits[0, 1, 1] = 5.0
    labels = torch.tensor([[0, 1]])
    heavy = torch.tensor([[True, False]])
    light = torch.tensor([[False, True]])

    both = compute_masked_cross_entropy(
        logits, labels, heavy_mask=heavy, light_mask=light
    )
    heavy_only = compute_masked_cross_entropy(logits[:, :1], labels[:, :1])
    light_only = compute_masked_cross_entropy(logits[:, 1:], labels[:, 1:])
    assert torch.allclose(both, heavy_only + light_only, atol=1e-6)

    leaked = compute_masked_cross_entropy(
        logits, labels, heavy_mask=heavy, light_mask=light, light_loss_weight=0.0
    )
    assert torch.allclose(leaked, heavy_only, atol=1e-6)


def test_generate_config_defaults_leave_baseline_disabled() -> None:
    config = BioSeqGenerateConfig()
    assert config.editing_threshold == 0.0
    assert config.self_correct is False
    assert config.max_post_steps == 16
    assert config.self_correct_temperature == 0.1
    # Existing callers that only pass old fields keep the default generate path.
    legacy = BioSeqGenerateConfig(max_iter=4, sampling_strategy="argmax")
    assert legacy.editing_threshold == 0.0
    assert legacy.self_correct is False


def test_apply_llada2_edits_only_committed_high_conf_disagreements() -> None:
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    pred = torch.tensor([[9, 8, 3, 7, 6]])
    confidence = torch.tensor([[0.99, 0.95, 0.99, 0.40, 0.99]])
    generation = torch.tensor([[False, True, True, True, True]])
    still_masked = torch.tensor([[False, False, False, False, True]])

    replace = llada2_edit_positions(
        tokens, pred, confidence, generation, 0.9, still_masked
    )
    # pos0 fixed, pos2 agrees, pos3 below threshold, pos4 still masked → only pos1
    assert replace.tolist() == [[False, True, False, False, False]]

    edited = apply_llada2_edits(
        tokens, pred, confidence, generation, 0.9, still_masked
    )
    assert edited.tolist() == [[1, 8, 3, 4, 5]]
    assert torch.equal(apply_llada2_edits(tokens, pred, confidence, generation, 0.0), tokens)


def test_gidd_self_correct_picks_highest_new_token_prob() -> None:
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    pred = torch.tensor([[9, 2, 7, 4], [5, 1, 2, 3]])
    pred_prob = torch.tensor([[0.40, 0.99, 0.70, 0.10], [0.20, 0.55, 0.80, 0.60]])
    generation = torch.tensor([[False, True, True, True], [True, True, True, False]])
    still_masked = torch.zeros_like(generation)

    replace = gidd_self_correct_positions(
        tokens, pred, pred_prob, generation, still_masked
    )
    # seq0: disagreements in gen = pos2 (0.70); pos0 is not generation
    # seq1: disagreements in gen = pos1 (0.55), pos2 (0.80) → pick pos2
    assert replace.tolist() == [[False, False, True, False], [False, False, True, False]]

    edited = apply_gidd_self_correct(tokens, pred, pred_prob, generation, still_masked)
    assert edited.tolist() == [[1, 2, 7, 4], [5, 6, 2, 8]]

    agree = apply_gidd_self_correct(tokens, tokens, pred_prob, generation)
    assert torch.equal(agree, tokens)
