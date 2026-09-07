"""CPU tests for multi-chain relation aux + per-row Ophiuchus t sampling.

No ESMC / LLaDA weights. Run::

    PYTHONPATH=dllm_test python dllm_test/scripts/diagnostics/test_relation_aux.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    sample_chain_conditioned_timesteps,
)
from dllm.pipelines.qwen3_vl_arch.relation_aux import (
    cognate_infonce,
    compute_relation_aux,
    generated_heavy_light_masks,
    infonce_pair,
    mean_token_logprob,
    pairing_auroc,
)


def _two_chain_batch(batch_size: int = 4, seq_len: int = 8) -> dict[str, torch.Tensor]:
    half = seq_len // 2
    residue_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    chain_ids[:, half:] = 1
    return {
        "input_ids": torch.arange(batch_size * seq_len).view(batch_size, seq_len),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "residue_mask": residue_mask,
        "chain_ids": chain_ids,
        "diffusion_eligible_mask": residue_mask.clone(),
        "diffusion_loss_mask": residue_mask.clone(),
    }


def test_generated_masks_ignore_antigen() -> None:
    # chain 0 = antigen (not eligible), 1 = heavy, 2 = light
    chain_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
    eligible = torch.tensor([[False, False, True, True, True, True]])
    residue = torch.ones_like(eligible)
    heavy, light = generated_heavy_light_masks(
        {
            "residue_mask": residue,
            "diffusion_eligible_mask": eligible,
            "chain_ids": chain_ids,
            "attention_mask": residue,
        }
    )
    assert bool(heavy[0].eq(torch.tensor([False, False, True, True, False, False])).all())
    assert bool(light[0].eq(torch.tensor([False, False, False, False, True, True])).all())


def test_infonce_matched_is_near_zero() -> None:
    hidden = torch.zeros(4, 4, 8)
    for i in range(4):
        hidden[i, 0] = torch.nn.functional.one_hot(torch.tensor(i), 8).float()
        hidden[i, 1] = hidden[i, 0].clone()
    heavy = torch.zeros(4, 4, dtype=torch.bool)
    light = torch.zeros(4, 4, dtype=torch.bool)
    heavy[:, 0] = True
    light[:, 1] = True
    loss = cognate_infonce(hidden, heavy, light, temperature=0.07)
    assert float(loss) < 1e-4, loss


def test_infonce_shuffled_is_large() -> None:
    hidden = torch.zeros(4, 4, 8)
    for i in range(4):
        hidden[i, 0] = torch.nn.functional.one_hot(torch.tensor(i), 8).float()
        hidden[i, 1] = torch.nn.functional.one_hot(torch.tensor((i + 1) % 4), 8).float()
    heavy = torch.zeros(4, 4, dtype=torch.bool)
    light = torch.zeros(4, 4, dtype=torch.bool)
    heavy[:, 0] = True
    light[:, 1] = True
    loss = cognate_infonce(hidden, heavy, light, temperature=0.07)
    assert float(loss) > 1.0, loss


def test_infonce_single_pair_is_zero() -> None:
    anchor = torch.randn(1, 8)
    assert float(infonce_pair(anchor, anchor, 0.07)) == 0.0
    hidden = torch.randn(1, 4, 8)
    heavy = torch.zeros(1, 4, dtype=torch.bool)
    light = torch.zeros(1, 4, dtype=torch.bool)
    heavy[:, 0] = True
    light[:, 1] = True
    assert float(cognate_infonce(hidden, heavy, light)) == 0.0


def test_compute_relation_aux_modes() -> None:
    hidden = torch.randn(3, 6, 8)
    heavy = torch.zeros(3, 6, dtype=torch.bool)
    light = torch.zeros(3, 6, dtype=torch.bool)
    heavy[:, :2] = True
    light[:, 2:4] = True
    none = compute_relation_aux("none", hidden, heavy, light)
    both = compute_relation_aux("both", hidden, heavy, light)
    assert float(none) == 0.0
    assert torch.isfinite(both)
    try:
        compute_relation_aux("nope", hidden, heavy, light)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid mode should raise")


def test_joint_ratio_always_joint() -> None:
    batch = _two_chain_batch(4, 8)
    ratios = {
        "single_chain_ratio": 0.0,
        "heavy2light_loss_ratio": 0.0,
        "light2heavy_loss_ratio": 0.0,
        "independent_loss_ratio": 0.0,
        "joint_loss_ratio": 1.0,
    }
    for _ in range(20):
        timed = sample_chain_conditioned_timesteps(batch, ratios=ratios, time_epsilon=1e-3)
        heavy_t = timed.token_t[:, 0]
        light_t = timed.token_t[:, 4]
        assert torch.allclose(heavy_t, light_t), (heavy_t, light_t)
        assert not timed.zero_loss_mask.any()


def test_nonzero_ratios_fire_at_batch_4() -> None:
    """The old int(B * ratio) path zeroed every non-joint bucket at B=4."""

    batch = _two_chain_batch(4, 8)
    ratios = {
        "single_chain_ratio": 0.1,
        "heavy2light_loss_ratio": 0.15,
        "light2heavy_loss_ratio": 0.15,
        "independent_loss_ratio": 0.1,
        "joint_loss_ratio": 0.5,
    }
    saw = {"h2l": 0, "l2h": 0, "single": 0, "indep": 0, "joint": 0}
    for _ in range(80):
        timed = sample_chain_conditioned_timesteps(
            batch, ratios=ratios, time_epsilon=1e-3, stage="train"
        )
        heavy_t = timed.token_t[:, 0]
        light_t = timed.token_t[:, 4]
        for row in range(4):
            ht = float(heavy_t[row])
            lt = float(light_t[row])
            if ht == 0.0 and lt > 0.0:
                saw["h2l"] += 1
            elif lt == 0.0 and ht > 0.0:
                saw["l2h"] += 1
            elif timed.zero_loss_mask[row].any():
                saw["single"] += 1
            elif abs(ht - lt) < 1e-6:
                saw["joint"] += 1
            else:
                saw["indep"] += 1
    missing = [name for name, count in saw.items() if count == 0]
    assert not missing, f"buckets never sampled at B=4: {missing} counts={saw}"


def test_val_stage_is_independent() -> None:
    batch = _two_chain_batch(8, 8)
    ratios = {
        "single_chain_ratio": 0.1,
        "heavy2light_loss_ratio": 0.15,
        "light2heavy_loss_ratio": 0.15,
        "independent_loss_ratio": 0.1,
        "joint_loss_ratio": 0.5,
    }
    timed = sample_chain_conditioned_timesteps(
        batch, ratios=ratios, time_epsilon=1e-3, stage="val"
    )
    assert not timed.zero_loss_mask.any()
    # Independent: t=0 / t=1 sentinels unused; heavy and light free.
    assert not (timed.token_t[:, 0] == 0).any()
    assert not (timed.token_t[:, 4] == 0).any()


def test_mean_logprob_and_auroc() -> None:
    logits = torch.zeros(2, 3, 5)
    logits[0, 0, 2] = 10.0
    logits[1, 1, 1] = 10.0
    target = torch.tensor([[2, 0, 0], [0, 1, 0]])
    mask = torch.tensor([[True, False, False], [False, True, False]])
    lp = mean_token_logprob(logits, target, mask)
    assert float(lp[0]) > -0.01
    assert float(lp[1]) > -0.01
    auc = pairing_auroc([1.0, 2.0, 3.0], [0.0, 0.5, 0.2])
    assert auc > 0.9, auc
    empty = pairing_auroc([], [0.0])
    assert empty != empty  # NaN


def main() -> None:
    tests = [
        test_generated_masks_ignore_antigen,
        test_infonce_matched_is_near_zero,
        test_infonce_shuffled_is_large,
        test_infonce_single_pair_is_zero,
        test_compute_relation_aux_modes,
        test_joint_ratio_always_joint,
        test_nonzero_ratios_fire_at_batch_4,
        test_val_stage_is_independent,
        test_mean_logprob_and_auroc,
    ]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"ALL {len(tests)} PASSED")


if __name__ == "__main__":
    main()
