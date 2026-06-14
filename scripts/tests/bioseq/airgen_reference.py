"""Minimal AirGen reference for multichain diffusion parity checks.

Logic copied from
``/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/dplm_multichain.py``
(``q_sample_comp``, ``construct_x_t``). Used when full AirGen imports are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Uniform


@dataclass
class AirGenTrainConfig:
    schedule: str = "uniform"
    single_chain_ratio: float = 0.0
    heavy2light_loss_ratio: float = 0.0
    light2heavy_loss_ratio: float = 0.0
    joint_loss_ratio: float = 1.0
    independent_loss_ratio: float = 0.0


class AirGenMultiChainReference:
    def __init__(self, cfg: AirGenTrainConfig | None = None, mask_id: int = 32) -> None:
        self.cfg = cfg or AirGenTrainConfig()
        self.mask_id = mask_id
        if self.cfg.schedule == "uniform":
            self.distribution = Uniform(0, 1)
        else:
            raise ValueError(self.cfg.schedule)

    def q_sample_comp(self, x_0, region, t, maskable_mask):
        u = torch.rand_like(x_0, dtype=torch.float)
        t_adaptive = torch.where(region > 0, t[:, None], t[:, None])
        t_mask = (u < t_adaptive) & maskable_mask
        t_mask_inv = (u >= t_adaptive) & maskable_mask
        t_mask_inv = torch.where((t_adaptive == 0) | (t_adaptive == 1), t_mask, t_mask_inv)
        x_t = x_0.masked_fill(t_mask, self.mask_id)
        x_t_inv = x_0.masked_fill(t_mask_inv, self.mask_id)
        return torch.cat([x_t, x_t_inv], dim=0), torch.cat([t_mask, t_mask_inv], dim=0)

    def get_non_special_sym_mask(self, output_tokens, partial_masks=None, *, pad_id=1, bos_id=0):
        non_special_sym_mask = output_tokens.ne(pad_id) & output_tokens.ne(bos_id)
        if partial_masks is not None:
            non_special_sym_mask &= ~partial_masks
        return non_special_sym_mask

    def construct_x_t(
        self,
        heavy_target,
        light_target,
        heavy_region,
        light_region,
        *,
        pad_id=1,
        bos_id=0,
        eps=1e-3,
        stage=None,
    ):
        bsz = heavy_target.size(0)
        heavy_t = self.distribution.sample((bsz,)).to(heavy_target.device)
        heavy_t = (1 - eps) * heavy_t + eps
        light_t = self.distribution.sample((bsz,)).to(heavy_target.device)
        light_t = (1 - eps) * light_t + eps

        assert (
            self.cfg.single_chain_ratio
            + self.cfg.heavy2light_loss_ratio
            + self.cfg.light2heavy_loss_ratio
            + self.cfg.joint_loss_ratio
            + self.cfg.independent_loss_ratio
            == 1.0
        )

        if stage == "val":
            split_sizes = [0, 0, 0, 0, bsz, 0]
        else:
            split_sizes = [
                int(bsz * self.cfg.single_chain_ratio / 2),
                int(bsz * self.cfg.single_chain_ratio / 2),
                int(bsz * self.cfg.heavy2light_loss_ratio),
                int(bsz * self.cfg.light2heavy_loss_ratio),
                int(bsz * self.cfg.independent_loss_ratio),
                int(bsz * self.cfg.joint_loss_ratio),
            ]
            split_sizes[-1] = bsz - sum(split_sizes[:-1])

        rand_index = torch.randperm(bsz).type_as(heavy_target)
        bool_index_list = []
        for int_index in torch.split(rand_index, split_sizes):
            bool_index = torch.zeros(bsz, dtype=torch.bool, device=heavy_target.device)
            bool_index[int_index] = True
            bool_index_list.append(bool_index)

        (
            mask_heavy_index,
            mask_light_index,
            heavy2light_index,
            light2heavy_index,
            independent_index,
            joint_index,
        ) = bool_index_list

        heavy_t = heavy_t.masked_fill(heavy2light_index, 0)
        heavy_t = heavy_t.masked_fill(mask_heavy_index, 1)
        heavy_x_t, heavy_loss_mask = self.q_sample_comp(
            heavy_target,
            heavy_region,
            heavy_t,
            maskable_mask=self.get_non_special_sym_mask(heavy_target, pad_id=pad_id, bos_id=bos_id),
        )

        light_t = light_t.masked_fill(light2heavy_index, 0)
        light_t = light_t.masked_fill(mask_light_index, 1)
        light_t = light_t.masked_scatter(joint_index, heavy_t[joint_index])
        light_x_t, light_loss_mask = self.q_sample_comp(
            light_target,
            light_region,
            light_t,
            maskable_mask=self.get_non_special_sym_mask(light_target, pad_id=pad_id, bos_id=bos_id),
        )

        heavy_t_inv = torch.where((heavy_t == 0) | (heavy_t == 1), heavy_t, 1 - heavy_t)
        light_t_inv = torch.where((light_t == 0) | (light_t == 1), light_t, 1 - light_t)

        return (
            {"t": torch.cat([heavy_t, heavy_t_inv], dim=0), "x_t": heavy_x_t, "mask": heavy_loss_mask},
            {"t": torch.cat([light_t, light_t_inv], dim=0), "x_t": light_x_t, "mask": light_loss_mask},
            mask_heavy_index.repeat(2),
            mask_light_index.repeat(2),
        )
