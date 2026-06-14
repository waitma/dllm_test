from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Uniform

from .model import OphiuchusAbBackbone
from .sampling import sample_from_categorical, stochastic_sample_from_categorical, topk_masking


@dataclass
class OphiuchusAbTrainConfig:
    schedule: str = "uniform"
    single_chain_ratio: float = 0.0
    heavy2light_loss_ratio: float = 0.0
    light2heavy_loss_ratio: float = 0.0
    joint_loss_ratio: float = 1.0
    independent_loss_ratio: float = 0.0


@dataclass
class OphiuchusAbGenerateConfig:
    max_iter: int = 500
    sampling_strategy: str = "gumbel_argmax"
    temperature: float = 1.0
    cfg_scale: float = 0.0
    disable_resample: bool = True
    resample_ratio: float = 0.25


class MultiChainOphiuchusAbModel(nn.Module):
    def __init__(
        self,
        net: OphiuchusAbBackbone | None = None,
        train_config: OphiuchusAbTrainConfig | None = None,
    ) -> None:
        super().__init__()
        self.net = net or OphiuchusAbBackbone()
        self.cfg = train_config or OphiuchusAbTrainConfig()
        self.tokenizer = self.net.tokenizer
        self._prepare_special_token()
        if self.cfg.schedule == "uniform":
            self.distribution = Uniform(0, 1)
        else:
            raise ValueError(f"Unsupported schedule: {self.cfg.schedule}")

    def _prepare_special_token(self) -> None:
        self.mask_id = self.net.mask_id
        self.pad_id = self.net.pad_id
        self.bos_id = self.net.bos_id
        self.eos_id = self.net.eos_id
        self.unk_id = self.net.unk_id
        self.x_id = self.net.x_id
        self.b_id = self.net.b_id
        self.u_id = self.net.u_id
        self.z_id = self.net.z_id
        self.o_id = self.net.o_id

    @property
    def special_token_list(self) -> list[int]:
        return [
            self.bos_id,
            self.mask_id,
            self.pad_id,
            self.unk_id,
            self.x_id,
            self.b_id,
            self.u_id,
            self.z_id,
            self.o_id,
        ]

    def forward(self, input_ids: torch.Tensor, **kwargs):
        return self.net(input_ids=input_ids, **kwargs)["logits"]

    def q_sample_comp(self, x_0, region, t, maskable_mask):
        u = torch.rand_like(x_0, dtype=torch.float)
        t_adaptive = torch.where(region > 0, t[:, None], t[:, None])
        t_mask = (u < t_adaptive) & maskable_mask
        t_mask_inv = (u >= t_adaptive) & maskable_mask
        t_mask_inv = torch.where((t_adaptive == 0) | (t_adaptive == 1), t_mask, t_mask_inv)
        x_t = x_0.masked_fill(t_mask, self.mask_id)
        x_t_inv = x_0.masked_fill(t_mask_inv, self.mask_id)
        return torch.cat([x_t, x_t_inv], dim=0), torch.cat([t_mask, t_mask_inv], dim=0)

    def construct_x_t(self, heavy_target, light_target, heavy_region, light_region, eps=1e-3, stage=None):
        bsz = heavy_target.size(0)
        heavy_t = self.distribution.sample((bsz,)).to(heavy_target.device)
        heavy_t = (1 - eps) * heavy_t + eps
        light_t = self.distribution.sample((bsz,)).to(heavy_target.device)
        light_t = (1 - eps) * light_t + eps

        ratios = (
            self.cfg.single_chain_ratio,
            self.cfg.heavy2light_loss_ratio,
            self.cfg.light2heavy_loss_ratio,
            self.cfg.independent_loss_ratio,
            self.cfg.joint_loss_ratio,
        )
        assert abs(sum(ratios) - 1.0) < 1e-6

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
            maskable_mask=self.get_non_special_sym_mask(heavy_target),
        )

        light_t = light_t.masked_fill(light2heavy_index, 0)
        light_t = light_t.masked_fill(mask_light_index, 1)
        light_t = light_t.masked_scatter(joint_index, heavy_t[joint_index])
        light_x_t, light_loss_mask = self.q_sample_comp(
            light_target,
            light_region,
            light_t,
            maskable_mask=self.get_non_special_sym_mask(light_target),
        )

        heavy_t_inv = torch.where((heavy_t == 0) | (heavy_t == 1), heavy_t, 1 - heavy_t)
        light_t_inv = torch.where((light_t == 0) | (light_t == 1), light_t, 1 - light_t)

        return (
            {
                "t": torch.cat([heavy_t, heavy_t_inv], dim=0),
                "x_t": heavy_x_t,
                "mask": heavy_loss_mask,
            },
            {
                "t": torch.cat([light_t, light_t_inv], dim=0),
                "x_t": light_x_t,
                "mask": light_loss_mask,
            },
            mask_heavy_index.repeat(2),
            mask_light_index.repeat(2),
        )

    def compute_loss(self, batch, weighting="reciprocal", gamma=None, eps=1e-3, stage=None):
        heavy_target = batch["heavy_tokens"]["targets"]
        light_target = batch["light_tokens"]["targets"]
        heavy_region = batch["heavy_tokens"]["regions"]
        light_region = batch["light_tokens"]["regions"]

        heavy_noised, light_noised, mask_heavy_index, mask_light_index = self.construct_x_t(
            heavy_target,
            light_target,
            heavy_region,
            light_region,
            stage=stage,
        )
        x_t = torch.concat([heavy_noised["x_t"], light_noised["x_t"]], dim=1)
        batch = dict(batch)
        batch["chain_ids"] = torch.cat(
            [
                batch["heavy_tokens"]["chain_ids"].repeat(2, 1),
                batch["light_tokens"]["chain_ids"].repeat(2, 1),
            ],
            dim=1,
        )
        heavy_target = heavy_target.repeat(2, 1)
        light_target = light_target.repeat(2, 1)

        logits = self.forward(x_t, **batch)
        heavy_logits, light_logits = logits.split([heavy_target.size(1), light_target.size(1)], dim=1)

        if weighting == "reciprocal":
            heavy_weight = 1 / (heavy_noised["t"] + (1 / gamma)) if gamma else 1 / heavy_noised["t"]
            light_weight = 1 / (light_noised["t"] + (1 / gamma)) if gamma else 1 / light_noised["t"]
        elif weighting == "linear":
            heavy_weight = 1 + eps - heavy_noised["t"]
            light_weight = 1 + eps - light_noised["t"]
        else:
            heavy_weight = torch.ones_like(heavy_noised["t"])
            light_weight = torch.ones_like(light_noised["t"])

        heavy_weight = heavy_weight[:, None].float().expand(heavy_target.size()).clone()
        heavy_weight[mask_heavy_index] = 0.0
        light_weight = light_weight[:, None].float().expand(light_target.size()).clone()
        light_weight[mask_light_index] = 0.0

        return (
            {"heavy": heavy_logits, "light": light_logits},
            {"heavy": heavy_target, "light": light_target},
            {"heavy": heavy_noised["mask"], "light": light_noised["mask"]},
            {
                "heavy": heavy_weight * batch["weights"].repeat(2, 1),
                "light": light_weight * batch["weights"].repeat(2, 1),
            },
        )

    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = output_tokens.ne(self.pad_id) & output_tokens.ne(self.bos_id)
        if partial_masks is not None:
            non_special_sym_mask &= ~partial_masks
        return non_special_sym_mask

    def initialize_output_tokens(self, batch, partial_masks=None, **kwargs):
        tokens = batch["input_ids"]
        output_mask = self.get_non_special_sym_mask(tokens, partial_masks=partial_masks)
        output_tokens = tokens.masked_fill(output_mask, self.mask_id)
        output_scores = torch.zeros_like(output_tokens, dtype=torch.float)
        return output_tokens, output_scores

    def forward_decoder(
        self,
        prev_decoder_out,
        partial_masks=None,
        sampling_strategy="gumbel_argmax",
        cfg_scale=0.0,
    ):
        output_tokens = prev_decoder_out["output_tokens"].clone()
        output_scores = prev_decoder_out["output_scores"].clone()
        chain_ids = prev_decoder_out["chain_ids"].clone()
        output_masks = prev_decoder_out["output_masks"].clone()
        step = prev_decoder_out["step"]
        max_step = prev_decoder_out["max_step"]
        temperature = prev_decoder_out["temperature"]
        history = prev_decoder_out["history"]

        if cfg_scale > 0.0:
            un_output_tokens = output_tokens.clone()
            un_output_tokens[partial_masks] = self.mask_id
            output_tokens_ = torch.cat([output_tokens, un_output_tokens], dim=0)
            chain_ids_ = chain_ids.repeat(2, 1)
            net_out = self.net(input_ids=output_tokens_, chain_ids=chain_ids_)
            logits = net_out["logits"]
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            net_out = self.net(input_ids=output_tokens, chain_ids=chain_ids)
            logits = net_out["logits"]

        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        logits[..., self.special_token_list] = float("-inf")
        logits[..., self.eos_id] += 1e-3 * np.log(step / max_step + 1e-3)

        if sampling_strategy == "vanilla":
            _tokens, _scores = sample_from_categorical(logits, temperature=temperature)
        elif sampling_strategy == "argmax":
            _scores, _tokens = logits.max(dim=-1)
        elif sampling_strategy == "gumbel_argmax":
            _tokens, _scores = stochastic_sample_from_categorical(logits, temperature=0.0, noise_scale=1.0)
        else:
            raise NotImplementedError(sampling_strategy)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        history.append(output_tokens.clone())

        return {
            "output_tokens": output_tokens,
            "output_scores": output_scores,
            "step": step + 1,
            "max_step": max_step,
            "history": history,
        }

    def _decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0,
        non_special_sym_mask,
        t,
        max_step,
        noise,
    ):
        remasking, topk_mode, schedule = decoding_strategy.split("-")

        if schedule == "linear":
            rate = 1 - t / max_step
        elif schedule == "cosine":
            rate = np.cos(t / max_step * np.pi * 0.5)
        elif schedule == "root":
            rate = 1 - (t / max_step) ** 0.5
        else:
            raise NotImplementedError(schedule)

        cutoff_len = (non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate).long()

        if remasking == "confidence":
            scores_for_topk = cur_scores.masked_fill(~xt_neq_x0, 1000.0)
        elif remasking == "random":
            scores_for_topk = torch.rand_like(cur_scores)
            scores_for_topk = scores_for_topk.masked_fill(~xt_neq_x0, 1000.0)
        else:
            raise NotImplementedError(remasking)

        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError(topk_mode)

        masked_to_x0 = xt_neq_x0 & ~lowest_k_mask
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
        return lowest_k_mask, output_tokens, output_scores

    def generate(
        self,
        batch,
        max_iter: int | None = None,
        temperature: float | None = None,
        partial_masks=None,
        sampling_strategy: str = "gumbel_argmax",
        cfg_scale: float = 0.0,
    ):
        max_iter = max_iter or 500
        temperature = 1.0 if temperature is None else temperature

        initial_output_tokens, initial_output_scores = self.initialize_output_tokens(batch, partial_masks=partial_masks)
        prev_decoder_out = {
            "output_tokens": initial_output_tokens,
            "output_scores": initial_output_scores,
            "chain_ids": batch["chain_ids"],
            "output_masks": None,
            "step": 0,
            "max_step": max_iter,
            "history": [initial_output_tokens.clone()],
            "temperature": temperature,
        }
        prev_decoder_out["output_masks"] = self.get_non_special_sym_mask(
            prev_decoder_out["output_tokens"],
            partial_masks=partial_masks,
        )

        for step in range(max_iter):
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy,
                    cfg_scale=cfg_scale,
                )

            output_tokens = decoder_out["output_tokens"]
            output_scores = decoder_out["output_scores"]
            non_special_sym_mask = self.get_non_special_sym_mask(
                prev_decoder_out["output_tokens"],
                partial_masks=partial_masks,
            )
            result_masks, result_tokens, result_scores = self._decoding(
                output_tokens=prev_decoder_out["output_tokens"].clone(),
                output_scores=prev_decoder_out["output_scores"].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy="confidence-deterministic-linear",
                xt_neq_x0=prev_decoder_out["output_masks"].clone(),
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
                noise=self.mask_id,
            )
            prev_decoder_out.update(
                output_masks=result_masks,
                output_tokens=result_tokens,
                output_scores=result_scores,
                step=step + 1,
                history=decoder_out["history"],
            )

        return prev_decoder_out["output_tokens"], prev_decoder_out["output_scores"]

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str | torch.device = "cpu"):
        model = cls()
        load_ophiuchus_checkpoint(model, checkpoint_path, device=device)
        return model


def load_ophiuchus_checkpoint(
    model: MultiChainOphiuchusAbModel,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("state_dict", payload)
    converted = OrderedDict()
    for key, value in state_dict.items():
        normalized = key[6:] if key.startswith("model.") else key
        converted[normalized] = value
    missing, unexpected = model.net.model.load_state_dict(converted, strict=strict)
    return list(missing), list(unexpected)
