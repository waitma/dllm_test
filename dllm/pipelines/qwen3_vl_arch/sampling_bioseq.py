"""Grammar-aware iterative denoising for BioSeq diffusion models.

Run smoke tests with::

    pytest scripts/tests/bioseq/test_sampling_bioseq.py -q
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dllm.pipelines.bioseq.ophiuchus.sampling import (
    sample_from_categorical,
    stochastic_sample_from_categorical,
    topk_masking,
)
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    BioSeqEncoderDiffusionModel,
    BioSeqNoEncoderDiffusionModel,
    apply_decoder_corruption_to_encoder,
    forbidden_diffusion_target_token_ids,
    mask_forbidden_target_logits,
)

BioSeqDiffusionModel = BioSeqEncoderDiffusionModel | BioSeqNoEncoderDiffusionModel


@dataclass
class BioSeqGenerateConfig:
    max_iter: int = 500
    sampling_strategy: str = "gumbel_argmax"
    temperature: float = 1.0
    decoding_strategy: str = "confidence-deterministic-linear"
    cfg_scale: float = 0.0


def resolve_partial_mask(
    batch: dict[str, Any],
    partial_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the visibility mask for inference.

    Positions marked True stay fixed at ``input_ids``; False positions are
    iteratively unmasked by the denoiser. Structure, relation, and
    ``fixed_context_mask`` positions are always forced visible.
    """

    attention = batch["attention_mask"].bool()
    forced_visible = batch.get("fixed_context_mask")
    if forced_visible is None:
        forced_visible = torch.zeros_like(attention, dtype=torch.bool)
    else:
        forced_visible = forced_visible.bool()

    for key in ("structure_token_mask", "relation_token_mask"):
        extra = batch.get(key)
        if extra is not None:
            forced_visible = forced_visible | extra.bool()

    if partial_mask is None:
        loss_mask = batch.get("diffusion_loss_mask", batch.get("diffusion_target_mask"))
        if loss_mask is not None:
            partial_mask = attention & ~loss_mask.bool()
        else:
            residue_mask = batch.get("residue_mask")
            partial_mask = attention & ~residue_mask.bool() if residue_mask is not None else ~attention
    else:
        partial_mask = partial_mask.bool()

    return (partial_mask | forced_visible) & attention


def build_generation_mask(
    batch: dict[str, Any],
    partial_mask: torch.Tensor,
) -> torch.Tensor:
    """Positions eligible for iterative unmasking."""

    attention = batch["attention_mask"].bool()
    eligible = batch.get("diffusion_eligible_mask", batch.get("diffusion_loss_mask"))
    if eligible is None:
        residue_mask = batch.get("residue_mask")
        eligible = residue_mask.bool() if residue_mask is not None else attention
    else:
        eligible = eligible.bool()
    return eligible & ~partial_mask & attention


def initialize_output_tokens(
    input_ids: torch.Tensor,
    generation_mask: torch.Tensor,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_tokens = input_ids.clone()
    output_tokens = output_tokens.masked_fill(generation_mask, int(mask_token_id))
    output_scores = torch.zeros_like(output_tokens, dtype=torch.float32)
    return output_tokens, output_scores


def _inference_timesteps(step: int, max_step: int, time_epsilon: float, device: torch.device) -> torch.Tensor:
    if max_step <= 0:
        return torch.ones(1, device=device, dtype=torch.float32)
    progress = float(step) / float(max_step)
    value = (1.0 - progress) * (1.0 - time_epsilon) + time_epsilon
    return torch.tensor([value], device=device, dtype=torch.float32)


def _model_logits(
    model: BioSeqDiffusionModel,
    batch: dict[str, Any],
    output_tokens: torch.Tensor,
    generation_mask: torch.Tensor,
    mask_token_id: int,
    timesteps: torch.Tensor,
    cfg_scale: float,
    partial_mask: torch.Tensor,
) -> torch.Tensor:
    corruption_mask = output_tokens.eq(int(mask_token_id)) & generation_mask
    forward_kwargs: dict[str, Any] = {
        "input_ids": output_tokens,
        "attention_mask": batch.get("attention_mask"),
        "position_ids_inner": batch.get("position_ids_inner"),
        "position_ids_chain": batch.get("position_ids_chain"),
        "timesteps": timesteps.expand(output_tokens.size(0)),
    }

    if isinstance(model, BioSeqEncoderDiffusionModel):
        encoder_batch = dict(batch)
        encoder_batch["input_ids"] = output_tokens
        noised_encoder_input_ids = apply_decoder_corruption_to_encoder(
            batch=encoder_batch,
            corruption_mask=corruption_mask,
            mask_token_id=int(mask_token_id),
        )
        forward_kwargs.update(
            {
                "residue_mask": batch.get("residue_mask"),
                "encoder_input_ids": noised_encoder_input_ids,
                "encoder_attention_mask": batch.get("encoder_attention_mask"),
                "encoder_residue_mask": batch.get("encoder_residue_mask"),
                "encoder_chain_mask": batch.get("encoder_chain_mask"),
                "encoder_position_ids": batch.get("encoder_position_ids"),
                "chain_ids": batch.get("chain_ids"),
            }
        )
        if cfg_scale > 0.0:
            unmasked_tokens = output_tokens.clone()
            unmasked_tokens[partial_mask] = int(mask_token_id)
            un_encoder_batch = dict(encoder_batch)
            un_encoder_batch["input_ids"] = unmasked_tokens
            un_corruption = unmasked_tokens.eq(int(mask_token_id)) & generation_mask
            un_noised_encoder = apply_decoder_corruption_to_encoder(
                batch=un_encoder_batch,
                corruption_mask=un_corruption,
                mask_token_id=int(mask_token_id),
            )
            cond_out = model(
                **forward_kwargs,
                input_ids=output_tokens,
                encoder_input_ids=noised_encoder_input_ids,
            )
            uncond_out = model(
                **forward_kwargs,
                input_ids=unmasked_tokens,
                encoder_input_ids=un_noised_encoder,
            )
            logits = uncond_out.logits + (cfg_scale + 1.0) * (cond_out.logits - uncond_out.logits)
            forbidden = forbidden_diffusion_target_token_ids(model.config)
            return mask_forbidden_target_logits(logits, forbidden)

        output = model(**forward_kwargs)
        forbidden = forbidden_diffusion_target_token_ids(model.config)
        return mask_forbidden_target_logits(output.logits, forbidden)

    output = model(**forward_kwargs)
    forbidden = forbidden_diffusion_target_token_ids(model.config)
    return mask_forbidden_target_logits(output.logits, forbidden)


def _sample_tokens(
    logits: torch.Tensor,
    sampling_strategy: str,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_strategy == "vanilla":
        return sample_from_categorical(logits, temperature=temperature)
    if sampling_strategy == "argmax":
        scores, tokens = logits.max(dim=-1)
        return tokens, scores.to(torch.float32)
    if sampling_strategy == "gumbel_argmax":
        return stochastic_sample_from_categorical(logits, temperature=0.0, noise_scale=1.0)
    raise NotImplementedError(sampling_strategy)


def _confidence_decoding(
    output_tokens: torch.Tensor,
    output_scores: torch.Tensor,
    cur_tokens: torch.Tensor,
    cur_scores: torch.Tensor,
    decoding_strategy: str,
    still_masked: torch.Tensor,
    generation_mask: torch.Tensor,
    step: int,
    max_step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    remasking, topk_mode, schedule = decoding_strategy.split("-")

    if schedule == "linear":
        rate = 1.0 - step / max_step
    elif schedule == "cosine":
        rate = float(np.cos(step / max_step * np.pi * 0.5))
    elif schedule == "root":
        rate = 1.0 - (step / max_step) ** 0.5
    else:
        raise NotImplementedError(schedule)

    active = generation_mask
    cutoff_len = (active.sum(dim=1, keepdim=True).type_as(output_scores) * rate).long()

    if remasking == "confidence":
        scores_for_topk = cur_scores.masked_fill(~still_masked, 1000.0)
    elif remasking == "random":
        scores_for_topk = torch.rand_like(cur_scores)
        scores_for_topk = scores_for_topk.masked_fill(~still_masked, 1000.0)
    else:
        raise NotImplementedError(remasking)

    if topk_mode.startswith("stochastic"):
        noise_scale = float(topk_mode.replace("stochastic", ""))
        lowest_k_mask = topk_masking(scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
    elif topk_mode == "deterministic":
        lowest_k_mask = topk_masking(scores_for_topk, cutoff_len, stochastic=False)
    else:
        raise NotImplementedError(topk_mode)

    commit = still_masked & ~lowest_k_mask
    output_tokens = output_tokens.masked_scatter(commit, cur_tokens[commit])
    output_scores = output_scores.masked_scatter(commit, cur_scores[commit])
    still_masked = still_masked & lowest_k_mask
    return still_masked, output_tokens, output_scores


@torch.no_grad()
def generate_bioseq(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    partial_mask: torch.Tensor | None = None,
    config: BioSeqGenerateConfig | None = None,
    return_history: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Iteratively denoise a grammar-rendered batch."""

    if not isinstance(model, (BioSeqEncoderDiffusionModel, BioSeqNoEncoderDiffusionModel)):
        raise TypeError("generate_bioseq expects BioSeqEncoderDiffusionModel or BioSeqNoEncoderDiffusionModel")

    config = config or BioSeqGenerateConfig()
    input_ids = batch["input_ids"]
    device = input_ids.device
    mask_token_id = int(model.config.mask_token_id)
    partial_mask = resolve_partial_mask(batch, partial_mask)
    generation_mask = build_generation_mask(batch, partial_mask)

    output_tokens, output_scores = initialize_output_tokens(input_ids, generation_mask, mask_token_id)
    still_masked = generation_mask.clone()
    history = [output_tokens.clone()]

    for step in range(config.max_iter):
        timesteps = _inference_timesteps(step + 1, config.max_iter, model.config.time_epsilon, device)
        logits = _model_logits(
            model=model,
            batch=batch,
            output_tokens=output_tokens,
            generation_mask=generation_mask,
            mask_token_id=mask_token_id,
            timesteps=timesteps,
            cfg_scale=config.cfg_scale,
            partial_mask=partial_mask,
        )
        sampled_tokens, sampled_scores = _sample_tokens(logits, config.sampling_strategy, config.temperature)
        output_tokens = output_tokens.masked_scatter(still_masked, sampled_tokens[still_masked])
        output_scores = output_scores.masked_scatter(still_masked, sampled_scores[still_masked])
        history.append(output_tokens.clone())

        still_masked, output_tokens, output_scores = _confidence_decoding(
            output_tokens=output_tokens,
            output_scores=output_scores,
            cur_tokens=output_tokens,
            cur_scores=output_scores,
            decoding_strategy=config.decoding_strategy,
            still_masked=still_masked,
            generation_mask=generation_mask,
            step=step + 1,
            max_step=config.max_iter,
        )
        if not still_masked.any():
            break

    if return_history:
        return output_tokens, output_scores, history
    return output_tokens, output_scores
