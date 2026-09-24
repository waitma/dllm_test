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
    apply_decoder_values_to_encoder,
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
    editing_threshold: float = 0.0
    max_post_steps: int = 16
    self_correct: bool = False
    self_correct_temperature: float = 0.1
    # Decoder-space allowlist for fixed-window infilling (EOS is not a residue).
    allowed_token_ids: tuple[int, ...] | None = None


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

    structure_mask = batch.get("structure_token_mask")
    if structure_mask is not None:
        forced_visible = forced_visible | structure_mask.bool()
    relation_mask = batch.get("relation_token_mask")
    relation_targets = batch.get("relation_target_mask")
    if relation_mask is not None:
        if relation_targets is None:
            forced_visible = forced_visible | relation_mask.bool()
        else:
            forced_visible = forced_visible | (relation_mask.bool() & ~relation_targets.bool())

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


def _encoder_input_from_output_tokens(
    model: BioSeqDiffusionModel,
    batch: dict[str, Any],
    output_tokens: torch.Tensor,
    generation_mask: torch.Tensor,
    mask_token_id: int,
    encoder_mask_id: int,
) -> torch.Tensor:
    """Mirror committed decoder residues, never reference targets, into ESMC.

    Every generated slot is overwritten on every call: pending or unsupported
    tokens become encoder MASK, and accepted residues are vocabulary-translated.
    Thus shrinking the pending mask cannot reveal the original clean targets.
    """

    fixed_canvas = "encoder_slot_mask" in batch
    slots = batch["chain_slot_mask"] if fixed_canvas else batch["residue_mask"]
    accepted = generation_mask & output_tokens.ne(mask_token_id) & slots.bool()
    inverse = getattr(model, "llada_to_grammar_ids", None)
    if inverse is not None:
        # Fusion uses LLaDA <res_A> IDs in the decoder but ESMC A IDs in
        # encoder_input_ids. Grammar delimiters must not be fed into ESMC.
        valid_ids = output_tokens.ge(0) & output_tokens.lt(inverse.numel())
        mapped = inverse[output_tokens.clamp(min=0, max=inverse.numel() - 1)]
        residue_ids = getattr(model, "_residue_token_ids", None)
        if residue_ids is None:
            raise ValueError("Remapped fusion inference requires decoder residue token IDs")
        supported = torch.isin(output_tokens, residue_ids)
        if fixed_canvas:
            eos_id = getattr(model.config, "decoder_chain_eos_token_id", None)
            if eos_id is None:
                raise ValueError("v2 fusion inference requires decoder_chain_eos_token_id")
            supported = supported | output_tokens.eq(int(eos_id))
        accepted = accepted & valid_ids & mapped.ge(0) & supported
    else:
        # Native BioSeq models share their residue vocabulary with the encoder.
        mapped = output_tokens
    encoder_values = torch.where(accepted, mapped, int(encoder_mask_id))
    return apply_decoder_values_to_encoder(batch, generation_mask, encoder_values)


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

    forward_kwargs: dict[str, Any] = {
        "input_ids": output_tokens,
        "attention_mask": batch.get("attention_mask"),
        "position_ids_inner": batch.get("position_ids_inner"),
        "position_ids_chain": batch.get("position_ids_chain"),
        "timesteps": timesteps.expand(output_tokens.size(0)),
    }

    if isinstance(model, BioSeqEncoderDiffusionModel):
        encoder_mask_id = int(
            getattr(getattr(model, "config", None), "encoder_mask_token_id", None)
            or mask_token_id
        )
        noised_encoder_input_ids = _encoder_input_from_output_tokens(
            model=model, batch=batch, output_tokens=output_tokens,
            generation_mask=generation_mask, mask_token_id=mask_token_id,
            encoder_mask_id=encoder_mask_id,
        )
        forward_kwargs.update(
            {
                "residue_mask": batch.get("chain_slot_mask", batch.get("residue_mask")) if "encoder_slot_mask" in batch else batch.get("residue_mask"),
                "encoder_input_ids": noised_encoder_input_ids,
                "encoder_attention_mask": batch.get("encoder_attention_mask"),
                "encoder_residue_mask": batch.get("encoder_slot_mask", batch.get("encoder_residue_mask")),
                "encoder_chain_mask": batch.get("encoder_chain_mask"),
                "encoder_position_ids": batch.get("encoder_position_ids"),
                "chain_ids": batch.get("chain_ids"),
            }
        )
        if "encoder_slot_mask" in batch:
            forward_kwargs["encoder_slot_mask"] = batch["encoder_slot_mask"]
            forward_kwargs["chain_slot_mask"] = batch["chain_slot_mask"]
        if cfg_scale > 0.0:
            # Residue-condition CFG: remove the observed sequence condition in
            # both streams, not grammar/type/termination/relation tokens or PAD.
            # The current generated state is identical in the two passes.
            visible_slots = batch["chain_slot_mask"] if "encoder_slot_mask" in batch else batch["residue_mask"]
            condition_mask = partial_mask & visible_slots.bool()
            unmasked_tokens = output_tokens.clone()
            unmasked_tokens[condition_mask] = int(mask_token_id)
            un_encoder_batch = {**batch, "encoder_input_ids": noised_encoder_input_ids}
            un_noised_encoder = apply_decoder_corruption_to_encoder(
                batch=un_encoder_batch,
                corruption_mask=condition_mask,
                mask_token_id=encoder_mask_id,
            )
            denoise = getattr(model, "_denoise", None)
            forward = denoise if callable(denoise) else model
            cond_out = forward(**forward_kwargs)
            uncond_kwargs = {**forward_kwargs, "input_ids": unmasked_tokens,
                             "encoder_input_ids": un_noised_encoder}
            uncond_out = forward(**uncond_kwargs)
            logits = uncond_out.logits + (cfg_scale + 1.0) * (cond_out.logits - uncond_out.logits)
            forbidden = forbidden_diffusion_target_token_ids(model.config)
            return mask_forbidden_target_logits(logits, forbidden)

        denoise = getattr(model, "_denoise", None)
        output = denoise(**forward_kwargs) if callable(denoise) else model(**forward_kwargs)
        forbidden = forbidden_diffusion_target_token_ids(model.config)
        return mask_forbidden_target_logits(output.logits, forbidden)

    denoise = getattr(model, "_denoise", None)
    output = denoise(**forward_kwargs) if callable(denoise) else model(**forward_kwargs)
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


def _committed_generation_mask(
    generation_mask: torch.Tensor,
    still_masked: torch.Tensor | None = None,
) -> torch.Tensor:
    committed = generation_mask.bool()
    if still_masked is not None:
        committed = committed & ~still_masked.bool()
    return committed


def llada2_edit_positions(
    tokens: torch.Tensor,
    pred: torch.Tensor,
    confidence: torch.Tensor,
    generation_mask: torch.Tensor,
    threshold: float,
    still_masked: torch.Tensor | None = None,
) -> torch.Tensor:
    """LLaDA2 edit-remask index: committed generation sites the model wants to rewrite.

    A position is selected iff it is in ``generation_mask``, is not still masked,
    ``pred`` differs from the current token, and ``confidence`` exceeds ``threshold``.
    Fixed context / remaining masks are never selected. ``threshold <= 0`` yields
    an empty mask so the default generate path stays a no-op.
    """

    if threshold <= 0.0:
        return torch.zeros_like(generation_mask, dtype=torch.bool)
    committed = _committed_generation_mask(generation_mask, still_masked)
    return committed & pred.ne(tokens) & confidence.gt(threshold)


def apply_llada2_edits(
    tokens: torch.Tensor,
    pred: torch.Tensor,
    confidence: torch.Tensor,
    generation_mask: torch.Tensor,
    threshold: float,
    still_masked: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one LLaDA2 threshold-edit step. Returns a new token tensor."""

    replace = llada2_edit_positions(
        tokens,
        pred,
        confidence,
        generation_mask,
        threshold,
        still_masked,
    )
    if not replace.any():
        return tokens
    return tokens.masked_scatter(replace, pred[replace])


def gidd_self_correct_positions(
    tokens: torch.Tensor,
    pred: torch.Tensor,
    pred_prob: torch.Tensor,
    generation_mask: torch.Tensor,
    still_masked: torch.Tensor | None = None,
) -> torch.Tensor:
    """GIDD self-correct index: at most one committed disagreement per sequence.

    Among ``generation_mask`` positions that are already committed and where
    ``pred != current``, pick the site with the highest model probability of
    the *new* token. Returns a boolean mask (empty if nothing disagrees).
    """

    committed = _committed_generation_mask(generation_mask, still_masked)
    eligible = committed & pred.ne(tokens)
    replace = torch.zeros_like(generation_mask, dtype=torch.bool)
    if not eligible.any():
        return replace
    scores = pred_prob.to(dtype=torch.float32).masked_fill(~eligible, float("-inf"))
    best = scores.argmax(dim=-1)
    has_any = eligible.any(dim=-1) & torch.isfinite(scores.max(dim=-1).values)
    if not has_any.any():
        return replace
    batch_idx = torch.arange(tokens.size(0), device=tokens.device)
    replace[batch_idx[has_any], best[has_any]] = True
    return replace


def apply_gidd_self_correct(
    tokens: torch.Tensor,
    pred: torch.Tensor,
    pred_prob: torch.Tensor,
    generation_mask: torch.Tensor,
    still_masked: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one GIDD self-correct step. Returns a new token tensor."""

    replace = gidd_self_correct_positions(
        tokens,
        pred,
        pred_prob,
        generation_mask,
        still_masked,
    )
    if not replace.any():
        return tokens
    return tokens.masked_scatter(replace, pred[replace])


def token_softmax_confidence(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Softmax probability of ``tokens`` under ``logits`` (``[B, S, V]`` → ``[B, S]``)."""

    probs = torch.softmax(logits.float(), dim=-1)
    return probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def _self_correct_prediction(
    logits: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = logits.float()
    if temperature > 0.0:
        scaled = scaled / temperature
    probs = torch.softmax(scaled, dim=-1)
    pred_prob, pred = probs.max(dim=-1)
    return pred, pred_prob


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

    allowed = config.allowed_token_ids
    if allowed is not None and (not allowed or min(allowed) < 0 or max(allowed) >= model.config.vocab_size):
        raise ValueError("allowed_token_ids must be a nonempty decoder-vocabulary subset")

    def constrain(logits):
        if allowed is None:
            return logits
        keep = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
        keep[list(allowed)] = True
        return logits.masked_fill(generation_mask.unsqueeze(-1) & ~keep, float("-inf"))

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
        logits = constrain(logits)
        sampled_tokens, sampled_scores = _sample_tokens(logits, config.sampling_strategy, config.temperature)
        # Keep the committed state separate from this step's proposals. Pending
        # sites must remain MASK in the actual next forward, not merely in the
        # bookkeeping mask. This also matches the original AirGen decoder.
        still_masked, output_tokens, output_scores = _confidence_decoding(
            output_tokens=output_tokens,
            output_scores=output_scores,
            cur_tokens=sampled_tokens,
            cur_scores=sampled_scores,
            decoding_strategy=config.decoding_strategy,
            still_masked=still_masked,
            generation_mask=generation_mask,
            step=step + 1,
            max_step=config.max_iter,
        )
        history.append(output_tokens.clone())
        if not still_masked.any():
            break

    if config.editing_threshold > 0.0 or config.self_correct:
        edit_timesteps = _inference_timesteps(
            config.max_iter,
            config.max_iter,
            model.config.time_epsilon,
            device,
        )
        if config.editing_threshold > 0.0:
            for _ in range(config.max_post_steps):
                logits = _model_logits(
                    model=model,
                    batch=batch,
                    output_tokens=output_tokens,
                    generation_mask=generation_mask,
                    mask_token_id=mask_token_id,
                    timesteps=edit_timesteps,
                    cfg_scale=config.cfg_scale,
                    partial_mask=partial_mask,
                )
                logits = constrain(logits)
                sampled_tokens, sampled_scores = _sample_tokens(
                    logits, config.sampling_strategy, config.temperature
                )
                confidence = token_softmax_confidence(logits, sampled_tokens)
                replace = llada2_edit_positions(
                    output_tokens,
                    sampled_tokens,
                    confidence,
                    generation_mask,
                    config.editing_threshold,
                    still_masked,
                )
                if not replace.any():
                    break
                output_tokens = output_tokens.masked_scatter(replace, sampled_tokens[replace])
                output_scores = output_scores.masked_scatter(replace, sampled_scores[replace])
                history.append(output_tokens.clone())

        if config.self_correct:
            for _ in range(config.max_post_steps):
                logits = _model_logits(
                    model=model,
                    batch=batch,
                    output_tokens=output_tokens,
                    generation_mask=generation_mask,
                    mask_token_id=mask_token_id,
                    timesteps=edit_timesteps,
                    cfg_scale=config.cfg_scale,
                    partial_mask=partial_mask,
                )
                logits = constrain(logits)
                pred, pred_prob = _self_correct_prediction(
                    logits, config.self_correct_temperature
                )
                replace = gidd_self_correct_positions(
                    output_tokens,
                    pred,
                    pred_prob,
                    generation_mask,
                    still_masked,
                )
                if not replace.any():
                    break
                output_tokens = output_tokens.masked_scatter(replace, pred[replace])
                output_scores = output_scores.masked_scatter(replace, pred_prob[replace])
                history.append(output_tokens.clone())

    if return_history:
        return output_tokens, output_scores, history
    return output_tokens, output_scores
