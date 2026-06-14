from __future__ import annotations

import torch


def topk_masking(scores: torch.Tensor, cutoff_len: torch.Tensor, stochastic: bool = False, temp: float = 1.0) -> torch.Tensor:
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        scored = scores + temp * gumbel_noise
    else:
        scored = scores
    sorted_index = scored.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    return scored < cutoff


def sample_from_categorical(logits: torch.Tensor | None = None, temperature: float = 1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def stochastic_sample_from_categorical(
    logits: torch.Tensor | None = None,
    temperature: float = 1.0,
    noise_scale: float = 1.0,
):
    logits = logits.to(torch.float64)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    logits_with_noise = logits + noise_scale * gumbel_noise
    probs = logits_with_noise.softmax(dim=-1)
    confidence, tokens = probs.max(dim=-1)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    confidence = sorted_probs[..., 0] - sorted_probs[..., 1]
    return tokens, confidence.to(torch.float32)
