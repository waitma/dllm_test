"""BioSeq masked-diffusion model stack for grammar-v1 foundation training.

Run training with::

    torchrun examples/bioseq/train_qwen3_vl_bioseq_ddp.py --model-type no_encoder ...

Tensor shape notation used throughout this module:

- ``B``: batch size
- ``S``: decoder sequence length (concatenated grammar record, padded)
- ``C``: max chains per record (encoder path; grammar-v1 proxy uses ``C=1``)
- ``L``: per-chain encoder length
- ``H``: decoder hidden size (``config.hidden_size``, e.g. 512)
- ``E``: encoder hidden size (``config.condition_hidden_size``, e.g. 960 for ESMC-300M)
- ``V``: vocabulary size (``config.vocab_size``, e.g. 56 for grammar + ESMC tokens)

High-level data flow (training)::

    collator batch
      -> sample_bioseq_diffusion_noise  (x_t, labels, corruption_mask, t)
      -> [encoder] ESMC(noisy proxy) -> gather -> encoder_condition [B, S, E]
      -> BioSeqDiffusionDecoder(x_t, t, encoder_condition?) -> logits [B, S, V]
      -> compute_masked_cross_entropy(logits, labels)

The decoder is a bidirectional transformer (not causal LM). ``lm_head.weight`` is
tied to ``token_embeddings.weight``. ESMC ``sequence_head`` is never used for
the diffusion objective; only ``last_hidden_state`` conditions the decoder.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class BioSeqDiffusionTransformerConfig:
    """Training config for the BioSeq diffusion decoder.

    The default vocabulary follows the local ESM2/MINT ids. For ESMC-tokenized
    training, set ``vocab_size`` to the encoder tokenizer vocabulary size.
    """

    vocab_size: int = 33
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.1
    max_position_embeddings: int = 4096
    max_chain_positions: int = 64
    max_chain_roles: int = 32  # unused; kept for checkpoint compatibility
    max_task_types: int = 32  # unused; kept for checkpoint compatibility
    pad_token_id: int = 1
    mask_token_id: int = 32
    forbidden_target_token_ids: tuple[int, ...] | None = None
    qk_norm: bool = False
    time_epsilon: float = 1e-3
    loss_norm: str = "token"
    condition_hidden_size: int | None = None
    use_condition_projection: bool = False
    condition_norm: bool = False
    gradient_checkpointing: bool = False
    initializer_range: float = 0.02
    # LLaDA2 MoE backbone knobs (only read by build_llada2_backbone /
    # BioSeqLLaDA2EncoderDiffusionModel). Defaults mirror inclusionAI/LLaDA2.0-mini
    # so the architecture stays faithful; scale them down via CLI for DDP-sized
    # from-scratch runs. ``num_hidden_layers`` / ``num_attention_heads`` /
    # ``hidden_size`` / ``intermediate_size`` are shared with the fields above.
    moe_num_experts: int = 256
    moe_num_experts_per_tok: int = 8
    moe_num_shared_experts: int = 1
    moe_intermediate_size: int = 512
    moe_n_group: int = 8
    moe_topk_group: int = 4
    moe_first_k_dense_replace: int = 1
    moe_num_key_value_heads: int = 4
    moe_head_dim: int = 128
    moe_partial_rotary_factor: float = 0.5
    moe_rope_theta: float = 600000.0
    moe_routed_scaling_factor: float = 2.5


@dataclass
class BioSeqDiffusionOutput:
    """Forward / loss bundle returned by decoder and top-level models.

    Typical shapes (grammar-v1, batch size ``B``, seq ``S``, vocab ``V``, hidden ``H``):

    - ``logits``: ``[B, S, V]`` — per-token vocabulary scores before softmax.
    - ``hidden_states``: ``[B, S, H]`` — final decoder representations.
    - ``loss``: scalar when ``compute_loss`` is used.
    - ``noised_input_ids`` / ``labels``: ``[B, S]``; labels use ``-100`` on non-corrupted positions.
    - ``corruption_mask``: ``[B, S]`` bool — positions replaced with ``<mask>``.
    - ``timesteps``: ``[B]`` float in ``(time_epsilon, 1]``.
    - ``noised_encoder_input_ids``: ``[B, C, L]`` (encoder path only).
    - ``encoder_condition``: ``[B, S, E]`` gathered ESMC features (encoder path only).
    - ``relation_aux_loss``: scalar pairing / chain-drop auxiliary (opt-in).
    """

    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: torch.Tensor
    noised_input_ids: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    corruption_mask: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    noised_encoder_input_ids: torch.Tensor | None = None
    encoder_condition: torch.Tensor | None = None
    relation_aux_loss: torch.Tensor | None = None


class BioSeqRMSNorm(nn.Module):
    """Root-mean-square layer norm (LLaMA-style). Preserves input rank."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input/output: ``[..., H]``."""
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states * self.weight


class BioSeqSwiGLU(nn.Module):
    """Feed-forward block: ``H -> I -> H`` with SiLU gating (SwiGLU)."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input/output: ``[B, S, H]``."""
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.down_proj(hidden_states)


class BioSeqSelfAttention(nn.Module):
    """Bidirectional self-attention for masked diffusion (``is_causal=False``).

    Projects ``H`` into multi-head Q/K/V, runs scaled dot-product attention over
    the full sequence, and projects back to ``H``. Padding positions are masked
    via ``attention_mask`` (1 = attend, 0 = ignore).
    """

    def __init__(self, config: BioSeqDiffusionTransformerConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.qk_norm = bool(getattr(config, "qk_norm", False))
        if self.qk_norm:
            self.q_norm = BioSeqRMSNorm(self.head_dim)
            self.k_norm = BioSeqRMSNorm(self.head_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Input ``hidden_states``: ``[B, S, H]``. Output: ``[B, S, H]``."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        query = self._shape(self.q_proj(hidden_states), batch_size, seq_len)
        key = self._shape(self.k_proj(hidden_states), batch_size, seq_len)
        value = self._shape(self.v_proj(hidden_states), batch_size, seq_len)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        key_mask = attention_mask.bool()[:, None, None, :] if attention_mask is not None else None
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=key_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(context)

    def _shape(self, states: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshape linear projection output to multi-head layout ``[B, heads, S, head_dim]``."""
        return states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


class BioSeqTransformerBlock(nn.Module):
    """Pre-norm transformer block: attention + SwiGLU MLP with residual connections."""

    def __init__(self, config: BioSeqDiffusionTransformerConfig) -> None:
        super().__init__()
        self.input_layernorm = BioSeqRMSNorm(config.hidden_size)
        self.self_attn = BioSeqSelfAttention(config)
        self.post_attention_layernorm = BioSeqRMSNorm(config.hidden_size)
        self.mlp = BioSeqSwiGLU(config.hidden_size, config.intermediate_size, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Input/output: ``[B, S, H]``."""
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), attention_mask=attention_mask)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + self.dropout(hidden_states)


class BioSeqTimestepEmbedding(nn.Module):
    """Sinusoidal diffusion timestep embedding, broadcast-added to every token."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Input ``timesteps``: ``[B]``. Output: ``[B, H]`` (added to all ``S`` positions)."""
        half_dim = self.hidden_size // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
            / max(half_dim - 1, 1)
        )
        args = timesteps[:, None] * frequencies[None]
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if embeddings.shape[-1] < self.hidden_size:
            embeddings = F.pad(embeddings, (0, self.hidden_size - embeddings.shape[-1]))
        return self.proj(embeddings)


def _diffusion_eligible_mask(
    batch: dict[str, Any],
    *,
    require_residue: bool = False,
) -> torch.Tensor:
    """Boolean ``[B, S]`` mask of tokens that may be corrupted / receive loss.

    Matches the historical ``sample_bioseq_diffusion_noise`` rule: when an
    explicit ``diffusion_eligible_mask`` is present it is trusted (residue is
    *not* AND-ed unless ``require_residue=True``). Otherwise residue tokens are
    required when ``residue_mask`` exists.
    """

    loss_mask = batch.get("diffusion_loss_mask", batch.get("diffusion_target_mask"))
    if loss_mask is None:
        raise KeyError("batch requires diffusion_loss_mask or diffusion_target_mask")
    explicit_eligible_mask = batch.get("diffusion_eligible_mask")
    eligible_mask = (
        explicit_eligible_mask.bool()
        if explicit_eligible_mask is not None
        else loss_mask.bool()
    )
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        eligible_mask = eligible_mask & attention_mask.bool()
    residue_mask = batch.get("residue_mask")
    if residue_mask is not None and (require_residue or explicit_eligible_mask is None):
        eligible_mask = eligible_mask & residue_mask.bool()
    return eligible_mask


def _sample_uniform_residues(
    current_ids: torch.Tensor,
    residue_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Draw a residue id different from ``current_ids`` when the pool allows it."""

    pool = residue_token_ids.reshape(-1)
    pool_size = int(pool.numel())
    if pool_size <= 0:
        raise ValueError("residue_token_ids must be non-empty")
    draw_index = torch.randint(pool_size, current_ids.shape, device=current_ids.device)
    picks = pool[draw_index]
    if pool_size > 1:
        same = picks.eq(current_ids)
        if same.any():
            picks = torch.where(same, pool[(draw_index + 1) % pool_size], picks)
    return picks


def sample_bioseq_diffusion_noise(
    batch: dict[str, Any],
    mask_token_id: int,
    time_epsilon: float = 1e-3,
    uniform_ratio: float = 0.0,
    gidd_gamma: float = 1.0,
    residue_token_ids: torch.Tensor | None = None,
    token_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample masked-diffusion corruption for one training step.

    Follows the AirGen/Ophiuchus-style forward process: per sequence sample
    ``t ~ Uniform(time_epsilon, 1)``, then independently mask each eligible
    token with probability ``t``. Fixed grammar context (``<fixs>...<fixd>``)
    is excluded via ``diffusion_loss_mask`` / ``diffusion_eligible_mask``.

    When ``uniform_ratio == 0`` (the default) this is exactly Bernoulli(``t``)
    absorbing-mask, including the per-row ``>=1`` mask guarantee. Optional
    ``token_t`` ``[B, S]`` replaces the per-sequence ``t`` on a per-token basis
    without changing that algorithm.

    When ``0 < uniform_ratio < 1``, eligible *residue* positions use the GIDD
    hybrid marginal (keep / ``<mask>`` / uniform other residue). Grammar and
    structure tokens are never uniformly replaced. Labels cover both mask and
    uniform sites; ``corruption_mask`` is True on both so encoder mirroring
    hides every corrupted target.

    Returns
    -------
    noised_input_ids : ``[B, S]`` — clean ids with corrupted positions rewritten.
    labels : ``[B, S]`` — clean ids on corrupted positions; ``-100`` elsewhere.
    corruption_mask : ``[B, S]`` bool — True on mask *and* uniform sites.
    timesteps : ``[B]`` float — per-sequence noise level (mean of ``token_t``
        on eligible positions when ``token_t`` is provided).
    """

    if not (0.0 < time_epsilon < 1.0):
        raise ValueError("time_epsilon must be in (0, 1)")
    if not (0.0 <= float(uniform_ratio) < 1.0):
        raise ValueError("uniform_ratio must be in [0, 1)")
    if float(gidd_gamma) < 0.0:
        raise ValueError("gidd_gamma must be >= 0")

    input_ids = batch["input_ids"]
    eligible_mask = _diffusion_eligible_mask(batch, require_residue=False)
    if not eligible_mask.any():
        raise ValueError("batch has no eligible diffusion target tokens")

    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    if token_t is None:
        # Per-sequence noise level t ~ U(eps, 1); each eligible token masked independently with prob t
        timesteps = torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(time_epsilon, 1.0)
        mask_probs = timesteps[:, None].expand(batch_size, seq_len)
    else:
        if tuple(token_t.shape) != (batch_size, seq_len):
            raise ValueError(
                f"token_t must have shape [B, S]=[{batch_size}, {seq_len}], got {tuple(token_t.shape)}"
            )
        mask_probs = token_t.to(device=device, dtype=torch.float32)
        eligible_float = eligible_mask.to(dtype=torch.float32)
        timesteps = (mask_probs * eligible_float).sum(dim=1) / eligible_float.sum(dim=1).clamp_min(1.0)

    use_gidd = 0.0 < float(uniform_ratio) < 1.0
    if not use_gidd:
        # Identical to the original Bernoulli(t) sampler (continue-train default).
        corruption_mask = (torch.rand(batch_size, seq_len, device=device) < mask_probs) & eligible_mask
        _guarantee_at_least_one_corruption(corruption_mask, eligible_mask, mask_probs, token_t is not None)
        noised_input_ids = input_ids.masked_fill(corruption_mask, int(mask_token_id))  # x_t [B,S]
        labels = input_ids.masked_fill(~corruption_mask, -100)  # targets only at masked sites [B,S]
        return noised_input_ids, labels, corruption_mask, timesteps

    residue_mask = batch.get("residue_mask")
    residue_eligible = (
        eligible_mask & residue_mask.bool()
        if residue_mask is not None
        else torch.zeros_like(eligible_mask)
    )
    other_eligible = eligible_mask & ~residue_eligible

    t = mask_probs.clamp(0.0, 1.0)
    gidd_b = (2.0 ** float(gidd_gamma)) * float(uniform_ratio) / (1.0 - float(uniform_ratio))
    half_gamma = 0.5 * float(gidd_gamma)
    c_t = gidd_b * t.pow(half_gamma) * (1.0 - t).pow(half_gamma)
    normalizer = 1.0 + c_t
    p_keep = (1.0 - t) / normalizer
    p_mask = t / normalizer
    roll = torch.rand(batch_size, seq_len, device=device)
    mask_draw = (roll >= p_keep) & (roll < (p_keep + p_mask))
    uniform_draw = roll >= (p_keep + p_mask)
    mask_sites = (mask_draw & residue_eligible) | ((roll < t) & other_eligible)
    uniform_sites = uniform_draw & residue_eligible
    if residue_token_ids is None or int(residue_token_ids.numel()) == 0:
        mask_sites = mask_sites | uniform_sites
        uniform_sites = torch.zeros_like(uniform_sites)

    corruption_mask = mask_sites | uniform_sites
    _guarantee_at_least_one_corruption(corruption_mask, eligible_mask, mask_probs, token_t is not None)
    mask_sites = mask_sites | (corruption_mask & ~uniform_sites)

    noised_input_ids = input_ids.clone()
    noised_input_ids[mask_sites] = int(mask_token_id)
    if uniform_sites.any():
        pool = residue_token_ids.to(device=device, dtype=noised_input_ids.dtype)
        noised_input_ids[uniform_sites] = _sample_uniform_residues(
            input_ids[uniform_sites],
            pool,
        )
    labels = input_ids.masked_fill(~corruption_mask, -100)
    return noised_input_ids, labels, corruption_mask, timesteps


def _guarantee_at_least_one_corruption(
    corruption_mask: torch.Tensor,
    eligible_mask: torch.Tensor,
    mask_probs: torch.Tensor,
    per_token_t: bool,
) -> None:
    """Force one corrupted site per row that can still receive noise.

    The historical sampler always forced ``>=1`` mask on rows with any eligible
    token. When ``token_t`` is provided, rows whose eligible tokens all have
    ``t == 0`` (Ophiuchus heavy2light / light2heavy clean side) stay clean.
    """

    batch_size = corruption_mask.shape[0]
    device = corruption_mask.device
    for row in range(batch_size):
        if not eligible_mask[row].any() or corruption_mask[row].any():
            continue
        if per_token_t and not (eligible_mask[row] & mask_probs[row].gt(0)).any():
            continue
        valid_positions = torch.nonzero(eligible_mask[row], as_tuple=False).flatten()
        choice = valid_positions[torch.randint(valid_positions.numel(), (1,), device=device)]
        corruption_mask[row, choice] = True


@dataclass
class ChainConditionedTimesteps:
    """Per-token noise levels and chain roles for Ophiuchus-style training."""

    token_t: torch.Tensor
    seq_t: torch.Tensor
    heavy_mask: torch.Tensor
    light_mask: torch.Tensor
    zero_loss_mask: torch.Tensor


def sample_chain_conditioned_timesteps(
    batch: dict[str, Any],
    *,
    ratios: dict[str, float],
    time_epsilon: float = 1e-3,
    stage: str | None = None,
) -> ChainConditionedTimesteps:
    """Sample per-token ``t`` from Ophiuchus multi-chain ratio buckets.

    Generated chains are the unique non-negative ``chain_ids`` on positions
    where ``diffusion_eligible_mask & residue_mask`` is True. The smallest id
    is treated as heavy (antibody_heavy / tcr_beta typically encode first);
    the next is light. Samples with fewer than two generated chains share one
    ``t`` on every eligible token.

    Ratio keys (must sum to 1):

    - ``single_chain_ratio``
    - ``heavy2light_loss_ratio``
    - ``light2heavy_loss_ratio``
    - ``independent_loss_ratio``
    - ``joint_loss_ratio``

    Each row is drawn independently from this categorical (not ``int(B * ratio)``,
    which floors every non-joint bucket to 0 at the usual per-device batch of
    2 or 4). ``joint_loss_ratio=1.0`` still always picks the shared-``t`` bucket,
    so existing continue-train runs are unchanged. ``stage=='val'`` assigns every
    sample to independent. ``t=0`` means no corruption on that chain; ``t=1``
    fully masks it. The fully-masked side of a ``single_chain`` draw is marked in
    ``zero_loss_mask``.
    """

    if not (0.0 < time_epsilon < 1.0):
        raise ValueError("time_epsilon must be in (0, 1)")

    ratio_keys = (
        "single_chain_ratio",
        "heavy2light_loss_ratio",
        "light2heavy_loss_ratio",
        "independent_loss_ratio",
        "joint_loss_ratio",
    )
    defaults = {
        "single_chain_ratio": 0.0,
        "heavy2light_loss_ratio": 0.0,
        "light2heavy_loss_ratio": 0.0,
        "independent_loss_ratio": 0.0,
        "joint_loss_ratio": 1.0,
    }
    parsed = {key: float(ratios.get(key, defaults[key])) for key in ratio_keys}
    ratio_sum = sum(parsed.values())
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ophiuchus ratios must sum to 1.0, got {ratio_sum}")

    input_ids = batch["input_ids"]
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    # Chain roles are defined on generated residues only.
    role_eligible = _diffusion_eligible_mask(batch, require_residue=True)
    eligible = _diffusion_eligible_mask(batch, require_residue=False)

    chain_ids = batch.get("chain_ids")
    heavy_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    light_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    if chain_ids is not None:
        sentinel = torch.iinfo(torch.int64).max
        cid = chain_ids.to(dtype=torch.int64)
        cid = cid.masked_fill((~role_eligible) | cid.lt(0), sentinel)
        heavy_id = cid.min(dim=1).values
        has_heavy = heavy_id.lt(sentinel)
        cid_without_heavy = cid.masked_fill(cid.eq(heavy_id.unsqueeze(1)), sentinel)
        light_id = cid_without_heavy.min(dim=1).values
        has_light = light_id.lt(sentinel)
        heavy_mask = role_eligible & has_heavy.unsqueeze(1) & chain_ids.eq(heavy_id.unsqueeze(1))
        light_mask = role_eligible & has_light.unsqueeze(1) & chain_ids.eq(light_id.unsqueeze(1))
    else:
        has_light = torch.zeros(batch_size, dtype=torch.bool, device=device)

    heavy_t = torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(time_epsilon, 1.0)
    light_t = torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(time_epsilon, 1.0)

    # Per-row categorical. ``int(B * ratio)`` floors every non-joint bucket to
    # 0 at per_device 2 or 4, so opening the Ophiuchus ratios would have been a
    # no-op on the real training jobs. joint=1.0 still always hits the last bin.
    bucket_probs = torch.tensor(
        [
            parsed["single_chain_ratio"] / 2.0,
            parsed["single_chain_ratio"] / 2.0,
            parsed["heavy2light_loss_ratio"],
            parsed["light2heavy_loss_ratio"],
            parsed["independent_loss_ratio"],
            parsed["joint_loss_ratio"],
        ],
        device=device,
        dtype=torch.float32,
    )
    if stage == "val":
        choice = torch.full((batch_size,), 4, device=device, dtype=torch.long)
    else:
        if float(bucket_probs.sum()) <= 0.0:
            raise ValueError("Ophiuchus ratio buckets are all zero")
        choice = torch.multinomial(bucket_probs.expand(batch_size, -1), num_samples=1).squeeze(1)
    mask_heavy_index = choice.eq(0)
    mask_light_index = choice.eq(1)
    heavy2light_index = choice.eq(2)
    light2heavy_index = choice.eq(3)
    _independent_index = choice.eq(4)
    joint_index = choice.eq(5)
    _ = _independent_index

    # t=0 → no corruption on that chain; t=1 → fully masked (AirGen construct_x_t).
    heavy_t = heavy_t.masked_fill(heavy2light_index, 0.0)
    heavy_t = heavy_t.masked_fill(mask_heavy_index, 1.0)
    light_t = light_t.masked_fill(light2heavy_index, 0.0)
    light_t = light_t.masked_fill(mask_light_index, 1.0)
    light_t = light_t.masked_scatter(joint_index, heavy_t[joint_index])

    token_t = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
    token_t = torch.where(heavy_mask, heavy_t[:, None].expand_as(token_t), token_t)
    token_t = torch.where(light_mask, light_t[:, None].expand_as(token_t), token_t)
    leftover = eligible & ~heavy_mask & ~light_mask
    token_t = torch.where(leftover, heavy_t[:, None].expand_as(token_t), token_t)

    two_chain = has_light
    zero_loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    zero_loss_mask = zero_loss_mask | (mask_heavy_index.unsqueeze(1) & heavy_mask & two_chain.unsqueeze(1))
    zero_loss_mask = zero_loss_mask | (mask_light_index.unsqueeze(1) & light_mask & two_chain.unsqueeze(1))

    chain_count = heavy_mask.any(dim=1).to(dtype=torch.float32) + light_mask.any(dim=1).to(dtype=torch.float32)
    seq_t = torch.where(
        chain_count.gt(0),
        (
            heavy_t * heavy_mask.any(dim=1).to(dtype=torch.float32)
            + light_t * light_mask.any(dim=1).to(dtype=torch.float32)
        )
        / chain_count.clamp_min(1.0),
        heavy_t,
    )
    return ChainConditionedTimesteps(
        token_t=token_t,
        seq_t=seq_t,
        heavy_mask=heavy_mask,
        light_mask=light_mask,
        zero_loss_mask=zero_loss_mask,
    )


def _residue_slot_positions(encoder_residue_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each chain's ``k``-th residue to its encoder-stream position.

    Vectorized replacement for ``torch.nonzero(residue_mask).flatten()[k]``: a
    single cumsum + scatter builds the whole ``slot -> position`` table so the
    gather / corruption-mirror paths need no per-chain Python loop.

    Parameters
    ----------
    encoder_residue_mask : ``[B, C, L]`` bool — 1 at residue token slots
        (``<cls>`` / ``<eos>`` / pad are 0).

    Returns
    -------
    slot_pos : ``[B, C, L]`` long — ``slot_pos[b, c, k]`` is the encoder position
        of the ``k``-th residue (ascending) in chain ``c``; entries with
        ``k >= count`` stay 0 (never indexed because callers gate on ``count``).
    counts : ``[B, C]`` long — residue count per chain.
    """

    batch_size, max_chains, chain_len = encoder_residue_mask.shape
    mask_long = encoder_residue_mask.to(torch.long)
    counts = mask_long.sum(dim=-1)  # [B, C]
    slot_index = mask_long.cumsum(dim=-1) - 1  # 0-based residue slot at residue positions
    positions = torch.arange(chain_len, device=encoder_residue_mask.device)
    positions = positions.view(1, 1, chain_len).expand(batch_size, max_chains, chain_len)
    # Route non-residue positions to a throwaway bucket column (index ``chain_len``)
    # so they never overwrite a real slot; residues keep their unique slot index.
    bucket = torch.full_like(slot_index, chain_len)
    scatter_index = torch.where(encoder_residue_mask, slot_index.clamp(min=0), bucket)
    slot_pos = encoder_residue_mask.new_zeros(batch_size, max_chains, chain_len + 1, dtype=torch.long)
    slot_pos.scatter_(dim=2, index=scatter_index, src=positions)
    return slot_pos[..., :chain_len], counts


def apply_decoder_corruption_to_encoder(
    batch: dict[str, Any],
    corruption_mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Mirror decoder corruption onto the ESMC encoder input stream.

    The collator builds ``encoder_input_ids`` as a per-chain (or single proxy)
    view of the current noisy state. This function writes ``<mask>`` at encoder
    positions that correspond to corrupted decoder tokens so ESMC never sees
    clean targets at masked sites.

    Parameters
    ----------
    corruption_mask : ``[B, S]`` — same mask returned by ``sample_bioseq_diffusion_noise``.

    Returns
    -------
    noised_encoder_input_ids : ``[B, C, L]`` — encoder ids with matching masks applied.
    """

    encoder_input_ids = batch["encoder_input_ids"]
    encoder_position_ids = batch.get("encoder_position_ids")
    if encoder_position_ids is not None:
        batch_size, max_chains, encoder_len = encoder_input_ids.shape
        if max_chains != 1:
            raise ValueError("Direct encoder_position_ids mapping requires a single proxy stream")
        # Corrupt every decoder token that maps to a valid proxy position in one scatter.
        valid = corruption_mask & encoder_position_ids.ge(0) & encoder_position_ids.lt(encoder_len)
        batch_index = (
            torch.arange(batch_size, device=encoder_input_ids.device).unsqueeze(1).expand_as(valid)
        )
        noised_encoder_input_ids = encoder_input_ids.clone()
        noised_encoder_input_ids[batch_index[valid], 0, encoder_position_ids[valid]] = int(mask_token_id)
        return noised_encoder_input_ids

    encoder_residue_mask = batch["encoder_residue_mask"]
    chain_ids = batch["chain_ids"]
    position_ids_inner = batch["position_ids_inner"]

    batch_size, max_chains, chain_len = encoder_input_ids.shape
    slot_pos, counts = _residue_slot_positions(encoder_residue_mask.bool())  # [B,C,L], [B,C]

    # Which corrupted decoder tokens map to a real residue slot in a real chain.
    safe_chain = chain_ids.clamp(min=0, max=max_chains - 1)
    token_residue_count = torch.gather(counts, 1, safe_chain)  # [B,S] residues in that chain
    valid = (
        corruption_mask
        & chain_ids.ge(0)
        & chain_ids.lt(max_chains)
        & position_ids_inner.ge(0)
        & position_ids_inner.lt(token_residue_count)
    )

    # Encoder position of each token's residue via flat (b*C + chain, inner) lookup.
    slot_pos_flat = slot_pos.reshape(batch_size * max_chains, chain_len)
    row = torch.arange(batch_size, device=encoder_input_ids.device).unsqueeze(1) * max_chains + safe_chain
    safe_inner = position_ids_inner.clamp(min=0, max=chain_len - 1)
    encoder_pos = slot_pos_flat[row.reshape(-1), safe_inner.reshape(-1)].reshape(batch_size, -1)

    noised_flat = encoder_input_ids.clone().reshape(batch_size * max_chains, chain_len)
    noised_flat[row[valid], encoder_pos[valid]] = int(mask_token_id)
    return noised_flat.reshape(batch_size, max_chains, chain_len)


def forbidden_diffusion_target_token_ids(config: BioSeqDiffusionTransformerConfig) -> tuple[int, ...]:
    """Token ids that must not be predicted during residue denoising.

    Masked diffusion feeds ``<mask>`` at corrupted positions. With tied input/output
    embeddings, the decoder can collapse to always predicting ``<mask>`` unless those
    logits are excluded from the denoising objective.
    """

    if config.forbidden_target_token_ids is not None:
        return config.forbidden_target_token_ids
    forbidden = {
        0,  # <cls>
        int(config.pad_token_id),
        2,  # <eos>
        3,  # <unk>
        int(config.mask_token_id),
    }
    return tuple(sorted(token_id for token_id in forbidden if 0 <= token_id < config.vocab_size))


def mask_forbidden_target_logits(
    logits: torch.Tensor,
    forbidden_token_ids: tuple[int, ...] | None,
) -> torch.Tensor:
    """Zero out logits for special tokens that must not be denoising targets.

    With tied input/output embeddings the model can collapse to predicting
    ``<mask>``; this sets forbidden vocab columns to ``finfo.min`` before CE.

    Input/output: ``[B, S, V]``.
    """
    if not forbidden_token_ids:
        return logits
    masked_logits = logits.clone()
    for token_id in forbidden_token_ids:
        if 0 <= token_id < masked_logits.size(-1):
            masked_logits[..., token_id] = torch.finfo(masked_logits.dtype).min
    return masked_logits


def _reduce_masked_token_loss(
    token_loss: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_norm: str,
    batch_size: int,
) -> torch.Tensor:
    """Reduce a ``[B, S]`` per-token NLL with the same rules as the original CE.

    ``loss_mask`` must be applied before the sum: ignored / off-chain positions
    can still hold nonzero NLL (``ignore_index`` only zeros ``labels == -100``).
    """

    masked = token_loss * loss_mask.to(dtype=token_loss.dtype)
    if loss_norm == "token":
        return masked.sum() / loss_mask.sum().clamp_min(1)
    if loss_norm == "sequence":
        per_sequence = masked.sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1)
        return per_sequence.mean()
    if loss_norm == "batch":
        return masked.sum() / batch_size
    raise ValueError(f"Unsupported loss_norm: {loss_norm}")


def compute_masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_norm: str = "token",
    forbidden_token_ids: tuple[int, ...] | None = None,
    focal: bool = False,
    focal_gamma: float = 1.0,
    token_weights: torch.Tensor | None = None,
    heavy_mask: torch.Tensor | None = None,
    light_mask: torch.Tensor | None = None,
    heavy_loss_weight: float = 1.0,
    light_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy on corrupted positions only (``labels != -100``).

    ``logits``: ``[B, S, V]``, ``labels``: ``[B, S]``. Returns a scalar loss.

    Defaults (``focal=False``, no ``token_weights``, no chain masks) are the
    original token/sequence/batch CE. Optional focal, reciprocal/token weights,
    and AirGen heavy/light reductions are opt-in.
    """
    logits = mask_forbidden_target_logits(logits, forbidden_token_ids)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    # ignore_index already zeros NLL on ``-100``; focal / weights keep those zeros.
    if focal:
        token_loss = token_loss * (1.0 - torch.exp(-token_loss)).pow(float(focal_gamma))
    if token_weights is not None:
        if tuple(token_weights.shape) != tuple(token_loss.shape):
            raise ValueError(
                f"token_weights must have shape {tuple(token_loss.shape)}, "
                f"got {tuple(token_weights.shape)}"
            )
        token_loss = token_loss * token_weights.to(dtype=token_loss.dtype)
    loss_mask = labels.ne(-100)
    batch_size = int(labels.shape[0])
    if heavy_mask is not None and light_mask is not None:
        heavy_valid = heavy_mask.bool() & loss_mask
        light_valid = light_mask.bool() & loss_mask
        neither_valid = loss_mask & ~heavy_mask.bool() & ~light_mask.bool()
        if heavy_valid.any() or light_valid.any():
            heavy_loss = _reduce_masked_token_loss(token_loss, heavy_valid, loss_norm, batch_size)
            light_loss = _reduce_masked_token_loss(token_loss, light_valid, loss_norm, batch_size)
            loss = float(heavy_loss_weight) * heavy_loss + float(light_loss_weight) * light_loss
            if neither_valid.any():
                # Positions in neither generated chain keep the single-stream reduction.
                loss = loss + _reduce_masked_token_loss(
                    token_loss, neither_valid, loss_norm, batch_size
                )
            return loss
    return _reduce_masked_token_loss(token_loss, loss_mask, loss_norm, batch_size)


class LocalESMCEncoder(nn.Module):
    """Thin wrapper exposing Biohub's native ESMC as a Hugging Face-like encoder.

    Used only for feature extraction in ``BioSeqEncoderDiffusionModel``. The
    returned ``logits`` (``sequence_head``) are not consumed by the diffusion
    training loop.
    """

    def __init__(self, esmc: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.esmc = esmc
        self.config = SimpleNamespace(hidden_size=hidden_size, d_model=hidden_size, model_type="esmc")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        diffusion_state: torch.Tensor | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        """Run ESMC and normalize outputs to HF-style names.

        Returns a namespace with:
        - ``last_hidden_state``: ``[B, L, E]`` token features used for conditioning.
        - ``hidden_states``: ``[n_layers, B, L, E]`` all layer outputs (optional).
        - ``logits``: ``[B, L, V_esmc]`` from ``sequence_head`` (unused in training).
        """
        if inputs_embeds is not None or diffusion_state is not None:
            if getattr(self.esmc, "_use_flash_attn", False):
                raise ValueError("ESMC inputs_embeds/diffusion_state path does not support flash attention")
            embeddings = inputs_embeds if inputs_embeds is not None else self._embed_diffusion_state(diffusion_state)
            if attention_mask is None:
                sequence_id = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
            else:
                sequence_id = attention_mask.bool()
            hidden_states, _, all_hidden_states = self.esmc.transformer(embeddings, sequence_id=sequence_id)
            output = SimpleNamespace(
                embeddings=hidden_states,
                hidden_states=torch.stack(all_hidden_states, dim=0),
                sequence_logits=self.esmc.sequence_head(hidden_states),
            )
        else:
            if input_ids is None:
                raise ValueError("LocalESMCEncoder requires input_ids, inputs_embeds, or diffusion_state")
            output = self.esmc(
                sequence_tokens=input_ids,
                sequence_id=attention_mask.bool() if attention_mask is not None else None,
            )
        return SimpleNamespace(
            last_hidden_state=output.embeddings,
            hidden_states=output.hidden_states,
            logits=output.sequence_logits,
        )

    def _embed_diffusion_state(self, diffusion_state: torch.Tensor | None) -> torch.Tensor:
        if diffusion_state is None:
            raise ValueError("diffusion_state is required")
        if not torch.is_floating_point(diffusion_state):
            return self.esmc.embed(diffusion_state.long())
        if diffusion_state.dim() != 3:
            raise ValueError(
                "Floating diffusion_state must have shape [batch, seq, vocab] "
                "or [batch, seq, hidden]"
            )
        if diffusion_state.shape[-1] == self.esmc.embed.num_embeddings:
            return torch.matmul(diffusion_state.to(self.esmc.embed.weight.dtype), self.esmc.embed.weight)
        if diffusion_state.shape[-1] == self.esmc.embed.embedding_dim:
            return diffusion_state.to(self.esmc.embed.weight.dtype)
        raise ValueError(
            "Floating diffusion_state last dimension must match ESMC vocab size "
            f"({self.esmc.embed.num_embeddings}) or hidden size ({self.esmc.embed.embedding_dim})"
        )


def _convert_biohub_esmc_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert Biohub HF-style ESMC safetensor keys to native ``esm`` keys."""

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("esmc."):
            new_key = key[len("esmc.") :]
        elif key.startswith("lm_head."):
            new_key = "sequence_head." + key[len("lm_head.") :]
        else:
            continue
        if new_key.endswith("._extra_state"):
            continue
        new_key = new_key.replace("attn.layernorm_qkv.layer_norm_weight", "attn.layernorm_qkv.0.weight")
        new_key = new_key.replace("attn.layernorm_qkv.layer_norm_bias", "attn.layernorm_qkv.0.bias")
        new_key = new_key.replace("attn.layernorm_qkv.weight", "attn.layernorm_qkv.1.weight")
        new_key = new_key.replace("ffn.layer_norm_weight", "ffn.0.weight")
        new_key = new_key.replace("ffn.layer_norm_bias", "ffn.0.bias")
        new_key = new_key.replace("ffn.fc1_weight", "ffn.1.weight")
        new_key = new_key.replace("ffn.fc2_weight", "ffn.3.weight")
        converted[new_key] = value
    return converted


def load_local_esmc_encoder(
    model_path: str | Path,
    use_flash_attn: bool = False,
) -> LocalESMCEncoder:
    """Load local Biohub ESMC safetensors without relying on HF AutoModel.

    The released local checkpoints under ``/c20250601/mj/model_weights/esmc``
    store a Hugging Face-style wrapper state dict, but the public ``esm``
    package exposes the native ``ESMC`` module. This loader bridges that naming
    difference and exposes ``last_hidden_state`` for BioSeq conditioning.
    """

    model_path = Path(model_path)
    try:
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "Loading local ESMC requires Biohub `esm` and `safetensors`. "
            "Install with `pip install esm==3.2.3 safetensors`."
        ) from exc

    config_path = model_path / "config.json"
    weights_path = model_path / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"ESMC config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"ESMC safetensors not found: {weights_path}")

    with config_path.open() as handle:
        config = json.load(handle)
    hidden_size = int(config["d_model"])
    # Flash-attention speeds up the ESMC input_ids forward (the path training
    # uses), but only if the `flash_attn` package is importable on this host.
    # Guard so `--encoder-use-flash-attn` is a safe no-op where flash_attn is
    # missing (falls back to SDPA) instead of crashing training. Parity of the
    # flash vs non-flash features is checked by
    # scripts/tests/bioseq/check_esmc_flash_parity.py before it is relied on.
    if use_flash_attn:
        try:
            import flash_attn  # noqa: F401
        except Exception:
            import warnings

            warnings.warn(
                "use_flash_attn=True but flash_attn is not importable; "
                "falling back to non-flash ESMC attention.",
                RuntimeWarning,
                stacklevel=2,
            )
            use_flash_attn = False
    esmc = ESMC(
        d_model=hidden_size,
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=use_flash_attn,
    )
    converted = _convert_biohub_esmc_state_dict(load_file(str(weights_path), device="cpu"))
    esmc.load_state_dict(converted, strict=True)
    return LocalESMCEncoder(esmc=esmc, hidden_size=hidden_size)


def _convert_esm2_masked_lm_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip ``EsmForMaskedLM`` prefixes and drop auxiliary heads for ``EsmModel``."""

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith("esm."):
            continue
        new_key = key[len("esm.") :]
        if new_key.startswith(("contact_head.", "pooler.")):
            continue
        converted[new_key] = value
    return converted


def _install_sklearn_import_stub() -> None:
    """Stub sklearn before transformers import on nodes with older glibc.

    Recent ``transformers`` versions import ``sklearn.metrics.roc_curve`` from
    ``generation.candidate_generator`` when loading ``EsmModel`` (via
    ``modeling_utils``). Some Volc cluster nodes only provide glibc < 2.32 while
    the shared conda sklearn wheel requires 2.32, so a real import fails before
    training starts. The stub satisfies the import-only dependency; ESM2 encoding
    never calls ``roc_curve``.
    """

    import importlib.util
    import sys
    import types

    existing = sys.modules.get("sklearn")
    if existing is not None and getattr(existing, "__file__", None):
        return

    for name in list(sys.modules):
        if name == "sklearn" or name.startswith("sklearn."):
            del sys.modules[name]

    sklearn = types.ModuleType("sklearn")
    sklearn.__spec__ = importlib.util.spec_from_loader("sklearn", loader=None)
    sklearn.__path__ = []  # type: ignore[attr-defined]
    metrics = types.ModuleType("sklearn.metrics")
    metrics.__spec__ = importlib.util.spec_from_loader("sklearn.metrics", loader=None)
    metrics.roc_curve = lambda *args, **kwargs: ([], [], [])
    sklearn.metrics = metrics
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.metrics"] = metrics


def load_local_esm2_encoder(model_path: str | Path) -> nn.Module:
    """Load local ESM2 weights as ``EsmModel`` without importing ``AutoModel``.

    Volc cluster nodes ship an older glibc than some conda-built sklearn wheels.
    Even ``transformers.models.esm.modeling_esm`` pulls in ``modeling_utils`` →
    ``generation`` → sklearn. ``_install_sklearn_import_stub`` avoids that import
    failure; weights are read from local ``config.json`` + ``model.safetensors``.
    """

    _install_sklearn_import_stub()
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    safetensors_path = model_path / "model.safetensors"
    bin_path = model_path / "pytorch_model.bin"
    if not config_path.is_file():
        raise FileNotFoundError(f"ESM2 config not found: {config_path}")
    if not safetensors_path.is_file() and not bin_path.is_file():
        raise FileNotFoundError(
            f"ESM2 weights not found under {model_path} (expected model.safetensors or pytorch_model.bin)"
        )

    try:
        from transformers.models.esm.configuration_esm import EsmConfig
        from transformers.models.esm.modeling_esm import EsmModel
    except ImportError as exc:
        raise ImportError("Loading local ESM2 requires transformers with ESM support.") from exc

    with config_path.open() as handle:
        config_dict = json.load(handle)
    skip_keys = {"architectures", "transformers_version", "torch_dtype", "vocab_list", "_name_or_path"}
    config = EsmConfig(**{key: value for key, value in config_dict.items() if key not in skip_keys})
    # ESM2's ``token_dropout`` rescales embeddings by ``1 / (1 - mask_ratio)``.
    # The diffusion encoder stream shares ``mask_token_id`` (32) with ESM2, and at
    # high noise levels a chain can be (almost) entirely masked, driving the ratio
    # to 1 -> division by zero -> NaN/inf at training step 1. The rescale assumes a
    # fixed MLM mask ratio (~0.12) that does not hold here, so disable it; masked
    # positions then use their normal mask-token embedding (numerically stable).
    config.token_dropout = False
    model = EsmModel(config)

    if safetensors_path.is_file():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device="cpu")
    else:
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    model.load_state_dict(_convert_esm2_masked_lm_state_dict(state_dict), strict=False)
    return model


class BioSeqDiffusionDecoder(nn.Module):
    """Bidirectional transformer denoiser for grammar token streams.

    Embeds the noisy decoder input ``x_t``, adds position / chain / timestep (and
    optional encoder) signals, runs ``num_hidden_layers`` transformer blocks,
    and predicts clean token logits via a tied ``lm_head``.

    Submodules
    ----------
    token_embeddings : ``[V, H]`` — shared with ``lm_head`` (weight tying).
    inner_position_embeddings : chain-local residue index (0, 1, 2, …).
    chain_position_embeddings : which chain slot (heavy=0, light=1, protein A/B, …).
    timestep_embeddings : diffusion noise level ``t`` per sequence.
    condition_proj : ``Linear(E, H)`` — only when ``condition_hidden_size`` is set.
    layers : ``num_hidden_layers`` × ``BioSeqTransformerBlock``.
    """

    def __init__(self, config: BioSeqDiffusionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.inner_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.chain_position_embeddings = nn.Embedding(config.max_chain_positions, config.hidden_size)
        self.timestep_embeddings = BioSeqTimestepEmbedding(config.hidden_size)
        self.condition_proj = (
            nn.Linear(config.condition_hidden_size, config.hidden_size, bias=False)
            if config.condition_hidden_size is not None and config.use_condition_projection
            else None
        )
        # Re-scale the encoder condition before injection. With replacement
        # injection the condition enters the *residual stream* un-normalized
        # (at residue positions h = cond + inner/chain/timestep, with cond the
        # token-identity term) and is carried by skip connections to the final
        # read-out: out = lm_head(RMSNorm(cond + Σ sublayer_outputs)).
        # The per-position RMSNorm only normalizes sublayer *inputs*, not the
        # condition's weight in that residual sum, so the condition's influence
        # on the output is set by its magnitude relative to the rest of the
        # stream. ESMC's final-LayerNorm output is tiny (per-pos L2 ~1.3, gamma
        # ~0.04) -> negligible in the residual -> the decoder ignores it
        # (ablation: raw ESMC condition ties zeroing it out; a fixed ×24 scale,
        # RMSNorm, or LayerNorm all fix it equally). ESM2 (L2 ~9.5) is already
        # large enough. A LayerNorm with gamma=1 brings any encoder's condition
        # to a usable, encoder-agnostic scale. Off by default for checkpoint
        # backward-compat.
        self.condition_norm = (
            nn.LayerNorm(config.condition_hidden_size)
            if config.condition_hidden_size is not None and config.condition_norm
            else None
        )
        self.layers = nn.ModuleList([BioSeqTransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = BioSeqRMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        residual_std = config.initializer_range / math.sqrt(2.0 * max(config.num_hidden_layers, 1))
        for layer in self.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=residual_std)
        self.lm_head.weight = self.token_embeddings.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, BioSeqRMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        diffusion_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids_inner: torch.Tensor | None = None,
        position_ids_chain: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        encoder_condition: torch.Tensor | None = None,
        encoder_condition_mask: torch.Tensor | None = None,
    ) -> BioSeqDiffusionOutput:
        """Forward pass over the concatenated grammar record.

        Parameters
        ----------
        input_ids : ``[B, S]`` — discrete noisy tokens (training uses ``<mask>`` at corrupted sites).
        diffusion_state : ``[B, S, V]`` or ``[B, S, H]`` — optional soft input instead of ids.
        attention_mask : ``[B, S]`` — 1 for real tokens, 0 for pad.
        position_ids_inner : ``[B, S]`` — residue index within each chain; ``-1`` for special tokens.
        position_ids_chain : ``[B, S]`` — chain slot index; ``-1`` for non-residue tokens.
        timesteps : ``[B]`` — diffusion time added to every token position.
        encoder_condition : ``[B, S, E]`` — per-token ESMC features (encoder models only).

        Returns ``BioSeqDiffusionOutput`` with ``logits [B, S, V]`` and ``hidden_states [B, S, H]``.
        """
        if input_ids is None and diffusion_state is None:
            raise ValueError("BioSeqDiffusionDecoder requires input_ids or diffusion_state")
        batch_size, seq_len = (diffusion_state.shape[:2] if diffusion_state is not None else input_ids.shape)
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        # Token embed: discrete ids or soft diffusion_state -> [B, S, H]
        hidden_states = (
            self._embed_diffusion_state(diffusion_state)
            if diffusion_state is not None
            else self.token_embeddings(input_ids)
        )

        # Replace ONLY the token-identity embedding at residue positions with the
        # encoder feature, BEFORE adding position / chain / timestep. Position,
        # chain-slot and timestep signals are then added on top of the encoder
        # condition (they must survive at residue sites, not be overwritten).
        # Projection mode adds the (projected) condition instead of replacing.
        if encoder_condition is not None:
            if self.condition_norm is not None:
                encoder_condition = self.condition_norm(encoder_condition.to(hidden_states.dtype))
            if self.config.use_condition_projection and self.condition_proj is not None:
                hidden_states = hidden_states + self.condition_proj(encoder_condition)
            elif encoder_condition_mask is not None:
                replace_mask = encoder_condition_mask.to(hidden_states.dtype).unsqueeze(-1)
                hidden_states = hidden_states * (1.0 - replace_mask) + encoder_condition.to(
                    hidden_states.dtype
                ) * replace_mask
            else:
                raise ValueError(
                    "encoder_condition requires encoder_condition_mask or condition projection"
                )

        # + chain-local position embed (residue index within each chain) -> [B, S, H]
        if position_ids_inner is None:
            position_ids_inner = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        safe_inner = position_ids_inner.clamp(min=0, max=self.config.max_position_embeddings - 1)
        inner_valid = position_ids_inner.ge(0).to(hidden_states.dtype).unsqueeze(-1)
        hidden_states = hidden_states + self.inner_position_embeddings(safe_inner) * inner_valid

        # + chain-slot embed (heavy/light, protein A/B, ...) -> [B, S, H]
        if position_ids_chain is not None:
            safe_chain = position_ids_chain.clamp(min=0, max=self.config.max_chain_positions - 1)
            chain_valid = position_ids_chain.ge(0).to(hidden_states.dtype).unsqueeze(-1)
            hidden_states = hidden_states + self.chain_position_embeddings(safe_chain) * chain_valid

        # + diffusion timestep embed [B, H] broadcast to all S positions -> [B, S, H]
        if timesteps is not None:
            hidden_states = hidden_states + self.timestep_embeddings(timesteps).unsqueeze(1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.to(hidden_states.dtype).unsqueeze(-1)

        # Bidirectional transformer stack: [B, S, H] -> [B, S, H]
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)
            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.to(hidden_states.dtype).unsqueeze(-1)

        hidden_states = self.final_layernorm(hidden_states)  # [B, S, H]
        logits = self.lm_head(hidden_states)  # [B, S, V] — tied with token_embeddings
        return BioSeqDiffusionOutput(loss=None, logits=logits, hidden_states=hidden_states)

    def _embed_diffusion_state(self, diffusion_state: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(diffusion_state):
            return self.token_embeddings(diffusion_state.long())
        if diffusion_state.dim() != 3:
            raise ValueError(
                "Floating diffusion_state must have shape [batch, seq, vocab] "
                "or [batch, seq, hidden]"
            )
        if diffusion_state.shape[-1] == self.config.vocab_size:
            return torch.matmul(diffusion_state.to(self.token_embeddings.weight.dtype), self.token_embeddings.weight)
        if diffusion_state.shape[-1] == self.config.hidden_size:
            return diffusion_state.to(self.token_embeddings.weight.dtype)
        raise ValueError(
            "Floating diffusion_state last dimension must match vocab_size "
            f"({self.config.vocab_size}) or hidden_size ({self.config.hidden_size})"
        )


class BioSeqNoEncoderDiffusionModel(nn.Module):
    """No-encoder BioSeq masked diffusion model.

    Wrapper around ``BioSeqDiffusionDecoder`` only. No ESMC/ESM backbone; all
    parameters are trained from scratch. Uses bidirectional self-attention over
    the full grammar record (not causal LM).

    Training entry point: ``compute_loss(batch)`` — samples noise, forward, CE.
    """

    def __init__(self, config: BioSeqDiffusionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = BioSeqDiffusionDecoder(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        diffusion_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids_inner: torch.Tensor | None = None,
        position_ids_chain: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        **_: Any,
    ) -> BioSeqDiffusionOutput:
        """Delegate to ``decoder``; see ``BioSeqDiffusionDecoder.forward`` for shapes."""
        return self.decoder(
            input_ids=input_ids,
            diffusion_state=diffusion_state,
            attention_mask=attention_mask,
            position_ids_inner=position_ids_inner,
            position_ids_chain=position_ids_chain,
            timesteps=timesteps,
        )

    def compute_loss(self, batch: dict[str, Any]) -> BioSeqDiffusionOutput:
        """Full training step: noise sampling -> forward -> masked CE."""
        # 1) Sample t and mask eligible tokens -> x_t [B,S], labels [B,S], mask [B,S]
        noised_input_ids, labels, corruption_mask, timesteps = sample_bioseq_diffusion_noise(
            batch=batch,
            mask_token_id=self.config.mask_token_id,
            time_epsilon=self.config.time_epsilon,
        )
        # 2) Decoder forward on noisy grammar stream -> logits [B,S,V]
        output = self.forward(
            input_ids=noised_input_ids,
            attention_mask=batch.get("attention_mask"),
            position_ids_inner=batch.get("position_ids_inner"),
            position_ids_chain=batch.get("position_ids_chain"),
            timesteps=timesteps,
        )
        # 3) CE only on corrupted positions; forbid pad/mask/cls/eos/unk logits
        forbidden = forbidden_diffusion_target_token_ids(self.config)
        loss = compute_masked_cross_entropy(
            output.logits,
            labels,
            loss_norm=self.config.loss_norm,
            forbidden_token_ids=forbidden,
        )
        return BioSeqDiffusionOutput(
            loss=loss,
            logits=output.logits,
            hidden_states=output.hidden_states,
            noised_input_ids=noised_input_ids,
            labels=labels,
            corruption_mask=corruption_mask,
            timesteps=timesteps,
        )


def infer_encoder_hidden_size(encoder: nn.Module) -> int:
    config = getattr(encoder, "config", None)
    for name in ("hidden_size", "d_model", "embed_dim", "encoder_embed_dim"):
        value = getattr(config, name, None) if config is not None else None
        if value is not None:
            return int(value)
    raise ValueError("Could not infer encoder hidden size; pass encoder_hidden_size explicitly")


class BioSeqEncoderDiffusionModel(nn.Module):
    """ESMC-conditioned BioSeq masked diffusion model.

    Two-tower layout:

    1. **Encoder** (pretrained ESMC): runs on noisy per-chain / proxy ``x_t``,
       outputs token features ``[B, C, L, E]``.
    2. **Decoder** (trainable BioSeqDiffusionDecoder): runs on the concatenated
       grammar stream ``[B, S]``, receives gathered encoder features as
       ``encoder_condition [B, S, E]``, predicts denoised tokens.

    Cross-chain reasoning happens in the decoder's bidirectional attention; the
    encoder processes chains (or the proxy stream) without cross-chain mixing.
    """

    def __init__(
        self,
        decoder_config: BioSeqDiffusionTransformerConfig,
        encoder: nn.Module,
        encoder_hidden_size: int | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        encoder_hidden_size = int(encoder_hidden_size or infer_encoder_hidden_size(encoder))
        use_projection = bool(decoder_config.use_condition_projection)
        if not use_projection and int(decoder_config.hidden_size) != encoder_hidden_size:
            raise ValueError(
                "decoder hidden_size must match encoder hidden size when use_condition_projection=False "
                f"(decoder={decoder_config.hidden_size}, encoder={encoder_hidden_size})"
            )
        self.config = replace(
            decoder_config,
            condition_hidden_size=encoder_hidden_size,
            use_condition_projection=use_projection,
        )
        self.encoder = encoder
        self.decoder = BioSeqDiffusionDecoder(self.config)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_esmc(
        cls,
        decoder_config: BioSeqDiffusionTransformerConfig,
        encoder_name_or_path: str,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        freeze_encoder: bool = False,
        use_flash_attn: bool = False,
    ) -> "BioSeqEncoderDiffusionModel":
        encoder_path = Path(encoder_name_or_path)
        config_path = encoder_path / "config.json"
        if config_path.is_file():
            try:
                with config_path.open() as handle:
                    config = json.load(handle)
            except json.JSONDecodeError:
                config = {}
            if str(config.get("model_type", "")).lower() == "esmc":
                encoder = load_local_esmc_encoder(encoder_name_or_path, use_flash_attn=use_flash_attn)
                return cls(decoder_config=decoder_config, encoder=encoder, freeze_encoder=freeze_encoder)

        try:
            from transformers import AutoModel
        except Exception:
            encoder = load_local_esmc_encoder(encoder_name_or_path, use_flash_attn=use_flash_attn)
            return cls(decoder_config=decoder_config, encoder=encoder, freeze_encoder=freeze_encoder)

        try:
            encoder = AutoModel.from_pretrained(
                encoder_name_or_path,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        except (OSError, ValueError, KeyError, RuntimeError):
            encoder = load_local_esmc_encoder(encoder_name_or_path, use_flash_attn=use_flash_attn)
        return cls(decoder_config=decoder_config, encoder=encoder, freeze_encoder=freeze_encoder)

    @classmethod
    def from_hf_encoder(
        cls,
        decoder_config: BioSeqDiffusionTransformerConfig,
        encoder_name_or_path: str,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        freeze_encoder: bool = False,
    ) -> "BioSeqEncoderDiffusionModel":
        """Load a Hugging Face ESM2 (or compatible) encoder for per-chain conditioning."""

        _ = local_files_only, trust_remote_code
        encoder = load_local_esm2_encoder(encoder_name_or_path)
        return cls(decoder_config=decoder_config, encoder=encoder, freeze_encoder=freeze_encoder)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        diffusion_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        chain_ids: torch.Tensor | None = None,
        position_ids_inner: torch.Tensor | None = None,
        position_ids_chain: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        encoder_input_ids: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_residue_mask: torch.Tensor | None = None,
        encoder_chain_mask: torch.Tensor | None = None,
        encoder_position_ids: torch.Tensor | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> BioSeqDiffusionOutput:
        """Encode noisy chains, gather conditions, then run the decoder.

        Requires ``encoder_input_ids [B, C, L]`` plus either:
        - ``encoder_position_ids [B, S]`` (grammar-v1 proxy path, ``C=1``), or
        - ``chain_ids [B, S]`` + ``position_ids_inner [B, S]`` (legacy per-chain path).

        Returns ``BioSeqDiffusionOutput`` including ``encoder_condition [B, S, E]``.
        """
        if input_ids is None and diffusion_state is None:
            raise ValueError("BioSeqEncoderDiffusionModel requires input_ids or diffusion_state")
        if encoder_input_ids is None:
            raise ValueError("encoder_input_ids are required for BioSeqEncoderDiffusionModel")
        if encoder_position_ids is None and chain_ids is None:
            raise ValueError("chain_ids or encoder_position_ids are required for encoder conditions")
        if encoder_position_ids is None and position_ids_inner is None:
            raise ValueError("position_ids_inner is required for per-chain encoder conditions")

        effective_encoder_residue_mask = encoder_residue_mask
        if encoder_position_ids is not None and residue_mask is not None:
            effective_encoder_residue_mask = residue_mask.unsqueeze(1)

        # Step 1: ESMC on noisy encoder stream [B,C,L] -> per-token features [B,C,L,E]
        chain_token_condition = self.encode_chain_tokens(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_residue_mask=effective_encoder_residue_mask,
            encoder_chain_mask=encoder_chain_mask,
            encoder_kwargs=encoder_kwargs,
        )
        # Step 2: align encoder features to decoder token positions -> [B,S,E]
        if encoder_position_ids is not None:
            token_condition = self.gather_proxy_token_condition(
                chain_token_condition,
                encoder_position_ids=encoder_position_ids,
                attention_mask=attention_mask,
                residue_mask=residue_mask,
            )
        else:
            token_condition = self.gather_token_condition(
                chain_token_condition,
                chain_ids=chain_ids,
                position_ids_inner=position_ids_inner,
                attention_mask=attention_mask,
                encoder_residue_mask=encoder_residue_mask,
            )
        encoder_condition_mask = self.build_encoder_condition_mask(
            chain_ids=chain_ids,
            position_ids_inner=position_ids_inner,
            attention_mask=attention_mask,
            encoder_position_ids=encoder_position_ids,
            residue_mask=residue_mask,
        )
        # Step 3: decoder denoises grammar stream x_t [B,S] with ESMC condition -> logits [B,S,V]
        output = self.decoder(
            input_ids=input_ids,
            diffusion_state=diffusion_state,
            attention_mask=attention_mask,
            position_ids_inner=position_ids_inner,
            position_ids_chain=position_ids_chain,
            timesteps=timesteps,
            encoder_condition=token_condition,
            encoder_condition_mask=encoder_condition_mask,
        )
        return replace(output, encoder_condition=token_condition)

    def encode_chain_tokens(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        encoder_residue_mask: torch.Tensor | None,
        encoder_chain_mask: torch.Tensor | None,
        encoder_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run the biological encoder on per-chain ``x_t`` and keep token features.

        Flattens ``[B, C, L]`` to ``[B*C, L]`` for a single encoder forward, then
        reshapes back and zeroes padding / non-residue / empty-chain positions.

        Returns
        -------
        chain_hidden : ``[B, C, L, E]`` — ESMC ``last_hidden_state`` per chain token.
        """
        batch_size, max_chains, chain_len = encoder_input_ids.shape
        # Flatten chains for one ESMC forward: [B,C,L] -> [B*C,L]
        flat_input_ids = encoder_input_ids.reshape(batch_size * max_chains, chain_len)
        flat_attention_mask = (
            encoder_attention_mask.reshape(batch_size * max_chains, chain_len)
            if encoder_attention_mask is not None
            else None
        )
        # Empty/padded chain rows (used to pad ragged chain counts) have an all-zero
        # attention mask. A transformer encoder attends over an all-masked row with a
        # softmax over -inf, producing NaN; the subsequent ``hidden * mask`` cannot
        # recover it (``NaN * 0 = NaN``). Give such rows a single valid attended
        # position so the encoder forward stays finite; their output is zeroed below.
        if flat_attention_mask is not None:
            empty_rows = flat_attention_mask.sum(dim=-1).eq(0)
            if empty_rows.any():
                flat_attention_mask = flat_attention_mask.clone()
                flat_attention_mask[empty_rows, 0] = 1
        call_kwargs: dict[str, Any] = {"input_ids": flat_input_ids}
        if flat_attention_mask is not None:
            call_kwargs["attention_mask"] = flat_attention_mask
        if encoder_kwargs:
            call_kwargs.update(encoder_kwargs)
        encoder_outputs = self.encoder(**call_kwargs)  # last_hidden_state [B*C,L,E]
        flat_hidden = getattr(encoder_outputs, "last_hidden_state", None)
        if flat_hidden is None:
            if isinstance(encoder_outputs, tuple):
                flat_hidden = encoder_outputs[0]
            else:
                raise ValueError("Encoder output must expose last_hidden_state or tuple[0]")

        hidden_size = flat_hidden.shape[-1]
        chain_hidden = flat_hidden.reshape(batch_size, max_chains, chain_len, hidden_size)  # [B,C,L,E]
        # Zero out pad / non-residue / empty-chain slots so they carry no condition signal
        if encoder_attention_mask is not None:
            chain_hidden = chain_hidden * encoder_attention_mask.to(chain_hidden.dtype).unsqueeze(-1)
        if encoder_residue_mask is not None:
            chain_hidden = chain_hidden * encoder_residue_mask.to(chain_hidden.dtype).unsqueeze(-1)
        if encoder_chain_mask is not None:
            chain_hidden = chain_hidden * encoder_chain_mask.to(chain_hidden.dtype).unsqueeze(-1).unsqueeze(-1)
        return chain_hidden

    def gather_token_condition(
        self,
        chain_token_condition: torch.Tensor,
        chain_ids: torch.Tensor,
        position_ids_inner: torch.Tensor,
        attention_mask: torch.Tensor | None,
        encoder_residue_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Map per-chain encoder features onto decoder token positions (legacy path).

        For each decoder residue token, looks up ``chain_ids`` and
        ``position_ids_inner`` to copy the matching ``[E]`` vector from
        ``chain_token_condition``. Grammar special tokens get zeros.

        Returns ``token_condition``: ``[B, S, E]``.
        """

        batch_size, seq_len = chain_ids.shape
        _, max_chains, chain_len, hidden_size = chain_token_condition.shape
        valid = chain_ids.ge(0) & chain_ids.lt(max_chains) & position_ids_inner.ge(0)
        if attention_mask is not None:
            valid = valid & attention_mask.bool()

        safe_chain = chain_ids.clamp(min=0, max=max_chains - 1)
        # Flat row index into the [B*C, L(, E)] views; one lookup per decoder token.
        row = torch.arange(batch_size, device=chain_ids.device).unsqueeze(1) * max_chains + safe_chain
        if encoder_residue_mask is None:
            # No residue mask: the k-th residue sits at encoder position k directly.
            valid = valid & position_ids_inner.lt(chain_len)
            encoder_pos = position_ids_inner.clamp(min=0, max=chain_len - 1)
        else:
            slot_pos, counts = _residue_slot_positions(encoder_residue_mask.bool())  # [B,C,L], [B,C]
            token_residue_count = torch.gather(counts, 1, safe_chain)
            valid = valid & position_ids_inner.lt(token_residue_count)
            safe_inner = position_ids_inner.clamp(min=0, max=chain_len - 1)
            slot_pos_flat = slot_pos.reshape(batch_size * max_chains, chain_len)
            encoder_pos = slot_pos_flat[row.reshape(-1), safe_inner.reshape(-1)].reshape(batch_size, seq_len)

        condition_flat = chain_token_condition.reshape(batch_size * max_chains, chain_len, hidden_size)
        gathered = condition_flat[row.reshape(-1), encoder_pos.reshape(-1)].reshape(
            batch_size, seq_len, hidden_size
        )
        return gathered * valid.to(gathered.dtype).unsqueeze(-1)

    def gather_proxy_token_condition(
        self,
        chain_token_condition: torch.Tensor,
        encoder_position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        residue_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather features from the single ESMC proxy stream (grammar-v1 path).

        ``encoder_position_ids[b, s]`` indexes into the proxy encoder sequence for
        each decoder token. Invalid / special positions are zeroed so structure
        and relation tokens receive no encoder signal.

        Returns ``token_condition``: ``[B, S, E]``.
        """

        batch_size, max_chains, encoder_len, hidden_size = chain_token_condition.shape
        if max_chains != 1:
            raise ValueError("Proxy token conditioning expects one encoder stream per record")
        batch_indices = torch.arange(batch_size, device=chain_token_condition.device).unsqueeze(1)
        valid = encoder_position_ids.ge(0) & encoder_position_ids.lt(encoder_len)
        if attention_mask is not None:
            valid = valid & attention_mask.bool()
        safe_positions = encoder_position_ids.clamp(min=0, max=encoder_len - 1)
        gathered = chain_token_condition[batch_indices, 0, safe_positions]
        valid_mask = valid.to(gathered.dtype).unsqueeze(-1)
        if residue_mask is not None:
            valid_mask = valid_mask * residue_mask.to(gathered.dtype).unsqueeze(-1)
        return gathered * valid_mask

    def build_encoder_condition_mask(
        self,
        *,
        chain_ids: torch.Tensor | None,
        position_ids_inner: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        encoder_position_ids: torch.Tensor | None,
        residue_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return ``[B, S]`` bool mask for residue positions that receive encoder embeddings."""

        if encoder_position_ids is not None:
            batch_size, seq_len = encoder_position_ids.shape
            valid = encoder_position_ids.ge(0)
            if attention_mask is not None:
                valid = valid & attention_mask.bool()
            if residue_mask is not None:
                valid = valid & residue_mask.bool()
            return valid

        if chain_ids is None or position_ids_inner is None:
            raise ValueError("chain_ids and position_ids_inner are required to build encoder condition mask")

        valid = chain_ids.ge(0) & position_ids_inner.ge(0)
        if attention_mask is not None:
            valid = valid & attention_mask.bool()
        if residue_mask is not None:
            valid = valid & residue_mask.bool()
        return valid

    def compute_loss(self, batch: dict[str, Any]) -> BioSeqDiffusionOutput:
        """Full training step: noise on decoder + mirrored encoder mask -> forward -> CE."""
        # 1) Mask decoder stream -> x_t [B,S]; mirror same mask onto encoder proxy [B,C,L]
        noised_input_ids, labels, corruption_mask, timesteps = sample_bioseq_diffusion_noise(
            batch=batch,
            mask_token_id=self.config.mask_token_id,
            time_epsilon=self.config.time_epsilon,
        )
        noised_encoder_input_ids = apply_decoder_corruption_to_encoder(
            batch=batch,
            corruption_mask=corruption_mask,
            mask_token_id=self.config.mask_token_id,
        )
        # 2) ESMC condition + decoder -> logits [B,S,V]
        output = self.forward(
            input_ids=noised_input_ids,
            attention_mask=batch.get("attention_mask"),
            chain_ids=batch.get("chain_ids"),
            position_ids_inner=batch.get("position_ids_inner"),
            position_ids_chain=batch.get("position_ids_chain"),
            timesteps=timesteps,
            residue_mask=batch.get("residue_mask"),
            encoder_input_ids=noised_encoder_input_ids,
            encoder_attention_mask=batch.get("encoder_attention_mask"),
            encoder_residue_mask=batch.get("encoder_residue_mask"),
            encoder_chain_mask=batch.get("encoder_chain_mask"),
            encoder_position_ids=batch.get("encoder_position_ids"),
        )
        # 3) CE on corrupted decoder positions only
        forbidden = forbidden_diffusion_target_token_ids(self.config)
        loss = compute_masked_cross_entropy(
            output.logits,
            labels,
            loss_norm=self.config.loss_norm,
            forbidden_token_ids=forbidden,
        )
        return BioSeqDiffusionOutput(
            loss=loss,
            logits=output.logits,
            hidden_states=output.hidden_states,
            noised_input_ids=noised_input_ids,
            labels=labels,
            corruption_mask=corruption_mask,
            timesteps=timesteps,
            noised_encoder_input_ids=noised_encoder_input_ids,
            encoder_condition=output.encoder_condition,
        )


def build_llada_backbone(config: "BioSeqDiffusionTransformerConfig", hidden_size: int) -> nn.Module:
    """Build a LLaDA (LLaMA-style, bidirectional masked-diffusion) backbone sized to BioSeq config.

    LLaDA supplies position through RoPE and takes no timestep input (RADD). The
    returned ``LLaDAModelLM`` exposes ``forward(inputs_embeds=...) -> logits [B, S, V]``
    with a grammar-sized (tied) vocabulary. Imported lazily so ``modeling_bioseq``
    stays importable in environments without the LLaDA/transformers stack.
    """

    from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
    from dllm.pipelines.llada.models.modeling_llada import (
        LLaDAModel,
        LLaDAModelLM,
        create_model_config_from_pretrained_config,
    )

    _ = hidden_size  # decoder width already equals encoder hidden via config.hidden_size
    llada_config = LLaDAConfig(
        d_model=int(config.hidden_size),
        n_heads=int(config.num_attention_heads),
        n_layers=int(config.num_hidden_layers),
        mlp_hidden_size=int(config.intermediate_size),
        # LLaMA block gates via explicit ff_proj(gate) * up_proj(value), so the
        # activation must be plain SiLU (output_multiplier 1.0), not the fused
        # SwiGLU (which would halve the width and mismatch up_proj).
        activation_type="silu",
        block_type="llama",
        layer_norm_type="rms",
        rms_norm_eps=1e-5,
        rope=True,
        rope_theta=10000.0,
        alibi=False,
        include_bias=False,
        include_qkv_bias=False,
        weight_tying=True,
        input_emb_norm=False,
        scale_logits=False,
        attention_dropout=float(config.dropout),
        residual_dropout=float(config.dropout),
        embedding_dropout=float(config.dropout),
        max_sequence_length=int(config.max_position_embeddings),
        vocab_size=int(config.vocab_size),
        embedding_size=int(config.vocab_size),
        pad_token_id=int(config.pad_token_id),
        mask_token_id=int(config.mask_token_id),
        use_cache=False,
        init_device="cpu",
    )
    # Build the inner model explicitly on CPU (LLaDAModelLM's auto path forces cuda).
    model_config = create_model_config_from_pretrained_config(llada_config)
    model_config.init_device = "cpu"
    inner = LLaDAModel(model_config, init_params=True)
    backbone = LLaDAModelLM(llada_config, model=inner)
    if bool(getattr(config, "gradient_checkpointing", False)):
        # Wire BioSeq's --gradient-checkpointing flag into LLaDA's activation
        # checkpointing (the in-house decoder used config.gradient_checkpointing
        # directly; LLaDA needs its own hook). Essential to fit large backbones.
        backbone.gradient_checkpointing_enable()
    return backbone


class BioSeqLLaDAEncoderDiffusionModel(BioSeqEncoderDiffusionModel):
    """ESMC/ESM2 encoder + **LLaDA** denoiser backbone.

    Reuses the exact per-chain encode + gather pipeline of
    ``BioSeqEncoderDiffusionModel`` (``encode_chain_tokens`` / ``gather_token_condition``
    / ``build_encoder_condition_mask``). The only change is the denoiser: instead of
    the in-house ``BioSeqDiffusionDecoder``, gathered encoder features **replace** the
    LLaDA token embedding (``wte``) at residue positions, and LLaDA denoises the flat
    grammar stream. Grammar special/structure/relation tokens keep their ``wte``
    embedding; position comes from LLaDA RoPE; there is no timestep input (LLaDA/RADD).

    The backbone is stored as ``self.decoder`` so optimizer grouping, DDP wrapping,
    and checkpoint plumbing that key on ``.decoder`` / ``.encoder`` keep working.
    """

    def __init__(
        self,
        decoder_config: BioSeqDiffusionTransformerConfig,
        encoder: nn.Module,
        encoder_hidden_size: int | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        encoder_hidden_size = int(encoder_hidden_size or infer_encoder_hidden_size(encoder))
        use_projection = bool(decoder_config.use_condition_projection)
        if not use_projection and int(decoder_config.hidden_size) != encoder_hidden_size:
            raise ValueError(
                "LLaDA decoder d_model must match encoder hidden size when use_condition_projection=False "
                f"(d_model={decoder_config.hidden_size}, encoder={encoder_hidden_size})"
            )
        self.config = replace(
            decoder_config,
            condition_hidden_size=encoder_hidden_size,
            use_condition_projection=use_projection,
        )
        self.encoder = encoder
        self.decoder = build_llada_backbone(self.config, encoder_hidden_size)
        self.condition_proj = (
            nn.Linear(encoder_hidden_size, int(decoder_config.hidden_size), bias=False)
            if use_projection
            else None
        )
        self.condition_norm = nn.LayerNorm(encoder_hidden_size) if self.config.condition_norm else None
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        diffusion_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        chain_ids: torch.Tensor | None = None,
        position_ids_inner: torch.Tensor | None = None,
        position_ids_chain: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        encoder_input_ids: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_residue_mask: torch.Tensor | None = None,
        encoder_chain_mask: torch.Tensor | None = None,
        encoder_position_ids: torch.Tensor | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> BioSeqDiffusionOutput:
        """Encode chains, gather condition, replace residue token embeddings, run LLaDA.

        ``position_ids_*`` and ``timesteps`` are accepted for a uniform training call
        signature but intentionally unused: LLaDA gets position from RoPE and needs no
        timestep. Returns ``BioSeqDiffusionOutput`` with ``logits [B, S, V]``.
        """
        if input_ids is None:
            raise ValueError("BioSeqLLaDAEncoderDiffusionModel requires input_ids")
        if encoder_input_ids is None:
            raise ValueError("encoder_input_ids are required for BioSeqLLaDAEncoderDiffusionModel")
        if encoder_position_ids is None and chain_ids is None:
            raise ValueError("chain_ids or encoder_position_ids are required for encoder conditions")

        effective_encoder_residue_mask = encoder_residue_mask
        if encoder_position_ids is not None and residue_mask is not None:
            effective_encoder_residue_mask = residue_mask.unsqueeze(1)

        # 1) per-chain encode -> [B, C, L, E]
        chain_token_condition = self.encode_chain_tokens(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_residue_mask=effective_encoder_residue_mask,
            encoder_chain_mask=encoder_chain_mask,
            encoder_kwargs=encoder_kwargs,
        )
        # 2) align to decoder token positions -> [B, S, E]
        if encoder_position_ids is not None:
            token_condition = self.gather_proxy_token_condition(
                chain_token_condition,
                encoder_position_ids=encoder_position_ids,
                attention_mask=attention_mask,
                residue_mask=residue_mask,
            )
        else:
            token_condition = self.gather_token_condition(
                chain_token_condition,
                chain_ids=chain_ids,
                position_ids_inner=position_ids_inner,
                attention_mask=attention_mask,
                encoder_residue_mask=encoder_residue_mask,
            )
        condition_mask = self.build_encoder_condition_mask(
            chain_ids=chain_ids,
            position_ids_inner=position_ids_inner,
            attention_mask=attention_mask,
            encoder_position_ids=encoder_position_ids,
            residue_mask=residue_mask,
        )

        # 3) inputs_embeds = LLaDA wte(x_t), with residue positions replaced by encoder features
        word_embeddings = self.decoder.get_input_embeddings()
        inputs_embeds = word_embeddings(input_ids)
        condition = token_condition.to(inputs_embeds.dtype)
        if self.condition_norm is not None:
            condition = self.condition_norm(condition)
        if self.config.use_condition_projection and self.condition_proj is not None:
            replace_mask = condition_mask.to(inputs_embeds.dtype).unsqueeze(-1)
            inputs_embeds = inputs_embeds + self.condition_proj(condition) * replace_mask
        else:
            replace_mask = condition_mask.to(inputs_embeds.dtype).unsqueeze(-1)
            inputs_embeds = inputs_embeds * (1.0 - replace_mask) + condition * replace_mask

        # 4) LLaDA denoiser (bidirectional; RoPE positions; no timestep) -> logits [B, S, V]
        llada_output = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return BioSeqDiffusionOutput(
            loss=None,
            logits=llada_output.logits,
            hidden_states=None,
            encoder_condition=token_condition,
        )


def build_llada2_backbone(config: "BioSeqDiffusionTransformerConfig", hidden_size: int) -> nn.Module:
    """Build a LLaDA2-MoE (inclusionAI LLaDA2.0) backbone sized to BioSeq config.

    Architecture is kept faithful to ``inclusionAI/LLaDA2.0-mini`` (MoE with
    group-limited sigmoid routing + shared expert, partial-RoPE attention, RMSNorm,
    ``is_causal=False``) — only from-scratch (no pretrained weights). The size knobs
    (layers / heads / experts / hidden) come from ``config`` so a run can be scaled
    down to fit plain DDP while the module classes/paths stay identical.

    Returns ``LLaDA2MoeModelLM`` exposing ``forward(inputs_embeds=..., attention_mask=4D)
    -> logits [B, S, V]`` with a grammar-sized vocabulary. The backbone is driven
    **bidirectionally** by passing a 4D padding mask (see
    ``BioSeqLLaDA2EncoderDiffusionModel.forward``), which bypasses the causal-mask
    construction in ``LLaDA2MoeModel.forward``. Imported lazily so ``modeling_bioseq``
    stays importable without the LLaDA2/transformers stack.
    """

    from dllm.pipelines.llada2.models.configuration_llada2_moe import LLaDA2MoeConfig
    from dllm.pipelines.llada2.models.modeling_llada2_moe import LLaDA2MoeModelLM

    _ = hidden_size  # decoder width already equals encoder hidden via config.hidden_size
    llada2_config = LLaDA2MoeConfig(
        vocab_size=int(config.vocab_size),
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.intermediate_size),
        num_hidden_layers=int(config.num_hidden_layers),
        num_attention_heads=int(config.num_attention_heads),
        num_key_value_heads=int(config.moe_num_key_value_heads),
        head_dim=int(config.moe_head_dim),
        hidden_act="silu",
        use_qkv_bias=False,
        use_qk_norm=bool(config.qk_norm),
        use_bias=False,
        rms_norm_eps=1e-5,
        embedding_dropout=float(config.dropout),
        attention_dropout=float(config.dropout),
        output_dropout=float(config.dropout),
        initializer_range=float(config.initializer_range),
        max_position_embeddings=int(config.max_position_embeddings),
        rope_theta=float(config.moe_rope_theta),
        partial_rotary_factor=float(config.moe_partial_rotary_factor),
        use_cache=False,
        num_experts=int(config.moe_num_experts),
        num_shared_experts=int(config.moe_num_shared_experts),
        num_experts_per_tok=int(config.moe_num_experts_per_tok),
        n_group=int(config.moe_n_group),
        topk_group=int(config.moe_topk_group),
        moe_intermediate_size=int(config.moe_intermediate_size),
        first_k_dense_replace=int(config.moe_first_k_dense_replace),
        routed_scaling_factor=float(config.moe_routed_scaling_factor),
        pad_token_id=int(config.pad_token_id),
        tie_word_embeddings=False,
    )
    # SDPA is required so the 4D bidirectional mask is used verbatim (the eager path
    # forces a causal mask). LLaDA2 attention already sets is_causal=False.
    llada2_config._attn_implementation = "sdpa"
    backbone = LLaDA2MoeModelLM(llada2_config)
    if bool(getattr(config, "gradient_checkpointing", False)):
        backbone.gradient_checkpointing_enable()
    return backbone


class BioSeqLLaDA2EncoderDiffusionModel(BioSeqEncoderDiffusionModel):
    """ESMC/ESM2 encoder + **LLaDA2-MoE** denoiser backbone (from scratch).

    Same wiring as :class:`BioSeqLLaDAEncoderDiffusionModel` (reuses the per-chain
    encode + gather pipeline; gathered encoder features **replace** the backbone
    ``word_embeddings`` at residue positions; grammar special/structure/relation
    tokens keep their embedding; position via RoPE; no timestep). The only change is
    the denoiser is LLaDA2-MoE, driven bidirectionally by a 4D padding mask.

    The backbone is stored as ``self.decoder`` so optimizer grouping, DDP wrapping,
    and checkpoint plumbing that key on ``.decoder`` / ``.encoder`` keep working.
    """

    def __init__(
        self,
        decoder_config: BioSeqDiffusionTransformerConfig,
        encoder: nn.Module,
        encoder_hidden_size: int | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        encoder_hidden_size = int(encoder_hidden_size or infer_encoder_hidden_size(encoder))
        use_projection = bool(decoder_config.use_condition_projection)
        if not use_projection and int(decoder_config.hidden_size) != encoder_hidden_size:
            raise ValueError(
                "LLaDA2 decoder hidden_size must match encoder hidden size when use_condition_projection=False "
                f"(hidden_size={decoder_config.hidden_size}, encoder={encoder_hidden_size})"
            )
        self.config = replace(
            decoder_config,
            condition_hidden_size=encoder_hidden_size,
            use_condition_projection=use_projection,
        )
        self.encoder = encoder
        self.decoder = build_llada2_backbone(self.config, encoder_hidden_size)
        self.condition_proj = (
            nn.Linear(encoder_hidden_size, int(decoder_config.hidden_size), bias=False)
            if use_projection
            else None
        )
        self.condition_norm = nn.LayerNorm(encoder_hidden_size) if self.config.condition_norm else None
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        diffusion_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        chain_ids: torch.Tensor | None = None,
        position_ids_inner: torch.Tensor | None = None,
        position_ids_chain: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        encoder_input_ids: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_residue_mask: torch.Tensor | None = None,
        encoder_chain_mask: torch.Tensor | None = None,
        encoder_position_ids: torch.Tensor | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> BioSeqDiffusionOutput:
        """Encode chains, gather condition, replace residue embeddings, run LLaDA2-MoE.

        ``position_ids_*`` and ``timesteps`` are accepted for a uniform training call
        signature but intentionally unused. Returns ``BioSeqDiffusionOutput`` with
        ``logits [B, S, V]``.
        """
        if input_ids is None:
            raise ValueError("BioSeqLLaDA2EncoderDiffusionModel requires input_ids")
        if encoder_input_ids is None:
            raise ValueError("encoder_input_ids are required for BioSeqLLaDA2EncoderDiffusionModel")
        if encoder_position_ids is None and chain_ids is None:
            raise ValueError("chain_ids or encoder_position_ids are required for encoder conditions")

        effective_encoder_residue_mask = encoder_residue_mask
        if encoder_position_ids is not None and residue_mask is not None:
            effective_encoder_residue_mask = residue_mask.unsqueeze(1)

        # 1) per-chain encode -> [B, C, L, E]
        chain_token_condition = self.encode_chain_tokens(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_residue_mask=effective_encoder_residue_mask,
            encoder_chain_mask=encoder_chain_mask,
            encoder_kwargs=encoder_kwargs,
        )
        # 2) align to decoder token positions -> [B, S, E]
        if encoder_position_ids is not None:
            token_condition = self.gather_proxy_token_condition(
                chain_token_condition,
                encoder_position_ids=encoder_position_ids,
                attention_mask=attention_mask,
                residue_mask=residue_mask,
            )
        else:
            token_condition = self.gather_token_condition(
                chain_token_condition,
                chain_ids=chain_ids,
                position_ids_inner=position_ids_inner,
                attention_mask=attention_mask,
                encoder_residue_mask=encoder_residue_mask,
            )
        condition_mask = self.build_encoder_condition_mask(
            chain_ids=chain_ids,
            position_ids_inner=position_ids_inner,
            attention_mask=attention_mask,
            encoder_position_ids=encoder_position_ids,
            residue_mask=residue_mask,
        )

        # 3) inputs_embeds = backbone wte(x_t), residue positions replaced by encoder features
        word_embeddings = self.decoder.get_input_embeddings()
        inputs_embeds = word_embeddings(input_ids)
        condition = token_condition.to(inputs_embeds.dtype)
        if self.condition_norm is not None:
            condition = self.condition_norm(condition)
        replace_mask = condition_mask.to(inputs_embeds.dtype).unsqueeze(-1)
        if self.config.use_condition_projection and self.condition_proj is not None:
            inputs_embeds = inputs_embeds + self.condition_proj(condition) * replace_mask
        else:
            inputs_embeds = inputs_embeds * (1.0 - replace_mask) + condition * replace_mask

        # 4) Bidirectional 4D padding mask (True = attend). Passing a 4D mask bypasses
        # LLaDA2's causal-mask construction; the attention itself is is_causal=False,
        # so the backbone runs as a full bidirectional masked-diffusion denoiser.
        mask_4d = None
        if attention_mask is not None:
            keep = attention_mask.to(torch.bool)
            batch_size, seq_len = keep.shape
            mask_4d = keep[:, None, None, :].expand(batch_size, 1, seq_len, seq_len).contiguous()

        llada2_output = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=mask_4d,
            use_cache=False,
            output_router_logits=False,
        )
        return BioSeqDiffusionOutput(
            loss=None,
            logits=llada2_output.logits,
            hidden_states=None,
            encoder_condition=token_condition,
        )
