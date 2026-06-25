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
    gradient_checkpointing: bool = False
    initializer_range: float = 0.02


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


def sample_bioseq_diffusion_noise(
    batch: dict[str, Any],
    mask_token_id: int,
    time_epsilon: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample masked-diffusion corruption for one training step.

    Follows the AirGen/Ophiuchus-style forward process: per sequence sample
    ``t ~ Uniform(time_epsilon, 1)``, then independently mask each eligible
    token with probability ``t``. Fixed grammar context (``<fixs>...<fixd>``)
    is excluded via ``diffusion_loss_mask`` / ``diffusion_eligible_mask``.

    Returns
    -------
    noised_input_ids : ``[B, S]`` — clean ids with corrupted positions set to ``mask_token_id``.
    labels : ``[B, S]`` — clean ids on corrupted positions; ``-100`` elsewhere.
    corruption_mask : ``[B, S]`` bool — True where ``<mask>`` was applied.
    timesteps : ``[B]`` float — sampled noise level per sequence.
    """

    if not (0.0 < time_epsilon < 1.0):
        raise ValueError("time_epsilon must be in (0, 1)")

    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    loss_mask = batch.get("diffusion_loss_mask", batch.get("diffusion_target_mask"))
    if loss_mask is None:
        raise KeyError("batch requires diffusion_loss_mask or diffusion_target_mask")

    explicit_eligible_mask = batch.get("diffusion_eligible_mask")
    eligible_mask = (
        explicit_eligible_mask.bool()
        if explicit_eligible_mask is not None
        else loss_mask.bool()
    )
    if attention_mask is not None:
        eligible_mask = eligible_mask & attention_mask.bool()
    residue_mask = batch.get("residue_mask")
    if explicit_eligible_mask is None and residue_mask is not None:
        eligible_mask = eligible_mask & residue_mask.bool()
    if not eligible_mask.any():
        raise ValueError("batch has no eligible diffusion target tokens")

    batch_size, seq_len = input_ids.shape
    # Per-sequence noise level t ~ U(eps, 1); each eligible token masked independently with prob t
    timesteps = torch.empty(batch_size, device=input_ids.device, dtype=torch.float32).uniform_(time_epsilon, 1.0)
    mask_probs = timesteps[:, None].expand(batch_size, seq_len)
    corruption_mask = (torch.rand(batch_size, seq_len, device=input_ids.device) < mask_probs) & eligible_mask

    # Guarantee >=1 masked token per row (avoid zero-loss microbatch)
    for row in range(batch_size):
        if eligible_mask[row].any() and not corruption_mask[row].any():
            valid_positions = torch.nonzero(eligible_mask[row], as_tuple=False).flatten()
            choice = valid_positions[torch.randint(valid_positions.numel(), (1,), device=input_ids.device)]
            corruption_mask[row, choice] = True

    noised_input_ids = input_ids.masked_fill(corruption_mask, int(mask_token_id))  # x_t [B,S]
    labels = input_ids.masked_fill(~corruption_mask, -100)  # targets only at masked sites [B,S]
    return noised_input_ids, labels, corruption_mask, timesteps


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
        noised_encoder_input_ids = encoder_input_ids.clone()
        batch_size, max_chains, encoder_len = encoder_input_ids.shape
        if max_chains != 1:
            raise ValueError("Direct encoder_position_ids mapping requires a single proxy stream")
        for batch_index in range(batch_size):
            decoder_positions = torch.nonzero(corruption_mask[batch_index], as_tuple=False).flatten()
            if decoder_positions.numel() == 0:
                continue
            encoder_positions = encoder_position_ids[batch_index, decoder_positions]
            valid = encoder_positions.ge(0) & encoder_positions.lt(encoder_len)
            noised_encoder_input_ids[
                batch_index,
                0,
                encoder_positions[valid],
            ] = int(mask_token_id)
        return noised_encoder_input_ids

    encoder_residue_mask = batch["encoder_residue_mask"]
    chain_ids = batch["chain_ids"]
    position_ids_inner = batch["position_ids_inner"]

    noised_encoder_input_ids = encoder_input_ids.clone()
    batch_size, max_chains, _ = encoder_input_ids.shape
    for batch_index in range(batch_size):
        corrupted_positions = torch.nonzero(corruption_mask[batch_index], as_tuple=False).flatten()
        for decoder_pos in corrupted_positions.tolist():
            chain_index = int(chain_ids[batch_index, decoder_pos].item())
            residue_index = int(position_ids_inner[batch_index, decoder_pos].item())
            if chain_index < 0 or chain_index >= max_chains or residue_index < 0:
                continue
            residue_token_positions = torch.nonzero(
                encoder_residue_mask[batch_index, chain_index],
                as_tuple=False,
            ).flatten()
            if residue_index >= residue_token_positions.numel():
                continue
            encoder_pos = residue_token_positions[residue_index]
            noised_encoder_input_ids[batch_index, chain_index, encoder_pos] = int(mask_token_id)
    return noised_encoder_input_ids


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


def compute_masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_norm: str = "token",
    forbidden_token_ids: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Cross-entropy on corrupted positions only (``labels != -100``).

    ``logits``: ``[B, S, V]``, ``labels``: ``[B, S]``. Returns a scalar loss.
    """
    logits = mask_forbidden_target_logits(logits, forbidden_token_ids)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    loss_mask = labels.ne(-100)
    if loss_norm == "token":
        return token_loss.sum() / loss_mask.sum().clamp_min(1)
    if loss_norm == "sequence":
        per_sequence = token_loss.sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1)
        return per_sequence.mean()
    if loss_norm == "batch":
        return token_loss.sum() / labels.shape[0]
    raise ValueError(f"Unsupported loss_norm: {loss_norm}")


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

        # + ESMC/ESM2 condition (encoder models): replace residue embeddings or add projection.
        if encoder_condition is not None:
            if encoder_condition_mask is not None and not self.config.use_condition_projection:
                replace_mask = encoder_condition_mask.to(hidden_states.dtype).unsqueeze(-1)
                hidden_states = hidden_states * (1.0 - replace_mask) + encoder_condition.to(
                    hidden_states.dtype
                ) * replace_mask
            elif self.condition_proj is not None:
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

        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(
            encoder_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
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
        _, max_chains, _, hidden_size = chain_token_condition.shape
        token_condition = chain_token_condition.new_zeros(batch_size, seq_len, hidden_size)
        valid_decoder = chain_ids.ge(0) & chain_ids.lt(max_chains) & position_ids_inner.ge(0)
        if attention_mask is not None:
            valid_decoder = valid_decoder & attention_mask.bool()

        for batch_index in range(batch_size):
            for chain_index in range(max_chains):
                decoder_positions = torch.nonzero(
                    valid_decoder[batch_index] & chain_ids[batch_index].eq(chain_index),
                    as_tuple=False,
                ).flatten()
                if decoder_positions.numel() == 0:
                    continue
                if encoder_residue_mask is None:
                    residue_token_positions = torch.arange(
                        chain_token_condition.shape[2],
                        device=chain_token_condition.device,
                    )
                else:
                    residue_token_positions = torch.nonzero(
                        encoder_residue_mask[batch_index, chain_index],
                        as_tuple=False,
                    ).flatten()
                residue_indices = position_ids_inner[batch_index, decoder_positions]
                in_bounds = residue_indices.lt(residue_token_positions.numel())
                if not in_bounds.any():
                    continue
                decoder_positions = decoder_positions[in_bounds]
                encoder_positions = residue_token_positions[residue_indices[in_bounds]]
                token_condition[batch_index, decoder_positions] = chain_token_condition[
                    batch_index,
                    chain_index,
                    encoder_positions,
                ]
        return token_condition

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
