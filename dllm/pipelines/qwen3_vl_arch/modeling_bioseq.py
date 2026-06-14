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
    max_chain_roles: int = 32
    max_task_types: int = 32
    pad_token_id: int = 1
    mask_token_id: int = 32
    time_epsilon: float = 1e-3
    loss_norm: str = "token"
    condition_hidden_size: int | None = None


@dataclass
class BioSeqDiffusionOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: torch.Tensor
    noised_input_ids: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    corruption_mask: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    noised_encoder_input_ids: torch.Tensor | None = None


class BioSeqRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states * self.weight


class BioSeqSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.down_proj(hidden_states)


class BioSeqSelfAttention(nn.Module):
    """Bidirectional self-attention for masked diffusion training."""

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

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        query = self._shape(self.q_proj(hidden_states), batch_size, seq_len)
        key = self._shape(self.k_proj(hidden_states), batch_size, seq_len)
        value = self._shape(self.v_proj(hidden_states), batch_size, seq_len)

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            key_mask = attention_mask.bool()[:, None, None, :]
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(context)

    def _shape(self, states: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


class BioSeqTransformerBlock(nn.Module):
    def __init__(self, config: BioSeqDiffusionTransformerConfig) -> None:
        super().__init__()
        self.input_layernorm = BioSeqRMSNorm(config.hidden_size)
        self.self_attn = BioSeqSelfAttention(config)
        self.post_attention_layernorm = BioSeqRMSNorm(config.hidden_size)
        self.mlp = BioSeqSwiGLU(config.hidden_size, config.intermediate_size, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), attention_mask=attention_mask)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + self.dropout(hidden_states)


class BioSeqTimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
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
    """Sample masked diffusion corruption from ``diffusion_loss_mask``.

    Only target residues selected by the collator/view sampler are eligible.
    Fixed context chains such as antigen, peptide, and MHC stay visible.
    """

    if not (0.0 < time_epsilon < 1.0):
        raise ValueError("time_epsilon must be in (0, 1)")

    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    residue_mask = batch.get("residue_mask")
    loss_mask = batch.get("diffusion_loss_mask", batch.get("diffusion_target_mask"))
    if loss_mask is None:
        raise KeyError("batch requires diffusion_loss_mask or diffusion_target_mask")

    eligible_mask = loss_mask.bool()
    if attention_mask is not None:
        eligible_mask = eligible_mask & attention_mask.bool()
    if residue_mask is not None:
        eligible_mask = eligible_mask & residue_mask.bool()
    if not eligible_mask.any():
        raise ValueError("batch has no eligible diffusion target residues")

    batch_size, seq_len = input_ids.shape
    timesteps = torch.empty(batch_size, device=input_ids.device, dtype=torch.float32).uniform_(time_epsilon, 1.0)
    mask_probs = timesteps[:, None].expand(batch_size, seq_len)
    corruption_mask = (torch.rand(batch_size, seq_len, device=input_ids.device) < mask_probs) & eligible_mask

    for row in range(batch_size):
        if eligible_mask[row].any() and not corruption_mask[row].any():
            valid_positions = torch.nonzero(eligible_mask[row], as_tuple=False).flatten()
            choice = valid_positions[torch.randint(valid_positions.numel(), (1,), device=input_ids.device)]
            corruption_mask[row, choice] = True

    noised_input_ids = input_ids.masked_fill(corruption_mask, int(mask_token_id))
    labels = input_ids.masked_fill(~corruption_mask, -100)
    return noised_input_ids, labels, corruption_mask, timesteps


def apply_decoder_corruption_to_encoder(
    batch: dict[str, Any],
    corruption_mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Mask encoder residues that correspond to corrupted decoder targets.

    The collator emits encoder inputs per chain and decoder inputs as a
    concatenated chain stream. This function maps each corrupted decoder residue
    back to its chain-local residue token in ``encoder_input_ids``.
    """

    encoder_input_ids = batch["encoder_input_ids"]
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


def compute_masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_norm: str = "token",
) -> torch.Tensor:
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
    """Thin wrapper exposing Biohub's native ESMC as a Hugging Face-like encoder."""

    def __init__(self, esmc: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.esmc = esmc
        self.config = SimpleNamespace(hidden_size=hidden_size, d_model=hidden_size, model_type="esmc")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        output = self.esmc(
            sequence_tokens=input_ids,
            sequence_id=attention_mask.bool() if attention_mask is not None else None,
        )
        return SimpleNamespace(
            last_hidden_state=output.embeddings,
            hidden_states=output.hidden_states,
            logits=output.sequence_logits,
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
    """Qwen-style bidirectional decoder used for BioSeq diffusion training."""

    def __init__(self, config: BioSeqDiffusionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.inner_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.chain_position_embeddings = nn.Embedding(config.max_chain_positions, config.hidden_size)
        self.chain_role_embeddings = nn.Embedding(config.max_chain_roles, config.hidden_size)
        self.task_embeddings = nn.Embedding(config.max_task_types, config.hidden_size)
        self.timestep_embeddings = BioSeqTimestepEmbedding(config.hidden_size)
        self.condition_proj = (
            nn.Linear(config.condition_hidden_size, config.hidden_size, bias=False)
            if config.condition_hidden_size is not None
            else None
        )
        self.layers = nn.ModuleList([BioSeqTransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = BioSeqRMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        chain_role_ids: torch.Tensor | None = None,
        position_ids_inner: torch.Tensor | None = None,
        position_ids_chain: torch.Tensor | None = None,
        task_type_ids: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        encoder_condition: torch.Tensor | None = None,
    ) -> BioSeqDiffusionOutput:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        hidden_states = self.token_embeddings(input_ids)

        if position_ids_inner is None:
            position_ids_inner = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        safe_inner = position_ids_inner.clamp(min=0, max=self.config.max_position_embeddings - 1)
        inner_valid = position_ids_inner.ge(0).to(hidden_states.dtype).unsqueeze(-1)
        hidden_states = hidden_states + self.inner_position_embeddings(safe_inner) * inner_valid

        if position_ids_chain is not None:
            safe_chain = position_ids_chain.clamp(min=0, max=self.config.max_chain_positions - 1)
            chain_valid = position_ids_chain.ge(0).to(hidden_states.dtype).unsqueeze(-1)
            hidden_states = hidden_states + self.chain_position_embeddings(safe_chain) * chain_valid

        if chain_role_ids is not None:
            safe_roles = chain_role_ids.clamp(min=0, max=self.config.max_chain_roles - 1)
            hidden_states = hidden_states + self.chain_role_embeddings(safe_roles)

        if task_type_ids is not None:
            safe_task = task_type_ids.clamp(min=0, max=self.config.max_task_types - 1)
            hidden_states = hidden_states + self.task_embeddings(safe_task).unsqueeze(1)

        if timesteps is not None:
            hidden_states = hidden_states + self.timestep_embeddings(timesteps).unsqueeze(1)

        if encoder_condition is not None:
            if self.condition_proj is None:
                raise ValueError("encoder_condition was provided, but condition_hidden_size is not configured")
            hidden_states = hidden_states + self.condition_proj(encoder_condition)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.to(hidden_states.dtype).unsqueeze(-1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.to(hidden_states.dtype).unsqueeze(-1)

        hidden_states = self.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states)
        return BioSeqDiffusionOutput(loss=None, logits=logits, hidden_states=hidden_states)


class BioSeqNoEncoderDiffusionModel(nn.Module):
    """Decoder-only BioSeq masked diffusion model."""

    def __init__(self, config: BioSeqDiffusionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = BioSeqDiffusionDecoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        chain_role_ids: torch.Tensor | None = None,
        position_ids_inner: torch.Tensor | None = None,
        position_ids_chain: torch.Tensor | None = None,
        task_type_ids: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        **_: Any,
    ) -> BioSeqDiffusionOutput:
        return self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chain_role_ids=chain_role_ids,
            position_ids_inner=position_ids_inner,
            position_ids_chain=position_ids_chain,
            task_type_ids=task_type_ids,
            timesteps=timesteps,
        )

    def compute_loss(self, batch: dict[str, Any]) -> BioSeqDiffusionOutput:
        noised_input_ids, labels, corruption_mask, timesteps = sample_bioseq_diffusion_noise(
            batch=batch,
            mask_token_id=self.config.mask_token_id,
            time_epsilon=self.config.time_epsilon,
        )
        output = self.forward(
            input_ids=noised_input_ids,
            attention_mask=batch.get("attention_mask"),
            chain_role_ids=batch.get("chain_role_ids"),
            position_ids_inner=batch.get("position_ids_inner"),
            position_ids_chain=batch.get("position_ids_chain"),
            task_type_ids=batch.get("task_type_ids"),
            timesteps=timesteps,
        )
        loss = compute_masked_cross_entropy(output.logits, labels, loss_norm=self.config.loss_norm)
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
    """ESM-family encoder conditioned BioSeq diffusion model."""

    def __init__(
        self,
        decoder_config: BioSeqDiffusionTransformerConfig,
        encoder: nn.Module,
        encoder_hidden_size: int | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        encoder_hidden_size = int(encoder_hidden_size or infer_encoder_hidden_size(encoder))
        self.config = replace(decoder_config, condition_hidden_size=encoder_hidden_size)
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
        try:
            from transformers import AutoModel
        except ImportError as exc:
            encoder = load_local_esmc_encoder(encoder_name_or_path, use_flash_attn=use_flash_attn)
            return cls(decoder_config=decoder_config, encoder=encoder, freeze_encoder=freeze_encoder)

        try:
            encoder = AutoModel.from_pretrained(
                encoder_name_or_path,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        except (OSError, ValueError, KeyError):
            encoder = load_local_esmc_encoder(encoder_name_or_path, use_flash_attn=use_flash_attn)
        return cls(decoder_config=decoder_config, encoder=encoder, freeze_encoder=freeze_encoder)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        chain_ids: torch.Tensor | None = None,
        chain_role_ids: torch.Tensor | None = None,
        position_ids_inner: torch.Tensor | None = None,
        position_ids_chain: torch.Tensor | None = None,
        task_type_ids: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        encoder_input_ids: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_residue_mask: torch.Tensor | None = None,
        encoder_chain_mask: torch.Tensor | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> BioSeqDiffusionOutput:
        if encoder_input_ids is None:
            raise ValueError("encoder_input_ids are required for BioSeqEncoderDiffusionModel")
        if chain_ids is None:
            raise ValueError("chain_ids are required to gather per-chain encoder conditions")

        chain_condition = self.encode_chains(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_residue_mask=encoder_residue_mask,
            encoder_chain_mask=encoder_chain_mask,
            encoder_kwargs=encoder_kwargs,
        )
        token_condition = self.gather_chain_condition(chain_condition, chain_ids=chain_ids, attention_mask=attention_mask)
        output = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chain_role_ids=chain_role_ids,
            position_ids_inner=position_ids_inner,
            position_ids_chain=position_ids_chain,
            task_type_ids=task_type_ids,
            timesteps=timesteps,
            encoder_condition=token_condition,
        )
        return output

    def encode_chains(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        encoder_residue_mask: torch.Tensor | None,
        encoder_chain_mask: torch.Tensor | None,
        encoder_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        batch_size, max_chains, chain_len = encoder_input_ids.shape
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
        encoder_outputs = self.encoder(**call_kwargs)
        flat_hidden = getattr(encoder_outputs, "last_hidden_state", None)
        if flat_hidden is None:
            if isinstance(encoder_outputs, tuple):
                flat_hidden = encoder_outputs[0]
            else:
                raise ValueError("Encoder output must expose last_hidden_state or tuple[0]")

        hidden_size = flat_hidden.shape[-1]
        chain_hidden = flat_hidden.reshape(batch_size, max_chains, chain_len, hidden_size)
        if encoder_residue_mask is None:
            pool_mask = encoder_attention_mask
        else:
            pool_mask = encoder_residue_mask
            if encoder_attention_mask is not None:
                pool_mask = pool_mask & encoder_attention_mask.bool()
        if pool_mask is None:
            pool_mask = torch.ones(
                batch_size,
                max_chains,
                chain_len,
                device=encoder_input_ids.device,
                dtype=torch.bool,
            )
        if encoder_chain_mask is not None:
            pool_mask = pool_mask & encoder_chain_mask.bool().unsqueeze(-1)

        pool_mask_float = pool_mask.to(chain_hidden.dtype).unsqueeze(-1)
        pooled = (chain_hidden * pool_mask_float).sum(dim=2) / pool_mask_float.sum(dim=2).clamp_min(1.0)
        if encoder_chain_mask is not None:
            pooled = pooled * encoder_chain_mask.to(pooled.dtype).unsqueeze(-1)
        return pooled

    def gather_chain_condition(
        self,
        chain_condition: torch.Tensor,
        chain_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        _, max_chains, hidden_size = chain_condition.shape
        safe_chain_ids = chain_ids.clamp(min=0, max=max_chains - 1)
        token_condition = torch.gather(
            chain_condition,
            dim=1,
            index=safe_chain_ids.unsqueeze(-1).expand(-1, -1, hidden_size),
        )
        valid = chain_ids.ge(0) & chain_ids.lt(max_chains)
        if attention_mask is not None:
            valid = valid & attention_mask.bool()
        return token_condition * valid.to(token_condition.dtype).unsqueeze(-1)

    def compute_loss(self, batch: dict[str, Any]) -> BioSeqDiffusionOutput:
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
        output = self.forward(
            input_ids=noised_input_ids,
            attention_mask=batch.get("attention_mask"),
            chain_ids=batch.get("chain_ids"),
            chain_role_ids=batch.get("chain_role_ids"),
            position_ids_inner=batch.get("position_ids_inner"),
            position_ids_chain=batch.get("position_ids_chain"),
            task_type_ids=batch.get("task_type_ids"),
            timesteps=timesteps,
            encoder_input_ids=noised_encoder_input_ids,
            encoder_attention_mask=batch.get("encoder_attention_mask"),
            encoder_residue_mask=batch.get("encoder_residue_mask"),
            encoder_chain_mask=batch.get("encoder_chain_mask"),
        )
        loss = compute_masked_cross_entropy(output.logits, labels, loss_norm=self.config.loss_norm)
        return BioSeqDiffusionOutput(
            loss=loss,
            logits=output.logits,
            hidden_states=output.hidden_states,
            noised_input_ids=noised_input_ids,
            labels=labels,
            corruption_mask=corruption_mask,
            timesteps=timesteps,
            noised_encoder_input_ids=noised_encoder_input_ids,
        )
