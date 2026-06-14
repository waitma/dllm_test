from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BioSeqModelConfig
from .data import SPECIAL_CHAIN_ID


@dataclass
class BioSeqModelOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor


class ChainAwareMultiheadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, use_multimer: bool = True) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.use_multimer = use_multimer

        self.self_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.multimer_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        chain_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        self_q, self_k, self_v = self._project(self.self_qkv(hidden_states), batch_size, seq_len)

        if self.use_multimer and chain_ids is not None:
            multi_q, multi_k, multi_v = self._project(self.multimer_qkv(hidden_states), batch_size, seq_len)
            same_chain = chain_ids[:, :, None].eq(chain_ids[:, None, :])
            valid_chain = chain_ids.ne(SPECIAL_CHAIN_ID)
            cross_chain = (~same_chain) & valid_chain[:, :, None] & valid_chain[:, None, :]
            self_scores = torch.matmul(self_q, self_k.transpose(-1, -2)) * self.scale
            multi_scores = torch.matmul(multi_q, multi_k.transpose(-1, -2)) * self.scale
            scores = torch.where(cross_chain[:, None, :, :], multi_scores, self_scores)
        else:
            multi_v = None
            cross_chain = None
            scores = torch.matmul(self_q, self_k.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask[:, None, None, :], torch.finfo(scores.dtype).min)

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, self_v)

        if multi_v is not None and cross_chain is not None:
            cross_probs = probs * cross_chain[:, None, :, :].to(probs.dtype)
            context = context + torch.matmul(cross_probs, multi_v - self_v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(context)

    def _project(
        self,
        qkv: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]


class BioSeqTransformerLayer(nn.Module):
    def __init__(self, config: BioSeqModelConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.attention = ChainAwareMultiheadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            use_multimer=config.use_multimer_attention,
        )
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        chain_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, chain_ids=chain_ids, attention_mask=attention_mask)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + self.dropout(hidden_states)


class NoEncoderBioDiffusionModel(nn.Module):
    def __init__(self, config: BioSeqModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = (
            nn.Embedding(config.max_position_embeddings, config.hidden_size)
            if config.use_position_embeddings
            else None
        )
        self.condition_proj = (
            nn.Linear(config.condition_hidden_size, config.hidden_size)
            if config.condition_hidden_size is not None
            else None
        )
        self.layers = nn.ModuleList(
            [BioSeqTransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        chain_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        condition_hidden_states: torch.Tensor | None = None,
        **_: Any,
    ) -> BioSeqModelOutput:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        hidden_states = self.token_embeddings(input_ids)
        if self.config.token_dropout:
            mask_positions = input_ids.eq(self.config.mask_token_id)
            hidden_states = hidden_states.masked_fill(mask_positions.unsqueeze(-1), 0.0)
            if attention_mask is None:
                src_lengths = input_ids.ne(self.config.pad_token_id).sum(dim=-1)
            else:
                src_lengths = attention_mask.sum(dim=-1)
            mask_ratio_train = 0.15 * 0.8
            mask_ratio_observed = mask_positions.sum(dim=-1).to(hidden_states.dtype) / src_lengths.clamp_min(1)
            scale = (1 - mask_ratio_train) / (1 - mask_ratio_observed).clamp_min(1e-6)
            hidden_states = hidden_states * scale[:, None, None]

        if self.position_embeddings is not None:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            hidden_states = hidden_states + self.position_embeddings(positions)

        if condition_hidden_states is not None:
            if self.condition_proj is None:
                raise ValueError("condition_hidden_states were provided, but condition_hidden_size is not configured")
            hidden_states = hidden_states + self.condition_proj(condition_hidden_states).unsqueeze(1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, chain_ids=chain_ids, attention_mask=attention_mask)

        hidden_states = self.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states)
        return BioSeqModelOutput(logits=logits, hidden_states=hidden_states)


class ESMCEncoderBioDiffusionModel(nn.Module):
    def __init__(
        self,
        decoder_config: BioSeqModelConfig,
        encoder_name_or_path: str,
        freeze_encoder: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("transformers is required to load the ESMC encoder") from exc

        self.encoder = AutoModel.from_pretrained(
            encoder_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        encoder_hidden_size = getattr(self.encoder.config, "hidden_size", None) or getattr(
            self.encoder.config, "d_model", None
        )
        if encoder_hidden_size is None:
            raise ValueError("Could not infer ESMC encoder hidden size from config")

        decoder_config.condition_hidden_size = int(encoder_hidden_size)
        self.decoder = NoEncoderBioDiffusionModel(decoder_config)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        chain_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_input_ids: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BioSeqModelOutput:
        if encoder_input_ids is None:
            raise ValueError("encoder_input_ids are required for ESMCEncoderBioDiffusionModel")
        encoder_outputs = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            **kwargs,
        )
        encoder_hidden = encoder_outputs.last_hidden_state
        if encoder_attention_mask is None:
            condition = encoder_hidden.mean(dim=1)
        else:
            mask = encoder_attention_mask.to(encoder_hidden.dtype).unsqueeze(-1)
            condition = (encoder_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        return self.decoder(
            input_ids=input_ids,
            chain_ids=chain_ids,
            attention_mask=attention_mask,
            condition_hidden_states=condition,
        )
