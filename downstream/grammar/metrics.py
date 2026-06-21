"""Shared metrics for grammar downstream eval."""

from __future__ import annotations

import torch

from dllm.pipelines.qwen3_vl_arch.data import GrammarTokenizer


def masked_token_accuracy(
    output_tokens: torch.Tensor,
    labels: torch.Tensor,
    masked_positions: torch.Tensor,
) -> torch.Tensor:
    """Per-row amino-acid recovery on masked positions (AAR)."""

    correct = (output_tokens == labels) & masked_positions
    counts = masked_positions.sum(dim=1).clamp_min(1)
    return correct.sum(dim=1).to(torch.float32) / counts.to(torch.float32)


def decode_residue_span(
    token_ids: torch.Tensor,
    tokenizer: GrammarTokenizer,
    start: int,
    end: int,
) -> str:
    chars: list[str] = []
    for token_id in token_ids[start:end].tolist():
        token = tokenizer.token(int(token_id))
        if len(token) == 1 and token.isalpha():
            chars.append(token)
    return "".join(chars)


def extract_chain_sequence(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    tokenizer: GrammarTokenizer,
    chain: str,
) -> str:
    from .masks import chain_residue_positions

    positions = chain_residue_positions(
        token_ids.unsqueeze(0),
        attention_mask.unsqueeze(0),
        residue_mask.unsqueeze(0),
        tokenizer,
        chain=chain,  # type: ignore[arg-type]
    )[0]
    if not positions:
        return ""
    indices = torch.tensor(positions, device=token_ids.device, dtype=torch.long)
    residue_ids = token_ids.index_select(0, indices)
    return decode_residue_span(residue_ids, tokenizer, 0, residue_ids.numel())
