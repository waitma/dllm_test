"""Shared metrics for grammar downstream eval.

Run focused checks with::

    python -m pytest scripts/tests/immune_llada/test_v2_inference_adapters.py -q
"""

from __future__ import annotations

from typing import Literal

import torch

from dllm.pipelines.immune_llada.data import GrammarTokenizer


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
    residue_ids = [int(token_id) for token_id in token_ids[start:end].tolist()]
    base = tokenizer.base_tokenizer
    decode = getattr(base, "decode", None)
    if callable(decode):
        decoded = decode(residue_ids, skip_special_tokens=True)
        if isinstance(decoded, str):
            return decoded

    inner = getattr(base, "tokenizer", None)
    raw = getattr(inner, "tokenizer", inner) if inner is not None else None
    if raw is not None and hasattr(raw, "decode"):
        decoded = raw.decode(residue_ids)
        if isinstance(decoded, str):
            return decoded.replace(" ", "")

    chars: list[str] = []
    for token_id in residue_ids:
        token = tokenizer.token(token_id)
        if len(token) == 1 and token.isalpha():
            chars.append(token)
    return "".join(chars)


def extract_chain_sequence(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    tokenizer: GrammarTokenizer,
    chain: Literal["heavy", "light"] | int,
    chain_eos_mask: torch.Tensor | None = None,
    *,
    position_ids_chain: torch.Tensor | None = None,
    chain_slot_mask: torch.Tensor | None = None,
) -> str:
    """Extract one generated chain and stop at its first generated EOS.

    ``chain_eos_mask`` is retained as a positional-API compatibility argument,
    but it describes clean renderer targets and is never used as an inference
    boundary. In v2, a chain's decoder canvas is selected by
    ``position_ids_chain`` + ``chain_slot_mask`` and the boundary is the first
    output token equal to grammar ``<eos>`` in that canvas. The decoder slots
    before that boundary are all decoded, including slots that were not clean
    amino-acid residues in ``residue_mask``. If no generated EOS is present,
    extraction is bounded by the entire selected canvas and does not force an
    artificial termination. This remains independent for every chain in a
    multi-chain row.
    """

    # Keep the import local: masks imports the sampling helper, and metrics is
    # also used by lightweight CPU-only downstream tools.
    from .masks import chain_slot_positions

    del chain_eos_mask
    position_ids_chain = (
        position_ids_chain.unsqueeze(0)
        if position_ids_chain is not None and position_ids_chain.ndim == 1
        else position_ids_chain
    )
    chain_slot_mask = (
        chain_slot_mask.unsqueeze(0)
        if chain_slot_mask is not None and chain_slot_mask.ndim == 1
        else chain_slot_mask
    )
    token_batch = token_ids.unsqueeze(0)
    attention_batch = attention_mask.unsqueeze(0)
    residue_batch = residue_mask.unsqueeze(0)

    # Clean residue targets cannot bound generated output; use them only for
    # legacy callers without a slot canvas.
    selected_positions = chain_slot_positions(
        token_batch,
        attention_batch,
        chain_slot_mask,
        tokenizer,
        chain=chain,
        position_ids_chain=position_ids_chain,
        residue_mask=residue_batch if chain_slot_mask is None else None,
    )[0]
    eos_id = int(tokenizer.eos_token_id)
    for index, position in enumerate(selected_positions):
        if int(token_ids[position].item()) == eos_id:
            selected_positions = selected_positions[:index]
            break
    if not selected_positions:
        return ""

    indices = torch.tensor(selected_positions, device=token_ids.device, dtype=torch.long)
    selected_ids = token_ids.index_select(0, indices)
    return decode_residue_span(selected_ids, tokenizer, 0, selected_ids.numel())
