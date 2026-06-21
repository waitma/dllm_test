"""Partial-mask helpers for grammar-v2 downstream eval."""

from __future__ import annotations

from typing import Literal

import torch

from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, GrammarTokenizer
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import resolve_partial_mask


ChainRole = Literal["heavy", "light"]


def chain_residue_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    tokenizer: GrammarTokenizer,
    chain: ChainRole,
) -> list[list[int]]:
    """Map each batch row to token indices inside one ``<prots>...<prote>`` span.

    Grammar-v2 uses repeated ``<prots>/<prote>`` blocks; span 0 is heavy/object A
    and span 1 is light/object B in antibody pairs.
    """

    start_id = tokenizer.special_id("<prots>")
    end_id = tokenizer.special_id("<prote>")
    target_span = 0 if chain == "heavy" else 1
    batch_positions: list[list[int]] = []
    for row in range(input_ids.size(0)):
        positions: list[int] = []
        in_chain = False
        span_index = -1
        for col in range(input_ids.size(1)):
            if not attention_mask[row, col]:
                continue
            token_id = int(input_ids[row, col].item())
            if token_id == start_id:
                span_index += 1
                in_chain = span_index == target_span
                continue
            if in_chain and token_id == end_id:
                break
            if in_chain and residue_mask[row, col]:
                positions.append(col)
        batch_positions.append(positions)
    return batch_positions


def light_chain_generation_partial_mask(
    batch: dict[str, torch.Tensor],
    tokenizer: GrammarTokenizer,
) -> torch.Tensor:
    """Keep heavy chain, structure, relation, and fixed context visible."""

    input_ids = batch["input_ids"]
    attention = batch["attention_mask"].bool()
    residue = batch["residue_mask"].bool()
    partial = attention.clone()
    light_positions = chain_residue_positions(input_ids, attention, residue, tokenizer, chain="light")
    for row, positions in enumerate(light_positions):
        for position in positions:
            partial[row, position] = False
    return resolve_partial_mask(batch, partial)


def cdr_generation_partial_mask(
    batch: dict[str, torch.Tensor],
    tokenizer: GrammarTokenizer,
    chain: BioSeqChain,
    chain_role: ChainRole,
    cdr_name: str,
    *,
    residue_span: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Mask only the requested CDR span on heavy or light chain."""

    attention = batch["attention_mask"].bool()
    residue = batch["residue_mask"].bool()
    partial = attention.clone()
    span = residue_span or chain.region_span(cdr_name.upper())
    if span is None:
        raise ValueError(f"Could not resolve {cdr_name} span for chain role {chain.role}")

    start_char, end_char = span
    row_positions = chain_residue_positions(
        batch["input_ids"],
        attention,
        residue,
        tokenizer,
        chain=chain_role,
    )[0]
    if end_char > len(row_positions):
        raise ValueError(
            f"{cdr_name} span [{start_char}, {end_char}) exceeds rendered chain length {len(row_positions)}"
        )
    for position in row_positions[start_char:end_char]:
        partial[0, position] = False
    return resolve_partial_mask(batch, partial)


def cdr_generation_partial_mask_from_subsequence(
    batch: dict[str, torch.Tensor],
    tokenizer: GrammarTokenizer,
    chain: BioSeqChain,
    chain_role: ChainRole,
    subsequence: str,
) -> torch.Tensor:
    normalized = "".join(str(subsequence or "").split()).upper().replace("J", "L")
    start = chain.sequence.find(normalized)
    if start < 0:
        raise ValueError(f"Could not locate subsequence in {chain.role}: {normalized!r}")
    return cdr_generation_partial_mask(
        batch,
        tokenizer,
        chain,
        chain_role,
        cdr_name="CDR",
        residue_span=(start, start + len(normalized)),
    )
