"""Partial-mask helpers for grammar-v2 downstream eval."""

from __future__ import annotations

from typing import Literal

import torch

from dllm.pipelines.qwen3_vl_arch.data import BioSeqChain, GrammarTokenizer
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import resolve_partial_mask


ChainRole = Literal["heavy", "light"]


def _target_chain_index(position_ids_chain: torch.Tensor, chain: ChainRole) -> int:
    """Resolve logical chain index for heavy/light in v2 records."""

    unique = sorted(
        {
            int(value)
            for value in position_ids_chain[position_ids_chain.ge(0)].tolist()
        }
    )
    if not unique:
        raise ValueError("No residue chain indices found in batch row")
    if chain == "heavy":
        return unique[1] if len(unique) >= 3 else unique[0]
    return unique[-1]


def chain_residue_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    tokenizer: GrammarTokenizer,
    chain: ChainRole,
    *,
    position_ids_chain: torch.Tensor | None = None,
) -> list[list[int]]:
    """Map each batch row to residue token indices for one chain (heavy or light)."""

    batch_positions: list[list[int]] = []
    for row in range(input_ids.size(0)):
        positions: list[int] = []
        if position_ids_chain is not None:
            target_index = _target_chain_index(position_ids_chain[row], chain)
            for col in range(input_ids.size(1)):
                if not attention_mask[row, col]:
                    continue
                if not residue_mask[row, col]:
                    continue
                if int(position_ids_chain[row, col].item()) == target_index:
                    positions.append(col)
            batch_positions.append(positions)
            continue

        prots_id = tokenizer.special_id("<prots>")
        protd_id = tokenizer.special_id("<protd>")
        dot_id = tokenizer.chain_separator_id()
        type_marker_ids = {tokenizer.special_id(token) for token in ("<ab>", "<tcr>", "<nb>", "<pep>")}
        target_span = 0 if chain == "heavy" else 1
        in_prots_block = False
        current_span = -1
        for col in range(input_ids.size(1)):
            if not attention_mask[row, col]:
                continue
            token_id = int(input_ids[row, col].item())
            if token_id == prots_id:
                in_prots_block = True
                current_span = -1
                continue
            if in_prots_block and token_id == protd_id:
                break
            if in_prots_block and token_id in type_marker_ids:
                continue
            if in_prots_block and token_id == dot_id:
                current_span += 1
                continue
            if in_prots_block and residue_mask[row, col]:
                if current_span < 0:
                    current_span = 0
                if current_span == target_span:
                    positions.append(col)
        batch_positions.append(positions)
    return batch_positions


def light_chain_generation_partial_mask(
    batch: dict[str, torch.Tensor],
    tokenizer: GrammarTokenizer,
    *,
    prompt_residues: int = 0,
) -> torch.Tensor:
    """Keep heavy chain, structure, relation, and fixed context visible."""

    input_ids = batch["input_ids"]
    attention = batch["attention_mask"].bool()
    residue = batch["residue_mask"].bool()
    partial = attention.clone()
    light_positions = chain_residue_positions(
        input_ids,
        attention,
        residue,
        tokenizer,
        chain="light",
        position_ids_chain=batch.get("position_ids_chain"),
    )
    prompt_residues = max(int(prompt_residues), 0)
    for row, positions in enumerate(light_positions):
        for position in positions:
            partial[row, position] = False
        if prompt_residues > 0:
            for position in positions[:prompt_residues]:
                partial[row, position] = True
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
        position_ids_chain=batch.get("position_ids_chain"),
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
