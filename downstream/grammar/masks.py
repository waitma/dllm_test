"""Partial-mask helpers for grammar-v2 downstream eval."""

from __future__ import annotations

from typing import Literal

import torch

from dllm.pipelines.immune_llada.data import (
    BioSeqChain,
    BioSeqRecord,
    GRAMMAR_NULL_CONTEXT_TOKEN,
    GrammarTokenizer,
)
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import resolve_partial_mask


ChainRole = Literal["heavy", "light"]


def _target_chain_index(
    position_ids_chain: torch.Tensor,
    chain: ChainRole | int,
) -> int:
    """Resolve a logical chain index, retaining the legacy heavy/light rules."""

    if isinstance(chain, int):
        return int(chain)
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


def _legacy_chain_residue_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    tokenizer: GrammarTokenizer,
    chain: ChainRole | int,
) -> list[list[int]]:
    """Parse legacy streams that predate position IDs on every slot."""

    batch_positions: list[list[int]] = []
    for row in range(input_ids.size(0)):
        positions: list[int] = []
        prots_id = tokenizer.special_id("<prots>")
        protd_id = tokenizer.special_id("<protd>")
        dot_id = tokenizer.chain_separator_id()
        type_marker_ids = {
            tokenizer.special_id(token) for token in ("<ab>", "<tcr>", "<nb>", "<pep>")
        }
        skip_marker_ids = set(type_marker_ids)
        skip_marker_ids.add(tokenizer.special_id(GRAMMAR_NULL_CONTEXT_TOKEN))
        target_span = int(chain) if isinstance(chain, int) else (0 if chain == "heavy" else 1)
        in_prots_block = False
        block_has_residues = False
        current_span = -1
        for col in range(input_ids.size(1)):
            if not attention_mask[row, col]:
                continue
            token_id = int(input_ids[row, col].item())
            if token_id == prots_id:
                in_prots_block = True
                block_has_residues = False
                current_span = -1
                continue
            if in_prots_block and token_id == protd_id:
                if block_has_residues:
                    break
                in_prots_block = False
                continue
            if in_prots_block and token_id in skip_marker_ids:
                continue
            if in_prots_block and token_id == dot_id:
                current_span += 1
                continue
            if in_prots_block and residue_mask[row, col]:
                block_has_residues = True
                if current_span < 0:
                    current_span = 0
                if current_span == target_span:
                    positions.append(col)
        batch_positions.append(positions)
    return batch_positions


def chain_residue_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    tokenizer: GrammarTokenizer,
    chain: ChainRole | int,
    *,
    position_ids_chain: torch.Tensor | None = None,
) -> list[list[int]]:
    """Map each batch row to amino-acid token indices for one chain.

    EOS slots are deliberately excluded. Use :func:`chain_slot_positions` for
    full-chain inference masks, where EOS is a generated slot rather than an
    amino-acid encoder residue.
    """

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
        else:
            batch_positions.append(
                _legacy_chain_residue_positions(
                    input_ids[row : row + 1],
                    attention_mask[row : row + 1],
                    residue_mask[row : row + 1],
                    tokenizer,
                    chain,
                )[0]
            )
    return batch_positions


def chain_slot_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chain_slot_mask: torch.Tensor | None,
    tokenizer: GrammarTokenizer,
    chain: ChainRole | int,
    *,
    position_ids_chain: torch.Tensor | None = None,
    residue_mask: torch.Tensor | None = None,
) -> list[list[int]]:
    """Map rows to all decoder slots for one chain, including decoder EOS.

    ``chain_slot_mask`` is the inference canvas: unlike ``chain_eos_mask`` it is
    independent of the clean target and remains valid after generation changes
    token values. The fallback is intentionally residue-only for legacy batches.
    """

    attention_mask = attention_mask.bool()
    if chain_slot_mask is None:
        if residue_mask is None:
            raise ValueError("residue_mask is required when chain_slot_mask is absent")
        return chain_residue_positions(
            input_ids,
            attention_mask,
            residue_mask.bool(),
            tokenizer,
            chain,
            position_ids_chain=position_ids_chain,
        )

    slots = chain_slot_mask.bool()
    if position_ids_chain is not None:
        result: list[list[int]] = []
        for row in range(input_ids.size(0)):
            target_index = _target_chain_index(position_ids_chain[row], chain)
            result.append(
                [
                    col
                    for col in range(input_ids.size(1))
                    if attention_mask[row, col]
                    and slots[row, col]
                    and int(position_ids_chain[row, col].item()) == target_index
                ]
            )
        return result

    # Without position IDs, parse the fixed grammar delimiters while selecting
    # all slots, not clean residues or token-value-dependent EOS positions.
    return _legacy_chain_residue_positions(
        input_ids, attention_mask, slots, tokenizer, chain
    )


def chain_slot_positions_by_chain(
    batch: dict[str, torch.Tensor],
) -> list[dict[int, list[int]]]:
    """Return all attention-visible decoder slots grouped by chain index."""

    attention = batch["attention_mask"].bool()
    slots = batch.get("chain_slot_mask")
    if slots is None:
        slots = batch.get("residue_mask")
    if slots is None:
        raise KeyError("batch requires chain_slot_mask or residue_mask")
    chain = batch.get("position_ids_chain", batch.get("chain_ids"))
    if chain is None:
        raise KeyError("batch requires position_ids_chain or chain_ids")
    rows: list[dict[int, list[int]]] = []
    for row in range(chain.size(0)):
        by_chain: dict[int, list[int]] = {}
        for col in range(chain.size(1)):
            if not attention[row, col] or not slots[row, col]:
                continue
            idx = int(chain[row, col].item())
            if idx >= 0:
                by_chain.setdefault(idx, []).append(col)
        rows.append(by_chain)
    return rows


def light_chain_generation_partial_mask(
    batch: dict[str, torch.Tensor],
    tokenizer: GrammarTokenizer,
    *,
    prompt_residues: int = 0,
) -> torch.Tensor:
    """Generate every light-chain decoder slot, retaining only the prompt."""

    input_ids = batch["input_ids"]
    attention = batch["attention_mask"].bool()
    residue = batch["residue_mask"].bool()
    slots = batch.get("chain_slot_mask")
    partial = attention.clone()
    light_slots = chain_slot_positions(
        input_ids,
        attention,
        slots,
        tokenizer,
        chain="light",
        position_ids_chain=batch.get("position_ids_chain"),
        residue_mask=residue,
    )
    light_residues = chain_residue_positions(
        input_ids,
        attention,
        residue,
        tokenizer,
        chain="light",
        position_ids_chain=batch.get("position_ids_chain"),
    )
    prompt_residues = max(int(prompt_residues), 0)
    for row, positions in enumerate(light_slots):
        partial[row, positions] = False
        prompt = set(light_residues[row][:prompt_residues])
        for position in positions:
            if position in prompt:
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


def residue_positions_by_chain(
    batch: dict[str, torch.Tensor],
) -> list[dict[int, list[int]]]:
    """Per row, map each rendered chain index to amino-acid token columns."""

    attention = batch["attention_mask"].bool()
    residue = batch["residue_mask"].bool()
    chain = batch.get("position_ids_chain", batch.get("chain_ids"))
    if chain is None:
        raise KeyError("batch requires position_ids_chain or chain_ids")
    rows: list[dict[int, list[int]]] = []
    for row in range(chain.size(0)):
        by_chain: dict[int, list[int]] = {}
        for col in range(chain.size(1)):
            if not attention[row, col] or not residue[row, col]:
                continue
            idx = int(chain[row, col].item())
            if idx < 0:
                continue
            by_chain.setdefault(idx, []).append(col)
        rows.append(by_chain)
    return rows


def tcr_generation_partial_mask(
    batch: dict[str, torch.Tensor],
    target_chain_indices: int | set[int] | list[int],
    *,
    prompt_residues: int = 0,
) -> torch.Tensor:
    """Generate all slots of the target TCR chains, including decoder EOS."""

    if isinstance(target_chain_indices, int):
        targets = {target_chain_indices}
    else:
        targets = set(int(i) for i in target_chain_indices)

    attention = batch["attention_mask"].bool()
    partial = attention.clone()
    prompt_residues = max(int(prompt_residues), 0)
    by_chain = chain_slot_positions_by_chain(batch)
    residues_by_chain = residue_positions_by_chain(batch)
    for row, slots in enumerate(by_chain):
        for idx in targets:
            positions = slots.get(idx, [])
            for position in positions:
                partial[row, position] = False
            prompt = set(residues_by_chain[row].get(idx, [])[:prompt_residues])
            for position in positions:
                if position in prompt:
                    partial[row, position] = True
    return resolve_partial_mask(batch, partial)


def cdr3b_span_partial_mask(
    batch: dict[str, torch.Tensor],
    chain_index: int,
    span: tuple[int, int] | list[tuple[int, int]],
) -> torch.Tensor:
    """Mask a residue span ``[start, end)`` within one TCR chain (infill style)."""

    attention = batch["attention_mask"].bool()
    partial = attention.clone()
    rows = residue_positions_by_chain(batch)
    if isinstance(span, tuple):
        spans = [span] * len(rows)
    else:
        spans = list(span)
        if len(spans) != len(rows):
            raise ValueError(f"got {len(spans)} spans for {len(rows)} batch rows")
    for row, by_chain in enumerate(rows):
        positions = by_chain.get(int(chain_index), [])
        start, end = spans[row]
        if end > len(positions):
            raise ValueError(
                f"span [{start},{end}) exceeds chain length {len(positions)} (row {row})"
            )
        for position in positions[start:end]:
            partial[row, position] = False
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


def framework_generation_partial_mask(
    batch: dict[str, torch.Tensor],
    tokenizer: GrammarTokenizer,
    records: list[BioSeqRecord],
    *,
    light_keep_c_terminal: int = 3,
) -> torch.Tensor:
    """Mask FR residues on heavy+light; keep CDRs (and optional light C-term) visible."""

    attention = batch["attention_mask"].bool()
    residue = batch["residue_mask"].bool()
    partial = attention.clone()
    keep_c = max(int(light_keep_c_terminal), 0)

    for row, record in enumerate(records):
        for chain_role, chain in (("heavy", record.chains[0]), ("light", record.chains[1])):
            positions = chain_residue_positions(
                batch["input_ids"][row : row + 1],
                attention[row : row + 1],
                residue[row : row + 1],
                tokenizer,
                chain=chain_role,  # type: ignore[arg-type]
                position_ids_chain=(
                    batch["position_ids_chain"][row : row + 1]
                    if "position_ids_chain" in batch
                    else None
                ),
            )[0]
            for region_name in ("FR1", "FR2", "FR3", "FR4"):
                span = chain.region_span(region_name)
                if span is None:
                    continue
                start, end = span
                if end > len(positions):
                    raise ValueError(
                        f"{region_name} span [{start},{end}) exceeds {chain_role} length "
                        f"{len(positions)} (row {row})"
                    )
                for position in positions[start:end]:
                    partial[row, position] = False
            if chain_role == "light" and keep_c > 0 and positions:
                for position in positions[-keep_c:]:
                    partial[row, position] = True
    return resolve_partial_mask(batch, partial)
