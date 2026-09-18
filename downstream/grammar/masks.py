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
        # The fixed no-context prefix renders as a residue-free
        # <prots> <null> <protd> block. Skip its marker, and on its closing
        # <protd> keep scanning rather than breaking: bailing out there would
        # return zero positions for every unconditional record.
        skip_marker_ids = set(type_marker_ids)
        skip_marker_ids.add(tokenizer.special_id(GRAMMAR_NULL_CONTEXT_TOKEN))
        target_span = 0 if chain == "heavy" else 1
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


def residue_positions_by_chain(
    batch: dict[str, torch.Tensor],
) -> list[dict[int, list[int]]]:
    """Per row, map each rendered chain index -> its residue token columns.

    Chain indices follow ``position_ids_chain`` from the grammar renderer, which
    numbers residue chains 0,1,2,... in render order. TCR records render as:
      * ``[tcr_beta]``                 -> beta = chain 0
      * ``[antigen, tcr_beta]``        -> antigen = 0, beta = 1
      * ``[antigen, tcr_alpha, beta]`` -> antigen = 0, alpha = 1, beta = 2
      * ``[tcr_alpha, tcr_beta]``      -> alpha = 0, beta = 1
    so the caller (which builds the records) knows the target index deterministically.
    """

    attention = batch["attention_mask"].bool()
    residue = batch["residue_mask"].bool()
    chain = batch["position_ids_chain"]
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
    """Mask (generate) the residues of the given TCR chain index/indices.

    Everything else -- structure tokens, relation tokens, fixed context (e.g. a
    conditioning epitope / MHC), and all non-target chains -- stays visible.
    ``prompt_residues`` keeps that many leading residues of each target chain
    visible (e.g. a fixed ``C`` anchor for CDR3b).
    """

    if isinstance(target_chain_indices, int):
        targets = {target_chain_indices}
    else:
        targets = set(int(i) for i in target_chain_indices)

    attention = batch["attention_mask"].bool()
    partial = attention.clone()
    prompt_residues = max(int(prompt_residues), 0)
    for row, by_chain in enumerate(residue_positions_by_chain(batch)):
        for idx in targets:
            positions = by_chain.get(idx, [])
            for position in positions:
                partial[row, position] = False
            if prompt_residues > 0:
                for position in positions[:prompt_residues]:
                    partial[row, position] = True
    return resolve_partial_mask(batch, partial)


def cdr3b_span_partial_mask(
    batch: dict[str, torch.Tensor],
    chain_index: int,
    span: tuple[int, int] | list[tuple[int, int]],
) -> torch.Tensor:
    """Mask a residue span ``[start, end)`` within one TCR chain (infill style).

    ``span`` is measured in within-chain residue coordinates (0-based). Use for
    CDR3b infilling inside a full-length beta chain while keeping the framework
    and (optionally) the epitope/MHC context fixed.

    Pass a single tuple to apply the same span to every row, or a list of one
    tuple per row when the rows differ in length (e.g. a centred window whose
    offset depends on each sequence's own length).
    """

    attention = batch["attention_mask"].bool()
    partial = attention.clone()
    rows = residue_positions_by_chain(batch)
    if isinstance(span, tuple):
        spans = [span] * len(rows)
    else:
        spans = list(span)
        if len(spans) != len(rows):
            raise ValueError(
                f"got {len(spans)} spans for {len(rows)} batch rows"
            )
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
    """Mask FR residues on heavy+light; keep CDRs (and optional light C-term) visible.

    Matches the Ophiuchus-Ab / AirGen humanization protocol: regenerate framework
    while preserving CDR identity, and keep the last ``light_keep_c_terminal``
    light-chain Fv residues native.
    """

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
