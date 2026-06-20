from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .esm_encoding import Esm2SequenceTokenizer, EsmTokenizerProtocol
from .mixture import bioseq_task_group
from .records import BioSeqRecord, CHAIN_ROLE_TO_ID, TASK_TYPE_TO_ID
from .view_sampler import BioSeqViewSampler, GenerationView, ResidueSpan

SPECIAL_CHAIN_ID = -1
SPECIAL_POSITION_ID = -1


def _span_contains(spans: list[ResidueSpan], chain_index: int, residue_index: int) -> bool:
    return any(span.chain_index == chain_index and span.start <= residue_index < span.end for span in spans)


@dataclass
class BioSeqQwenDataCollator:
    """Collate BioSeq records for foundation diffusion with ESM-family ids.

    The decoder stream and the per-chain encoder tensors use the same tokenizer
    by default. This keeps the batch compatible with ESM2/MINT today and allows
    swapping in a local ESMC/Hugging Face tokenizer adapter later.
    """

    tokenizer: EsmTokenizerProtocol = field(default_factory=Esm2SequenceTokenizer)
    view_sampler: BioSeqViewSampler = field(default_factory=BioSeqViewSampler)
    max_chain_length: int | None = None
    max_sequence_length: int | None = None
    chain_role_to_id: dict[str, int] = field(default_factory=lambda: dict(CHAIN_ROLE_TO_ID))
    task_type_to_id: dict[str, int] = field(default_factory=lambda: dict(TASK_TYPE_TO_ID))
    single_view_per_batch: bool = False
    require_homogeneous_task: bool = False

    def __call__(self, records: list[BioSeqRecord | dict[str, Any]]) -> dict[str, Any]:
        normalized = [self._coerce_record(record) for record in records]
        if self.require_homogeneous_task:
            task_groups = {bioseq_task_group(record) for record in normalized}
            if len(task_groups) > 1:
                raise ValueError(f"BioSeq batch contains mixed task groups: {sorted(task_groups)}")
        views = self.view_sampler.sample_batch(normalized) if self.single_view_per_batch else [
            self.view_sampler.sample(record) for record in normalized
        ]
        encoded = []
        resolved_views = []
        for record, view in zip(normalized, views):
            row = self._encode_decoder(record, view)
            if not any(row["diffusion_loss_mask"]):
                fallback_view = self.view_sampler.full_denoise(record)
                fallback_row = self._encode_decoder(record, fallback_view)
                if any(fallback_row["diffusion_loss_mask"]):
                    view = fallback_view
                    row = fallback_row
            resolved_views.append(view)
            encoded.append(row)
        views = resolved_views
        encoder = [self._encode_encoder(record) for record in normalized]

        decoder_batch = self._pad_decoder(encoded)
        encoder_batch = self._pad_encoder(encoder)
        decoder_batch.update(encoder_batch)
        decoder_batch["view_names"] = [view.name for view in views]
        decoder_batch["task_groups"] = [bioseq_task_group(record) for record in normalized]
        decoder_batch["task_types"] = [record.task_type for record in normalized]
        decoder_batch["sources"] = [record.source for record in normalized]
        decoder_batch["weights"] = torch.tensor([record.weight for record in normalized], dtype=torch.float32).unsqueeze(-1)
        return decoder_batch

    def _coerce_record(self, record: BioSeqRecord | dict[str, Any]) -> BioSeqRecord:
        if isinstance(record, BioSeqRecord):
            return record
        from .sources import processed_json_to_record

        converted = processed_json_to_record(record)
        if converted is None:
            raise ValueError(f"Cannot coerce record into BioSeqRecord: {record}")
        return converted

    def _encode_decoder(self, record: BioSeqRecord, view: GenerationView) -> dict[str, list[int]]:
        input_ids: list[int] = []
        chain_ids: list[int] = []
        chain_role_ids: list[int] = []
        position_ids_inner: list[int] = []
        position_ids_chain: list[int] = []
        residue_mask: list[int] = []
        target_mask: list[int] = []
        fixed_context_mask: list[int] = []
        special_token_mask: list[int] = []

        for chain_index, chain in enumerate(record.chains):
            token_ids, residue_flags = self.tokenizer.encode_chain(chain.sequence, max_length=self.max_chain_length)
            role_id = self.chain_role_to_id.get(chain.role, self.chain_role_to_id["unknown"])
            residue_index = 0
            for token_id, is_residue in zip(token_ids, residue_flags):
                is_target = bool(is_residue and _span_contains(view.target_spans, chain_index, residue_index))
                input_ids.append(token_id)
                chain_ids.append(chain_index)
                chain_role_ids.append(role_id)
                position_ids_chain.append(chain_index)
                if is_residue:
                    position_ids_inner.append(residue_index)
                    residue_mask.append(1)
                    target_mask.append(int(is_target))
                    fixed_context_mask.append(int(not is_target))
                    special_token_mask.append(0)
                    residue_index += 1
                else:
                    position_ids_inner.append(SPECIAL_POSITION_ID)
                    residue_mask.append(0)
                    target_mask.append(0)
                    fixed_context_mask.append(0)
                    special_token_mask.append(1)

        if self.max_sequence_length is not None:
            input_ids = input_ids[: self.max_sequence_length]
            chain_ids = chain_ids[: self.max_sequence_length]
            chain_role_ids = chain_role_ids[: self.max_sequence_length]
            position_ids_inner = position_ids_inner[: self.max_sequence_length]
            position_ids_chain = position_ids_chain[: self.max_sequence_length]
            residue_mask = residue_mask[: self.max_sequence_length]
            target_mask = target_mask[: self.max_sequence_length]
            fixed_context_mask = fixed_context_mask[: self.max_sequence_length]
            special_token_mask = special_token_mask[: self.max_sequence_length]

        attention_mask = [1] * len(input_ids)
        visible_mask = [int(bool(fixed) or bool(special)) for fixed, special in zip(fixed_context_mask, special_token_mask)]
        return {
            "input_ids": input_ids,
            "labels": list(input_ids),
            "attention_mask": attention_mask,
            "chain_ids": chain_ids,
            "chain_role_ids": chain_role_ids,
            "position_ids_inner": position_ids_inner,
            "position_ids_chain": position_ids_chain,
            "residue_mask": residue_mask,
            "visible_mask": visible_mask,
            "fixed_context_mask": fixed_context_mask,
            "diffusion_target_mask": target_mask,
            "diffusion_loss_mask": list(target_mask),
            "task_type_id": [self.task_type_to_id.get(record.task_type, self.task_type_to_id["generic"])],
        }

    def _encode_encoder(self, record: BioSeqRecord) -> list[dict[str, list[int]]]:
        chains = []
        for chain in record.chains:
            token_ids, residue_flags = self.tokenizer.encode_chain(chain.sequence, max_length=self.max_chain_length)
            chains.append(
                {
                    "input_ids": token_ids,
                    "attention_mask": [1] * len(token_ids),
                    "residue_mask": residue_flags,
                    "chain_role_id": self.chain_role_to_id.get(chain.role, self.chain_role_to_id["unknown"]),
                }
            )
        return chains

    def _pad_decoder(self, rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in rows)
        pad_id = self.tokenizer.pad_token_id
        padded: dict[str, list[list[int]]] = {key: [] for key in rows[0] if key != "task_type_id"}
        task_type_ids = []

        for row in rows:
            pad_len = max_len - len(row["input_ids"])
            for key in padded:
                if key in {"input_ids", "labels"}:
                    pad_value = pad_id if key == "input_ids" else -100
                elif key == "chain_ids":
                    pad_value = SPECIAL_CHAIN_ID
                elif key in {"position_ids_inner", "position_ids_chain"}:
                    pad_value = SPECIAL_POSITION_ID
                else:
                    pad_value = 0
                padded[key].append(row[key] + [pad_value] * pad_len)
            task_type_ids.append(row["task_type_id"][0])

        batch = {key: torch.tensor(value, dtype=torch.long) for key, value in padded.items()}
        bool_keys = {
            "attention_mask",
            "residue_mask",
            "visible_mask",
            "fixed_context_mask",
            "diffusion_target_mask",
            "diffusion_loss_mask",
        }
        for key in bool_keys:
            batch[key] = batch[key].bool()
        batch["task_type_ids"] = torch.tensor(task_type_ids, dtype=torch.long)
        return batch

    def _pad_encoder(self, rows: list[list[dict[str, list[int]]]]) -> dict[str, torch.Tensor]:
        max_chains = max(len(row) for row in rows)
        max_len = max(len(chain["input_ids"]) for row in rows for chain in row)
        ids = []
        attention = []
        residue = []
        chain_mask = []
        role_ids = []

        for row in rows:
            row_ids = []
            row_attention = []
            row_residue = []
            row_chain_mask = []
            row_role_ids = []
            for chain in row:
                pad_len = max_len - len(chain["input_ids"])
                row_ids.append(chain["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
                row_attention.append(chain["attention_mask"] + [0] * pad_len)
                row_residue.append(chain["residue_mask"] + [0] * pad_len)
                row_chain_mask.append(1)
                row_role_ids.append(chain["chain_role_id"])
            while len(row_ids) < max_chains:
                row_ids.append([self.tokenizer.pad_token_id] * max_len)
                row_attention.append([0] * max_len)
                row_residue.append([0] * max_len)
                row_chain_mask.append(0)
                row_role_ids.append(self.chain_role_to_id["unknown"])
            ids.append(row_ids)
            attention.append(row_attention)
            residue.append(row_residue)
            chain_mask.append(row_chain_mask)
            role_ids.append(row_role_ids)

        return {
            "encoder_input_ids": torch.tensor(ids, dtype=torch.long),
            "encoder_attention_mask": torch.tensor(attention, dtype=torch.bool),
            "encoder_residue_mask": torch.tensor(residue, dtype=torch.bool),
            "encoder_chain_mask": torch.tensor(chain_mask, dtype=torch.bool),
            "encoder_chain_role_ids": torch.tensor(role_ids, dtype=torch.long),
        }
