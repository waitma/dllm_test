from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..data import Esm2ProteinTokenizer, OPHIUCHUS_AB_CHAIN_LENGTHS, TASK_TYPE_TO_ID


@dataclass
class MultiChainDynamicCollator:
    """Variable-length two-slot collator for the exact Ophiuchus-Ab model.

    Unlike :class:`OphiuchusAbTrainingCollator`, this collator does not pad to
    the fixed ``(150, 128)`` Ophiuchus lengths. Each batch is padded to the
    longest chain-1 / chain-2 sequence in that batch using the real ``<pad>``
    token, so the ESM2 backbone's padding mask removes the padded positions from
    attention, token dropout and the diffusion loss.

    It also accepts single-chain examples (e.g. nanobody VHH). A single-chain
    example fills the chain-1 slot and uses a minimal ``[<cls>, <eos>]``
    placeholder for the chain-2 slot, so the existing two-slot
    ``MultiChainOphiuchusAbModel.compute_loss`` keeps working unchanged.
    """

    tokenizer: Esm2ProteinTokenizer = field(default_factory=Esm2ProteinTokenizer)
    max_length: int | None = 512
    min_chain2_length: int = 2

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        chain1_encoded: list[list[int]] = []
        chain2_encoded: list[list[int]] = []
        weights: list[float] = []

        for example in examples:
            chains = OphiuchusAbTrainingCollator._extract_chains(example)
            if not chains:
                raise ValueError("MultiChainDynamicCollator requires at least one chain")

            chain1 = chains[0]
            chain2 = chains[1] if len(chains) >= 2 and chains[1] else ""

            chain1_ids, _ = self.tokenizer.encode_chain(chain1, max_length=self.max_length)
            if chain2:
                chain2_ids, _ = self.tokenizer.encode_chain(chain2, max_length=self.max_length)
            else:
                chain2_ids = [self.tokenizer.cls_token_id, self.tokenizer.eos_token_id]

            chain1_encoded.append(chain1_ids)
            chain2_encoded.append(chain2_ids)
            weights.append(float(example.get("weight", 1.0)))

        chain1_max = max(len(ids) for ids in chain1_encoded)
        chain2_max = max(max(len(ids) for ids in chain2_encoded), self.min_chain2_length)

        heavy_targets = torch.stack([self._pad(ids, chain1_max) for ids in chain1_encoded], dim=0)
        light_targets = torch.stack([self._pad(ids, chain2_max) for ids in chain2_encoded], dim=0)

        return {
            "heavy_tokens": {
                "targets": heavy_targets,
                "regions": torch.full_like(heavy_targets, -1),
                "chain_ids": torch.zeros_like(heavy_targets),
            },
            "light_tokens": {
                "targets": light_targets,
                "regions": torch.full_like(light_targets, -1),
                "chain_ids": torch.ones_like(light_targets),
            },
            "weights": torch.tensor(weights, dtype=torch.float32).unsqueeze(-1),
        }

    def _pad(self, encoded: list[int], target_length: int) -> torch.Tensor:
        pad_len = target_length - len(encoded)
        padded = encoded + [self.tokenizer.pad_token_id] * pad_len
        return torch.tensor(padded[:target_length], dtype=torch.long)


@dataclass
class OphiuchusAbTrainingCollator:
    tokenizer: Esm2ProteinTokenizer = field(default_factory=Esm2ProteinTokenizer)
    chain_lengths: tuple[int, int] = OPHIUCHUS_AB_CHAIN_LENGTHS

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        heavy_targets = []
        light_targets = []
        heavy_regions = []
        light_regions = []
        heavy_chain_ids = []
        light_chain_ids = []
        weights = []

        for example in examples:
            chains = self._extract_chains(example)
            if len(chains) < 2:
                raise ValueError("OphiuchusAbTrainingCollator requires heavy and light chains")

            heavy_encoded, _ = self.tokenizer.encode_chain(chains[0], max_length=self.chain_lengths[0])
            light_encoded, _ = self.tokenizer.encode_chain(chains[1], max_length=self.chain_lengths[1])

            heavy_tensor = self._pad_chain(heavy_encoded, self.chain_lengths[0])
            light_tensor = self._pad_chain(light_encoded, self.chain_lengths[1])

            heavy_targets.append(heavy_tensor)
            light_targets.append(light_tensor)
            heavy_regions.append(torch.full_like(heavy_tensor, -1))
            light_regions.append(torch.full_like(light_tensor, -1))
            heavy_chain_ids.append(torch.zeros_like(heavy_tensor))
            light_chain_ids.append(torch.ones_like(light_tensor))
            weights.append(1.0)

        return {
            "heavy_tokens": {
                "targets": torch.stack(heavy_targets, dim=0),
                "regions": torch.stack(heavy_regions, dim=0),
                "chain_ids": torch.stack(heavy_chain_ids, dim=0),
            },
            "light_tokens": {
                "targets": torch.stack(light_targets, dim=0),
                "regions": torch.stack(light_regions, dim=0),
                "chain_ids": torch.stack(light_chain_ids, dim=0),
            },
            "weights": torch.tensor(weights, dtype=torch.float32).unsqueeze(-1),
        }

    def _pad_chain(self, encoded: list[int], chain_length: int) -> torch.Tensor:
        padded = encoded + [self.tokenizer.eos_token_id] * (chain_length - len(encoded))
        return torch.tensor(padded[:chain_length], dtype=torch.long)

    @staticmethod
    def _extract_chains(example: dict[str, Any]) -> list[str]:
        if "chains" in example:
            chains = example["chains"]
            if isinstance(chains, str):
                return [chains]
            return [str(chain) for chain in chains if str(chain)]

        ordered_keys = (
            "vh_protein_sequence",
            "vl_protein_sequence",
            "heavy",
            "light",
        )
        return [str(example[key]) for key in ordered_keys if key in example and str(example[key])]


@dataclass
class OphiuchusAbInferenceCollator:
    tokenizer: Esm2ProteinTokenizer = field(default_factory=Esm2ProteinTokenizer)
    chain_lengths: tuple[int, int] = OPHIUCHUS_AB_CHAIN_LENGTHS
    task_type_to_id: dict[str, int] = field(default_factory=lambda: dict(TASK_TYPE_TO_ID))

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = []
        chain_ids = []
        attention_mask = []
        partial_masks = []
        task_type_ids = []

        for example in examples:
            chains = OphiuchusAbTrainingCollator._extract_chains(example)
            if len(chains) < 2:
                raise ValueError("OphiuchusAbInferenceCollator requires heavy and light chains")

            row_ids = []
            row_chain_ids = []
            row_attention = []
            row_partial = []

            for chain_index, (sequence, chain_length) in enumerate(zip(chains[:2], self.chain_lengths)):
                encoded, _ = self.tokenizer.encode_chain(sequence, max_length=chain_length)
                pad_len = chain_length - len(encoded)
                row_ids.extend(encoded + [self.tokenizer.eos_token_id] * pad_len)
                row_chain_ids.extend([chain_index] * chain_length)
                row_attention.extend([1] * len(encoded) + [0] * pad_len)
                fixed = bool(example.get("fix_chain_indices")) and chain_index in set(example.get("fix_chain_indices", []))
                row_partial.extend([fixed] * chain_length)

            input_ids.append(row_ids)
            chain_ids.append(row_chain_ids)
            attention_mask.append(row_attention)
            partial_masks.append(row_partial)
            task_type = str(example.get("task_type", "antibody"))
            task_type_ids.append(self.task_type_to_id.get(task_type, self.task_type_to_id["generic"]))

        tokens = torch.tensor(input_ids, dtype=torch.long)
        partial_mask = torch.tensor(partial_masks, dtype=torch.bool)
        return {
            "input_ids": tokens,
            "chain_ids": torch.tensor(chain_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "partial_masks": partial_mask,
            "task_type_ids": torch.tensor(task_type_ids, dtype=torch.long),
        }
