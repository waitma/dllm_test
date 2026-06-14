from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch


SPECIAL_TOKENS = ("<pad>", "<cls>", "<eos>", "<mask>", "<unk>", "<sep>")
PROTEIN_TOKENS = tuple("ACDEFGHIKLMNPQRSTVWY") + ("X", "B", "U", "Z", "O", ".", "-")
ESM2_PROTEIN_TOKENS = (
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
)
OPHIUCHUS_AB_CHAIN_LENGTHS = (150, 128)
TASK_TYPE_TO_ID = {
    "antibody": 0,
    "tcr": 1,
    "antigen": 2,
    "tcr_pmhc": 3,
    "ppi": 4,
    "generic": 5,
}
SPECIAL_CHAIN_ID = -1


class ProteinTokenizer:
    def __init__(self) -> None:
        self.tokens = SPECIAL_TOKENS + PROTEIN_TOKENS
        self.special_tokens = SPECIAL_TOKENS
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.cls_token_id = self.token_to_id["<cls>"]
        self.eos_token_id = self.token_to_id["<eos>"]
        self.mask_token_id = self.token_to_id["<mask>"]
        self.unk_token_id = self.token_to_id["<unk>"]
        self.sep_token_id = self.token_to_id["<sep>"]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode_residues(self, sequence: str) -> list[int]:
        return [self.token_to_id.get(residue.upper(), self.unk_token_id) for residue in sequence]

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        pieces = []
        special = set(self.special_tokens)
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), "<unk>")
            if skip_special_tokens and token in special:
                continue
            pieces.append(token)
        return "".join(pieces)


class Esm2ProteinTokenizer(ProteinTokenizer):
    """ESM-1b/ESM2 token ids used by Ophiuchus-Ab and facebook/esm2_t33_650M_UR50D."""

    def __init__(self) -> None:
        self.tokens = ("<cls>", "<pad>", "<eos>", "<unk>") + ESM2_PROTEIN_TOKENS + ("<null_1>", "<mask>")
        self.special_tokens = ("<cls>", "<pad>", "<eos>", "<unk>", "<null_1>", "<mask>")
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.cls_token_id = self.token_to_id["<cls>"]
        self.eos_token_id = self.token_to_id["<eos>"]
        self.mask_token_id = self.token_to_id["<mask>"]
        self.unk_token_id = self.token_to_id["<unk>"]
        self.sep_token_id = self.eos_token_id

    def encode_residues(self, sequence: str) -> list[int]:
        normalized = sequence.replace("J", "L").replace("j", "L")
        return [self.token_to_id.get(residue.upper(), self.unk_token_id) for residue in normalized]

    def encode(self, sequence: str) -> list[int]:
        normalized = sequence.replace("J", "L").replace("j", "L")
        if "<mask>" not in normalized:
            return [self.cls_token_id] + self.encode_residues(normalized) + [self.eos_token_id]
        pieces: list[int] = [self.cls_token_id]
        cursor = 0
        while cursor < len(normalized):
            if normalized.startswith("<mask>", cursor):
                pieces.append(self.mask_token_id)
                cursor += len("<mask>")
                continue
            pieces.append(self.token_to_id.get(normalized[cursor].upper(), self.unk_token_id))
            cursor += 1
        pieces.append(self.eos_token_id)
        return pieces

    def encode_chain(self, sequence: str, max_length: int | None = None) -> tuple[list[int], list[int]]:
        residue_ids = self.encode_residues(sequence)
        if max_length is not None:
            if max_length < 2:
                raise ValueError("max_length must leave room for <cls> and <eos>")
            residue_ids = residue_ids[: max_length - 2]
        input_ids = [self.cls_token_id] + residue_ids + [self.eos_token_id]
        loss_mask = [0] + [1] * len(residue_ids) + [0]
        return input_ids, loss_mask


@dataclass
class BioSeqCollator:
    tokenizer: ProteinTokenizer = field(default_factory=ProteinTokenizer)
    max_length: int | None = None
    task_type_to_id: dict[str, int] = field(default_factory=lambda: dict(TASK_TYPE_TO_ID))

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode_example(example) for example in examples]
        target_length = max(len(item["input_ids"]) for item in encoded)
        if self.max_length is not None:
            target_length = min(target_length, self.max_length)

        input_ids = []
        chain_ids = []
        attention_mask = []
        loss_mask = []
        task_type_ids = []

        for item in encoded:
            ids = item["input_ids"][:target_length]
            chains = item["chain_ids"][:target_length]
            trainable = item["loss_mask"][:target_length]
            pad_len = target_length - len(ids)

            input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            chain_ids.append(chains + [SPECIAL_CHAIN_ID] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            loss_mask.append(trainable + [0] * pad_len)
            task_type_ids.append(self.task_type_to_id.get(item["task_type"], self.task_type_to_id["generic"]))

        tokens = torch.tensor(input_ids, dtype=torch.long)
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "chain_ids": torch.tensor(chain_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
            "task_type_ids": torch.tensor(task_type_ids, dtype=torch.long),
        }

    def _encode_example(self, example: dict[str, Any]) -> dict[str, Any]:
        chains = self._extract_chains(example)
        if not chains:
            raise ValueError("BioSeq examples must contain at least one non-empty chain")

        input_ids = [self.tokenizer.cls_token_id]
        chain_ids = [SPECIAL_CHAIN_ID]
        loss_mask = [0]

        for chain_index, sequence in enumerate(chains):
            residue_ids = self.tokenizer.encode_residues(sequence)
            input_ids.extend(residue_ids)
            chain_ids.extend([chain_index] * len(residue_ids))
            loss_mask.extend([1] * len(residue_ids))
            input_ids.append(self.tokenizer.eos_token_id)
            chain_ids.append(SPECIAL_CHAIN_ID)
            loss_mask.append(0)

        return {
            "input_ids": input_ids,
            "chain_ids": chain_ids,
            "loss_mask": loss_mask,
            "task_type": str(example.get("task_type", "generic")),
        }

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
            "tra",
            "trb",
            "tcr_alpha",
            "tcr_beta",
            "peptide",
            "mhc",
            "antigen",
            "protein_a",
            "protein_b",
            "sequence",
        )
        return [str(example[key]) for key in ordered_keys if key in example and str(example[key])]


@dataclass
class OphiuchusAbCollator:
    tokenizer: Esm2ProteinTokenizer = field(default_factory=Esm2ProteinTokenizer)
    chain_lengths: tuple[int, int] = OPHIUCHUS_AB_CHAIN_LENGTHS
    task_type_to_id: dict[str, int] = field(default_factory=lambda: dict(TASK_TYPE_TO_ID))

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = []
        chain_ids = []
        attention_mask = []
        loss_mask = []
        task_type_ids = []

        for example in examples:
            chains = BioSeqCollator._extract_chains(example)
            if len(chains) < 2:
                raise ValueError("OphiuchusAbCollator requires heavy and light chains")

            row_ids = []
            row_chain_ids = []
            row_attention = []
            row_loss = []

            for chain_index, (sequence, chain_length) in enumerate(zip(chains[:2], self.chain_lengths)):
                encoded, trainable = self.tokenizer.encode_chain(sequence, max_length=chain_length)
                pad_len = chain_length - len(encoded)
                row_ids.extend(encoded + [self.tokenizer.eos_token_id] * pad_len)
                row_chain_ids.extend([chain_index] * chain_length)
                row_attention.extend([1] * len(encoded) + [0] * pad_len)
                row_loss.extend(trainable + [0] * pad_len)

            input_ids.append(row_ids)
            chain_ids.append(row_chain_ids)
            attention_mask.append(row_attention)
            loss_mask.append(row_loss)
            task_type = str(example.get("task_type", "antibody"))
            task_type_ids.append(self.task_type_to_id.get(task_type, self.task_type_to_id["generic"]))

        tokens = torch.tensor(input_ids, dtype=torch.long)
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "chain_ids": torch.tensor(chain_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
            "task_type_ids": torch.tensor(task_type_ids, dtype=torch.long),
        }
