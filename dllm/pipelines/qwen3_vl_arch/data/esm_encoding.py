from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


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


class EsmTokenizerProtocol(Protocol):
    pad_token_id: int
    cls_token_id: int
    eos_token_id: int
    mask_token_id: int

    def encode_chain(self, sequence: str, max_length: int | None = None) -> tuple[list[int], list[int]]:
        ...


class Esm2SequenceTokenizer:
    """ESM-1b/ESM2-compatible token ids used by MINT/Ophiuchus-Ab."""

    def __init__(self) -> None:
        self.tokens = ("<cls>", "<pad>", "<eos>", "<unk>") + ESM2_PROTEIN_TOKENS + ("<null_1>", "<mask>")
        self.special_tokens = ("<cls>", "<pad>", "<eos>", "<unk>", "<null_1>", "<mask>")
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.cls_token_id = self.token_to_id["<cls>"]
        self.pad_token_id = self.token_to_id["<pad>"]
        self.eos_token_id = self.token_to_id["<eos>"]
        self.unk_token_id = self.token_to_id["<unk>"]
        self.mask_token_id = self.token_to_id["<mask>"]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode_residues(self, sequence: str) -> list[int]:
        normalized = sequence.replace("J", "L").replace("j", "L")
        return [self.token_to_id.get(residue.upper(), self.unk_token_id) for residue in normalized]

    def encode_chain(self, sequence: str, max_length: int | None = None) -> tuple[list[int], list[int]]:
        residue_ids = self.encode_residues(sequence)
        if max_length is not None:
            if max_length < 2:
                raise ValueError("max_length must leave room for <cls> and <eos>")
            residue_ids = residue_ids[: max_length - 2]
        token_ids = [self.cls_token_id] + residue_ids + [self.eos_token_id]
        residue_mask = [0] + [1] * len(residue_ids) + [0]
        return token_ids, residue_mask

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        special = set(self.special_tokens)
        pieces = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), "<unk>")
            if skip_special_tokens and token in special:
                continue
            pieces.append(token)
        return "".join(pieces)


@dataclass
class HuggingFaceEsmTokenizerAdapter:
    """Adapter for local ESM-family Hugging Face tokenizers.

    Use this when the encoder comes from a local ESM2/ESMC snapshot. The adapter
    keeps the data loader independent of the model implementation while making
    the emitted encoder ids match the encoder tokenizer.
    """

    tokenizer: object

    @classmethod
    def from_pretrained(cls, path: str | Path, local_files_only: bool = True) -> "HuggingFaceEsmTokenizerAdapter":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=local_files_only)
        return cls(tokenizer)

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    @property
    def cls_token_id(self) -> int:
        return int(self.tokenizer.cls_token_id)

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def mask_token_id(self) -> int:
        return int(self.tokenizer.mask_token_id)

    def encode_chain(self, sequence: str, max_length: int | None = None) -> tuple[list[int], list[int]]:
        kwargs = {"add_special_tokens": True}
        if max_length is not None:
            kwargs.update({"max_length": max_length, "truncation": True})
        token_ids = list(self.tokenizer.encode(sequence, **kwargs))
        residue_mask = [0] * len(token_ids)
        if len(token_ids) >= 2:
            for index in range(1, len(token_ids) - 1):
                residue_mask[index] = 1
        return token_ids, residue_mask

