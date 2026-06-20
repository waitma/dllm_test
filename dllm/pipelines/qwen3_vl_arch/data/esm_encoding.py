from __future__ import annotations

import json
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


class TokenizersEsmTokenizer:
    """Small wrapper for local ``tokenizer.json`` snapshots.

    Some ESMC releases declare ``ESMCTokenizer``, which older Transformers
    versions cannot import. The underlying ``tokenizer.json`` is still a normal
    tokenizers file, so this wrapper provides the small API the collator needs.
    """

    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer
        self.cls_token_id = self._token_id("<cls>")
        self.pad_token_id = self._token_id("<pad>")
        self.eos_token_id = self._token_id("<eos>")
        self.unk_token_id = self._token_id("<unk>")
        self.mask_token_id = self._token_id("<mask>")

    @classmethod
    def from_file(cls, path: str | Path) -> "TokenizersEsmTokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing required token: {token}")
        return int(token_id)

    def encode(self, sequence: str, add_special_tokens: bool = True, **kwargs) -> list[int]:
        encoding = self.tokenizer.encode(sequence, add_special_tokens=add_special_tokens)
        token_ids = list(encoding.ids)
        max_length = kwargs.get("max_length")
        if max_length is not None and len(token_ids) > int(max_length):
            max_length = int(max_length)
            if max_length < 2:
                raise ValueError("max_length must leave room for <cls> and <eos>")
            if add_special_tokens and token_ids[0] == self.cls_token_id and token_ids[-1] == self.eos_token_id:
                token_ids = [token_ids[0]] + token_ids[1 : max_length - 1] + [token_ids[-1]]
            else:
                token_ids = token_ids[:max_length]
        return token_ids


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
        path = Path(path)
        tokenizer_json = path / "tokenizer.json"
        if tokenizer_json.is_file() and _prefer_tokenizer_json(path):
            return cls(TokenizersEsmTokenizer.from_file(tokenizer_json))

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=local_files_only)
        except Exception as exc:
            if not tokenizer_json.is_file():
                raise
            tokenizer = TokenizersEsmTokenizer.from_file(tokenizer_json)
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

    @property
    def vocab_size(self) -> int:
        value = getattr(self.tokenizer, "vocab_size", None)
        if value is not None:
            return int(value)
        return int(len(self.tokenizer))

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


def _prefer_tokenizer_json(path: Path) -> bool:
    config_path = path / "config.json"
    tokenizer_config_path = path / "tokenizer_config.json"
    for candidate in (config_path, tokenizer_config_path):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text())
        except json.JSONDecodeError:
            continue
        model_type = str(data.get("model_type", "")).lower()
        tokenizer_class = str(data.get("tokenizer_class", "")).lower()
        if model_type == "esmc" or tokenizer_class == "esmctokenizer":
            return True
    return False
