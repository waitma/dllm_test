"""Grammar-v1 BioSeq serialization and collation.

Build the Arrow cache with::

    python scripts/data/build_bioseq_grammar_v1.py --splits train,valid

Inspect one encoded batch with::

    python -c "from dllm.pipelines.qwen3_vl_arch.data import GrammarTokenizer; print(GrammarTokenizer().vocab_size)"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset

from .esm_encoding import Esm2SequenceTokenizer, EsmTokenizerProtocol
from .mixture import distributed_worker_shard
from .records import BioSeqChain, BioSeqRecord, TASK_TYPE_TO_ID


GRAMMAR_STRUCTURE_TOKENS = (
    "<fixs>",
    "<fixd>",
    "<generate>",
    "<proas>",
    "<proae>",
    "<probs>",
    "<probd>",
    "<peptides>",
    "<peptided>",
    "<protas>",
    "<protad>",
    "<protbs>",
    "<protbd>",
)
GRAMMAR_RELATIONS = (
    "binding",
    "activation",
    "inhibition",
    "catalysis",
    "reaction",
    "expression",
    "ptmod",
    "neutralization",
    "nonbinding",
    "unknown_relation",
)
GRAMMAR_RELATION_TOKENS = tuple(f"<{relation}>" for relation in GRAMMAR_RELATIONS)
GRAMMAR_TOKENS = GRAMMAR_STRUCTURE_TOKENS + GRAMMAR_RELATION_TOKENS

TOKEN_CLASS_PAD = 0
TOKEN_CLASS_RESIDUE = 1
TOKEN_CLASS_STRUCTURE = 2
TOKEN_CLASS_RELATION = 3
TOKEN_CLASS_NAMES = {
    TOKEN_CLASS_RESIDUE: "residue",
    TOKEN_CLASS_STRUCTURE: "structure",
    TOKEN_CLASS_RELATION: "relation",
}

DEFAULT_GRAMMAR_DATA_DIR = Path(
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/bioseq_grammar_v1"
)


class GrammarTokenizer:
    """Append grammar tokens to an ESM-family residue vocabulary."""

    def __init__(self, base_tokenizer: EsmTokenizerProtocol | None = None) -> None:
        self.base_tokenizer = base_tokenizer or Esm2SequenceTokenizer()
        self.base_vocab_size = int(getattr(self.base_tokenizer, "vocab_size"))
        self.token_to_id = {
            token: self.base_vocab_size + index for index, token in enumerate(GRAMMAR_TOKENS)
        }
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}
        self.pad_token_id = int(self.base_tokenizer.pad_token_id)
        self.mask_token_id = int(self.base_tokenizer.mask_token_id)
        self.cls_token_id = int(self.base_tokenizer.cls_token_id)
        self.eos_token_id = int(self.base_tokenizer.eos_token_id)

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(GRAMMAR_TOKENS)

    def special_id(self, token: str) -> int:
        try:
            return self.token_to_id[token]
        except KeyError as exc:
            raise KeyError(f"Unknown grammar token: {token}") from exc

    def encode_residues(self, sequence: str) -> list[int]:
        token_ids, residue_mask = self.base_tokenizer.encode_chain(sequence)
        return [int(token_id) for token_id, is_residue in zip(token_ids, residue_mask) if is_residue]

    def token(self, token_id: int) -> str:
        if int(token_id) in self.id_to_token:
            return self.id_to_token[int(token_id)]
        base_id_to_token = getattr(self.base_tokenizer, "id_to_token", {})
        return str(base_id_to_token.get(int(token_id), f"<base:{int(token_id)}>"))

    def decode_tokens(self, token_ids: list[int]) -> list[str]:
        return [self.token(token_id) for token_id in token_ids if int(token_id) != self.pad_token_id]


def _relation_token(relation: str | None) -> str:
    normalized = str(relation or "binding").strip().lower().replace(" ", "_")
    if normalized not in GRAMMAR_RELATIONS:
        normalized = "unknown_relation"
    return f"<{normalized}>"


def _first_chain(record: BioSeqRecord, roles: set[str]) -> BioSeqChain | None:
    return next((chain for chain in record.chains if chain.role.lower() in roles), None)


@dataclass
class GrammarRenderer:
    """Render a BioSeqRecord into the flat grammar-v1 token stream."""

    tokenizer: GrammarTokenizer
    ppi_max_protein_length: int = 1024

    def encode(self, record: BioSeqRecord) -> dict[str, Any]:
        ids: list[int] = []
        fixed: list[int] = []
        classes: list[int] = []

        def special(token: str, is_fixed: bool = False) -> None:
            ids.append(self.tokenizer.special_id(token))
            fixed.append(int(is_fixed))
            classes.append(
                TOKEN_CLASS_RELATION if token in GRAMMAR_RELATION_TOKENS else TOKEN_CLASS_STRUCTURE
            )

        def sequence(sequence_value: str, is_fixed: bool = False, cap: int | None = None) -> None:
            normalized = sequence_value[:cap] if cap is not None else sequence_value
            residue_ids = self.tokenizer.encode_residues(normalized)
            ids.extend(residue_ids)
            fixed.extend([int(is_fixed)] * len(residue_ids))
            classes.extend([TOKEN_CLASS_RESIDUE] * len(residue_ids))

        def fixed_sequence(chain: BioSeqChain) -> None:
            special("<fixs>", is_fixed=True)
            sequence(chain.sequence, is_fixed=True)
            special("<fixd>", is_fixed=True)

        relation = _relation_token(record.labels.get("relation") or record.metadata.get("relation"))
        roles = {chain.role.lower() for chain in record.chains}
        grammar_name = "generic_pair"

        if record.task_type == "ppi" or {"protein_a", "protein_b"} <= roles:
            protein_a = _first_chain(record, {"protein_a", "other"}) or record.chains[0]
            protein_b = _first_chain(record, {"protein_b"}) or record.chains[1]
            special("<protas>")
            sequence(protein_a.sequence, cap=self.ppi_max_protein_length)
            special("<protad>")
            special(relation)
            special("<protbs>")
            sequence(protein_b.sequence, cap=self.ppi_max_protein_length)
            special("<protbd>")
            grammar_name = "ppi_pair"
        else:
            mhc = _first_chain(record, {"mhc", "pmhc", "hla"})
            antigen = _first_chain(record, {"antigen"})
            peptide = _first_chain(record, {"peptide", "epitope"})
            heavy = _first_chain(record, {"antibody_heavy", "nanobody_vhh"})
            light = _first_chain(record, {"antibody_light"})
            alpha = _first_chain(record, {"tcr_alpha"})
            beta = _first_chain(record, {"tcr_beta"})
            if record.task_type == "antibody" and len(record.chains) >= 2:
                heavy, light = record.chains[:2]
            if record.task_type == "tcr" and len(record.chains) >= 2 and (
                alpha is None or beta is None
            ):
                beta, alpha = record.chains[:2]

            if record.task_type in {"antibody_antigen", "nanobody_antigen"} and antigen is not None:
                fixed_sequence(antigen)
                special("<generate>")
                self._append_receptor_pair(special, sequence, heavy, light, relation)
                grammar_name = "antigen_antibody"
            elif alpha is not None or beta is not None or record.task_type.startswith("tcr"):
                if mhc is not None:
                    fixed_sequence(mhc)
                    special("<binding>")
                tcr_peptide = peptide or antigen
                if tcr_peptide is not None:
                    special("<peptides>")
                    sequence(tcr_peptide.sequence)
                    special("<peptided>")
                special("<generate>")
                self._append_receptor_pair(special, sequence, alpha, beta, relation)
                grammar_name = "tcr_pmhc" if mhc is not None else (
                    "tcr_peptide" if tcr_peptide is not None else "tcr_pair"
                )
            else:
                special("<generate>")
                self._append_receptor_pair(special, sequence, heavy, light, relation)
                grammar_name = "antibody_pair"

        if not ids:
            raise ValueError(f"Grammar renderer produced an empty record for {record.source}")
        diffusion_mask = [int(not is_fixed) for is_fixed in fixed]
        return {
            "input_ids": ids,
            "fixed_context_mask": fixed,
            "diffusion_loss_mask": diffusion_mask,
            "diffusion_eligible_mask": diffusion_mask,
            "token_class_ids": classes,
            "task_type": record.task_type,
            "source": record.source,
            "grammar_name": grammar_name,
            "weight": float(record.weight),
        }

    @staticmethod
    def _append_receptor_pair(
        special: Any,
        sequence: Any,
        chain_a: BioSeqChain | None,
        chain_b: BioSeqChain | None,
        relation: str,
    ) -> None:
        if chain_a is not None:
            special("<proas>")
            sequence(chain_a.sequence)
            special("<proae>")
        if chain_a is not None and chain_b is not None:
            special(relation)
        if chain_b is not None:
            special("<probs>")
            sequence(chain_b.sequence)
            special("<probd>")
        if chain_a is None and chain_b is None:
            raise ValueError("Receptor grammar requires at least one target chain")


def grammar_record_from_arrow(row: dict[str, Any]) -> BioSeqRecord:
    chains = [
        BioSeqChain(sequence=sequence, role=role)
        for sequence, role in zip(row["chains"], row["roles"])
    ]
    return BioSeqRecord(
        chains=chains,
        task_type=str(row["task_type"]),
        source=str(row["source"]),
        split=str(row.get("split") or "") or None,
        labels={"relation": row.get("relation", "binding")},
        weight=float(row.get("weight", 1.0)),
    )


@dataclass(frozen=True)
class GrammarArrowSourceConfig:
    name: str
    path: Path = DEFAULT_GRAMMAR_DATA_DIR
    split: str = "train"
    weight: float = 1.0
    max_records: int | None = None


class GrammarArrowSource(IterableDataset):
    """Stream semantic grammar records from preprocessed Hugging Face Arrow shards."""

    def __init__(self, config: GrammarArrowSourceConfig) -> None:
        super().__init__()
        self.config = config
        self.path = config.path / config.name / config.split
        if not self.path.exists():
            raise FileNotFoundError(
                f"Grammar Arrow source not found: {self.path}. "
                "Run scripts/data/build_bioseq_grammar_v1.py first."
            )

    def iter_records(self, shard_index: int = 0, num_shards: int = 1) -> Iterator[BioSeqRecord]:
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError("GrammarArrowSource requires the `datasets` package") from exc

        dataset = load_from_disk(str(self.path))
        dataset = dataset.shard(num_shards=num_shards, index=shard_index, contiguous=True)
        kept = 0
        for row in dataset:
            row = dict(row)
            record = grammar_record_from_arrow(row)
            if self.config.weight != 1.0:
                record = BioSeqRecord(
                    chains=record.chains,
                    task_type=record.task_type,
                    source=record.source,
                    split=record.split,
                    metadata=record.metadata,
                    labels=record.labels,
                    weight=self.config.weight,
                )
            yield record
            kept += 1
            if self.config.max_records is not None and kept >= self.config.max_records:
                break

    def __iter__(self) -> Iterator[BioSeqRecord]:
        shard_index, num_shards = distributed_worker_shard()
        yield from self.iter_records(shard_index=shard_index, num_shards=num_shards)


def _base_token_id(tokenizer: EsmTokenizerProtocol, token: str, default: int) -> int:
    token_to_id = getattr(tokenizer, "token_to_id", None)
    if isinstance(token_to_id, dict) and token in token_to_id:
        return int(token_to_id[token])
    wrapped = getattr(tokenizer, "tokenizer", None)
    nested = getattr(wrapped, "tokenizer", wrapped)
    lookup = getattr(nested, "token_to_id", None)
    if callable(lookup):
        value = lookup(token)
        if value is not None:
            return int(value)
    return int(default)


@dataclass
class GrammarBioSeqCollator:
    """Pad complete grammar records and construct the ESMC proxy stream."""

    tokenizer: GrammarTokenizer
    max_sequence_length: int = 2112
    task_type_to_id: dict[str, int] = field(default_factory=lambda: dict(TASK_TYPE_TO_ID))

    def __post_init__(self) -> None:
        self.renderer = GrammarRenderer(self.tokenizer)
        self.encoder_separator_token_id = _base_token_id(
            self.tokenizer.base_tokenizer,
            "|",
            default=31,
        )

    def __call__(self, records: list[BioSeqRecord | dict[str, Any]]) -> dict[str, Any]:
        rows = [
            record if isinstance(record, dict) and "input_ids" in record else self.renderer.encode(record)
            for record in records
        ]
        lengths = [len(row["input_ids"]) for row in rows]
        if max(lengths) > self.max_sequence_length:
            raise ValueError(
                f"Grammar record length {max(lengths)} exceeds max_sequence_length "
                f"{self.max_sequence_length}; records are never grammar-truncated"
            )
        max_len = max(lengths)
        pad_id = self.tokenizer.pad_token_id
        batch: dict[str, list[list[int]]] = {
            key: []
            for key in (
                "input_ids",
                "labels",
                "attention_mask",
                "fixed_context_mask",
                "diffusion_loss_mask",
                "diffusion_eligible_mask",
                "residue_mask",
                "structure_token_mask",
                "relation_token_mask",
                "token_class_ids",
                "position_ids_inner",
                "position_ids_chain",
                "encoder_position_ids",
            )
        }
        encoder_ids: list[list[list[int]]] = []
        encoder_attention: list[list[list[int]]] = []

        for row in rows:
            input_ids = list(row["input_ids"])
            classes = list(row["token_class_ids"])
            attention = [1] * len(input_ids)
            pad_len = max_len - len(input_ids)
            proxy = [
                token_id if token_id < self.tokenizer.base_vocab_size else self.encoder_separator_token_id
                for token_id in input_ids
            ]
            batch["input_ids"].append(input_ids + [pad_id] * pad_len)
            batch["labels"].append(input_ids + [-100] * pad_len)
            batch["attention_mask"].append(attention + [0] * pad_len)
            batch["fixed_context_mask"].append(list(row["fixed_context_mask"]) + [0] * pad_len)
            batch["diffusion_loss_mask"].append(list(row["diffusion_loss_mask"]) + [0] * pad_len)
            batch["diffusion_eligible_mask"].append(list(row["diffusion_eligible_mask"]) + [0] * pad_len)
            batch["residue_mask"].append(
                [int(value == TOKEN_CLASS_RESIDUE) for value in classes] + [0] * pad_len
            )
            batch["structure_token_mask"].append(
                [int(value == TOKEN_CLASS_STRUCTURE) for value in classes] + [0] * pad_len
            )
            batch["relation_token_mask"].append(
                [int(value == TOKEN_CLASS_RELATION) for value in classes] + [0] * pad_len
            )
            batch["token_class_ids"].append(classes + [TOKEN_CLASS_PAD] * pad_len)
            batch["position_ids_inner"].append(list(range(len(input_ids))) + [-1] * pad_len)
            batch["position_ids_chain"].append([0] * len(input_ids) + [-1] * pad_len)
            batch["encoder_position_ids"].append(list(range(len(input_ids))) + [-1] * pad_len)
            encoder_ids.append([proxy + [pad_id] * pad_len])
            encoder_attention.append([attention + [0] * pad_len])

        result = {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
        for key in (
            "attention_mask",
            "fixed_context_mask",
            "diffusion_loss_mask",
            "diffusion_eligible_mask",
            "residue_mask",
            "structure_token_mask",
            "relation_token_mask",
        ):
            result[key] = result[key].bool()
        result["task_type_ids"] = torch.tensor(
            [self.task_type_to_id.get(row["task_type"], self.task_type_to_id["generic"]) for row in rows],
            dtype=torch.long,
        )
        result["encoder_input_ids"] = torch.tensor(encoder_ids, dtype=torch.long)
        result["encoder_attention_mask"] = torch.tensor(encoder_attention, dtype=torch.bool)
        result["encoder_residue_mask"] = result["encoder_attention_mask"].clone()
        result["encoder_chain_mask"] = torch.ones(len(rows), 1, dtype=torch.bool)
        result["grammar_names"] = [str(row["grammar_name"]) for row in rows]
        result["view_names"] = ["grammar_v1"] * len(rows)
        result["task_groups"] = [str(row["task_type"]) for row in rows]
        result["task_types"] = [str(row["task_type"]) for row in rows]
        result["sources"] = [str(row["source"]) for row in rows]
        result["weights"] = torch.tensor(
            [float(row.get("weight", 1.0)) for row in rows],
            dtype=torch.float32,
        ).unsqueeze(-1)
        return result
