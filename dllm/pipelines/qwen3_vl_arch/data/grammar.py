"""Grammar-v2 BioSeq serialization and collation.

Training still reads semantic records from ``bioseq_grammar_v1`` Arrow shards;
``GrammarRenderer`` applies the v2 token layout at encode time.

Build the Arrow cache with::

    python scripts/data/build_bioseq_grammar_v1.py --splits train,valid

Inspect one encoded batch with::

    python -c "from dllm.pipelines.qwen3_vl_arch.data import GrammarTokenizer; print(GrammarTokenizer().vocab_size)"
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset

from .esm_encoding import Esm2SequenceTokenizer, EsmTokenizerProtocol
from .mixture import distributed_worker_shard
from .records import BioSeqChain, BioSeqRecord, TASK_TYPE_TO_ID

# v2 structure tokens (no <fixs>/<fixd>/<generate>/<prote>/<pairs>).
# Multi-chain objects use repeated <prots>...<protd> blocks; peptide uses <pep>.
GRAMMAR_STRUCTURE_TOKENS = (
    "<ab>",
    "<tcr>",
    "<nb>",
    "<pep>",
    "<prots>",
    "<protd>",
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
    "unknown",
)

GRAMMAR_RELATION_TOKENS = tuple(f"<{relation}>" for relation in GRAMMAR_RELATIONS)
GRAMMAR_TOKENS = GRAMMAR_STRUCTURE_TOKENS + GRAMMAR_RELATION_TOKENS
GRAMMAR_TYPE_MARKERS = frozenset({"<ab>", "<tcr>", "<nb>", "<pep>"})

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

# Process-local cache: load_from_disk is very slow on multi-million-row grammar shards.
_GRAMMAR_ARROW_DATASET_CACHE: dict[str, object] = {}


def _cached_grammar_arrow_dataset(path: Path):
    key = str(path.resolve())
    cached = _GRAMMAR_ARROW_DATASET_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise ImportError("GrammarArrowSource requires the `datasets` package") from exc
    cached = load_from_disk(key)
    _GRAMMAR_ARROW_DATASET_CACHE[key] = cached
    return cached


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

    def chain_separator_id(self) -> int:
        """Base-vocabulary id for the literal ``.`` chain separator."""

        base = self.base_tokenizer
        if hasattr(base, "token_id"):
            return int(base.token_id("."))
        token_to_id = getattr(base, "token_to_id", None)
        if isinstance(token_to_id, dict) and "." in token_to_id:
            return int(token_to_id["."])
        inner = getattr(base, "tokenizer", None)
        if inner is not None and hasattr(inner, "token_to_id"):
            token_id = inner.token_to_id(".")
            if token_id is not None:
                return int(token_id)
        raise AttributeError("Base tokenizer must expose '.' for chain separation")


def _relation_token(relation: str | None) -> str:
    raw = str(relation or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not raw:
        normalized = "unknown"
    elif raw in {"unknown_relation", "unk", "unknown"}:
        normalized = "unknown"
    elif raw in GRAMMAR_RELATIONS:
        normalized = raw
    else:
        normalized = "unknown"
    return f"<{normalized}>"


def _first_chain(record: BioSeqRecord, roles: set[str]) -> BioSeqChain | None:
    return next((chain for chain in record.chains if chain.role.lower() in roles), None)


def _record_seed(record: BioSeqRecord) -> int:
    payload = (
        record.source,
        record.task_type,
        tuple((chain.role, chain.sequence) for chain in record.chains),
    )
    return hash(payload) & 0xFFFFFFFF


def _grammar_position_ids(
    input_ids: list[int],
    classes: list[int],
    tokenizer: GrammarTokenizer,
) -> tuple[list[int], list[int]]:
    """Assign per-residue chain index and within-chain position for decoder embeddings."""

    position_ids_chain: list[int] = []
    position_ids_inner: list[int] = []
    chain_index = 0
    inner_index = 0
    in_protein_block = False
    separator_id = tokenizer.chain_separator_id()

    for token_id, class_id in zip(input_ids, classes):
        token = tokenizer.token(token_id)
        if class_id == TOKEN_CLASS_RESIDUE:
            position_ids_chain.append(chain_index)
            position_ids_inner.append(inner_index)
            inner_index += 1
            continue

        position_ids_chain.append(-1)
        position_ids_inner.append(-1)
        if token == "<prots>":
            in_protein_block = True
            inner_index = 0
        elif token == "<protd>":
            in_protein_block = False
            chain_index += 1
            inner_index = 0
        elif in_protein_block and int(token_id) == separator_id:
            chain_index += 1
            inner_index = 0

    return position_ids_chain, position_ids_inner


def _build_per_chain_encoder_inputs(
    input_ids: list[int],
    classes: list[int],
    position_ids_chain: list[int],
    position_ids_inner: list[int],
    tokenizer: GrammarTokenizer,
) -> tuple[list[list[int]], list[list[int]], list[int], list[int]]:
    """Build per-chain ``<cls> seq <eos>`` encoder streams and decoder ``chain_ids``."""

    residue_groups: dict[int, list[tuple[int, int]]] = {}
    for index, (class_id, chain_id, inner_id, token_id) in enumerate(
        zip(classes, position_ids_chain, position_ids_inner, input_ids)
    ):
        if class_id != TOKEN_CLASS_RESIDUE or chain_id < 0 or inner_id < 0:
            continue
        residue_groups.setdefault(chain_id, []).append((inner_id, token_id))

    sorted_chain_ids = sorted(residue_groups)
    encoder_chains: list[list[int]] = []
    encoder_residue_masks: list[list[int]] = []
    logical_to_encoder: dict[int, int] = {}

    for encoder_index, logical_id in enumerate(sorted_chain_ids):
        residue_tokens = [token_id for _, token_id in sorted(residue_groups[logical_id], key=lambda item: item[0])]
        encoded_ids, encoded_mask = tokenizer.base_tokenizer.encode_chain(
            _residue_ids_to_sequence(residue_tokens, tokenizer)
        )
        encoder_chains.append([int(token_id) for token_id in encoded_ids])
        encoder_residue_masks.append([int(value) for value in encoded_mask])
        logical_to_encoder[logical_id] = encoder_index

    decoder_chain_ids = [-1] * len(input_ids)
    decoder_inner_ids = [-1] * len(input_ids)
    for index, (class_id, chain_id, inner_id) in enumerate(
        zip(classes, position_ids_chain, position_ids_inner)
    ):
        if class_id != TOKEN_CLASS_RESIDUE or chain_id < 0:
            continue
        encoder_chain = logical_to_encoder.get(chain_id)
        if encoder_chain is None:
            continue
        decoder_chain_ids[index] = encoder_chain
        decoder_inner_ids[index] = inner_id

    return encoder_chains, encoder_residue_masks, decoder_chain_ids, decoder_inner_ids


def _residue_ids_to_sequence(residue_ids: list[int], tokenizer: GrammarTokenizer) -> str:
    pieces: list[str] = []
    base_id_to_token = getattr(tokenizer.base_tokenizer, "id_to_token", {})
    for token_id in residue_ids:
        if token_id in tokenizer.id_to_token:
            continue
        token = base_id_to_token.get(int(token_id), "X")
        if token.startswith("<"):
            continue
        pieces.append(str(token))
    return "".join(pieces)


@dataclass
class GrammarRenderer:
    """Render a BioSeqRecord into the flat grammar-v2 token stream."""

    tokenizer: GrammarTokenizer
    ppi_max_protein_length: int = 1024
    rng: random.Random | None = None

    def encode(self, record: BioSeqRecord) -> dict[str, Any]:
        ids: list[int] = []
        fixed: list[int] = []
        classes: list[int] = []
        separator_id = self.tokenizer.chain_separator_id()

        def special(token: str, is_fixed: bool = False) -> None:
            ids.append(self.tokenizer.special_id(token))
            fixed.append(int(is_fixed))
            classes.append(
                TOKEN_CLASS_RELATION if token in GRAMMAR_RELATION_TOKENS else TOKEN_CLASS_STRUCTURE
            )

        def literal(token_id: int, is_fixed: bool = False) -> None:
            ids.append(int(token_id))
            fixed.append(int(is_fixed))
            classes.append(TOKEN_CLASS_STRUCTURE)

        def sequence(sequence_value: str, is_fixed: bool = False, cap: int | None = None) -> None:
            normalized = sequence_value[:cap] if cap is not None else sequence_value
            residue_ids = self.tokenizer.encode_residues(normalized)
            ids.extend(residue_ids)
            fixed.extend([int(is_fixed)] * len(residue_ids))
            classes.extend([TOKEN_CLASS_RESIDUE] * len(residue_ids))

        def append_protein_block(
            chains: list[BioSeqChain],
            *,
            type_marker: str | None = None,
            is_fixed: bool = False,
            type_marker_fixed: bool | None = None,
            cap: int | None = None,
        ) -> None:
            if not chains:
                raise ValueError("Protein block requires at least one chain")
            special("<prots>", is_fixed=is_fixed)
            if type_marker is not None:
                marker_fixed = type_marker_fixed if type_marker_fixed is not None else is_fixed
                special(type_marker, is_fixed=marker_fixed)
            for chain_index, chain in enumerate(chains):
                if chain_index > 0:
                    literal(separator_id, is_fixed=is_fixed)
                sequence(chain.sequence, is_fixed=is_fixed, cap=cap)
            special("<protd>", is_fixed=is_fixed)

        def append_peptide_block(chain: BioSeqChain, *, is_fixed: bool) -> None:
            append_protein_block(
                [chain],
                type_marker="<pep>",
                is_fixed=is_fixed,
                type_marker_fixed=True,
            )

        relation = _relation_token(record.labels.get("relation") or record.metadata.get("relation"))
        roles = {chain.role.lower() for chain in record.chains}
        grammar_name = "generic"

        if record.task_type == "ppi" or {"protein_a", "protein_b"} <= roles:
            protein_a = _first_chain(record, {"protein_a", "other"}) or record.chains[0]
            protein_b = _first_chain(record, {"protein_b"}) or record.chains[1]
            append_protein_block([protein_a], is_fixed=True, cap=self.ppi_max_protein_length)
            special(relation, is_fixed=True)
            append_protein_block([protein_b], is_fixed=False, cap=self.ppi_max_protein_length)
            grammar_name = "ppi_conditional"
        else:
            mhc_chains = [chain for chain in record.chains if chain.role.lower() in {"mhc", "pmhc", "hla"}]
            antigen = _first_chain(record, {"antigen"})
            peptide = _first_chain(record, {"peptide", "epitope"})
            heavy = _first_chain(record, {"antibody_heavy", "nanobody_vhh"})
            light = _first_chain(record, {"antibody_light"})
            alpha = _first_chain(record, {"tcr_alpha"})
            beta = _first_chain(record, {"tcr_beta"})

            if record.task_type == "antibody" and len(record.chains) >= 2:
                heavy, light = record.chains[0], record.chains[1]
            if record.task_type == "tcr" and len(record.chains) >= 2 and (alpha is None or beta is None):
                alpha, beta = record.chains[1], record.chains[0]

            if record.task_type in {
                "antibody_antigen",
                "nanobody_antigen",
                "antibody_neutralization",
            } and antigen is not None:
                append_protein_block([antigen], is_fixed=True)
                special(relation, is_fixed=True)
                ab_chains = [chain for chain in (heavy, light) if chain is not None]
                if record.task_type == "nanobody_antigen" or (
                    light is None and "nanobody_vhh" in roles
                ):
                    receptor_type = "<nb>"
                    grammar_name = "antigen_nanobody"
                else:
                    receptor_type = "<ab>"
                    grammar_name = "antigen_antibody"
                append_protein_block(
                    ab_chains if ab_chains else record.chains,
                    type_marker=receptor_type,
                    is_fixed=False,
                    type_marker_fixed=True,
                )
            elif record.task_type in {"nanobody"} or (
                heavy is not None and light is None and "nanobody_vhh" in roles and antigen is None
            ):
                append_protein_block(
                    [heavy or record.chains[0]],
                    type_marker="<nb>",
                    is_fixed=False,
                    type_marker_fixed=True,
                )
                grammar_name = "nanobody"
            elif alpha is not None or beta is not None or record.task_type.startswith("tcr"):
                tcr_peptide = peptide or antigen
                if mhc_chains:
                    append_protein_block(mhc_chains, is_fixed=True)
                    special("<binding>", is_fixed=True)
                if tcr_peptide is not None:
                    append_peptide_block(tcr_peptide, is_fixed=True)
                    special("<binding>", is_fixed=True)
                receptor = [chain for chain in (alpha, beta) if chain is not None]
                if len(receptor) == 1:
                    append_protein_block(
                        receptor,
                        type_marker="<tcr>",
                        is_fixed=False,
                        type_marker_fixed=True,
                    )
                    grammar_name = "tcr_single"
                else:
                    append_protein_block(
                        receptor,
                        type_marker="<tcr>",
                        is_fixed=False,
                        type_marker_fixed=True,
                    )
                    grammar_name = "tcr_pmhc" if mhc_chains else (
                        "tcr_peptide" if tcr_peptide is not None else "tcr_pair"
                    )
            elif record.task_type == "antibody" or (heavy is not None and light is not None):
                ab_chains = [chain for chain in (heavy, light) if chain is not None]
                append_protein_block(
                    ab_chains if ab_chains else record.chains,
                    type_marker="<ab>",
                    is_fixed=False,
                    type_marker_fixed=True,
                )
                grammar_name = "antibody_pair"
            elif record.task_type == "tcr":
                append_protein_block(
                    record.chains,
                    type_marker="<tcr>",
                    is_fixed=False,
                    type_marker_fixed=True,
                )
                grammar_name = "tcr_pair"
            else:
                append_protein_block(record.chains, is_fixed=False)
                grammar_name = "single_entity"

        if not ids:
            raise ValueError(f"Grammar renderer produced an empty record for {record.source}")
        diffusion_mask = [int(not is_fixed) for is_fixed in fixed]
        position_ids_chain, position_ids_inner = _grammar_position_ids(ids, classes, self.tokenizer)
        return {
            "input_ids": ids,
            "fixed_context_mask": fixed,
            "diffusion_loss_mask": diffusion_mask,
            "diffusion_eligible_mask": diffusion_mask,
            "token_class_ids": classes,
            "position_ids_chain": position_ids_chain,
            "position_ids_inner": position_ids_inner,
            "task_type": record.task_type,
            "source": record.source,
            "grammar_name": grammar_name,
            "weight": float(record.weight),
        }


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
        labels={"relation": row.get("relation", "unknown")},
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
        dataset = _cached_grammar_arrow_dataset(self.path)
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


@dataclass
class GrammarBioSeqCollator:
    """Pad grammar records and build per-chain encoder inputs for ESMC/ESM2."""

    tokenizer: GrammarTokenizer
    max_sequence_length: int = 2112
    task_type_to_id: dict[str, int] = field(default_factory=lambda: dict(TASK_TYPE_TO_ID))

    def __post_init__(self) -> None:
        self.renderer = GrammarRenderer(self.tokenizer)

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
                "chain_ids",
            )
        }
        encoder_ids: list[list[list[int]]] = []
        encoder_attention: list[list[list[int]]] = []
        encoder_residue: list[list[list[int]]] = []
        encoder_chain_mask: list[list[int]] = []

        for row in rows:
            input_ids = list(row["input_ids"])
            classes = list(row["token_class_ids"])
            attention = [1] * len(input_ids)
            pad_len = max_len - len(input_ids)
            position_ids_chain = list(row["position_ids_chain"])
            position_ids_inner = list(row["position_ids_inner"])
            chain_enc, chain_residue_masks, decoder_chain_ids, decoder_inner_ids = _build_per_chain_encoder_inputs(
                input_ids,
                classes,
                position_ids_chain,
                position_ids_inner,
                self.tokenizer,
            )

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
            batch["position_ids_inner"].append(decoder_inner_ids + [-1] * pad_len)
            batch["position_ids_chain"].append(position_ids_chain + [-1] * pad_len)
            batch["chain_ids"].append(decoder_chain_ids + [-1] * pad_len)

            max_chain_len = max((len(chain) for chain in chain_enc), default=2)
            padded_chains = [chain + [pad_id] * (max_chain_len - len(chain)) for chain in chain_enc]
            padded_masks = [mask + [0] * (max_chain_len - len(mask)) for mask in chain_residue_masks]
            if not padded_chains:
                padded_chains = [[self.tokenizer.cls_token_id, self.tokenizer.eos_token_id]]
                padded_masks = [[0, 0]]
            encoder_ids.append(padded_chains)
            encoder_attention.append([[1 if token_id != pad_id else 0 for token_id in chain] for chain in padded_chains])
            encoder_residue.append(padded_masks)
            encoder_chain_mask.append([1] * len(padded_chains))

        max_chains = max(len(chains) for chains in encoder_ids)
        max_chain_len = max(len(chain) for chains in encoder_ids for chain in chains)
        padded_encoder_ids: list[list[list[int]]] = []
        padded_encoder_attention: list[list[list[int]]] = []
        padded_encoder_residue: list[list[list[int]]] = []
        padded_encoder_chain_mask: list[list[int]] = []
        for chains, masks, residue_masks, chain_mask in zip(
            encoder_ids, encoder_attention, encoder_residue, encoder_chain_mask
        ):
            padded_chains = chains + [[pad_id] * max_chain_len] * (max_chains - len(chains))
            padded_attn = masks + [[0] * max_chain_len] * (max_chains - len(masks))
            padded_res = residue_masks + [[0] * max_chain_len] * (max_chains - len(residue_masks))
            for chain_index, chain in enumerate(padded_chains):
                if len(chain) < max_chain_len:
                    padded_chains[chain_index] = chain + [pad_id] * (max_chain_len - len(chain))
                    padded_attn[chain_index] = padded_attn[chain_index] + [0] * (
                        max_chain_len - len(padded_attn[chain_index])
                    )
                    padded_res[chain_index] = padded_res[chain_index] + [0] * (
                        max_chain_len - len(padded_res[chain_index])
                    )
            padded_encoder_ids.append(padded_chains)
            padded_encoder_attention.append(padded_attn)
            padded_encoder_residue.append(padded_res)
            padded_encoder_chain_mask.append(chain_mask + [0] * (max_chains - len(chain_mask)))

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
        result["encoder_input_ids"] = torch.tensor(padded_encoder_ids, dtype=torch.long)
        result["encoder_attention_mask"] = torch.tensor(padded_encoder_attention, dtype=torch.bool)
        result["encoder_residue_mask"] = torch.tensor(padded_encoder_residue, dtype=torch.bool)
        result["encoder_chain_mask"] = torch.tensor(padded_encoder_chain_mask, dtype=torch.bool)
        result["grammar_names"] = [str(row["grammar_name"]) for row in rows]
        result["view_names"] = ["grammar_v2"] * len(rows)
        result["task_groups"] = [str(row["task_type"]) for row in rows]
        result["task_types"] = [str(row["task_type"]) for row in rows]
        result["sources"] = [str(row["source"]) for row in rows]
        result["weights"] = torch.tensor(
            [float(row.get("weight", 1.0)) for row in rows],
            dtype=torch.float32,
        ).unsqueeze(-1)
        return result
