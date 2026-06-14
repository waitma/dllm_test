from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


BIOSEQ_SCHEMA_VERSION = "bioseq.v1"
VALID_PROTEIN_CHARS = set("ACDEFGHIKLMNPQRSTVWYXBZUO.-")
REGION_FIELDS = ("FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4")


@dataclass
class BioSeqJsonlExample:
    chains: list[str]
    task_type: str
    source: str
    chain_roles: list[str] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    split: str | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    regions: dict[str, dict[str, str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = BIOSEQ_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, {}, [])}

    def validate(self) -> None:
        if not self.chains:
            raise ValueError("BioSeq example must contain at least one chain")
        if self.chain_roles and len(self.chain_roles) != len(self.chains):
            raise ValueError("chain_roles length must match chains length")
        if not self.targets:
            self.targets = list(range(len(self.chains)))
        for index, chain in enumerate(self.chains):
            if not is_valid_protein_sequence(chain):
                raise ValueError(f"Invalid protein sequence at chain index {index}")


def normalize_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


def is_valid_protein_sequence(sequence: str) -> bool:
    return bool(sequence) and all(residue in VALID_PROTEIN_CHARS for residue in sequence)


def normalize_source(value: Any, fallback: str) -> str:
    source = str(value or "").strip()
    return source if source else fallback


def receptor_regions(row: dict[str, Any], prefix: str) -> dict[str, str]:
    regions: dict[str, str] = {}
    for field_name in REGION_FIELDS:
        value = normalize_sequence(row.get(f"{prefix}_{field_name}", ""))
        if value:
            regions[field_name] = value
    return regions


def antibody_role(row: dict[str, Any], prefix: str, fallback: str) -> str:
    anarci_type = str(row.get(f"{prefix}_anarci_type", "")).strip().upper()
    chain_type = str(row.get(f"{prefix}_type", "")).strip().lower()
    if anarci_type == "H" or chain_type == "heavy":
        return "antibody_heavy"
    if anarci_type in {"L", "K"} or chain_type in {"light", "kappa", "lambda"}:
        return "antibody_light"
    return fallback


def tcr_role(row: dict[str, Any], prefix: str, fallback: str) -> str:
    anarci_type = str(row.get(f"{prefix}_anarci_type", "")).strip().upper()
    chain_type = str(row.get(f"{prefix}_type", "")).strip().lower()
    if anarci_type == "B" or chain_type in {"beta", "trb"}:
        return "tcr_beta"
    if anarci_type == "A" or chain_type in {"alpha", "tra"}:
        return "tcr_alpha"
    return fallback


def type_to_role(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "alpha": "tcr_alpha",
        "a": "tcr_alpha",
        "tra": "tcr_alpha",
        "beta": "tcr_beta",
        "b": "tcr_beta",
        "trb": "tcr_beta",
        "antigen": "antigen",
        "peptide": "peptide",
        "mhc": "mhc",
        "heavy": "antibody_heavy",
        "light": "antibody_light",
        "h": "antibody_heavy",
        "l": "antibody_light",
    }.get(normalized, normalized or "other")


def oas_paired_row_to_example(row: dict[str, Any]) -> BioSeqJsonlExample:
    chains = [normalize_sequence(row.get("cleaned_chain1_seq")), normalize_sequence(row.get("cleaned_chain2_seq"))]
    example = BioSeqJsonlExample(
        chains=chains,
        chain_roles=[
            antibody_role(row, "chain1", "antibody_chain_1"),
            antibody_role(row, "chain2", "antibody_chain_2"),
        ],
        task_type="antibody",
        source=normalize_source(row.get("source_file"), "oas_paired"),
        split=str(row.get("split") or "").strip() or None,
        targets=[0, 1],
        regions={
            "0": receptor_regions(row, "chain1"),
            "1": receptor_regions(row, "chain2"),
        },
        metadata=compact_metadata(
            row,
            (
                "species",
                "data_type",
                "chain1_type",
                "chain2_type",
                "chain1_v",
                "chain1_j",
                "chain2_v",
                "chain2_j",
                "chain1_cluster",
                "chain2_cluster",
                "pair_cluster",
                "cluster_id",
            ),
        ),
    )
    example.validate()
    return example


def ots_paired_row_to_example(row: dict[str, Any]) -> BioSeqJsonlExample:
    chains = [normalize_sequence(row.get("cleaned_chain1_seq")), normalize_sequence(row.get("cleaned_chain2_seq"))]
    example = BioSeqJsonlExample(
        chains=chains,
        chain_roles=[
            tcr_role(row, "chain1", "tcr_chain_1"),
            tcr_role(row, "chain2", "tcr_chain_2"),
        ],
        task_type="tcr",
        source=normalize_source(row.get("source_file"), "ots_paired"),
        split=str(row.get("split") or "").strip() or None,
        targets=[0, 1],
        regions={
            "0": receptor_regions(row, "chain1"),
            "1": receptor_regions(row, "chain2"),
        },
        metadata=compact_metadata(
            row,
            (
                "species",
                "data_type",
                "chain1_type",
                "chain2_type",
                "chain1_v",
                "chain1_j",
                "chain2_v",
                "chain2_j",
                "chain1_cluster",
                "chain2_cluster",
                "pair_cluster",
                "cluster_id",
            ),
        ),
    )
    example.validate()
    return example


def nanobody_row_to_example(row: dict[str, Any]) -> BioSeqJsonlExample:
    chain = normalize_sequence(row.get("cleaned_seq") or row.get("vhh_seq"))
    example = BioSeqJsonlExample(
        chains=[chain],
        chain_roles=["nanobody_vhh"],
        task_type="antibody",
        source=normalize_source(row.get("source"), "nanobody"),
        split=str(row.get("split") or "").strip() or None,
        targets=[0],
        regions={"0": {field_name: normalize_sequence(row.get(field_name)) for field_name in REGION_FIELDS if row.get(field_name)}},
        metadata=compact_metadata(row, ("anarci_chain_type", "cluster_id")),
    )
    example.validate()
    return example


def processed_json_row_to_example(row: dict[str, Any]) -> BioSeqJsonlExample:
    chains = [normalize_sequence(chain) for chain in row.get("chains", [])]
    chain_roles = [type_to_role(value) for value in row.get("types", [])]
    example = BioSeqJsonlExample(
        chains=chains,
        chain_roles=chain_roles,
        task_type=task_type_from_source(row.get("source")),
        source=normalize_source(row.get("source"), "processed"),
        targets=[int(index) for index in row.get("targets", list(range(len(chains))))],
        metadata=compact_metadata(row, ("source",)),
    )
    example.validate()
    return example


def task_type_from_source(source: Any) -> str:
    normalized = str(source or "").strip().lower()
    if normalized == "ppi":
        return "ppi"
    if normalized in {"vdjdb", "mcpas", "mira", "iedb", "atlas", "deepinsight", "tcrdesign"}:
        return "tcr_pmhc"
    return "generic"


def compact_metadata(row: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            metadata[key] = value
    return metadata


def iter_csv_examples(
    input_path: Path | str,
    row_adapter: Callable[[dict[str, Any]], BioSeqJsonlExample],
    limit: int | None = None,
) -> Iterator[BioSeqJsonlExample]:
    count = 0
    with Path(input_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row_adapter(row)
            count += 1
            if limit is not None and count >= limit:
                break


def iter_processed_jsonl_examples(input_path: Path | str, limit: int | None = None) -> Iterator[BioSeqJsonlExample]:
    count = 0
    with Path(input_path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            yield processed_json_row_to_example(json.loads(line))
            count += 1
            if limit is not None and count >= limit:
                break


def write_jsonl(examples: Iterable[BioSeqJsonlExample], output_path: Path | str) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w") as handle:
        for example in examples:
            example.validate()
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")
            written += 1
    return written


def load_jsonl(input_path: Path | str) -> list[dict[str, Any]]:
    with Path(input_path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
