"""Exact and single-edit leakage audit against frozen AB/TCR benchmarks.

Invoke it through::

    conda run -n pllm python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py audit
"""

from __future__ import annotations

import csv
import json
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .schema import CanonicalRecord, is_valid_sequence, normalize_sequence


_SEQUENCE_KEY_EXCLUSIONS = (
    "id",
    "score",
    "label",
    "prediction",
    "metric",
    "name",
    "source",
)


def _role_from_key(key: str) -> str | None:
    normalized = key.lower().replace("-", "_").replace(" ", "_")
    if normalized.startswith("train_"):
        return None
    if any(token in normalized for token in _SEQUENCE_KEY_EXCLUSIONS) and not any(
        token in normalized
        for token in ("cdr3", "sequence", "seq", "peptide", "epitope")
    ):
        return None
    if any(
        token in normalized
        for token in (
            "cdr3a",
            "cdr3_alpha",
            "tra_",
            "tcr_a",
            "tcra",
            "alpha",
        )
    ):
        return "tcr_alpha"
    if any(
        token in normalized
        for token in (
            "cdr3b",
            "cdr3_beta",
            "trb_",
            "tcr_b",
            "tcrb",
            "beta",
        )
    ):
        return "tcr_beta"
    if any(token in normalized for token in ("peptide", "epitope", "pep_seq")):
        return "peptide"
    if any(
        token in normalized
        for token in (
            "cdrh",
            "h_cdr",
            "heavy",
            "vh_seq",
            "vh_sequence",
            "h_sequence",
        )
    ):
        return "antibody_heavy"
    if any(
        token in normalized
        for token in (
            "cdrl",
            "l_cdr",
            "light",
            "vl_seq",
            "vl_sequence",
            "l_sequence",
        )
    ):
        return "antibody_light"
    if any(token in normalized for token in ("antigen_seq", "ag_seq")):
        return "antigen"
    if "binder" in normalized or normalized in {"cdr3", "tcr"}:
        return "tcr_beta"
    return None


def _sequence_from_value(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    sequence = normalize_sequence(value)
    if len(sequence) < 6 or not is_valid_sequence(sequence):
        return ""
    return sequence


def _normalize_mhc(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "")
    if not text:
        return ""
    if not text.startswith("HLA-") and text[0] in "ABCDEFG":
        text = "HLA-" + text
    head, separator, tail = text.partition("-")
    if separator and "*" not in tail and ":" in tail:
        gene = "".join(character for character in tail if character.isalpha())
        digits = tail[len(gene) :]
        if gene and digits:
            text = f"{head}-{gene}*{digits}"
    return text


def _iter_mapping_sequences(
    value: Any, source_path: str
) -> Iterator[tuple[str, str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower().startswith("train_"):
                continue
            role = _role_from_key(str(key))
            if role:
                values = item if isinstance(item, list) else [item]
                for part in values:
                    sequence = _sequence_from_value(part)
                    if sequence:
                        yield role, sequence, source_path
            yield from _iter_mapping_sequences(item, source_path)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_mapping_sequences(item, source_path)


_AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


def _mmcif_polymer_sequences(path: Path) -> dict[str, str]:
    """Read author-chain polymer sequences from a standard mmCIF loop."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    required = {
        "_pdbx_poly_seq_scheme.seq_id",
        "_pdbx_poly_seq_scheme.mon_id",
        "_pdbx_poly_seq_scheme.pdb_strand_id",
    }
    residues: dict[str, dict[int, str]] = defaultdict(dict)
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        header_index = index + 1
        headers: list[str] = []
        while header_index < len(lines) and lines[header_index].lstrip().startswith("_"):
            headers.append(lines[header_index].strip())
            header_index += 1
        if not required.issubset(headers):
            index = header_index
            continue
        seq_index = headers.index("_pdbx_poly_seq_scheme.seq_id")
        mon_index = headers.index("_pdbx_poly_seq_scheme.mon_id")
        chain_index = headers.index("_pdbx_poly_seq_scheme.pdb_strand_id")
        row_index = header_index
        while row_index < len(lines):
            line = lines[row_index].strip()
            if not line:
                row_index += 1
                continue
            if line.startswith(("#", "loop_", "_", "data_")):
                break
            values = shlex.split(line)
            if len(values) >= len(headers):
                try:
                    sequence_index = int(values[seq_index])
                except ValueError:
                    row_index += 1
                    continue
                residue = _AA3_TO_1.get(values[mon_index].upper(), "X")
                for chain in values[chain_index].split(","):
                    chain = chain.strip()
                    if chain and chain not in {".", "?"}:
                        residues[chain].setdefault(sequence_index, residue)
            row_index += 1
        return {
            chain: "".join(sequence[index] for index in sorted(sequence))
            for chain, sequence in residues.items()
        }
    return {}


def _iter_humanization_cif_sequences(
    path: Path,
) -> Iterator[tuple[str, str, str]]:
    """Resolve the released humanization H/L chain map against its mmCIF."""

    chain_map = path.parent.parent / "test_chains.csv"
    if not chain_map.exists():
        return
    wanted: list[str] = []
    with chain_map.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0].strip().lower() == path.stem.lower():
                wanted = [value.strip() for value in row[1].split("-") if value.strip()]
                break
    if len(wanted) != 2:
        return
    sequences = _mmcif_polymer_sequences(path)
    source_path = str(path.resolve())
    for role, chain in zip(("antibody_heavy", "antibody_light"), wanted):
        sequence = _sequence_from_value(sequences.get(chain, ""))
        if sequence:
            yield role, sequence, source_path


def iter_benchmark_sequences(path: Path) -> Iterator[tuple[str, str, str]]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open(
                encoding="utf-8-sig", errors="replace", newline=""
            ) as handle:
                for row in csv.DictReader(handle, delimiter=delimiter):
                    for key, value in row.items():
                        role = _role_from_key(key)
                        if not role:
                            continue
                        sequence = _sequence_from_value(value)
                        if sequence:
                            yield role, sequence, str(path.resolve())
        elif suffix == ".jsonl":
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.strip():
                        yield from _iter_mapping_sequences(
                            json.loads(line), str(path.resolve())
                        )
        elif suffix == ".json":
            text = path.read_text(encoding="utf-8", errors="replace")
            try:
                values = [json.loads(text)]
            except json.JSONDecodeError:
                values = [json.loads(line) for line in text.splitlines() if line.strip()]
            for value in values:
                yield from _iter_mapping_sequences(value, str(path.resolve()))
        elif suffix == ".cif":
            yield from _iter_humanization_cif_sequences(path)
        elif suffix == ".txt" and any(
            token in path.name.lower() for token in ("cdr3b", "tcrb", "beta")
        ):
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    sequence = _sequence_from_value(line.strip())
                    if sequence:
                        yield "tcr_beta", sequence, str(path.resolve())
    except (csv.Error, json.JSONDecodeError, OSError):
        return


def _is_eval_item(item: Mapping[str, Any]) -> bool:
    partition = item.get("partition")
    if partition:
        return partition in {"benchmark_eval", "benchmark_all"}
    return "train" not in Path(item["path"]).name.lower()


def build_benchmark_sequence_bank(quarantine: Mapping[str, Any]) -> dict:
    bank: dict[str, dict[str, str]] = defaultdict(dict)
    for item in quarantine.get("files", []):
        if not _is_eval_item(item):
            continue
        path = Path(item["path"])
        for role, sequence, source_path in iter_benchmark_sequences(path):
            bank[role].setdefault(sequence, source_path)
    return {role: dict(values) for role, values in bank.items()}


def _append_sequence(group: dict[str, list[str] | str], role: str, value: Any) -> None:
    sequence = _sequence_from_value(value)
    if not sequence:
        return
    sequences = group.setdefault(role, [])
    if isinstance(sequences, list) and sequence not in sequences:
        sequences.append(sequence)


def _group_from_mapping(
    row: Mapping[str, Any], *, context_key: str = ""
) -> dict[str, list[str] | str]:
    group: dict[str, list[str] | str] = {}
    for key, value in row.items():
        normalized_key = str(key).lower().replace("-", "_").replace(" ", "_")
        if normalized_key.startswith("train_"):
            continue
        role = _role_from_key(str(key))
        if role:
            for item in value if isinstance(value, list) else [value]:
                _append_sequence(group, role, item)
        if (
            "mhc" in normalized_key
            or "hla" in normalized_key
            or normalized_key == "allele"
        ) and isinstance(value, str):
            group["mhc"] = _normalize_mhc(value)
    if not group.get("peptide") and context_key:
        peptide_candidate = context_key.split("_HLA-")[0]
        _append_sequence(group, "peptide", peptide_candidate)
        if "_HLA-" in context_key and not group.get("mhc"):
            group["mhc"] = _normalize_mhc(
                "HLA-" + context_key.split("_HLA-", 1)[1]
            )
    return group


def _iter_json_groups(
    value: Any, *, context_key: str = ""
) -> Iterator[dict[str, list[str] | str]]:
    if isinstance(value, Mapping):
        group = _group_from_mapping(value, context_key=context_key)
        if any(
            group.get(role)
            for role in ("tcr_alpha", "tcr_beta", "antibody_heavy")
        ):
            yield group
        for key, item in value.items():
            if str(key).lower().startswith("train_"):
                continue
            if isinstance(item, (Mapping, list)):
                yield from _iter_json_groups(item, context_key=str(key))
    elif isinstance(value, list):
        for item in value:
            yield from _iter_json_groups(item, context_key=context_key)


def iter_benchmark_groups(path: Path) -> Iterator[dict[str, list[str] | str]]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open(
                encoding="utf-8-sig", errors="replace", newline=""
            ) as handle:
                for row in csv.DictReader(handle, delimiter=delimiter):
                    group = _group_from_mapping(row)
                    if group:
                        yield group
        elif suffix == ".json":
            text = path.read_text(encoding="utf-8", errors="replace")
            try:
                values = [json.loads(text)]
            except json.JSONDecodeError:
                values = [json.loads(line) for line in text.splitlines() if line.strip()]
            for value in values:
                yield from _iter_json_groups(value)
        elif suffix == ".jsonl":
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.strip():
                        yield from _iter_json_groups(json.loads(line))
    except (csv.Error, json.JSONDecodeError, OSError):
        return


def build_benchmark_interaction_bank(quarantine: Mapping[str, Any]) -> dict:
    receptor_peptide: dict[tuple[str, str, str], str] = {}
    paired_tcr_peptide: dict[tuple[str, str, str], str] = {}
    receptor_pmhc: dict[tuple[str, str, str, str], str] = {}
    for item in quarantine.get("files", []):
        if not _is_eval_item(item):
            continue
        path = Path(item["path"])
        source_path = str(path.resolve())
        for group in iter_benchmark_groups(path):
            peptides = group.get("peptide", [])
            alpha = group.get("tcr_alpha", [])
            beta = group.get("tcr_beta", [])
            peptides = peptides if isinstance(peptides, list) else []
            alpha = alpha if isinstance(alpha, list) else []
            beta = beta if isinstance(beta, list) else []
            mhc = group.get("mhc", "")
            mhc = mhc if isinstance(mhc, str) else ""
            for peptide in peptides:
                for role, sequences in (("tcr_alpha", alpha), ("tcr_beta", beta)):
                    for sequence in sequences:
                        receptor_peptide.setdefault(
                            (role, sequence, peptide), source_path
                        )
                        if mhc:
                            receptor_pmhc.setdefault(
                                (role, sequence, peptide, mhc), source_path
                            )
                for alpha_sequence in alpha:
                    for beta_sequence in beta:
                        paired_tcr_peptide.setdefault(
                            (alpha_sequence, beta_sequence, peptide), source_path
                        )
    return {
        "receptor_peptide": receptor_peptide,
        "paired_tcr_peptide": paired_tcr_peptide,
        "receptor_pmhc": receptor_pmhc,
    }


def _single_edit_indices(
    sequences: Iterable[str],
) -> tuple[dict[str, str], dict[str, str]]:
    substitutions: dict[str, str] = {}
    deletions: dict[str, str] = {}
    for sequence in sequences:
        for index in range(len(sequence)):
            substitutions.setdefault(
                f"{len(sequence)}:{sequence[:index]}*{sequence[index + 1:]}",
                sequence,
            )
            deletions.setdefault(sequence[:index] + sequence[index + 1 :], sequence)
    return substitutions, deletions


def _receptor_candidates(record: CanonicalRecord, role: str) -> list[str]:
    entity = record.entity(role)
    if not entity:
        return []
    candidates = [entity.sequence]
    cdr3 = _sequence_from_value(entity.regions.get("CDR3"))
    if cdr3 and cdr3 not in candidates:
        candidates.append(cdr3)
    for sequence in list(candidates):
        if sequence.startswith("C") and len(sequence) > 6:
            candidates.append(sequence[1:])
    return candidates


def audit_benchmark_leakage(
    records: Iterable[CanonicalRecord],
    quarantine: Mapping[str, Any],
    *,
    benchmark_bank: Mapping[str, Mapping[str, str]] | None = None,
    interaction_bank: Mapping[str, Mapping[tuple, str]] | None = None,
) -> dict:
    """Audit entity matches, composite interactions, and CDR3/peptide edit distance."""

    rows = list(records)
    bank = (
        {role: dict(values) for role, values in benchmark_bank.items()}
        if benchmark_bank is not None
        else build_benchmark_sequence_bank(quarantine)
    )
    interactions = (
        {name: dict(values) for name, values in interaction_bank.items()}
        if interaction_bank is not None
        else build_benchmark_interaction_bank(quarantine)
    )
    exact_samples: list[dict] = []
    near_samples: list[dict] = []
    composite_samples: list[dict] = []
    exact_record_ids: set[str] = set()
    exact_receptor_records: set[str] = set()
    exact_peptide_records: set[str] = set()
    near_record_ids: set[str] = set()
    receptor_peptide_records: set[str] = set()
    paired_tcr_peptide_records: set[str] = set()
    receptor_pmhc_records: set[str] = set()
    exact_by_role: Counter[str] = Counter()
    near_by_role: Counter[str] = Counter()
    near_roles = {"tcr_alpha", "tcr_beta", "peptide"}
    record_roles = {entity.role for record in rows for entity in record.entities}
    indices = {
        role: _single_edit_indices(values)
        for role, values in bank.items()
        if role in near_roles and role in record_roles
    }
    for record in rows:
        for entity in record.entities:
            role_bank = bank.get(entity.role, {})
            benchmark_path = role_bank.get(entity.sequence)
            if benchmark_path:
                exact_record_ids.add(record.record_id)
                exact_by_role[entity.role] += 1
                if entity.role in {
                    "tcr_alpha",
                    "tcr_beta",
                    "antibody_heavy",
                    "antibody_light",
                }:
                    exact_receptor_records.add(record.record_id)
                if entity.role == "peptide":
                    exact_peptide_records.add(record.record_id)
                if len(exact_samples) < 200:
                    exact_samples.append(
                        {
                            "record_id": record.record_id,
                            "entity_id": entity.entity_id,
                            "role": entity.role,
                            "sequence": entity.sequence,
                            "benchmark_path": benchmark_path,
                        }
                    )
                continue
            if entity.role not in indices:
                continue
            substitutions, deletions = indices[entity.role]
            neighbor = ""
            for index in range(len(entity.sequence)):
                pattern = (
                    f"{len(entity.sequence)}:"
                    f"{entity.sequence[:index]}*{entity.sequence[index + 1:]}"
                )
                if pattern in substitutions:
                    neighbor = substitutions[pattern]
                    break
            if not neighbor and entity.sequence in deletions:
                neighbor = deletions[entity.sequence]
            if not neighbor:
                for index in range(len(entity.sequence)):
                    shortened = entity.sequence[:index] + entity.sequence[index + 1 :]
                    if shortened in role_bank:
                        neighbor = shortened
                        break
            if neighbor:
                near_record_ids.add(record.record_id)
                near_by_role[entity.role] += 1
                if len(near_samples) < 200:
                    near_samples.append(
                        {
                            "record_id": record.record_id,
                            "entity_id": entity.entity_id,
                            "role": entity.role,
                            "sequence": entity.sequence,
                            "benchmark_neighbor": neighbor,
                            "distance": 1,
                            "benchmark_path": role_bank.get(neighbor, ""),
                        }
                    )

        peptide = record.entity("peptide")
        if not peptide:
            continue
        alpha = _receptor_candidates(record, "tcr_alpha")
        beta = _receptor_candidates(record, "tcr_beta")
        allele = _normalize_mhc(record.context.get("mhc_allele"))
        pair_sources: set[str] = set()
        pmhc_sources: set[str] = set()
        for role, candidates in (("tcr_alpha", alpha), ("tcr_beta", beta)):
            for sequence in candidates:
                pair_key = (role, sequence, peptide.sequence)
                source = interactions.get("receptor_peptide", {}).get(pair_key)
                if source:
                    receptor_peptide_records.add(record.record_id)
                    pair_sources.add(source)
                pmhc_key = (role, sequence, peptide.sequence, allele)
                source = interactions.get("receptor_pmhc", {}).get(pmhc_key)
                if allele and source:
                    receptor_pmhc_records.add(record.record_id)
                    pmhc_sources.add(source)
        paired_sources = {
            source
            for alpha_sequence in alpha
            for beta_sequence in beta
            if (
                source := interactions.get("paired_tcr_peptide", {}).get(
                    (alpha_sequence, beta_sequence, peptide.sequence)
                )
            )
        }
        if paired_sources:
            paired_tcr_peptide_records.add(record.record_id)
        if (pair_sources or paired_sources or pmhc_sources) and len(composite_samples) < 200:
            composite_samples.append(
                {
                    "record_id": record.record_id,
                    "receptor_peptide": bool(pair_sources),
                    "paired_tcr_peptide": bool(paired_sources),
                    "receptor_pmhc": bool(pmhc_sources),
                    "benchmark_paths": sorted(
                        pair_sources | paired_sources | pmhc_sources
                    ),
                }
            )
    records_by_id = {record.record_id: record for record in rows}

    def counts_by_pack(record_ids: set[str]) -> dict[str, int]:
        counts = Counter(
            records_by_id[record_id].eligibility.pack
            for record_id in record_ids
        )
        return dict(sorted(counts.items()))

    def counts_by_source(record_ids: set[str]) -> dict[str, int]:
        counts = Counter(
            source.source_name
            for record_id in record_ids
            for source in records_by_id[record_id].source_records
        )
        return dict(sorted(counts.items()))

    match_sets = {
        "any_exact_entity": exact_record_ids,
        "exact_receptor_entity": exact_receptor_records,
        "exact_peptide_entity": exact_peptide_records,
        "exact_receptor_peptide_pair": receptor_peptide_records,
        "exact_paired_tcr_peptide": paired_tcr_peptide_records,
        "exact_receptor_pmhc": receptor_pmhc_records,
        "any_single_edit_entity": near_record_ids,
    }
    return {
        "schema_version": "immune_receptor_leakage_audit.v2",
        "record_count": len(rows),
        "benchmark_sequence_counts": {
            role: len(values) for role, values in sorted(bank.items())
        },
        "exact_match_record_count": len(exact_record_ids),
        "exact_match_record_count_definition": "records with any exact sequence entity",
        "records_with_exact_receptor_entity": len(exact_receptor_records),
        "exact_receptor_record_ids": sorted(exact_receptor_records),
        "records_with_exact_peptide_entity": len(exact_peptide_records),
        "exact_peptide_record_ids": sorted(exact_peptide_records),
        "exact_match_entity_counts_by_role": dict(sorted(exact_by_role.items())),
        "exact_match_record_ids": sorted(exact_record_ids),
        "exact_match_sample": exact_samples,
        "exact_receptor_peptide_pair_records": len(receptor_peptide_records),
        "exact_receptor_peptide_pair_record_ids": sorted(receptor_peptide_records),
        "exact_paired_tcr_peptide_records": len(paired_tcr_peptide_records),
        "exact_paired_tcr_peptide_record_ids": sorted(paired_tcr_peptide_records),
        "exact_receptor_pmhc_records": len(receptor_pmhc_records),
        "exact_receptor_pmhc_record_ids": sorted(receptor_pmhc_records),
        "match_record_counts_by_pack": {
            name: counts_by_pack(record_ids)
            for name, record_ids in match_sets.items()
        },
        "match_record_counts_by_source": {
            name: counts_by_source(record_ids)
            for name, record_ids in match_sets.items()
        },
        "composite_match_sample": composite_samples,
        "near_match_method": (
            "Levenshtein distance exactly 1 by substitution/insertion/deletion; "
            "applied only to TCR alpha/beta and peptide sequence entities. Full-chain "
            "near identity requires a later MMseqs2 audit."
        ),
        "near_match_record_count": len(near_record_ids),
        "near_match_entity_counts_by_role": dict(sorted(near_by_role.items())),
        "near_match_record_ids": sorted(near_record_ids),
        "near_match_sample": near_samples,
    }
