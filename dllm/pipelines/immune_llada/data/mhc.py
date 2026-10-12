"""Sequence-first MHC inputs. Test: python -m pytest scripts/tests/immune_llada/test_mhc.py -q.

Offline lookup is explicit; legacy adapters never perform network or allele lookup.
The table is content-pinned by the dataset builder. Ambiguity is never resolved by
choosing the first allele, truncating field resolution, or imputing from peptide.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from .records import BioSeqChain

POLICY = "sequence_first_mhc_v1"
AA = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO")


def clean_sequence(value: str) -> str:
    """Remove formatting only; never replace a provided residue with another."""
    value = "".join(str(value or "").split()).upper()
    if value and not set(value) <= AA:
        raise ValueError("MHC sequence contains unsupported residues/alignment gaps")
    return value


def allele_key(name: str) -> str:
    name = str(name or "").strip().upper()
    if re.match(r"^[ABCEFG]\*?\d", name):
        name = "HLA-" + name
    if name.startswith("H-2-"):
        name = "H2-" + name[4:]
    return name.replace("*", "").replace(":", "")


class PseudoTable:
    def __init__(self, path: Path | str):
        path = Path(path)
        data = path.read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.path = str(path.resolve())
        self.by_name = defaultdict(set)
        for line in data.decode().splitlines():
            if not line.strip():
                continue
            name, seq = line.split()
            seq = clean_sequence(seq)
            if len(seq) != 34 or "X" in seq:
                raise ValueError(f"Invalid pseudo reference for {name}")
            self.by_name[allele_key(name)].add(seq)

    def lookup(self, name: str) -> tuple[str, str]:
        key = allele_key(name)
        sequences = self.by_name.get(key, set())
        if not key:
            return "", "missing_allele"
        if not sequences:
            return "", "allele_not_in_reference"
        if len(sequences) != 1:
            return "", "conflicting_reference"
        return next(iter(sequences)), "exact_unique_pseudo"


def chain_entry(sequence: str, *, representation: str, origin: str,
                component: str = "mhc", scope: str | None = None) -> dict:
    sequence = clean_sequence(sequence)
    if not sequence or set(sequence) == {"X"}:
        raise ValueError("Empty/all-X MHC is missing context, not a supplied chain")
    return {"sequence": sequence, "metadata": {
        "mhc_representation": representation, "mhc_origin": origin,
        "mhc_component": component,
        "sequence_scope": scope or ("fragment" if representation.endswith("pseudo34") else "unknown"),
        "mhc_policy": POLICY,
    }}


def resolve_mhc(*, provided: list[dict], allele: str, table: PseudoTable) -> dict:
    chains = []
    for item in provided:
        seq = clean_sequence(item.get("sequence", ""))
        if not seq or set(seq) == {"X"}:
            continue
        # Supplied strings are kept even if partially unknown or absent from the table.
        chains.append(chain_entry(seq, representation=item.get("representation", "provided_sequence"),
                                  origin=item.get("origin", "source_sequence"),
                                  component=item.get("component", "mhc"), scope=item.get("scope")))
    info = {"policy": POLICY, "allele": str(allele or ""), "reference_sha256": table.sha256}
    if chains:
        info["status"] = "provided_sequence_kept"
    else:
        seq, status = table.lookup(allele)
        info["status"] = status
        if seq:
            chains = [chain_entry(seq, representation="allele_pseudo34", origin="DTU_NetMHCpan_4.1")]
    return {"chains": chains, "resolution": info}


def explicit_mhc_chains(row: dict) -> list[BioSeqChain] | None:
    """None means legacy schema; [] means an explicitly unresolved MHC."""
    raw = row.get("mhc_chains_json")
    if raw in (None, ""):
        return None
    items = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(items, list):
        raise ValueError("mhc_chains_json must be a list")
    result = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
            raise ValueError("MHC entry requires sequence and provenance metadata")
        seq = clean_sequence(item["sequence"])
        if not seq or set(seq) == {"X"}:
            raise ValueError("Explicit MHC entries cannot be empty/all-X")
        result.append(BioSeqChain(seq, "mhc", metadata=dict(item["metadata"])))
    return result
