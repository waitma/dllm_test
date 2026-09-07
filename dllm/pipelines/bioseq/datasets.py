"""Dataset loaders that feed the exact Ophiuchus-Ab diffusion model.

This module loads the three local immune-sequence corpora used for BioSeq
training and normalizes every row into a minimal ``{"chains", "task_type",
"source", "weight"}`` record that :class:`MultiChainDynamicCollator` understands:

* OAS paired antibody  -> two chains (heavy oriented to slot 0)
* OTS paired TCR        -> two chains (beta oriented to slot 0)
* nanobody (VHH)        -> one chain

All paths must be absolute, following the project rule. The loaders stream the
CSV files and stop after ``max_rows`` valid rows, so the multi-GB ``train.csv``
files never need to be fully materialized for a smoke run.
"""

from __future__ import annotations

import csv
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from torch.utils.data import ConcatDataset, Dataset

from .adapters import is_valid_protein_sequence, normalize_sequence

logger = logging.getLogger(__name__)

# csv fields such as full FR/CDR region strings can be long; lift the limit once.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DEFAULT_DATA_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/data")

OAS_DEFAULT_DIR = DEFAULT_DATA_ROOT / "oas_previous_clean/splits"
OTS_DEFAULT_DIR = DEFAULT_DATA_ROOT / "ots_paired_clean/final"
NANOBODY_DEFAULT_DIR = DEFAULT_DATA_ROOT / "nanobody_processed/step7_clean"
OAS_LABEL_FILE_TEMPLATE = "cleaned_merged_data_step_clustered_{split}_oas_label.csv"


@dataclass
class ImmuneSourceSpec:
    name: str
    path: Path
    row_to_record: Callable[[dict[str, Any]], dict[str, Any] | None]
    task_type: str
    weight: float = 1.0


def _is_heavy(anarci_type: str, chain_type: str) -> bool:
    return anarci_type.strip().upper() == "H" or chain_type.strip().lower() == "heavy"


def _is_beta(anarci_type: str, chain_type: str) -> bool:
    return anarci_type.strip().upper() == "B" or chain_type.strip().lower() in {"beta", "trb"}


# Canonical relations understood by the TCR/antibody *recognition* grammar plane.
# That plane collapses anything it does not recognize to ``<binding>`` (grammar's
# ``_relation_token`` maps unknown -> ``<unknown>``, and the recognition renderer
# then falls back to ``<binding>``). That fallback is intentional for *unlabeled*
# pairs but is a silent-corruption trap for a *labeled* negative written with the
# wrong spelling (e.g. "non-binding", "False", "0"): it would become a false
# positive with no error. We therefore normalize known spellings here and FAIL
# LOUD on anything unexpected, at the source, instead of letting the grammar
# swallow it. (The shared grammar stays lenient on purpose: other lines such as
# immune_receptor_v2 emit relations like "specificity"/"affinity" that rely on
# the unknown-degrade behaviour.)
_RELATION_POSITIVE = {"binding", "binder", "positive", "pos", "true", "1", "yes"}
_RELATION_NEGATIVE = {
    "nonbinding",
    "non_binding",
    "nonbinder",
    "non_binder",
    "negative",
    "neg",
    "false",
    "0",
    "no",
}


def canonical_recognition_relation(value: Any, *, default: str = "binding") -> str:
    """Map a raw relation cell to exactly ``"binding"`` or ``"nonbinding"``.

    Empty/absent -> ``default`` (matches "unlabeled pair defaults to binding").
    A non-empty value that is neither a known positive nor a known negative
    spelling raises, so a mislabeled row surfaces instead of silently rendering
    as ``<binding>``.
    """

    raw = str(value if value is not None else "").strip().lower().replace(" ", "_").replace("-", "_")
    if not raw:
        return default
    if raw in _RELATION_POSITIVE:
        return "binding"
    if raw in _RELATION_NEGATIVE:
        return "nonbinding"
    raise ValueError(
        f"Unrecognized recognition relation {value!r}; expected binding/nonbinding "
        "(or a known synonym). Fix the data source: an unknown value would silently "
        "render as <binding> in the TCR/antibody recognition plane."
    )


def _canonical_csv_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized in {"val", "valid", "validation"}:
        return "valid"
    return normalized or "train"


def _source_split_path(spec: ImmuneSourceSpec, split: str) -> Path:
    if spec.path.is_file():
        return spec.path
    if spec.name == "oas":
        return spec.path / OAS_LABEL_FILE_TEMPLATE.format(split=_canonical_csv_split(split))
    return spec.path / f"{split}.csv"


def _is_oas_label_row(row: dict[str, Any]) -> bool:
    return "cleaned_h_sequence" in row and "cleaned_l_sequence" in row


def oas_paired_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    if _is_oas_label_row(row):
        heavy = normalize_sequence(row.get("cleaned_h_sequence"))
        light = normalize_sequence(row.get("cleaned_l_sequence"))
        if not is_valid_protein_sequence(heavy) or not is_valid_protein_sequence(light):
            return None
        return {"chains": [heavy, light], "task_type": "antibody", "source": "oas_paired"}

    chain1 = normalize_sequence(row.get("cleaned_chain1_seq"))
    chain2 = normalize_sequence(row.get("cleaned_chain2_seq"))
    if not is_valid_protein_sequence(chain1) or not is_valid_protein_sequence(chain2):
        return None
    chain2_heavy = _is_heavy(str(row.get("chain2_anarci_type", "")), str(row.get("chain2_type", "")))
    chain1_heavy = _is_heavy(str(row.get("chain1_anarci_type", "")), str(row.get("chain1_type", "")))
    if chain2_heavy and not chain1_heavy:
        chains = [chain2, chain1]
    else:
        chains = [chain1, chain2]
    return {"chains": chains, "task_type": "antibody", "source": "oas_paired"}


def ots_paired_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    chain1 = normalize_sequence(row.get("cleaned_chain1_seq"))
    chain2 = normalize_sequence(row.get("cleaned_chain2_seq"))
    if not is_valid_protein_sequence(chain1) or not is_valid_protein_sequence(chain2):
        return None
    chain2_beta = _is_beta(str(row.get("chain2_anarci_type", "")), str(row.get("chain2_type", "")))
    chain1_beta = _is_beta(str(row.get("chain1_anarci_type", "")), str(row.get("chain1_type", "")))
    if chain2_beta and not chain1_beta:
        chains = [chain2, chain1]
    else:
        chains = [chain1, chain2]
    return {"chains": chains, "task_type": "tcr", "source": "ots_paired"}


def nanobody_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    chain = normalize_sequence(row.get("cleaned_seq") or row.get("vhh_seq"))
    if not is_valid_protein_sequence(chain):
        return None
    return {"chains": [chain], "task_type": "antibody", "source": "nanobody"}


def asd_antibody_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """ASD antibody<->antigen row -> recognition record.

    Emits explicit ``roles`` (antigen + heavy + optional light) and a
    ``relation`` label so the grammar renderer builds the antibody_antigen
    plane: ``[antigen (context)] <relation> [<ab> heavy (+light)]``. Column
    layout is produced by ``downstream/asd/scripts/step6_split.py``.
    """

    antigen = normalize_sequence(row.get("antigen_seq"))
    heavy = normalize_sequence(row.get("heavy_fv"))
    light = normalize_sequence(row.get("light_fv"))
    if not is_valid_protein_sequence(antigen) or not is_valid_protein_sequence(heavy):
        return None
    chains = [antigen, heavy]
    roles = ["antigen", "antibody_heavy"]
    if is_valid_protein_sequence(light):
        chains.append(light)
        roles.append("antibody_light")
    return {
        "chains": chains,
        "roles": roles,
        "task_type": "antibody_antigen",
        "source": "asd_antibody",
        "relation": canonical_recognition_relation(row.get("relation")),
    }


def trait_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """TRAIT TCR–pMHC row -> recognition record.

    Layout from ``downstream/trait/scripts/step4_split.py``: epitope + optional
    HLA pseudo-sequence + CDR3α/β. Grammar renders
    ``[mhc] <binding> [peptide] <binding|nonbinding> [<tcr> α . β]``.
    """

    epitope = normalize_sequence(row.get("epitope_seq"))
    cdr3a = normalize_sequence(row.get("cdr3a"))
    cdr3b = normalize_sequence(row.get("cdr3b"))
    mhc = normalize_sequence(row.get("mhc_seq"))
    if not is_valid_protein_sequence(epitope):
        return None
    if not is_valid_protein_sequence(cdr3a) and not is_valid_protein_sequence(cdr3b):
        return None
    chains: list[str] = []
    roles: list[str] = []
    if is_valid_protein_sequence(mhc):
        chains.append(mhc)
        roles.append("mhc")
    chains.append(epitope)
    roles.append("peptide")
    if is_valid_protein_sequence(cdr3a):
        chains.append(cdr3a)
        roles.append("tcr_alpha")
    if is_valid_protein_sequence(cdr3b):
        chains.append(cdr3b)
        roles.append("tcr_beta")
    return {
        "chains": chains,
        "roles": roles,
        "task_type": "tcr_pmhc" if is_valid_protein_sequence(mhc) else "tcr_epitope",
        "source": "trait",
        "relation": canonical_recognition_relation(row.get("relation")),
    }


def _cdr3_core_key(seq: Any) -> str:
    """Anchor-free CDR3 core key (drop conserved leading C + trailing F/W) so a
    full IMGT junction ``C..F`` and an ANARCI loop compare equal."""

    s = normalize_sequence(seq)
    if not s:
        return ""
    if len(s) >= 5:
        if s[0] == "C":
            s = s[1:]
        if s and s[-1] in "FW":
            s = s[:-1]
    return s


def tcr_native_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """Unified tiered tcr_native row -> recognition record.

    Prefers full-length Fv (``alpha_fv``/``beta_fv``); falls back to the CDR3
    loop (``cdr3a``/``cdr3b``) when Fv is absent (completeness-gate fallback).
    Emits ``[mhc?] peptide [tcr_alpha?] [tcr_beta?]`` with a canonical relation,
    matching the TRAIT/grammar recognition layout. Tier A/B (with MHC) render as
    ``tcr_pmhc``; tier C (no MHC) renders as ``tcr_epitope``.
    """

    epitope = normalize_sequence(row.get("epitope_seq"))
    mhc = normalize_sequence(row.get("mhc_seq"))
    alpha = normalize_sequence(row.get("alpha_fv")) or normalize_sequence(row.get("cdr3a"))
    beta = normalize_sequence(row.get("beta_fv")) or normalize_sequence(row.get("cdr3b"))
    if not is_valid_protein_sequence(epitope):
        return None
    if not is_valid_protein_sequence(alpha) and not is_valid_protein_sequence(beta):
        return None
    chains: list[str] = []
    roles: list[str] = []
    if is_valid_protein_sequence(mhc):
        chains.append(mhc)
        roles.append("mhc")
    chains.append(epitope)
    roles.append("peptide")
    if is_valid_protein_sequence(alpha):
        chains.append(alpha)
        roles.append("tcr_alpha")
    if is_valid_protein_sequence(beta):
        chains.append(beta)
        roles.append("tcr_beta")
    return {
        "chains": chains,
        "roles": roles,
        "task_type": "tcr_pmhc" if is_valid_protein_sequence(mhc) else "tcr_epitope",
        "source": "tcr_native",
        "relation": canonical_recognition_relation(row.get("relation")),
    }


def tcr_repertoire_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """Unlabeled single-chain CDR3b row -> bare generation record.

    No epitope and no MHC, so this renders as the ``tcr_single`` grammar layout
    -- the layout the T4 Setting-A unconditional benchmark decodes with, and the
    one the rest of the mix never produces (OTS ships paired full-length chains,
    which render as ``tcr_pair``).

    Decontamination already happened at corpus-build time
    (``scripts/data/tcr_native/build_repertoire.py``); there is no epitope here,
    so the ``(cdr3b_core|epitope)`` exclusion keys used by the other TCR sources
    do not apply and are not re-checked.
    """

    cdr3b = normalize_sequence(row.get("cdr3b"))
    if not is_valid_protein_sequence(cdr3b):
        return None
    return {
        "chains": [cdr3b],
        "roles": ["tcr_beta"],
        "task_type": "tcr",
        "source": "tcr_repertoire",
    }


# Explicit opt-out values for a blocklist argument. An *empty* string is NOT
# accepted: the 2026-08-27 audit found all four immune-fusion ymls passed
# ``--asd_antibody_benchmark_blocklist ""``, which the old silent-empty-set
# behaviour turned into a no-op, disabling antibody decontamination for a whole
# training run with no trace in the logs. Same failure mode silently voided the
# T4 generation filter (blocklist file created after training had started).
DISABLED_BLOCKLIST_SENTINELS = frozenset({"none", "off", "disabled", "-"})


def load_exclusion_keys(
    path: Path | str | None, *, allow_missing: bool = False
) -> set[str]:
    """Load a newline-delimited exclusion-key blocklist (``replaces_trait``).

    Fails loudly rather than silently returning an empty set: a blocklist that
    quietly does nothing is indistinguishable from a working one at training
    time. Pass one of :data:`DISABLED_BLOCKLIST_SENTINELS` (or ``None``) to
    disable a filter on purpose; use ``allow_missing=True`` only for smoke tests.
    """

    if path is None:
        return set()
    raw = str(path).strip()
    if not raw:
        raise ValueError(
            "Empty blocklist path. This used to silently disable the filter; "
            f"pass one of {sorted(DISABLED_BLOCKLIST_SENTINELS)} to disable it "
            "explicitly, or give a real path."
        )
    if raw.lower() in DISABLED_BLOCKLIST_SENTINELS:
        return set()
    p = Path(raw)
    if not p.is_file():
        if allow_missing:
            return set()
        raise FileNotFoundError(
            f"Blocklist not found: {p}. Generate it before training -- a missing "
            "blocklist used to be a silent no-op."
        )
    keys = {line.strip() for line in p.read_text().splitlines() if line.strip()}
    if not keys:
        raise ValueError(f"Blocklist {p} exists but contains no keys.")
    return keys


def trait_exclusion_key(row: dict[str, Any]) -> str:
    """Stateless (CDR3b-core | epitope) key used to drop TRAIT rows superseded by
    native full-length Fv. Must match the key emitted by the replaces_trait
    blocklist artifact."""

    return f"{_cdr3_core_key(row.get('cdr3b'))}|{normalize_sequence(row.get('epitope_seq'))}"


def tcr_native_exclusion_key(row: dict[str, Any]) -> str:
    """(CDR3b-core | epitope) key for the unified tier schema shared by
    ``tcr_native`` and ``tcr_papers``.

    Unlike :func:`trait_exclusion_key` this does *not* strip anchors: the
    unified ``cdr3b`` column already stores the anchor-free IMGT core (see
    ``scripts/data/tcr_native/ingest_repo.py``), so stripping again would
    corrupt cores that happen to start with C and end with F/W."""

    return f"{normalize_sequence(row.get('cdr3b'))}|{normalize_sequence(row.get('epitope_seq'))}"


def tcr_repertoire_exclusion_key(row: dict[str, Any]) -> str:
    """Bare CDR3b-core key for the unlabeled repertoire source.

    The epitope-conditioned blocklists are keyed ``(cdr3b_core|epitope)``, which
    can never match a row that has no epitope, so the *blocklist* has to be
    projected down to bare cores by the caller (see ``_core_only`` in
    ``build_repertoire.py`` and the projection in ``build_immune_specs``).
    Deliberately over-aggressive: it drops a benchmark CDR3b regardless of which
    epitope it was an answer for.

    This is defence in depth. ``build_repertoire.py`` already applied the same
    projection at corpus-build time, but the blocklists grow (the 2026-08-28
    audit added Setting B's answer key, +1,757 cores), and re-reading 40M rows
    to rebuild the corpus for a handful of new keys is not worth it when the
    filter can be applied statelessly at load time.
    """

    return normalize_sequence(row.get("cdr3b"))


def ots_exclusion_key(row: dict[str, Any]) -> str:
    """Stateless CDR3b key used to drop OTS rows that leak into the downstream
    TCR-binding benchmark. The OTS ``chain*_cdr3`` is already the ANARCI loop
    (anchor-free core), so the key is the normalized beta-chain CDR3 -- matching
    the ots_benchmark_blocklist entries emitted by ``decontam.py --mode ots``
    (which uses the identical extractor)."""

    key = ""
    for idx in ("1", "2"):
        if str(row.get(f"chain{idx}_anarci_type", "")).strip().upper() == "B":
            key = normalize_sequence(row.get(f"chain{idx}_cdr3"))
    return key


def with_exclusion_filter(
    row_to_record: Callable[[dict[str, Any]], dict[str, Any] | None],
    exclusion_keys: set[str],
    key_fn: Callable[[dict[str, Any]], str],
) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    """Wrap a ``row_to_record`` so rows whose ``key_fn`` is blocklisted are
    dropped (returns ``None``). Stateless: keyed purely on row content."""

    if not exclusion_keys:
        return row_to_record

    def _wrapped(row: dict[str, Any]) -> dict[str, Any] | None:
        record = row_to_record(row)
        if record is None:
            return None
        if key_fn(row) in exclusion_keys:
            return None
        return record

    return _wrapped


def with_exclusion_filter_multi(
    row_to_record: Callable[[dict[str, Any]], dict[str, Any] | None],
    exclusion_keys: set[str],
    keys_fn: Callable[[dict[str, Any]], Iterable[str]],
) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    """Like :func:`with_exclusion_filter` but ``keys_fn`` returns *several*
    prefixed keys per row (e.g. ``h3:``/``H:``); the row is dropped if
    ANY of them is blocklisted. Used for antibody benchmark decontamination
    (CDR-H3 core OR heavy chain; light-only matching is not used)."""

    if not exclusion_keys:
        return row_to_record

    def _wrapped(row: dict[str, Any]) -> dict[str, Any] | None:
        record = row_to_record(row)
        if record is None:
            return None
        for key in keys_fn(row):
            if key in exclusion_keys:
                return None
        return record

    return _wrapped


# Antibody/nanobody CDR-H3 junction (Cys ... [FW]-G-X-G at the FR4 boundary).
_AB_JUNCTION_RE = re.compile(r"C[A-Z]{2,32}?[FW]G[A-Z]G")


def _h3_core_from_chain(chain: Any) -> str:
    """Extract the anchor-free CDR-H3 core from a full heavy/VHH variable chain
    via the [FW]G-X-G motif (stateless; mirrors the derivation script)."""

    s = normalize_sequence(chain)
    best = ""
    for match in _AB_JUNCTION_RE.finditer(s):
        junction = match.group(0)[:-3]  # drop trailing G-X-G, keep C...[FW]
        if 5 <= len(junction) <= 35:
            best = junction
    return _cdr3_core_key(best) if best else ""


def oas_benchmark_keys(row: dict[str, Any]) -> list[str]:
    """Prefixed benchmark keys for an OAS paired row: CDR-H3 core (0.80) and
    heavy (0.95). Light-only matching is dropped (shared germline lights)."""

    if _is_oas_label_row(row):
        heavy = normalize_sequence(row.get("cleaned_h_sequence"))
        h3 = _cdr3_core_key(row.get("h_cdr3")) or _h3_core_from_chain(heavy)
    else:
        chain1 = normalize_sequence(row.get("cleaned_chain1_seq"))
        chain2 = normalize_sequence(row.get("cleaned_chain2_seq"))
        c1h = _is_heavy(str(row.get("chain1_anarci_type", "")), str(row.get("chain1_type", "")))
        c2h = _is_heavy(str(row.get("chain2_anarci_type", "")), str(row.get("chain2_type", "")))
        heavy = chain2 if (c2h and not c1h) else chain1
        h3 = _h3_core_from_chain(heavy)
    keys: list[str] = []
    if h3:
        keys.append(f"h3:{h3}")
    if heavy:
        keys.append(f"H:{heavy}")
    return keys


def asd_antibody_benchmark_keys(row: dict[str, Any]) -> list[str]:
    heavy = normalize_sequence(row.get("heavy_fv"))
    h3 = _h3_core_from_chain(heavy)
    keys: list[str] = []
    if h3:
        keys.append(f"h3:{h3}")
    if heavy:
        keys.append(f"H:{heavy}")
    return keys


def asd_nanobody_benchmark_keys(row: dict[str, Any]) -> list[str]:
    """Unused in the active mix (asd_nanobody dropped); kept for completeness."""

    vhh = normalize_sequence(row.get("vhh_fv"))
    h3 = _h3_core_from_chain(vhh)
    keys: list[str] = []
    if h3:
        keys.append(f"h3:{h3}")
    if vhh:
        keys.append(f"H:{vhh}")  # VHH is banked with heavy chains
    return keys


def trait_benchmark_key(row: dict[str, Any]) -> str:
    """Anchor-free CDR3b core for TRAIT rows (TCR binding-benchmark decontam)."""

    return _cdr3_core_key(row.get("cdr3b"))


def asd_nanobody_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """ASD nanobody<->antigen row -> recognition record (single VHH chain)."""

    antigen = normalize_sequence(row.get("antigen_seq"))
    vhh = normalize_sequence(row.get("vhh_fv"))
    if not is_valid_protein_sequence(antigen) or not is_valid_protein_sequence(vhh):
        return None
    return {
        "chains": [antigen, vhh],
        "roles": ["antigen", "nanobody_vhh"],
        "task_type": "nanobody_antigen",
        "source": "asd_nanobody",
        "relation": canonical_recognition_relation(row.get("relation")),
    }


def _iter_csv_records(
    path: Path,
    row_to_record: Callable[[dict[str, Any]], dict[str, Any] | None],
    max_rows: int | None,
    stats: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield records, optionally recording ``read``/``kept`` counts into ``stats``.

    ``read`` counts CSV rows consumed and ``kept`` the records that survived
    ``row_to_record`` (which is where the decontamination filters live), so a
    caller can log "raw -> kept" per source and see whether a blocklist actually
    did anything. Note that with ``max_rows`` set, iteration stops early, so
    ``read`` is a prefix count rather than the file's full length.
    """

    kept = 0
    read = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            read += 1
            record = row_to_record(row)
            if record is None:
                continue
            yield record
            kept += 1
            if max_rows is not None and kept >= max_rows:
                break
    if stats is not None:
        stats["read"] = read
        stats["kept"] = kept


def _reservoir_sample(
    stream: Iterator[dict[str, Any]], k: int, seed: int
) -> list[dict[str, Any]]:
    """Uniform sample of ``k`` items from ``stream`` in one pass (Algorithm R)."""

    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
            continue
        j = rng.randrange(i + 1)
        if j < k:
            reservoir[j] = item
    return reservoir


class ImmuneCsvDataset(Dataset):
    """In-memory map-style dataset built from one immune CSV source.

    ``sample_seed`` switches ``max_rows`` from "take the first N rows" to
    "uniformly sample N rows" (reservoir sampling, single pass, memory bounded by
    N). The prefix behaviour silently destroys composition when a CSV is sorted:
    ``tcr_native/valid.csv`` is ordered by its ``source`` column, so capping it at
    2,000 rows kept only ``piste`` + ``tcr_pmhc_fulllength`` and dropped
    ``minervina`` / ``tenx`` / ``covidvac`` (982 rows) from every ``eval_loss``
    ever computed -- while the log still reported "2000 rows". See
    downstream/benchmark/audit_2026_08_27/RETRAIN_PLAN.md §3a.
    """

    def __init__(
        self,
        spec: ImmuneSourceSpec,
        split: str = "train",
        max_rows: int | None = None,
        sample_seed: int | None = None,
    ) -> None:
        path = _source_split_path(spec, split)
        if not path.is_file():
            raise FileNotFoundError(f"Immune CSV not found: {path}")
        self.name = spec.name
        self.weight = spec.weight
        self.stats: dict[str, int] = {}
        sampled = max_rows is not None and sample_seed is not None
        stream = _iter_csv_records(
            path,
            spec.row_to_record,
            None if sampled else max_rows,
            self.stats,
        )
        if sampled:
            self.records = _reservoir_sample(stream, int(max_rows), int(sample_seed))
        else:
            self.records = list(stream)
        for record in self.records:
            record["weight"] = spec.weight
        if not self.records:
            raise ValueError(f"No valid rows loaded from {path}")
        read = self.stats.get("read", 0)
        kept = self.stats.get("kept", 0)
        dropped = read - kept
        if sampled:
            note = f" [sampled {len(self.records)}/{kept}, seed={sample_seed}]"
        elif max_rows is not None:
            note = f" [prefix-truncated at max_rows={max_rows}]"
        else:
            note = ""
        logger.info(
            "[%s/%s] rows %d -> %d filtered out %d (%.2f%%)%s",
            spec.name,
            split,
            read,
            kept,
            dropped,
            100.0 * dropped / read if read else 0.0,
            note,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def default_immune_specs(
    oas_dir: Path | str = OAS_DEFAULT_DIR,
    ots_dir: Path | str = OTS_DEFAULT_DIR,
    nanobody_dir: Path | str = NANOBODY_DEFAULT_DIR,
    oas_weight: float = 1.0,
    ots_weight: float = 1.0,
    nanobody_weight: float = 1.0,
) -> list[ImmuneSourceSpec]:
    return [
        ImmuneSourceSpec("oas", Path(oas_dir), oas_paired_row_to_record, "antibody", oas_weight),
        ImmuneSourceSpec("ots", Path(ots_dir), ots_paired_row_to_record, "tcr", ots_weight),
        ImmuneSourceSpec("nanobody", Path(nanobody_dir), nanobody_row_to_record, "antibody", nanobody_weight),
    ]


def build_mixed_immune_dataset(
    specs: list[ImmuneSourceSpec] | None = None,
    split: str = "train",
    max_rows_per_source: int | None = None,
    sample_seed: int | None = None,
) -> tuple[ConcatDataset, dict[str, int], list[ImmuneCsvDataset]]:
    """Build a concatenated dataset over the immune sources.

    ``sample_seed`` makes ``max_rows_per_source`` a uniform random sample instead
    of a prefix (see :class:`ImmuneCsvDataset`).

    Returns the ``ConcatDataset``, a ``{source_name: num_rows}`` summary, and the
    per-source datasets (so callers can evaluate each source separately).
    """

    specs = specs or default_immune_specs()
    datasets: list[ImmuneCsvDataset] = []
    counts: dict[str, int] = {}
    for spec in specs:
        dataset = ImmuneCsvDataset(
            spec,
            split=split,
            max_rows=max_rows_per_source,
            sample_seed=sample_seed,
        )
        datasets.append(dataset)
        counts[spec.name] = len(dataset)
    return ConcatDataset(datasets), counts, datasets
