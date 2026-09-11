"""Adapt the seven active immune CSV sources into canonical ``BioSeqRecord`` objects.

Import ``row_to_record`` and pass a source name plus a CSV row.  Invalid schema or
sequences return ``None``; unknown non-empty recognition relations raise.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .records import (
    BioSeqChain,
    BioSeqRecord,
    compact_metadata,
    is_valid_protein_sequence,
    normalize_sequence,
)
from .profiles import default_region_profile, sample_region_lengths

_RELATION_POSITIVE = {"binding", "binder", "positive", "pos", "true", "1", "yes"}
_RELATION_NEGATIVE = {
    "nonbinding", "non_binding", "nonbinder", "non_binder", "negative", "neg",
    "false", "0", "no",
}
_RELATION_UNKNOWN = {"unknown", "unk", "na", "n/a", "none", "null"}
_REGION_ORDER = ("FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4")
_AB_JUNCTION_RE = re.compile(r"C[A-Z]{2,32}?[FW]G[A-Z]G")

#: Sources whose receptor chains are completed from a region-length profile.
#: They receive ``profile``/``row_index`` from ``row_to_record``; passing no
#: profile keeps the raw chains, which is what the profile-learning pass needs.
COMPLETION_SOURCES = ("trait", "tcr_native", "tcr_papers", "tcr_repertoire")


def canonical_recognition_relation(value: Any, *, default: str = "binding") -> str:
    raw = str(value if value is not None else "").strip().lower().replace(" ", "_").replace("-", "_")
    if not raw:
        return default
    if raw in _RELATION_POSITIVE:
        return "binding"
    if raw in _RELATION_NEGATIVE:
        return "nonbinding"
    if raw in _RELATION_UNKNOWN:
        return "unknown"
    raise ValueError(
        f"Unrecognized recognition relation {value!r}; expected binding/nonbinding "
        "or a known synonym"
    )


def _is_heavy(row: dict[str, Any], prefix: str) -> bool:
    return (
        str(row.get(f"{prefix}_anarci_type", "")).strip().upper() == "H"
        or str(row.get(f"{prefix}_type", "")).strip().lower() == "heavy"
    )


def _is_beta(row: dict[str, Any], prefix: str) -> bool:
    return (
        str(row.get(f"{prefix}_anarci_type", "")).strip().upper() == "B"
        or str(row.get(f"{prefix}_type", "")).strip().lower() in {"beta", "trb"}
    )


def _relation_is_supervised(row: dict[str, Any]) -> bool:
    value = row.get("relation")
    if value is None or not str(value).strip():
        return False
    return canonical_recognition_relation(value, default="unknown") in {"binding", "nonbinding"}


def _regions(row: dict[str, Any], prefix: str) -> dict[str, str]:
    return {
        region: normalize_sequence(row.get(f"{prefix}_{region}"))
        for region in _REGION_ORDER
        if normalize_sequence(row.get(f"{prefix}_{region}"))
    }


def _oas_regions(row: dict[str, Any], prefix: str) -> dict[str, str]:
    fields = {"FR1": "fwr1", "CDR1": "cdr1", "FR2": "fwr2", "CDR2": "cdr2", "FR3": "fwr3", "CDR3": "cdr3", "FR4": "fwr4"}
    return {
        region: normalize_sequence(row.get(f"{prefix}_{field}"))
        for region, field in fields.items()
        if normalize_sequence(row.get(f"{prefix}_{field}"))
    }


def _chain_metadata(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return compact_metadata(row, (
        f"{prefix}_anarci_type", f"{prefix}_type", f"{prefix}_v",
        f"{prefix}_j", f"{prefix}_cluster", f"{prefix}_cdr3",
    ))


def _oas_chain_metadata(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    keys = ("h_v_call", "h_d_call", "h_j_call", "H_cluster_id", "h_region_labels") if prefix == "h" else ("l_locus", "l_v_call", "l_j_call", "L_cluster_id", "l_region_labels")
    return compact_metadata(row, keys)


def _split(row: dict[str, Any], split: str | None) -> str | None:
    return str(row.get("split") or split or "").strip() or None


def _cdr3_core_key(value: Any) -> str:
    sequence = normalize_sequence(value)
    if len(sequence) >= 5:
        sequence = sequence.removeprefix("C")
        if sequence.endswith(("F", "W")):
            sequence = sequence[:-1]
    return sequence


def _cdr3b_is_full_junction(row: dict[str, Any]) -> bool:
    """Is this row's ``cdr3b`` a full IMGT junction (``C...F/W``) or an anchor-free core?

    Decided from provenance, never from the sequence's own first/last character:
    an anchor-free core can legitimately start with C or end with F/W (e.g.
    ``AAAATGAGEQF``), so a character heuristic silently eats a real residue on
    core-form corpora. ``data/tcr_repertoire_junc80`` writes full junctions and
    tags them ``fv_source=tcrdesign2026_cdr3_junc80``; the older
    ``data/tcr_repertoire`` build stores stripped cores as ``tcrdesign2026_cdr3``.
    """
    markers = (
        str(row.get("fv_source") or ""),
        str(row.get("provenance") or ""),
        str(row.get("schema_version") or ""),
    )
    return any("junc" in marker.lower() for marker in markers)


def _split_cdr3_anchors(value: Any, *, is_junction: bool) -> tuple[str, str, str]:
    """Return ``(core, leading-C, trailing-F/W)`` using IMGT anchor semantics.

    The CDR3 loop excludes the conserved FR3-terminal C and FR4-initial F/W, so
    those anchors are peeled off and re-placed into the framework regions. When
    the input is already an anchor-free core (``is_junction=False``) it is passed
    through untouched -- no anchors are invented and no residue is removed.
    """
    sequence = normalize_sequence(value)
    if not sequence:
        return "", "", ""
    if not is_junction:
        return sequence, "", ""
    leading = "C" if len(sequence) >= 5 and sequence.startswith("C") else ""
    trailing = sequence[-1] if len(sequence) >= 5 and sequence[-1] in {"F", "W"} else ""
    start = len(leading)
    end = len(sequence) - len(trailing) if trailing else len(sequence)
    return sequence[start:end], leading, trailing


def _is_placeholder_sequence(sequence: str) -> bool:
    """Is this an all-``X`` upstream placeholder rather than a real sequence?

    ``tcrdesign26_pmhc`` (Zenodo 14545852) writes ``X`` for missing values per its
    own README, so a context chain can arrive as ``X`` * 34 for "allele unknown"
    or ``X`` * 21 for "epitope unknown". Those are holes, not sequences, and the
    residue alphabet check cannot catch them because ``X`` is a legal residue.
    """
    return bool(sequence) and set(sequence) == {"X"}


def is_fully_synthesized_chain(chain: BioSeqChain) -> bool:
    """True when this chain was synthesized outright (no observed residues).

    Completed chains carry ``receptor_completion`` or ``beta_only_completion``
    plus ``synthetic_regions`` and a ``synthetic_residue_mask``. A fully
    synthesized partner — the missing chain on an alpha-only or beta-only row —
    has an all-1 mask. A CDR3-scaffolded chain has 0s on the observed loop (and
    any peeled anchors) and is not counted here.
    """
    metadata = chain.metadata
    if not (metadata.get("receptor_completion") or metadata.get("beta_only_completion")):
        return False
    if not metadata.get("synthetic_regions"):
        return False
    mask = metadata.get("synthetic_residue_mask")
    return isinstance(mask, list) and bool(mask) and all(int(flag) == 1 for flag in mask)


def partner_completion_flags(record: BioSeqRecord) -> tuple[bool, bool]:
    """Return ``(beta_only_completed, alpha_only_completed)`` for a kept record.

    ``beta_only_completed`` means a fully synthesized alpha was added (the
    observed chain was beta). ``alpha_only_completed`` means a fully
    synthesized beta was added. A row with both CDR3s or both Fvs has neither.
    """
    beta_only = False
    alpha_only = False
    for chain in record.chains:
        if not is_fully_synthesized_chain(chain):
            continue
        if chain.role == "tcr_alpha":
            beta_only = True
        elif chain.role == "tcr_beta":
            alpha_only = True
    return beta_only, alpha_only


def row_stripped_placeholder_mhc(row: Mapping[str, Any], record: BioSeqRecord) -> bool:
    """True when the raw row carried an all-``X`` MHC that was then stripped.

    Empty ``mhc_seq`` is not a downgrade — there was never an allele to drop.
    The live ``downgraded_all_x_mhc`` counter uses this check. Prepared JSONL
    cannot recover the raw MHC, so a post-hoc recount has to use
    ``is_downgraded_mhc_layout`` and may include never-had-MHC rows.
    """
    mhc = normalize_sequence(row.get("mhc_seq"))
    return _is_placeholder_sequence(mhc) and "mhc" not in record.chain_roles


def is_downgraded_mhc_layout(record: BioSeqRecord) -> bool:
    """Prepared-row stand-in for ``downgraded_all_x_mhc``.

    A ``tcr_pmhc``-eligible peptide with no MHC chain. This is what can be
    recovered from published JSONL without the raw CSV.
    """
    roles = record.chain_roles
    return "peptide" in roles and "mhc" not in roles


def complete_tcr_chain(
    role: str,
    *,
    profile: dict[str, Any],
    source: str,
    row_index: int,
    core: str = "",
    leading_anchor: str = "",
    trailing_anchor: str = "",
    extra_metadata: dict[str, Any] | None = None,
) -> BioSeqChain:
    """Build a full-length TCR chain scaffold around an observed CDR3 ``core``.

    Region lengths are drawn from ``profile`` deterministically per
    ``(seed, source, row_index)``. Every synthesized position is ``X`` and is
    reported in ``metadata['synthetic_residue_mask']`` so the grammar keeps it out
    of both the diffusion loss and the corruption. An empty ``core`` synthesizes
    the whole chain, which is how a missing partner chain is represented: it
    costs no supervision but gives the receptor a stable two-chain layout and
    lets ``position_ids_chain`` distinguish alpha from beta, which a single-chain
    block cannot (there is no ``<tcra>``/``<tcrb>`` marker).
    """
    lengths = sample_region_lengths(
        profile, role, seed=int(profile.get("seed", 42)), source=source, row_index=row_index,
    )
    regions = {region: "X" * length for region, length in lengths.items()}
    observed_anchor_regions: dict[str, str] = {}
    if core:
        # IMGT places the conserved C at the end of FR3 and the F/W at the start
        # of FR4; the loop itself carries neither. Anchors overwrite the last/first
        # synthetic position instead of extending the region, so the sampled
        # region length is preserved exactly.
        if leading_anchor:
            if len(regions["FR3"]) < 1:
                raise ValueError(f"{role} FR3 profile must have room for the leading C anchor")
            regions["FR3"] = regions["FR3"][:-1] + leading_anchor
            observed_anchor_regions["FR3"] = leading_anchor
        if trailing_anchor:
            if len(regions["FR4"]) < 1:
                raise ValueError(f"{role} FR4 profile must have room for the trailing F/W anchor")
            regions["FR4"] = trailing_anchor + regions["FR4"][1:]
            observed_anchor_regions["FR4"] = trailing_anchor
        regions["CDR3"] = core
    sequence = "".join(regions[region] for region in _REGION_ORDER)
    synthetic_mask = [1] * len(sequence)
    cursor = 0
    for region in _REGION_ORDER:
        value = regions[region]
        if core and region == "CDR3":
            for index in range(cursor, cursor + len(value)):
                synthetic_mask[index] = 0
        if region in observed_anchor_regions:
            synthetic_mask[cursor if region == "FR4" else cursor + len(value) - 1] = 0
        cursor += len(value)
    observed_regions = ["CDR3"] if core else []
    # ``extra_metadata`` leads so callers can keep their own completion marker as
    # the first key and reproduce previously published records byte for byte.
    metadata: dict[str, Any] = {
        **(extra_metadata or {}),
        "synthetic_regions": [region for region in _REGION_ORDER if region not in observed_regions],
        "observed_regions": observed_regions,
        "observed_anchor_regions": observed_anchor_regions,
        "synthetic_residue_mask": synthetic_mask,
        "completion_profile_seed": int(profile.get("seed", 42)),
    }
    return BioSeqChain(sequence, role, regions, metadata)


def _receptor_chains(
    *,
    alpha_fv: str,
    beta_fv: str,
    alpha_cdr3: str,
    beta_cdr3: str,
    is_junction: bool,
    profile: dict[str, Any] | None,
    source: str,
    row_index: int,
) -> list[BioSeqChain]:
    """Return receptor chains beta-first, completed to full length when possible.

    Beta precedes alpha to match ``_ots`` (and ``_oas`` heavy-first): the
    chain-conditioned timestep sampler treats the smallest chain id as the
    heavy/beta chain.

    Without a ``profile`` the chains are passed through exactly as given, which
    keeps direct adapter callers and the profile-learning pass free of synthesized
    scaffolding. With a profile, a real full-length Fv is kept verbatim -- we have
    no IMGT numbering for it, so no region annotation is invented -- while a bare
    CDR3 is expanded and a missing chain is synthesized outright.
    """
    if profile is None:
        chains: list[BioSeqChain] = []
        for role, value in (("tcr_beta", beta_fv or beta_cdr3), ("tcr_alpha", alpha_fv or alpha_cdr3)):
            if is_valid_protein_sequence(value):
                chains.append(BioSeqChain(value, role))
        return chains
    completed: list[BioSeqChain] = []
    for role, full_length, cdr3 in (
        ("tcr_beta", beta_fv, beta_cdr3),
        ("tcr_alpha", alpha_fv, alpha_cdr3),
    ):
        if is_valid_protein_sequence(full_length):
            completed.append(BioSeqChain(full_length, role))
            continue
        core, leading, trailing = _split_cdr3_anchors(cdr3, is_junction=is_junction)
        completed.append(
            complete_tcr_chain(
                role,
                profile=profile,
                source=source,
                row_index=row_index,
                core=core,
                leading_anchor=leading,
                trailing_anchor=trailing,
                extra_metadata={"receptor_completion": True},
            )
        )
    return completed


def _h3_core_from_chain(value: Any) -> str:
    best = ""
    for match in _AB_JUNCTION_RE.finditer(normalize_sequence(value)):
        junction = match.group(0)[:-3]
        if 5 <= len(junction) <= 35:
            best = junction
    return _cdr3_core_key(best) if best else ""


def _pair_identifiers(heavy: str, h3: str = "") -> dict[str, str]:
    identifiers = {"heavy_sequence": heavy}
    if h3:
        identifiers["heavy_h3_core"] = h3
        identifiers["benchmark_h3_key"] = f"h3:{h3}"
    identifiers["benchmark_heavy_key"] = f"H:{heavy}"
    return identifiers


def _recognition_identifiers(row: dict[str, Any], *, strip_anchors: bool) -> dict[str, str]:
    cdr3b = _cdr3_core_key(row.get("cdr3b")) if strip_anchors else normalize_sequence(row.get("cdr3b"))
    epitope = normalize_sequence(row.get("epitope_seq"))
    identifiers: dict[str, str] = {}
    if cdr3b:
        identifiers["cdr3b_core"] = cdr3b
    if epitope:
        identifiers["epitope"] = epitope
    if cdr3b or epitope:
        pair_key = f"{cdr3b}|{epitope}"
        identifiers["pair_key"] = pair_key
        identifiers["cdr3b_core|epitope"] = pair_key
    return identifiers


def _oas(row: dict[str, Any], split: str | None, weight: float) -> BioSeqRecord | None:
    if "cleaned_h_sequence" in row and "cleaned_l_sequence" in row:
        heavy = normalize_sequence(row.get("cleaned_h_sequence"))
        light = normalize_sequence(row.get("cleaned_l_sequence"))
        if not is_valid_protein_sequence(heavy) or not is_valid_protein_sequence(light):
            return None
        light_role = "antibody_heavy" if str(row.get("l_locus", "")).strip().upper() == "H" else "antibody_light"
        h3 = _cdr3_core_key(row.get("h_cdr3")) or _h3_core_from_chain(heavy)
        identifiers = _pair_identifiers(heavy, h3)
        identifiers["source_row_id"] = str(row.get("sequence_id") or row.get("id") or row.get("pair_id") or "")
        return BioSeqRecord(
            [BioSeqChain(heavy, "antibody_heavy", _oas_regions(row, "h"), _oas_chain_metadata(row, "h")), BioSeqChain(light, light_role, _oas_regions(row, "l"), _oas_chain_metadata(row, "l"))],
            "antibody", "oas_paired", _split(row, split),
            compact_metadata(row, ("species", "source", "l_locus", "ab_cluster_key", "ab_cluster_id", "ab_cluster_id_counts")),
            weight=weight, identifiers={k: v for k, v in identifiers.items() if v},
        )
    chain1 = normalize_sequence(row.get("cleaned_chain1_seq"))
    chain2 = normalize_sequence(row.get("cleaned_chain2_seq"))
    if not is_valid_protein_sequence(chain1) or not is_valid_protein_sequence(chain2):
        return None
    first = BioSeqChain(chain1, "antibody_heavy" if _is_heavy(row, "chain1") else "antibody_light", _regions(row, "chain1"), _chain_metadata(row, "chain1"))
    second = BioSeqChain(chain2, "antibody_heavy" if _is_heavy(row, "chain2") else "antibody_light", _regions(row, "chain2"), _chain_metadata(row, "chain2"))
    chains = [second, first] if second.role == "antibody_heavy" and first.role != "antibody_heavy" else [first, second]
    heavy_chain = next((chain for chain in chains if chain.role == "antibody_heavy"), chains[0])
    heavy = heavy_chain.sequence
    return BioSeqRecord(chains, "antibody", "oas_paired", _split(row, split), compact_metadata(row, ("species", "data_type", "source_file", "pair_cluster", "cluster_id")), weight=weight, identifiers=_pair_identifiers(heavy, _h3_core_from_chain(heavy)))


def _ots(row: dict[str, Any], split: str | None, weight: float) -> BioSeqRecord | None:
    chain1, chain2 = normalize_sequence(row.get("cleaned_chain1_seq")), normalize_sequence(row.get("cleaned_chain2_seq"))
    if not is_valid_protein_sequence(chain1) or not is_valid_protein_sequence(chain2):
        return None
    first = BioSeqChain(chain1, "tcr_beta" if _is_beta(row, "chain1") else "tcr_alpha", _regions(row, "chain1"), _chain_metadata(row, "chain1"))
    second = BioSeqChain(chain2, "tcr_beta" if _is_beta(row, "chain2") else "tcr_alpha", _regions(row, "chain2"), _chain_metadata(row, "chain2"))
    chains = [second, first] if second.role == "tcr_beta" and first.role != "tcr_beta" else [first, second]
    beta_key = ""
    for index in ("1", "2"):
        if str(row.get(f"chain{index}_anarci_type", "")).strip().upper() == "B":
            beta_key = normalize_sequence(row.get(f"chain{index}_cdr3"))
    identifiers = {"cdr3b_core": beta_key, "benchmark_key": beta_key}
    return BioSeqRecord(chains, "tcr", "ots_paired", _split(row, split), compact_metadata(row, ("species", "data_type", "source_file", "pair_cluster", "cluster_id")), weight=weight, identifiers={k: v for k, v in identifiers.items() if v})


def _nanobody(row: dict[str, Any], split: str | None, weight: float) -> BioSeqRecord | None:
    sequence = normalize_sequence(row.get("cleaned_seq") or row.get("vhh_seq"))
    if not is_valid_protein_sequence(sequence):
        return None
    return BioSeqRecord([BioSeqChain(sequence, "nanobody_vhh", {region: normalize_sequence(row.get(region)) for region in _REGION_ORDER if normalize_sequence(row.get(region))}, compact_metadata(row, ("source", "anarci_chain_type", "cluster_id")))], "antibody", "nanobody", _split(row, split), compact_metadata(row, ("source", "cluster_id")), weight=weight, identifiers={"heavy_sequence": sequence, "heavy_h3_core": _h3_core_from_chain(sequence), "benchmark_heavy_key": f"H:{sequence}"})


def _asd_antibody(row: dict[str, Any], split: str | None, weight: float) -> BioSeqRecord | None:
    antigen, heavy, light = normalize_sequence(row.get("antigen_seq")), normalize_sequence(row.get("heavy_fv")), normalize_sequence(row.get("light_fv"))
    if not is_valid_protein_sequence(antigen) or not is_valid_protein_sequence(heavy):
        return None
    chains = [BioSeqChain(antigen, "antigen"), BioSeqChain(heavy, "antibody_heavy")]
    if is_valid_protein_sequence(light):
        chains.append(BioSeqChain(light, "antibody_light"))
    h3 = _h3_core_from_chain(heavy)
    return BioSeqRecord(chains, "antibody_antigen", "asd_antibody", _split(row, split), metadata={"relation_supervised": _relation_is_supervised(row)}, labels={"relation": canonical_recognition_relation(row.get("relation"))}, weight=weight, identifiers={k: v for k, v in {"heavy_sequence": heavy, "heavy_h3_core": h3, "benchmark_h3_key": f"h3:{h3}" if h3 else "", "benchmark_heavy_key": f"H:{heavy}"}.items() if v})


def _context_chain(sequence: str, role: str) -> BioSeqChain | None:
    """Wrap a context sequence, dropping all-``X`` upstream placeholders.

    An all-``X`` MHC is "allele unknown", not a pseudo-sequence: conditioning on
    34 ``X`` teaches a fake restriction and the value cannot be recovered (the
    ``mhc_allele_norm`` column is empty on exactly those rows). Dropping the chain
    degrades the record to the existing ``tcr_peptide`` layout, which keeps the
    epitope -- the dominant conditioning signal -- fully intact.
    """
    if not is_valid_protein_sequence(sequence) or _is_placeholder_sequence(sequence):
        return None
    return BioSeqChain(sequence, role)


def _trait(
    row: dict[str, Any],
    split: str | None,
    weight: float,
    *,
    profile: dict[str, Any] | None = None,
    row_index: int = 0,
) -> BioSeqRecord | None:
    epitope, alpha, beta, mhc = (normalize_sequence(row.get(key)) for key in ("epitope_seq", "cdr3a", "cdr3b", "mhc_seq"))
    if not is_valid_protein_sequence(epitope) or (not is_valid_protein_sequence(alpha) and not is_valid_protein_sequence(beta)):
        return None
    chains: list[BioSeqChain] = []
    mhc_chain = _context_chain(mhc, "mhc")
    if mhc_chain is not None: chains.append(mhc_chain)
    chains.append(BioSeqChain(epitope, "peptide"))
    # ``downstream/trait/step4_final`` stores full IMGT junctions (C...F/W) in
    # cdr3a/cdr3b, unlike the anchor-free cores in tcr_native/tcr_papers. Without
    # this the C would land both at the end of FR3 and at the head of CDR3.
    chains.extend(_receptor_chains(
        alpha_fv="", beta_fv="", alpha_cdr3=alpha, beta_cdr3=beta,
        is_junction=True, profile=profile, source="trait", row_index=row_index,
    ))
    metadata: dict[str, Any] = {"relation_supervised": _relation_is_supervised(row)}
    if profile is not None: metadata["receptor_completion"] = True
    return BioSeqRecord(chains, "tcr_pmhc" if mhc_chain is not None else "tcr_epitope", "trait", _split(row, split), metadata, labels={"relation": canonical_recognition_relation(row.get("relation"))}, weight=weight, identifiers=_recognition_identifiers(row, strip_anchors=True))


def _native(
    row: dict[str, Any],
    split: str | None,
    weight: float,
    source: str = "tcr_native",
    *,
    profile: dict[str, Any] | None = None,
    row_index: int = 0,
) -> BioSeqRecord | None:
    epitope, mhc = normalize_sequence(row.get("epitope_seq")), normalize_sequence(row.get("mhc_seq"))
    alpha_fv, beta_fv = normalize_sequence(row.get("alpha_fv")), normalize_sequence(row.get("beta_fv"))
    alpha_cdr3, beta_cdr3 = normalize_sequence(row.get("cdr3a")), normalize_sequence(row.get("cdr3b"))
    alpha, beta = alpha_fv or alpha_cdr3, beta_fv or beta_cdr3
    if not is_valid_protein_sequence(epitope) or (not is_valid_protein_sequence(alpha) and not is_valid_protein_sequence(beta)):
        return None
    chains: list[BioSeqChain] = []
    mhc_chain = _context_chain(mhc, "mhc")
    if mhc_chain is not None: chains.append(mhc_chain)
    chains.append(BioSeqChain(epitope, "peptide"))
    if profile is None:
        # Preserve the historical alpha-then-beta order and raw region columns for
        # direct adapter callers and the profile-learning pass.
        if is_valid_protein_sequence(alpha):
            chains.append(BioSeqChain(alpha, "tcr_alpha", _regions(row, "alpha")))
        if is_valid_protein_sequence(beta):
            chains.append(BioSeqChain(beta, "tcr_beta", _regions(row, "beta")))
    else:
        # cdr3a/cdr3b here are anchor-free cores, so no anchor is peeled off; a
        # populated alpha_fv/beta_fv is a real Fv and is kept verbatim.
        chains.extend(_receptor_chains(
            alpha_fv=alpha_fv, beta_fv=beta_fv, alpha_cdr3=alpha_cdr3, beta_cdr3=beta_cdr3,
            is_junction=False, profile=profile, source=source, row_index=row_index,
        ))
    metadata = compact_metadata(row, ("source", "tier", "dataset", "paper", "source_file"))
    if source != "tcr_native": metadata["dataset_source"] = source
    metadata["relation_supervised"] = _relation_is_supervised(row)
    if profile is not None: metadata["receptor_completion"] = True
    return BioSeqRecord(chains, "tcr_pmhc" if mhc_chain is not None else "tcr_epitope", "tcr_native", _split(row, split), metadata, {"relation": canonical_recognition_relation(row.get("relation"))}, weight, _recognition_identifiers(row, strip_anchors=False))


def _repertoire(
    row: dict[str, Any],
    split: str | None,
    weight: float,
    *,
    profile: dict[str, Any] | None = None,
    row_index: int = 0,
) -> BioSeqRecord | None:
    beta = normalize_sequence(row.get("cdr3b"))
    if not is_valid_protein_sequence(beta):
        return None
    # The stored value may be a full junction (C...F/W) or an anchor-free core;
    # provenance decides which, never the sequence's own first/last character.
    # The model layout follows IMGT: C belongs to FR3, F/W belongs to FR4, and
    # only the middle loop is CDR3. Blocklist keys remain anchor-free cores.
    beta_core, leading_anchor, trailing_anchor = _split_cdr3_anchors(
        beta, is_junction=_cdr3b_is_full_junction(row)
    )
    core = beta_core
    if profile is None:
        # Direct adapter callers retain a shape-compatible fallback. The offline
        # preprocessing pipeline always supplies a profile learned from current
        # complete TCR rows and records its digest in the manifest.
        profile = default_region_profile()

    def completed_chain(role: str, observed_core: str = "") -> BioSeqChain:
        return complete_tcr_chain(
            role,
            profile=profile,
            source="tcr_repertoire",
            row_index=row_index,
            core=observed_core,
            leading_anchor=leading_anchor if observed_core else "",
            trailing_anchor=trailing_anchor if observed_core else "",
            extra_metadata={"beta_only_completion": True},
        )

    alpha = completed_chain("tcr_alpha")
    completed_beta = completed_chain("tcr_beta", beta_core)
    relation_supervised = _relation_is_supervised(row)
    labels = {"relation": canonical_recognition_relation(row.get("relation"))}
    return BioSeqRecord(
        # Beta first, matching _ots (and _oas heavy-first): the chain-conditioned
        # timestep sampler treats the smallest chain id as the heavy/beta chain.
        [completed_beta, alpha],
        "tcr",
        "tcr_repertoire",
        _split(row, split),
        metadata={"beta_only_completion": True, "relation_supervised": relation_supervised},
        labels=labels,
        weight=weight,
        identifiers={"cdr3b_core": core, "benchmark_key": core},
    )


def _asd_nanobody(row: dict[str, Any], split: str | None, weight: float) -> BioSeqRecord | None:
    antigen, vhh = normalize_sequence(row.get("antigen_seq")), normalize_sequence(row.get("vhh_fv"))
    if not is_valid_protein_sequence(antigen) or not is_valid_protein_sequence(vhh):
        return None
    h3 = _h3_core_from_chain(vhh)
    return BioSeqRecord([BioSeqChain(antigen, "antigen"), BioSeqChain(vhh, "nanobody_vhh")], "nanobody_antigen", "asd_nanobody", _split(row, split), labels={"relation": canonical_recognition_relation(row.get("relation"))}, weight=weight, identifiers={k: v for k, v in {"heavy_sequence": vhh, "heavy_h3_core": h3, "benchmark_h3_key": f"h3:{h3}" if h3 else "", "benchmark_heavy_key": f"H:{vhh}"}.items() if v})


def row_to_record(
    source: str,
    row: dict[str, Any],
    *,
    split: str | None = None,
    weight: float = 1.0,
    profile: dict[str, Any] | None = None,
    row_index: int = 0,
) -> BioSeqRecord | None:
    """Dispatch one raw row; schema-invalid rows return ``None``."""
    adapters = {"oas": _oas, "ots": _ots, "nanobody": _nanobody, "asd_antibody": _asd_antibody, "asd_nanobody": _asd_nanobody, "trait": _trait, "tcr_native": _native, "tcr_papers": lambda r, s, w, **kw: _native(r, s, w, "tcr_papers", **kw), "tcr_repertoire": _repertoire}
    try:
        adapter = adapters[source]
    except KeyError as exc:
        raise ValueError(f"Unknown immune source {source!r}; known: {sorted(adapters)}") from exc
    if source in COMPLETION_SOURCES:
        return adapter(row, split, float(weight), profile=profile, row_index=row_index)
    return adapter(row, split, float(weight))


# Public named adapters are useful to callers migrating one source at a time.
def oas_row_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    return row_to_record("oas", row, split=split, weight=weight)


def ots_row_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    return row_to_record("ots", row, split=split, weight=weight)


def nanobody_row_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    return row_to_record("nanobody", row, split=split, weight=weight)


def asd_antibody_row_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    return row_to_record("asd_antibody", row, split=split, weight=weight)


def asd_nanobody_row_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    return row_to_record("asd_nanobody", row, split=split, weight=weight)


def trait_row_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    return row_to_record("trait", row, split=split, weight=weight)


def tcr_native_row_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    return row_to_record("tcr_native", row, split=split, weight=weight)


def tcr_papers_row_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    return row_to_record("tcr_papers", row, split=split, weight=weight)


def tcr_repertoire_row_to_record(row: dict[str, Any], split: str | None = None, weight: float = 1.0) -> BioSeqRecord | None:
    return row_to_record("tcr_repertoire", row, split=split, weight=weight)


__all__ = [
    "COMPLETION_SOURCES", "asd_antibody_row_to_record", "asd_nanobody_row_to_record",
    "canonical_recognition_relation", "complete_tcr_chain",
    "is_downgraded_mhc_layout", "is_fully_synthesized_chain",
    "nanobody_row_to_record", "oas_row_to_record", "ots_row_to_record",
    "partner_completion_flags", "row_stripped_placeholder_mhc", "row_to_record",
    "tcr_native_row_to_record", "tcr_papers_row_to_record",
    "tcr_repertoire_row_to_record", "trait_row_to_record",
]
