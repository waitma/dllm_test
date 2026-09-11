"""Region-length profiles used to complete beta-only TCR records."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable
from typing import Any

from .records import BioSeqRecord, REGION_ORDER

TCR_REGION_ORDER = REGION_ORDER


def _stable_seed(seed: int, source: str, row_index: int) -> int:
    payload = f"{seed}:{source}:{row_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def empty_tcr_region_lengths() -> dict[str, dict[str, Counter[int]]]:
    return {
        "tcr_alpha": {region: Counter() for region in TCR_REGION_ORDER},
        "tcr_beta": {region: Counter() for region in TCR_REGION_ORDER},
    }


def add_tcr_region_lengths(
    counts: dict[str, dict[str, Counter[int]]],
    record: BioSeqRecord,
) -> None:
    """Accumulate observed region lengths, ignoring synthesized scaffolding.

    Completed chains carry ``metadata['synthetic_regions']`` for every region the
    profile itself generated. Counting those would feed the profile back into
    itself and collapse it toward whatever it already emits, so they are skipped
    and only genuinely observed regions contribute.
    """
    for chain in record.chains:
        role = chain.role.lower()
        if role not in counts:
            continue
        synthetic = set(chain.metadata.get("synthetic_regions") or ())
        for region in TCR_REGION_ORDER:
            if region in synthetic:
                continue
            value = chain.regions.get(region, "")
            if value:
                counts[role][region][len(value)] += 1


def collect_tcr_region_lengths(records: Iterable[BioSeqRecord]) -> dict[str, dict[str, Counter[int]]]:
    result = empty_tcr_region_lengths()
    for record in records:
        add_tcr_region_lengths(result, record)
    return result


_FALLBACK_REGION_LENGTHS = {
    "tcr_alpha": {"FR1": 25, "CDR1": 6, "FR2": 15, "CDR2": 6, "FR3": 38, "CDR3": 13, "FR4": 14},
    "tcr_beta": {"FR1": 25, "CDR1": 6, "FR2": 15, "CDR2": 6, "FR3": 38, "CDR3": 13, "FR4": 14},
}


def finalize_region_profile(
    counts: dict[str, dict[str, Counter[int]]],
    *,
    seed: int,
    sources: list[str],
    source_policy: str = "current_complete_tcr_rows",
    observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    distributions: dict[str, dict[str, dict[str, int]]] = {}
    fallback_regions: dict[str, list[str]] = {}
    for role, regions in counts.items():
        distributions[role] = {}
        fallback_regions[role] = []
        for region in TCR_REGION_ORDER:
            counter = regions.get(region, Counter())
            if counter:
                distributions[role][region] = {
                    str(length): int(count) for length, count in sorted(counter.items())
                }
            else:
                distributions[role][region] = {str(_FALLBACK_REGION_LENGTHS[role][region]): 1}
                fallback_regions[role].append(region)
    return {
        "schema_version": "immune_llada.tcr_region_profile.v1",
        "seed": int(seed),
        "sources": list(sources),
        "source_policy": str(source_policy),
        "fallback_regions": fallback_regions,
        "region_distributions": distributions,
        "observations": dict(observations or {}),
    }


def sample_region_lengths(profile: dict[str, Any], role: str, *, seed: int, source: str, row_index: int) -> dict[str, int]:
    distributions = profile.get("region_distributions", {}).get(role, {})
    rng = random.Random(_stable_seed(seed, source, row_index))
    result: dict[str, int] = {}
    for region in TCR_REGION_ORDER:
        values = distributions.get(region, {})
        if not values:
            raise ValueError(f"Region profile has no lengths for {role}.{region}")
        # Sort numerically rather than trusting dict order. ``finalize_region_profile``
        # builds these numerically sorted, but ``atomic_json_dump`` writes the
        # manifest with ``sort_keys=True``, so a reloaded profile iterates
        # lexicographically ('10' before '2'). Weights are looked up by length so
        # the distribution is correct either way, but ``rng.choices`` maps a given
        # uniform draw onto whichever length sits at that cumulative position --
        # so without this a profile round-tripped through the manifest cannot
        # reproduce the records it originally generated. Only regions whose support
        # spans single and double digits are affected, i.e. CDR3.
        lengths = sorted(int(length) for length in values)
        weights = [int(values[str(length)]) for length in lengths]
        result[region] = rng.choices(lengths, weights=weights, k=1)[0]
    return result


def profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for role, regions in profile.get("region_distributions", {}).items():
        summary[role] = {}
        for region, values in regions.items():
            total = sum(int(value) for value in values.values())
            summary[role][region] = {
                "count": total,
                "min": min(map(int, values)) if values else None,
                "max": max(map(int, values)) if values else None,
                "fallback": region in set(profile.get("fallback_regions", {}).get(role, [])),
            }
    return summary


def default_region_profile() -> dict[str, Any]:
    """Small deterministic fallback for direct adapter calls, never training preparation."""
    lengths = {
        "tcr_alpha": {region: {"1": 1} for region in TCR_REGION_ORDER},
        "tcr_beta": {region: {"1": 1} for region in TCR_REGION_ORDER},
    }
    return {
        "schema_version": "immune_llada.tcr_region_profile.v1",
        "seed": 42,
        "sources": ["direct_adapter_fallback"],
        "source_policy": "direct_adapter_fallback",
        "fallback_regions": {role: list(TCR_REGION_ORDER) for role in lengths},
        "region_distributions": lengths,
        "observations": {},
    }


def fallback_region_profile(*, seed: int = 42) -> dict[str, Any]:
    """Plausible-length profile for runs with no complete-TCR source to learn from.

    ``default_region_profile`` emits length-1 regions, which is only safe as a
    shape check. Completion actually writes these lengths into training
    sequences, so the degenerate path uses the documented per-region medians and
    says so in ``source_policy``.
    """
    return {
        "schema_version": "immune_llada.tcr_region_profile.v1",
        "seed": int(seed),
        "sources": ["fallback_region_lengths"],
        "source_policy": "fallback_region_lengths",
        "fallback_regions": {role: list(TCR_REGION_ORDER) for role in _FALLBACK_REGION_LENGTHS},
        "region_distributions": {
            role: {region: {str(length): 1} for region, length in regions.items()}
            for role, regions in _FALLBACK_REGION_LENGTHS.items()
        },
        "observations": {},
    }


def profile_digest(profile: dict[str, Any]) -> str:
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "add_tcr_region_lengths",
    "collect_tcr_region_lengths",
    "default_region_profile",
    "empty_tcr_region_lengths",
    "fallback_region_profile",
    "finalize_region_profile",
    "profile_digest",
    "profile_summary",
    "sample_region_lengths",
]
