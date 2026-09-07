#!/usr/bin/env python3
"""ANARCI IMGT segmentation + full-length-Fv completeness gate.

Runs in the ``flow`` conda env (has ``anarci`` + HMMER backend). Given a raw
amino-acid chain (a native contig ``aa_sequence`` or a stitchr full chain that
still carries signal peptide / constant region), ANARCI numbers ONLY the
variable domain, so the concatenated FR1..FR4 is exactly the Fv aligned to the
OTS pre-training representation.

Completeness gate for a "full-length Fv":
  * all four framework regions (FR1..FR4) present and non-empty;
  * FR1 begins near IMGT position 1 (first numbered position <= FR1_START_MAX);
  * FR4 matches the ``[FW]G.G`` motif;
  * Fv length within [FV_MIN, FV_MAX].
A row that fails the gate is NOT dropped: the caller degrades it to
``sequence_scope=cdr3`` using the ANARCI CDR3 loop (IMGT 105-117, anchor-free).
"""

from __future__ import annotations

import re
from typing import Iterable

from anarci import anarci

# IMGT region boundaries (inclusive), CDR3 loop = 105-117 (anchors 104/118 out).
_REGIONS = {
    "fr1": (1, 26),
    "cdr1": (27, 38),
    "fr2": (39, 55),
    "cdr2": (56, 65),
    "fr3": (66, 104),
    "cdr3": (105, 117),
    "fr4": (118, 128),
}
FR1_START_MAX = 6
FV_MIN = 100
FV_MAX = 135
_FGXG = re.compile(r"^[FW]G.G")


def _region(numbering, lo: int, hi: int) -> str:
    return "".join(aa for (pos, _ins), aa in numbering if lo <= pos <= hi and aa != "-")


def segment_batch(
    items: list[tuple[str, str]],
    *,
    allowed_chain_types: Iterable[str] = ("A", "B", "D", "G"),
    ncpu: int = 8,
) -> dict[str, dict]:
    """Segment a batch of ``(id, sequence)`` items.

    Returns ``{id -> result}`` where result is ``None`` if ANARCI found no
    variable domain, else a dict with FR/CDR segments, ``fv``, ``cdr3``
    (anchor-free loop), ``chain_type``, and ``gate_pass`` / ``gate_reasons``.
    """

    allowed = {c.upper() for c in allowed_chain_types}
    clean = [(sid, str(seq).strip().upper()) for sid, seq in items if seq]
    out: dict[str, dict] = {}
    if not clean:
        return out
    numbered, details, _ = anarci(clean, scheme="imgt", output=False, ncpu=ncpu)
    for (sid, seq), domains, det in zip(clean, numbered, details):
        if not domains:
            out[sid] = None
            continue
        # pick the best (first) domain
        numbering, _start, _end = domains[0]
        ctype = (det[0].get("chain_type") if det else "") or ""
        ctype = ctype.upper()
        segs = {name: _region(numbering, lo, hi) for name, (lo, hi) in _REGIONS.items()}
        fv = "".join(segs[n] for n in ("fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"))
        first_pos = numbering[0][0][0] if numbering else 999
        reasons = []
        if allowed and ctype not in allowed:
            reasons.append(f"chain_type={ctype}")
        for fr in ("fr1", "fr2", "fr3", "fr4"):
            if not segs[fr]:
                reasons.append(f"empty_{fr}")
        if first_pos > FR1_START_MAX:
            reasons.append(f"fr1_start={first_pos}")
        if not _FGXG.match(segs["fr4"]):
            reasons.append("fr4_no_FGXG")
        if not (FV_MIN <= len(fv) <= FV_MAX):
            reasons.append(f"fv_len={len(fv)}")
        out[sid] = {
            "chain_type": ctype,
            "fv": fv,
            "cdr3": segs["cdr3"],
            "segments": segs,
            "fr1_start": first_pos,
            "gate_pass": not reasons,
            "gate_reasons": reasons,
        }
    return out


if __name__ == "__main__":
    import json
    import sys

    demo = [
        ("beta_full", "MSIGLLCCAALSLLWAGPVNAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYSGSMGELFFGEGSRLTVLEDLKNVFPPEVAVFEPSEAEISHTQKATLVCLATGFYPDHVELSWWVNGKEVHSGVSTDPQPLKEQPALNDSRYCLSSRLRVSATFWQNPRNHFRCQVQFYGLSENDEWTQDRAKPVTQIVSAEAWGRADCGFTSESYQQGVLSATILYEILLGKATLYAVLVSALVLMAMVKRKDSRG"),
    ]
    res = segment_batch(demo)
    print(json.dumps(res, indent=2))
