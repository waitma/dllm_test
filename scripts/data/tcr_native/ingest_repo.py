#!/usr/bin/env python3
"""Ingest repo-local TCR recognition sources into unified tier rows.

  PISTE  -> tier B  (CDR3b + peptide + HLA pseudo)     [no ANARCI]
  full   -> tier A  (ANARCI-extracted alpha/beta Fv + peptide + HLA pseudo)

Outputs one unified CSV per source under ``data/tcr_native/unified/``.

The full-length source stores full V+C+TM chains; ANARCI numbers only the
variable domain so the concatenated FR1..FR4 is the OTS-aligned Fv. Run the
full-length ingestion in the ``flow`` conda env (has ANARCI).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    DATA,
    UNIFIED_COLUMNS,
    canonical_relation,
    cdr3_core,
    empty_row,
    is_valid_protein_sequence,
    load_hla_pseudo,
    norm_allele,
    normalize_sequence,
    record_id,
)

OUT_DIR = DATA / "tcr_native/unified"
PISTE_DIR = DATA / "ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/random"
FULL_JSONL = DATA / "tcr_pmhc_fulllength/records_train.jsonl"


def _stable_rank(*parts) -> int:
    return int(hashlib.sha1(("|".join(str(p) for p in parts)).encode()).hexdigest(), 16)


# --------------------------------------------------------------------------- #
# PISTE  (tier B)
# --------------------------------------------------------------------------- #

def ingest_piste(
    *,
    neg_ratio: float = 1.0,
    neg_floor: int = 50,
    dominant_epitope: str = "KLGGALQAK",
    seed: int = 42,
) -> tuple[list[dict], dict]:
    hla = load_hla_pseudo()
    rows_pos: list[dict] = []
    rows_neg: list[dict] = []
    stats = Counter()
    seen_keys: set[tuple] = set()

    for split_file in ("train_data.csv", "val_data.csv"):
        p = PISTE_DIR / split_file
        with p.open(newline="") as handle:
            for row in csv.DictReader(handle):
                stats["read"] += 1
                cdr3_full = normalize_sequence(row.get("CDR3"))
                core = cdr3_core(cdr3_full, has_anchors=True)
                epitope = normalize_sequence(row.get("MT_pep"))
                allele = norm_allele(row.get("HLA_type"))
                mhc_seq = normalize_sequence(row.get("HLA_sequence")) or hla.get(allele, "")
                if not is_valid_protein_sequence(core) or not is_valid_protein_sequence(epitope):
                    stats["drop_invalid"] += 1
                    continue
                if not mhc_seq:
                    stats["drop_no_hla"] += 1
                    continue
                relation = canonical_relation(row.get("Label"))
                # exact dedup on (core, epitope, allele, relation)
                key = (core, epitope, allele, relation)
                if key in seen_keys:
                    stats["dup"] += 1
                    continue
                seen_keys.add(key)
                out = empty_row()
                out.update(
                    record_id=record_id("piste", core, epitope, allele, relation),
                    source="piste",
                    fv_source="piste_cdr3",
                    tier="B",
                    task_type="tcr_pmhc",
                    relation=relation,
                    sequence_scope="cdr3",
                    cdr3b=core,
                    epitope_seq=epitope,
                    mhc_seq=mhc_seq,
                    mhc_allele_norm=allele,
                    provenance=f"piste:{split_file}",
                )
                (rows_pos if relation == "binding" else rows_neg).append((epitope, core, out))

    # Cap negatives per epitope to neg_ratio x positives (floor neg_floor);
    # the dominant epitope is further capped to <=1 negative per clonotype.
    pos_per_ep = Counter(ep for ep, _, _ in rows_pos)
    kept: list[dict] = []
    for _, _, out in rows_pos:
        kept.append(out)
    neg_by_ep: dict[str, list] = defaultdict(list)
    for ep, core, out in rows_neg:
        neg_by_ep[ep].append((core, out))
    dominant_clono: set[str] = set()
    for ep, items in neg_by_ep.items():
        if ep == dominant_epitope:
            # one negative per unique clonotype (core)
            chosen = []
            for core, out in sorted(items, key=lambda t: _stable_rank(seed, t[0])):
                if core in dominant_clono:
                    continue
                dominant_clono.add(core)
                chosen.append(out)
            cap = int(max(pos_per_ep.get(ep, 0) * neg_ratio, neg_floor))
            chosen = chosen[:cap]
        else:
            cap = int(max(pos_per_ep.get(ep, 0) * neg_ratio, neg_floor))
            ordered = sorted(items, key=lambda t: _stable_rank(seed, ep, t[0]))
            chosen = [out for _, out in ordered[:cap]]
        kept.extend(chosen)
        stats[f"neg_kept"] += len(chosen)

    stats["pos_kept"] = len(rows_pos)
    stats["neg_available"] = len(rows_neg)
    stats["final"] = len(kept)
    report = {
        "source": "piste",
        "counts": dict(stats),
        "neg_ratio": neg_ratio,
        "neg_floor": neg_floor,
        "dominant_epitope": dominant_epitope,
        "dominant_epitope_clonotypes_kept_neg": len(dominant_clono),
        "unique_epitopes": len(set(list(pos_per_ep) + list(neg_by_ep))),
    }
    return kept, report


# --------------------------------------------------------------------------- #
# full-length TCR-pMHC (tier A) -- needs ANARCI
# --------------------------------------------------------------------------- #

def ingest_fulllength(*, batch_size: int = 4000, ncpu: int = 8) -> tuple[list[dict], dict]:
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent))
    from anarci_fv import segment_batch

    hla = load_hla_pseudo()
    stats = Counter()

    # Load records, collect chains to segment.
    records: list[dict] = []
    with FULL_JSONL.open() as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            roles = rec.get("chain_roles") or rec.get("roles") or []
            chains = rec.get("chains") or []
            alpha = beta = ""
            for ro, se in zip(roles, chains):
                if ro == "tcr_alpha":
                    alpha = normalize_sequence(se)
                elif ro == "tcr_beta":
                    beta = normalize_sequence(se)
                elif ro == "peptide":
                    pass
            peptide = ""
            for ro, se in zip(roles, chains):
                if ro == "peptide":
                    peptide = normalize_sequence(se)
            meta = rec.get("metadata", {})
            records.append(
                {
                    "idx": i,
                    "alpha": alpha,
                    "beta": beta,
                    "peptide": peptide,
                    "allele": meta.get("mhc_a", ""),
                    "relation": rec.get("labels", {}).get("relation", "binding"),
                    "origin": meta.get("origin", ""),
                }
            )
    stats["read"] = len(records)

    # Batch ANARCI over all alpha/beta chains.
    seg: dict[str, dict] = {}
    items: list[tuple[str, str]] = []
    for r in records:
        if r["alpha"]:
            items.append((f"a{r['idx']}", r["alpha"]))
        if r["beta"]:
            items.append((f"b{r['idx']}", r["beta"]))
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        seg.update(segment_batch(chunk, ncpu=ncpu))
        stats["anarci_done"] += len(chunk)

    kept: list[dict] = []
    for r in records:
        a = seg.get(f"a{r['idx']}")
        b = seg.get(f"b{r['idx']}")
        epitope = r["peptide"]
        if not is_valid_protein_sequence(epitope):
            stats["drop_no_peptide"] += 1
            continue
        allele = norm_allele(r["allele"])
        mhc_seq = hla.get(allele, "")

        # beta is required (it is the primary receptor chain)
        beta_fv = beta_cdr3 = ""
        beta_scope = ""
        if b:
            beta_cdr3 = b["cdr3"]
            if b["gate_pass"]:
                beta_fv = b["fv"]
                beta_scope = "fv"
            elif is_valid_protein_sequence(beta_cdr3):
                beta_scope = "cdr3"
        if not (beta_fv or is_valid_protein_sequence(beta_cdr3)):
            stats["drop_no_beta"] += 1
            continue

        alpha_fv = alpha_cdr3 = ""
        if a:
            alpha_cdr3 = a["cdr3"]
            if a["gate_pass"]:
                alpha_fv = a["fv"]
            # if alpha gate fails we still keep alpha_cdr3 when valid

        # tier: A requires alpha + beta + peptide + MHC-mapped; else fall to C
        has_alpha = bool(alpha_fv or is_valid_protein_sequence(alpha_cdr3))
        if mhc_seq and has_alpha:
            tier = "A"
            task_type = "tcr_pmhc"
        elif mhc_seq:
            tier = "B"
            task_type = "tcr_pmhc"
        else:
            tier = "C"
            task_type = "tcr_epitope"
            stats["allele_unmapped_to_C"] += 1

        # overall sequence_scope: fv if beta_fv present else cdr3
        scope = "fv" if beta_fv else "cdr3"
        if scope == "fv":
            stats["scope_fv"] += 1
        else:
            stats["scope_cdr3"] += 1

        relation = canonical_relation(r["relation"])
        out = empty_row()
        out.update(
            record_id=record_id("full", r["idx"]),
            source="tcr_pmhc_fulllength",
            fv_source="stitchr",
            tier=tier,
            task_type=task_type,
            relation=relation,
            sequence_scope=scope,
            alpha_fv=alpha_fv,
            beta_fv=beta_fv,
            cdr3a=alpha_cdr3,
            cdr3b=beta_cdr3,
            epitope_seq=epitope,
            mhc_seq=mhc_seq,
            mhc_allele_norm=allele,
            provenance=f"fulllength:{r['origin']}",
        )
        kept.append(out)

    stats["final"] = len(kept)
    stats["tier_A"] = sum(1 for r in kept if r["tier"] == "A")
    stats["tier_B"] = sum(1 for r in kept if r["tier"] == "B")
    stats["tier_C"] = sum(1 for r in kept if r["tier"] == "C")
    report = {"source": "tcr_pmhc_fulllength", "counts": dict(stats)}
    return kept, report


def write_unified(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIFIED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=["piste", "fulllength"])
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--ncpu", type=int, default=8)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.source == "piste":
        rows, report = ingest_piste(neg_ratio=args.neg_ratio)
        write_unified(rows, OUT_DIR / "piste.csv")
    else:
        rows, report = ingest_fulllength(ncpu=args.ncpu)
        write_unified(rows, OUT_DIR / "fulllength.csv")

    (OUT_DIR / f"{args.source}_ingest_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
