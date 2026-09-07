#!/usr/bin/env python3
"""Parse the three native single-cell TCR-pMHC sources into unified tier rows.

  minervina : pogorely/COVID_vax_CD8 cd8_only_dextr_rev_clean.tsv (dextramer+TCR)
  tenx      : 10x Zenodo 6952657 contig CSV + binarized dextramer matrix
  covidvac  : Kocher/Drost Zenodo 15691612 02_dex_annotated_cd8.h5ad (scirpy)

All three are dextramer/multimer single-cell assays: an assigned epitope means
the cell's TCR is a *binder* (positive). We do NOT have native full-length Fv for
these sources here (10x full aa is only in multi-GB JSON; Minervina/CovidVac
carry only CDR3 in the released tables), so ``sequence_scope=cdr3`` -- a
plan-sanctioned completeness-gate fallback. HLA pseudo-sequence is mapped when
the restricting allele is resolvable, else the row degrades to tier C.

Peptide resolution for Minervina uses ONLY peptide->HLA pairs citable from
published sources (paper text + MDPI Vaccines 2024 12(6):679 + Nat Med 2020
41591-020-01143-2 Suppl. Table 1). Unresolvable dextramer codes are dropped and
reported (never fabricated).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from common import (
    DATA,
    PROJECT_ROOT,
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
NATIVE = DATA / "tcr_native"

# --------------------------------------------------------------------------- #
# Minervina dextramer-code -> (peptide, restricting HLA allele). CITABLE only.
# --------------------------------------------------------------------------- #
MINERVINA_PEPTIDES: dict[str, tuple[str, str]] = {
    # A*01:01
    "A01_TTD": ("TTDPSFLGRY", "HLA-A*01:01"),  # ORF1ab1637; MDPI 2024 12(6):679
    "A01_FTS": ("FTSDYYQLY", "HLA-A*01:01"),   # ORF3a207; MDPI 2024 12(6):679
    "A01_LTD": ("LTDEMIAQY", "HLA-A*01:01"),   # S865; MDPI + NatMed2020 SupplT1
    "A01_DTD": ("DTDFVNEFY", "HLA-A*01:01"),   # ORF1ab5130; NatMed2020 SupplT1
    # A*02:01
    "A02_YLQ": ("YLQPRTFLL", "HLA-A*02:01"),   # S269; Minervina text + MDPI
    # A*24:02
    "A24_VYI": ("VYIGDPAQL", "HLA-A*24:02"),   # ORF1ab5840; MDPI 2024 12(6):679
    "A24_QYI": ("QYIKWPWYI", "HLA-A*24:02"),   # S1208; MDPI 2024 12(6):679
    # B*15:01
    "B15_NQK": ("NQKLIANQF", "HLA-B*15:01"),   # Minervina text (SARS-CoV-2 var)
}


def write_unified(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIFIED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in UNIFIED_COLUMNS})


def _tier_and_task(has_alpha: bool, has_mhc: bool) -> tuple[str, str]:
    if has_mhc and has_alpha:
        return "A", "tcr_pmhc"
    if has_mhc:
        return "B", "tcr_pmhc"
    return "C", "tcr_epitope"


# --------------------------------------------------------------------------- #
# Minervina
# --------------------------------------------------------------------------- #

def parse_minervina() -> tuple[list[dict], dict]:
    tsv = NATIVE / "github/COVID_vax_CD8-master/cd8_only_dextr_rev_clean.tsv"
    hla = load_hla_pseudo()
    stats = Counter()
    unresolved = Counter()
    rows: list[dict] = []
    seen: set[tuple] = set()
    csv.field_size_limit(2**31 - 1)
    with tsv.open(newline="") as handle:
        for r in csv.DictReader(handle, delimiter="\t"):
            stats["read"] += 1
            code = (r.get("epitope") or "").strip()
            if code not in MINERVINA_PEPTIDES:
                unresolved[code] += 1
                stats["drop_unresolved_epitope"] += 1
                continue
            peptide, allele_full = MINERVINA_PEPTIDES[code]
            allele = norm_allele(allele_full)
            mhc_seq = hla.get(allele, "")
            cb_full = normalize_sequence(r.get("cdr3b"))
            ca_full = normalize_sequence(r.get("cdr3a"))
            cb = cdr3_core(cb_full, has_anchors=True) if cb_full and cb_full != "NA" else ""
            ca = cdr3_core(ca_full, has_anchors=True) if ca_full and ca_full != "NA" else ""
            if not is_valid_protein_sequence(cb) and not is_valid_protein_sequence(ca):
                stats["drop_no_cdr3"] += 1
                continue
            has_alpha = is_valid_protein_sequence(ca)
            tier, task = _tier_and_task(has_alpha, bool(mhc_seq))
            key = (cb, ca, peptide, allele, "binding")
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            out = empty_row()
            out.update(
                record_id=record_id("minervina", cb, ca, peptide, r.get("donor", "")),
                source="minervina",
                fv_source="minervina_cdr3",
                tier=tier,
                task_type=task,
                relation="binding",
                sequence_scope="cdr3",
                cdr3a=ca,
                cdr3b=cb,
                epitope_seq=peptide,
                mhc_seq=mhc_seq,
                mhc_allele_norm=allele if mhc_seq else "",
                donor=r.get("donor", ""),
                provenance=f"minervina:{code}:clone={r.get('clonotype_id','')}",
            )
            rows.append(out)
    stats["final"] = len(rows)
    stats["tier_A"] = sum(1 for r in rows if r["tier"] == "A")
    stats["tier_B"] = sum(1 for r in rows if r["tier"] == "B")
    stats["tier_C"] = sum(1 for r in rows if r["tier"] == "C")
    report = {
        "source": "minervina",
        "counts": dict(stats),
        "resolved_codes": sorted(MINERVINA_PEPTIDES),
        "unresolved_codes": dict(unresolved),
    }
    return rows, report


# --------------------------------------------------------------------------- #
# 10x  (Zenodo 6952657)
# --------------------------------------------------------------------------- #
_HLA_CODE = re.compile(r"^([ABC])(\d{2})(\d{2})$")


def _hla_from_code(code: str) -> str:
    m = _HLA_CODE.match(code)
    if not m:
        return ""
    return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"


# 10x dextramer column: <HLAcode>_<peptide>_<free-text-name>_binder (boolean).
_BINDER_COL = re.compile(r"^([ABC]\d{4})_([ACDEFGHIKLMNPQRSTVWY]{7,20})_.+_binder$")


def _tcr_from_clono(cell_clono: str) -> tuple[str, str]:
    """Parse cell_clono_cdr3_aa 'TRA:...;TRB:...' -> (cdr3a_core, cdr3b_core).

    Uses the first TRA and first TRB CDR3 (full junctions in the file)."""

    ca = cb = ""
    for tok in str(cell_clono or "").split(";"):
        tok = tok.strip()
        if tok.startswith("TRA:") and not ca:
            ca = cdr3_core(tok[4:], has_anchors=True)
        elif tok.startswith("TRB:") and not cb:
            cb = cdr3_core(tok[4:], has_anchors=True)
    return ca, cb


def parse_tenx(*, neg_cap_per_pmhc: int = 5, neg_cap_total: int = 4000) -> tuple[list[dict], dict]:
    hla = load_hla_pseudo()
    stats = Counter()
    rows: list[dict] = []
    seen: set[tuple] = set()
    pos_per_pmhc: Counter = Counter()
    neg_candidates: list = []
    for donor in ("donor1", "donor2", "donor3", "donor4"):
        binz = NATIVE / f"10x/{donor}_binarized_matrix.csv"
        if not binz.is_file():
            stats[f"skip_{donor}_missing_file"] += 1
            continue
        with binz.open(newline="") as handle:
            reader = csv.DictReader(handle)
            cols = reader.fieldnames or []
            dex_defs: dict[str, tuple[str, str]] = {}
            for col in cols:
                m = _BINDER_COL.match(col)
                if m:
                    dex_defs[col] = (m.group(2), norm_allele(_hla_from_code(m.group(1))))
            stats["n_dextramers"] = len(dex_defs)
            clono_col = "cell_clono_cdr3_aa" if "cell_clono_cdr3_aa" in cols else None
            for row in reader:
                stats["cells_read"] += 1
                ca, cb = _tcr_from_clono(row.get(clono_col, "")) if clono_col else ("", "")
                if not is_valid_protein_sequence(cb) and not is_valid_protein_sequence(ca):
                    continue
                for dex_col, (pep_raw, allele) in dex_defs.items():
                    val = str(row.get(dex_col, "")).strip().lower()
                    is_binder = val in ("true", "1", "1.0", "yes")
                    peptide = normalize_sequence(pep_raw)
                    if not is_valid_protein_sequence(peptide):
                        continue
                    mhc_seq = hla.get(allele, "")
                    has_alpha = is_valid_protein_sequence(ca)
                    tier, task = _tier_and_task(has_alpha, bool(mhc_seq))
                    relation = "binding" if is_binder else "nonbinding"
                    key = (cb, ca, peptide, allele, relation)
                    if key in seen:
                        continue
                    out = empty_row()
                    out.update(
                        record_id=record_id("tenx", cb, ca, peptide, allele, relation),
                        source="tenx",
                        fv_source="10x_cdr3",
                        tier=tier,
                        task_type=task,
                        relation=relation,
                        sequence_scope="cdr3",
                        cdr3a=ca,
                        cdr3b=cb,
                        epitope_seq=peptide,
                        mhc_seq=mhc_seq,
                        mhc_allele_norm=allele if mhc_seq else "",
                        donor=donor,
                        provenance=f"10x:{donor}:{dex_col}",
                    )
                    if is_binder:
                        seen.add(key)
                        rows.append(out)
                        pos_per_pmhc[(peptide, allele)] += 1
                        stats["pos"] += 1
                    else:
                        neg_candidates.append((key, out, (peptide, allele)))
    # cap negatives: <= neg_cap_per_pmhc x positives per pMHC, <= neg_cap_total
    neg_per_pmhc: Counter = Counter()
    neg_kept = 0
    neg_candidates.sort(key=lambda t: t[0])  # deterministic
    for key, out, pmhc in neg_candidates:
        if neg_kept >= neg_cap_total:
            break
        if key in seen:
            continue
        cap = max(1, neg_cap_per_pmhc * pos_per_pmhc.get(pmhc, 0))
        if neg_per_pmhc[pmhc] >= cap:
            continue
        seen.add(key)
        rows.append(out)
        neg_per_pmhc[pmhc] += 1
        neg_kept += 1
    stats["neg_kept"] = neg_kept
    stats["neg_candidates"] = len(neg_candidates)
    stats["final"] = len(rows)
    for t in ("A", "B", "C"):
        stats[f"tier_{t}"] = sum(1 for r in rows if r["tier"] == t)
    return rows, {"source": "tenx", "counts": dict(stats),
                  "n_pmhc": len(pos_per_pmhc)}


# --------------------------------------------------------------------------- #
# CovidVac  (Zenodo 15691612 h5ad)
# --------------------------------------------------------------------------- #

# CovidVac binding_ct stores the peptide sequence directly. HLA is not in obs;
# map peptide->restricting allele ONLY for citable epitopes (else tier C).
COVIDVAC_PEPTIDE_HLA: dict[str, str] = {
    "YLQPRTFLL": "HLA-A*02:01",  # S269; MDPI 2024 12(6):679 + Minervina text
    "LTDEMIAQY": "HLA-A*01:01",  # S865; MDPI + NatMed2020 SupplT1
    "KCYGVSPTK": "HLA-A*03:01",  # S378; MDPI 2024 12(6):679
    "QYIKWPWYI": "HLA-A*24:02",  # S1208; MDPI 2024 12(6):679
}


def parse_covidvac() -> tuple[list[dict], dict]:
    import anndata as ad

    h5 = NATIVE / "covidvac/02_dex_annotated_cd8.h5ad"
    hla = load_hla_pseudo()
    stats = Counter()
    epitope_hits = Counter()
    unmapped_pep = Counter()
    rows: list[dict] = []
    seen: set[tuple] = set()
    adata = ad.read_h5ad(h5, backed="r")
    obs = adata.obs
    cols = set(obs.columns)
    BIND, BETA, ALPHA = "binding_ct", "IR_VDJ_1_junction_aa", "IR_VJ_1_junction_aa"
    BLOCUS, ALOCUS = "IR_VDJ_1_locus", "IR_VJ_1_locus"
    meta = {"n_obs": int(adata.n_obs),
            "columns_used": {"binding": BIND, "beta": BETA, "alpha": ALPHA},
            "have_cols": {c: (c in cols) for c in (BIND, BETA, ALPHA, BLOCUS, ALOCUS)}}
    if BIND not in cols or BETA not in cols:
        return [], {"source": "covidvac", "counts": dict(stats), "meta": meta,
                    "error": "required columns (binding_ct / IR_VDJ_1_junction_aa) not found"}

    def sval(v) -> str:
        s = str(v).strip()
        return "" if s.lower() in ("nan", "none", "na", "") else s

    for _, r in obs.iterrows():
        stats["read"] += 1
        peptide = normalize_sequence(sval(r.get(BIND, "")))
        if not peptide or peptide == "NOBINDING" or not is_valid_protein_sequence(peptide) or not (7 <= len(peptide) <= 15):
            stats["drop_no_binding"] += 1
            continue
        # respect locus if present (VDJ=beta/TRB, VJ=alpha/TRA)
        blocus = sval(r.get(BLOCUS, "")).upper() if BLOCUS in cols else "TRB"
        alocus = sval(r.get(ALOCUS, "")).upper() if ALOCUS in cols else "TRA"
        cb = cdr3_core(sval(r.get(BETA, "")), has_anchors=True) if blocus in ("TRB", "") else ""
        ca = cdr3_core(sval(r.get(ALPHA, "")), has_anchors=True) if alocus in ("TRA", "") else ""
        if not is_valid_protein_sequence(cb) and not is_valid_protein_sequence(ca):
            stats["drop_no_cdr3"] += 1
            continue
        allele_full = COVIDVAC_PEPTIDE_HLA.get(peptide, "")
        allele = norm_allele(allele_full)
        mhc_seq = hla.get(allele, "")
        if not mhc_seq:
            unmapped_pep[peptide] += 1
        has_alpha = is_valid_protein_sequence(ca)
        tier, task = _tier_and_task(has_alpha, bool(mhc_seq))
        key = (cb, ca, peptide, allele, "binding")
        if key in seen:
            stats["dup"] += 1
            continue
        seen.add(key)
        epitope_hits[peptide] += 1
        out = empty_row()
        out.update(
            record_id=record_id("covidvac", cb, ca, peptide, allele),
            source="covidvac",
            fv_source="covidvac_cdr3",
            tier=tier,
            task_type=task,
            relation="binding",
            sequence_scope="cdr3",
            cdr3a=ca,
            cdr3b=cb,
            epitope_seq=peptide,
            mhc_seq=mhc_seq,
            mhc_allele_norm=allele if mhc_seq else "",
            provenance=f"covidvac:{peptide}",
        )
        rows.append(out)
    stats["final"] = len(rows)
    for t in ("A", "B", "C"):
        stats[f"tier_{t}"] = sum(1 for r in rows if r["tier"] == t)
    return rows, {"source": "covidvac", "counts": dict(stats), "meta": meta,
                  "epitopes": dict(epitope_hits),
                  "unmapped_hla_peptides": dict(unmapped_pep),
                  "mapped_hla_peptides": sorted(COVIDVAC_PEPTIDE_HLA)}


PARSERS = {"minervina": parse_minervina, "tenx": parse_tenx, "covidvac": parse_covidvac}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=sorted(PARSERS))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, report = PARSERS[args.source]()
    write_unified(rows, OUT_DIR / f"{args.source}.csv")
    (OUT_DIR / f"{args.source}_parse_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
