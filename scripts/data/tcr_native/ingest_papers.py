#!/usr/bin/env python3
"""Ingest the training corpora of published TCR design / generation papers into
the unified tcr_native tier schema.

All five corpora already ship inside the cloned baseline repos under
``downstream/benchmark/baselines/``, so nothing is downloaded here.

  tcrt5        TCRT5 (Nat Mach Intell 2025)  peptide + 34aa HLA pseudo + CDR3b
  tcrdiff      TCRDiff (2026)                peptide + HLA allele + paired
                                             CDR1/2/3 alpha & beta + binding label
  gratcr_tep   GRATCR  TEP.csv               CDR3b + epitope        (no MHC)
  gratcr_mira  GRATCR  MIRA.csv (ImmuneCODE) CDR3b + epitope        (no MHC)
  epidiff      TCR-epiDiff                   CDR3b + epitope        (no MHC)

Output goes to a *separate* corpus root (``data/tcr_papers/``) rather than into
``data/tcr_native/``: the latter is what the current checkpoints trained on and
overwriting it would break their reproducibility. The new corpus is registered
as its own source token so it can be mixed and weighted independently.

Tier follows the existing convention: A = alpha + beta + peptide + MHC,
B = beta + peptide + MHC, C = no MHC (renders without an MHC grammar block).

Usage
-----
    python ingest_papers.py --source all
    python ingest_papers.py --finalize
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
    strip_alignment_gaps,
)


def _aa(value):
    """Uppercase + drop alignment gaps that RemapCollator cannot map."""
    return strip_alignment_gaps(normalize_sequence(value))


BASELINES = PROJECT_ROOT / "downstream/benchmark/baselines"
CORPUS = DATA / "tcr_papers"
UNIFIED_DIR = CORPUS / "unified"
DATASET_DIR = CORPUS / "dataset"

TCRT5_DIR = BASELINES / "TCRT5_tcr_translate"
TCRDIFF_CSV = BASELINES / "TCRDiff/data/tcrpmhc/train_tcr_pmhc_data.csv"
GRATCR_TEP = BASELINES / "GRATCR/Data/TEP.csv"
GRATCR_MIRA = BASELINES / "GRATCR/Data/MIRA/MIRA.csv"
EPIDIFF_CSV = (
    BASELINES / "TCR-epiDiff/Model/Dataset/TCR-epiDiff_Training_Dataset.csv"
)

# TcrDesign 2026 (Zenodo 14545852). The only audited candidate that is not
# already saturated by our corpus: +898 net-new epitopes on antigen_beta alone,
# versus +3 for UniPMT / VDJdb / McPAS / GLIPH combined.
#
# STALE for VDJdb (2026-08-28): the "+3" was measured against the 2025-12-29 dump.
# The 2026-06-03 release grew vdjdb_full.txt from 139,745 to 192,754 rows (+37.9%)
# and is now on disk at data/tcr/vdjdb_2026_06_03/. Re-run assess_candidate.py
# against it before reusing this line to justify skipping VDJdb.
TCRDESIGN26 = DATA / "tcr_papers/raw/tcrdesign2026"
TD26_BETA = TCRDESIGN26 / "tcrdesign_G/antigen_beta.csv"
TD26_PAIRED = TCRDESIGN26 / "tcrdesign_G/antigen_beta_alpha.csv"
TD26_PMHC = TCRDESIGN26 / "tcrdesign_B/pMHC_TCR_train.tsv"

# pMHC_TCR_train.tsv ships no header. Columns, verified by inspection:
# alpha V / alpha J / CDR3a / beta V / beta J / CDR3b / epitope / MHC 34mer / label
TD26_PMHC_COLUMNS = [
    "trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b", "epitope", "mhc_pseudo", "label",
]

MIN_CDR3, MAX_CDR3 = 4, 40
MIN_EPITOPE, MAX_EPITOPE = 6, 25


def _drop_placeholder(seq: str | None) -> str:
    """TcrDesign-2026 marks an absent chain with a run of ``X`` (e.g. the 15-char
    ``XXXXXXXXXXXXXXX``), which is not a sequence.

    This matters: ``cdr3_core`` leaves such a token untouched (it neither starts
    with C nor ends with F/W), so without this guard the placeholder is ingested
    as if it were a real CDR3 and inflates every downstream count.
    """

    text = (seq or "").strip().upper()
    return "" if not text or set(text) <= {"X"} else text


def _stable_rank(*parts) -> int:
    """Deterministic shuffle key (same helper as ingest_repo.py)."""

    return int(hashlib.sha1(("|".join(str(p) for p in parts)).encode()).hexdigest(), 16)


def ok_cdr3(seq: str) -> bool:
    return MIN_CDR3 <= len(seq) <= MAX_CDR3 and is_valid_protein_sequence(seq)


def ok_epitope(seq: str) -> bool:
    return MIN_EPITOPE <= len(seq) <= MAX_EPITOPE and is_valid_protein_sequence(seq)


def _pseudo_tables() -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(allele -> pseudo, pseudo -> allele)``.

    PISTE's ``common_hla_sequence.csv`` is the project-canonical table (10,413
    alleles) and TCRT5's ``refs/hla_pseudo_seqs.csv`` adds 70 more; the 2,951
    shared alleles agree on 2,950, so the two use the same NetMHCpan 34aa
    convention and can be merged.
    """

    allele2pseudo = load_hla_pseudo()
    extra = TCRT5_DIR / "refs/hla_pseudo_seqs.csv"
    if extra.is_file():
        with extra.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = norm_allele(row["allele"])
                seq = row["pseudo-sequence"].strip()
                if key and seq:
                    allele2pseudo.setdefault(key, seq)
    # Several alleles share a pseudo-sequence; pick the lexicographically
    # smallest so the reverse map is deterministic.
    pseudo2allele: dict[str, str] = {}
    for allele, pseudo in sorted(allele2pseudo.items()):
        pseudo2allele.setdefault(pseudo, allele)
    return allele2pseudo, pseudo2allele


def _tier(*, has_alpha: bool, has_mhc: bool) -> tuple[str, str]:
    if not has_mhc:
        return "C", "tcr_epitope"
    return ("A" if has_alpha else "B"), "tcr_pmhc"


# --------------------------------------------------------------------------- #
# TCRT5
# --------------------------------------------------------------------------- #

def ingest_tcrt5() -> tuple[list[dict], dict]:
    """``pmhc2tcr_topk_train_source`` lines are ``"<peptide> <hla_pseudo>"`` and
    the aligned ``_target`` lines are the full CDR3b junction."""

    _, pseudo2allele = _pseudo_tables()
    src = TCRT5_DIR / "data/pmhc2tcr_topk_train_source_dedup_uf.txt"
    tgt = TCRT5_DIR / "data/pmhc2tcr_topk_train_target_dedup_uf.txt"
    stats = Counter()
    seen: set[tuple] = set()
    rows: list[dict] = []

    with src.open() as fs, tgt.open() as ft:
        for line_s, line_t in zip(fs, ft):
            stats["read"] += 1
            parts = line_s.split()
            if len(parts) < 2:
                stats["drop_malformed_source"] += 1
                continue
            epitope = _aa(parts[0])
            pseudo = _aa(parts[1])
            core = cdr3_core(line_t.strip(), has_anchors=True)
            if not ok_epitope(epitope) or not ok_cdr3(core):
                stats["drop_invalid"] += 1
                continue
            if len(pseudo) != 34:
                stats["drop_bad_pseudo"] += 1
                continue
            allele = pseudo2allele.get(pseudo, "")
            if not allele:
                stats["pseudo_allele_unmapped"] += 1
            key = (core, epitope, allele, "binding")
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            out = empty_row()
            out.update(
                record_id=record_id("tcrt5", core, epitope, allele),
                source="tcrt5",
                fv_source="tcrt5_cdr3",
                tier="B",
                task_type="tcr_pmhc",
                relation="binding",
                sequence_scope="cdr3",
                cdr3b=core,
                epitope_seq=epitope,
                mhc_seq=pseudo,
                mhc_allele_norm=allele,
                provenance="tcrt5:pmhc2tcr_topk_train_dedup_uf",
            )
            rows.append(out)

    stats["final"] = len(rows)
    return rows, {"source": "tcrt5", "counts": dict(stats),
                  "unique_epitopes": len({r["epitope_seq"] for r in rows})}


# --------------------------------------------------------------------------- #
# TCRDiff
# --------------------------------------------------------------------------- #

def ingest_tcrdiff(*, neg_ratio: float = 1.0, neg_floor: int = 50, seed: int = 42
                   ) -> tuple[list[dict], dict]:
    """TCRDiff's combined ``train_tcr_pmhc_data.csv``.

    Two corpus-scope decisions, both reported:

    * **Human class I only.** 83,469 rows carry a class II (DR/DQ/DP, 35,991) or
      mouse H-2 (47,478) allele. Our MHC grammar block is the NetMHCpan class I
      34aa pseudo-sequence, so those alleles have no representation; demoting
      them to tier C would smuggle mouse / class II biology into the class I
      plane with the MHC signal silently dropped. They are dropped instead.
    * **Negatives capped** to ``neg_ratio`` x positives per epitope (floor
      ``neg_floor``), matching :func:`ingest_piste`. The raw file is 522,545
      nonbinding vs 103,021 binding; left uncapped the negatives would dominate
      the epitope-conditioned plane.
    """

    allele2pseudo, _ = _pseudo_tables()
    stats = Counter()
    seen: set[tuple] = set()
    pos: list[dict] = []
    neg: list[tuple[str, str, dict]] = []

    with TCRDIFF_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            stats["read"] += 1
            epitope = _aa(row.get("Peptide"))
            beta = cdr3_core(row.get("CDR3B"), has_anchors=True)
            alpha = cdr3_core(row.get("CDR3A"), has_anchors=True)
            if not ok_epitope(epitope) or not ok_cdr3(beta):
                stats["drop_no_or_bad_cdr3b"] += 1
                continue
            if not ok_cdr3(alpha):
                alpha = ""
            allele = norm_allele(row.get("MHC") or row.get("MHCA"))
            mhc_seq = allele2pseudo.get(allele, "")
            if not mhc_seq:
                if any(k in allele for k in ("DR", "DQ", "DP")):
                    stats["drop_class_ii"] += 1
                elif allele.upper().startswith("H2") or allele.upper().startswith("H-2"):
                    stats["drop_mouse_h2"] += 1
                else:
                    stats["drop_allele_unmapped"] += 1
                continue
            try:
                relation = canonical_relation(row.get("binding"))
            except ValueError:
                stats["drop_bad_label"] += 1
                continue
            tier, task_type = _tier(has_alpha=bool(alpha), has_mhc=True)
            key = (beta, alpha, epitope, allele, relation)
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            out = empty_row()
            out.update(
                record_id=record_id("tcrdiff", beta, alpha, epitope, allele, relation),
                source="tcrdiff",
                fv_source="tcrdiff_cdr3",
                tier=tier,
                task_type=task_type,
                relation=relation,
                sequence_scope="cdr3",
                cdr3a=alpha,
                cdr3b=beta,
                epitope_seq=epitope,
                mhc_seq=mhc_seq,
                mhc_allele_norm=allele,
                provenance=f"tcrdiff:train_tcr_pmhc_data:{row.get('Organism', '')}",
            )
            if relation == "binding":
                pos.append(out)
            else:
                neg.append((epitope, beta, out))

    pos_per_ep = Counter(r["epitope_seq"] for r in pos)
    neg_by_ep: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for epitope, beta, out in neg:
        neg_by_ep[epitope].append((beta, out))
    rows = list(pos)
    for epitope, items in neg_by_ep.items():
        cap = int(max(pos_per_ep.get(epitope, 0) * neg_ratio, neg_floor))
        ordered = sorted(items, key=lambda t: _stable_rank(seed, epitope, t[0]))
        rows.extend(out for _, out in ordered[:cap])

    stats["pos_kept"] = len(pos)
    stats["neg_available"] = len(neg)
    stats["neg_kept"] = len(rows) - len(pos)
    stats["final"] = len(rows)
    return rows, {
        "source": "tcrdiff",
        "counts": dict(stats),
        "neg_ratio": neg_ratio,
        "neg_floor": neg_floor,
        "scope": "human class I only (class II + mouse H-2 dropped)",
        "unique_epitopes": len({r["epitope_seq"] for r in rows}),
        "by_tier": dict(Counter(r["tier"] for r in rows)),
        "by_relation": dict(Counter(r["relation"] for r in rows)),
        "paired_alpha_beta": sum(1 for r in rows if r["cdr3a"]),
    }


# --------------------------------------------------------------------------- #
# CDR3b-only corpora (no MHC -> tier C)
# --------------------------------------------------------------------------- #

def _ingest_cdr3b_epitope(
    name: str, path: Path, beta_col: str, epitope_col: str, provenance: str
) -> tuple[list[dict], dict]:
    stats = Counter()
    seen: set[tuple] = set()
    rows: list[dict] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            stats["read"] += 1
            epitope = _aa(row.get(epitope_col))
            beta = cdr3_core(row.get(beta_col), has_anchors=True)
            if not ok_epitope(epitope) or not ok_cdr3(beta):
                stats["drop_invalid"] += 1
                continue
            key = (beta, epitope)
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            out = empty_row()
            out.update(
                record_id=record_id(name, beta, epitope),
                source=name,
                fv_source=f"{name}_cdr3",
                tier="C",
                task_type="tcr_epitope",
                relation="binding",
                sequence_scope="cdr3",
                cdr3b=beta,
                epitope_seq=epitope,
                provenance=provenance,
            )
            rows.append(out)
    stats["final"] = len(rows)
    return rows, {"source": name, "counts": dict(stats),
                  "unique_epitopes": len({r["epitope_seq"] for r in rows})}


# --------------------------------------------------------------------------- #
# TcrDesign 2026
# --------------------------------------------------------------------------- #

def ingest_td26_paired() -> tuple[list[dict], dict]:
    """``antigen_beta_alpha.csv``: epitope + CDR3b + CDR3a, no MHC -> tier C."""

    stats = Counter()
    seen: set[tuple] = set()
    rows: list[dict] = []
    with TD26_PAIRED.open(newline="") as handle:
        for row in csv.DictReader(handle):
            stats["read"] += 1
            epitope = _aa(row.get("antigen"))
            beta = cdr3_core(_drop_placeholder(row.get("betaCDR3")), has_anchors=True)
            alpha = cdr3_core(_drop_placeholder(row.get("alphaCDR3")), has_anchors=True)
            if not ok_epitope(epitope) or not ok_cdr3(beta):
                stats["drop_invalid"] += 1
                continue
            if not ok_cdr3(alpha):
                alpha = ""
                stats["alpha_missing"] += 1
            key = (beta, alpha, epitope)
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            out = empty_row()
            out.update(
                record_id=record_id("tcrdesign26_paired", beta, alpha, epitope),
                source="tcrdesign26_paired",
                fv_source="tcrdesign26_cdr3",
                tier="C",
                task_type="tcr_epitope",
                relation="binding",
                sequence_scope="cdr3",
                cdr3a=alpha,
                cdr3b=beta,
                epitope_seq=epitope,
                provenance="tcrdesign2026:tcrdesign_G/antigen_beta_alpha.csv",
            )
            rows.append(out)
    stats["final"] = len(rows)
    return rows, {"source": "tcrdesign26_paired", "counts": dict(stats),
                  "unique_epitopes": len({r["epitope_seq"] for r in rows})}


def ingest_td26_pmhc(*, neg_ratio: float = 1.0, neg_floor: int = 50, seed: int = 42
                     ) -> tuple[list[dict], dict]:
    """``pMHC_TCR_train.tsv``: V/J + CDR3a/b + epitope + 34aa MHC + binding label.

    Three corpus decisions, all reported:

    * **Rows without a beta CDR3 are dropped.** The file mixes beta-only
      (228,310), alpha-only (147,227) and paired (367,581) rows; our
      epitope-conditioned plane is keyed on CDR3b, and an alpha-only row has no
      beta to condition on.
    * **Label conflicts resolve to binding.** 12,944 ``(CDR3b, epitope)`` pairs
      appear with both labels. A measured binder is stronger evidence than a
      sampled negative, so the positive wins and the negative copy is dropped.
    * **Negatives capped** at ``neg_ratio`` x positives per epitope (floor
      ``neg_floor``), matching :func:`ingest_tcrdiff`. Raw is 375,180 negative vs
      91,205 positive unique pairs; uncapped they would swamp the plane.
    """

    _, pseudo2allele = _pseudo_tables()
    stats = Counter()
    pos_keys: set[tuple[str, str]] = set()
    pos: list[dict] = []
    neg: list[tuple[str, str, dict]] = []
    seen: set[tuple] = set()

    def build(row: dict) -> tuple[tuple, dict] | None:
        epitope = _aa(row.get("epitope"))
        beta = cdr3_core(_drop_placeholder(row.get("cdr3b")), has_anchors=True)
        alpha = cdr3_core(_drop_placeholder(row.get("cdr3a")), has_anchors=True)
        if not ok_cdr3(beta):
            stats["drop_no_beta"] += 1
            return None
        if not ok_epitope(epitope):
            stats["drop_bad_epitope"] += 1
            return None
        if not ok_cdr3(alpha):
            alpha = ""
        pseudo = _aa(row.get("mhc_pseudo"))
        if len(pseudo) != 34:
            stats["drop_bad_pseudo"] += 1
            return None
        allele = pseudo2allele.get(pseudo, "")
        if not allele:
            stats["pseudo_allele_unmapped"] += 1
        try:
            relation = canonical_relation(row.get("label"))
        except ValueError:
            stats["drop_bad_label"] += 1
            return None
        tier, task_type = _tier(has_alpha=bool(alpha), has_mhc=True)
        out = empty_row()
        out.update(
            record_id=record_id("tcrdesign26_pmhc", beta, alpha, epitope, allele, relation),
            source="tcrdesign26_pmhc",
            fv_source="tcrdesign26_cdr3",
            tier=tier,
            task_type=task_type,
            relation=relation,
            sequence_scope="cdr3",
            cdr3a=alpha,
            cdr3b=beta,
            epitope_seq=epitope,
            mhc_seq=pseudo,
            mhc_allele_norm=allele,
            provenance="tcrdesign2026:tcrdesign_B/pMHC_TCR_train.tsv",
        )
        return (beta, alpha, epitope, allele, relation), out

    with TD26_PMHC.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", fieldnames=TD26_PMHC_COLUMNS)
        for raw in reader:
            stats["read"] += 1
            built = build(raw)
            if built is None:
                continue
            key, out = built
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            if out["relation"] == "binding":
                pos_keys.add((out["cdr3b"], out["epitope_seq"]))
                pos.append(out)
            else:
                neg.append((out["epitope_seq"], out["record_id"], out))

    kept_neg: list[dict] = []
    per_epitope: dict[str, list[dict]] = defaultdict(list)
    for epitope, _, out in neg:
        if (out["cdr3b"], epitope) in pos_keys:
            stats["neg_dropped_label_conflict"] += 1
            continue
        per_epitope[epitope].append(out)
    pos_per_epitope = Counter(r["epitope_seq"] for r in pos)
    for epitope, group in per_epitope.items():
        cap = max(neg_floor, int(neg_ratio * pos_per_epitope.get(epitope, 0)))
        group.sort(key=lambda r: _stable_rank(seed, r["record_id"]))
        kept_neg.extend(group[:cap])
        stats["neg_dropped_over_cap"] += max(0, len(group) - cap)

    rows = pos + kept_neg
    stats["final_binding"] = len(pos)
    stats["final_nonbinding"] = len(kept_neg)
    stats["final"] = len(rows)
    return rows, {"source": "tcrdesign26_pmhc", "counts": dict(stats),
                  "unique_epitopes": len({r["epitope_seq"] for r in rows}),
                  "tier_counts": dict(Counter(r["tier"] for r in rows))}


INGESTORS = {
    "tcrt5": ingest_tcrt5,
    "tcrdiff": ingest_tcrdiff,
    "tcrdesign26_beta": lambda: _ingest_cdr3b_epitope(
        "tcrdesign26_beta", TD26_BETA, "betaCDR3", "antigen",
        "tcrdesign2026:tcrdesign_G/antigen_beta.csv",
    ),
    "tcrdesign26_paired": ingest_td26_paired,
    "tcrdesign26_pmhc": ingest_td26_pmhc,
    "gratcr_tep": lambda: _ingest_cdr3b_epitope(
        "gratcr_tep", GRATCR_TEP, "beta", "epitope", "gratcr:TEP.csv"
    ),
    "gratcr_mira": lambda: _ingest_cdr3b_epitope(
        "gratcr_mira", GRATCR_MIRA, "beta", "epitope", "gratcr:MIRA.csv"
    ),
    "epidiff": lambda: _ingest_cdr3b_epitope(
        "epidiff", EPIDIFF_CSV, "CDR3", "Epitope", "tcr_epidiff:Training_Dataset.csv"
    ),
}


def write_unified(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIFIED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in UNIFIED_COLUMNS})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=[*INGESTORS, "all"], default=None)
    args = ap.parse_args()
    if not args.source:
        ap.error("--source is required")

    names = list(INGESTORS) if args.source == "all" else [args.source]
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name in names:
        rows, report = INGESTORS[name]()
        write_unified(rows, UNIFIED_DIR / f"{name}.csv")
        reports[name] = report
        print(json.dumps({name: report}, indent=2, sort_keys=True), flush=True)

    path = UNIFIED_DIR / "ingest_papers_report.json"
    existing = json.loads(path.read_text()) if path.is_file() else {}
    existing.update(reports)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    print(f"\nWROTE {path}")


if __name__ == "__main__":
    main()
