#!/usr/bin/env python3
"""Probe a CDR-L3 decontamination bucket, which does not exist today.

The antibody banks only carry ``ab_cdrh3.txt`` (heavy CDR3), ``ab_heavy.fasta``
and ``ab_light.fasta``. There is no ``ab_cdrl3`` bank and no light-CDR3
extractor anywhere, so the light side is currently protected only by the
full-light-chain bucket -- which was disabled on 2026-08-16 because germline
sharing made it fire on 305,665 OAS rows whose heavy chain and CDR-H3 were both
clean.

Dropping the full-light bucket was right, but it removed the light side
entirely instead of replacing it with the specific analogue of CDR-H3. This
script measures what a real CDR-L3 bucket would cost, so the two can be
compared on equal footing.

Bank sources for CDR-L3 (benchmarks that actually score light chains):
  * Kong SAbDab CDR-infilling -- ``cdrl3_seq`` field (L1/L2/L3 are scored on
    the SAb23H2 table, and the pairing task generates the whole light chain)
  * comp_chain OAS holdout    -- ``l_cdr3`` column (light-pairing benchmark)
  * FLAb                      -- full light chains only, no CDR3 column, so
                                 CDR-L3 is regex-extracted
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/data/tcr_native"))
sys.path.insert(0, str(PROJECT_ROOT))

from decontam import (  # noqa: E402
    BANK_DIR,
    DATA,
    MMSEQS_BIN,
    canon_core,
    is_cdr3,
    is_valid_protein_sequence,
    normalize_sequence,
)
from decontam_extra import ASD_AB_TRAIN, OAS_TRAIN  # noqa: E402

from dllm.pipelines.bioseq.immune_receptor_v2.export import (  # noqa: E402
    ClusterRule,
    cluster_with_mmseqs,
)

csv.field_size_limit(2**31 - 1)

KONG = DATA / "downstream/cdr_infilling/sabdab_kong"
OAS_HOLDOUT = DATA / "downstream/comp_chain/test_data_oas_holdout.csv"

# Light junction: Cys(104) ... [FW]-G-X-G at FR4 start. Light CDR3s are shorter
# than heavy, so the inner span is tighter than the heavy-chain pattern.
_L_JUNCTION_RE = re.compile(r"C[A-Z]{2,24}?[FW]G[A-Z]G")


def l3_core(chain) -> str:
    """Anchor-free CDR-L3 core from a full light variable domain."""
    s = normalize_sequence(chain)
    best = ""
    for m in _L_JUNCTION_RE.finditer(s):
        junction = m.group(0)[:-3]
        if 5 <= len(junction) <= 25:
            best = junction
    return canon_core(best, has_anchors=True) if best else ""


def load_kong() -> list[dict]:
    rows: dict[str, dict] = {}
    for fold in range(10):
        p = KONG / "cdrh3" / f"fold_{fold}" / "test.json"
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                d = json.loads(line)
                rows[f"{d.get('pdb')}_{d.get('heavy_chain')}"] = d
    return list(rows.values())


def cluster_hits(query: set[str], bank: set[str], tag: str, threads: int,
                 min_id: float) -> set[str]:
    if not query or not bank:
        return set()
    out_dir = DATA / "dedup/clusters_cdrl3" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    res = cluster_with_mmseqs(
        rule=ClusterRule("ab_l3", min_id, 0.80),
        query_sequences=query, benchmark_sequences=bank,
        output_dir=out_dir, mmseqs_bin=Path(MMSEQS_BIN), threads=threads,
    )
    return {q for q in query if res.sequence_to_cluster.get(q, "") in res.benchmark_clusters}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    report: dict = {}
    kong = load_kong()

    # --- extractor sanity: does the regex reproduce the annotated cdrl3_seq? ---
    n = agree = 0
    for d in kong:
        light = normalize_sequence(d.get("light_chain_seq"))
        loop = normalize_sequence(d.get("cdrl3_seq"))
        if not light or not loop:
            continue
        n += 1
        if l3_core(light) == loop:
            agree += 1
    report["extractor_agreement_on_kong"] = {
        "n": n, "agree": agree, "rate": round(agree / n, 4) if n else 0.0,
    }

    # --- build the CDR-L3 bank ---
    bank_l3: set[str] = set()
    prov: dict[str, int] = {}
    before = 0
    for d in kong:
        loop = normalize_sequence(d.get("cdrl3_seq"))
        if loop and is_cdr3(loop, max_len=45):
            bank_l3.add(loop)
    prov["sabdab_kong.cdrl3_seq"] = len(bank_l3) - before
    before = len(bank_l3)

    if OAS_HOLDOUT.is_file():
        with OAS_HOLDOUT.open(newline="") as fh:
            for row in csv.DictReader(fh):
                c = canon_core(row.get("l_cdr3"), has_anchors=True)
                if c and is_cdr3(c, max_len=45):
                    bank_l3.add(c)
    prov["comp_chain_oas_holdout.l_cdr3"] = len(bank_l3) - before
    before = len(bank_l3)

    for line in (BANK_DIR / "ab_light.fasta").read_text().splitlines():
        if line and not line.startswith(">"):
            c = l3_core(line)
            if c and is_cdr3(c, max_len=45):
                bank_l3.add(c)
    prov["ab_light.fasta (regex)"] = len(bank_l3) - before

    report["bank_l3"] = {"total": len(bank_l3), "provenance_new_keys": prov}

    # --- query side: OAS (has l_cdr3 column) and ASD (regex on light_fv) ---
    for src, path, extract in (
        ("oas", OAS_TRAIN, "col"),
        ("asd_antibody", ASD_AB_TRAIN, "regex"),
    ):
        per_row: list[str] = []
        cores: set[str] = set()
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if extract == "col":
                    c = canon_core(row.get("l_cdr3"), has_anchors=True)
                    if not c:
                        c = l3_core(row.get("cleaned_l_sequence") or row.get("l_sequence"))
                else:
                    c = l3_core(row.get("light_fv"))
                per_row.append(c)
                if c and is_cdr3(c, max_len=45):
                    cores.add(c)
        n_rows = len(per_row)
        entry = {
            "n_rows": n_rows,
            "rows_with_l3": sum(1 for c in per_row if c),
            "unique_l3_cores": len(cores),
            "exact_hits": len(cores & bank_l3),
            "rows_exact": sum(1 for c in per_row if c in (cores & bank_l3)),
            "sweep": {},
        }
        for mid in (0.80, 0.90):
            hit = cluster_hits(cores, bank_l3, f"{src}_{int(mid * 100)}", args.threads, mid)
            entry["sweep"][f"{mid:.2f}"] = {
                "cores_hit": len(hit),
                "rows_removed": sum(1 for c in per_row if c in hit),
                "row_frac": round(sum(1 for c in per_row if c in hit) / n_rows, 4),
            }
        report[src] = entry
        print(json.dumps({src: entry}, indent=2), flush=True)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
