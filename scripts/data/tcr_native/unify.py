#!/usr/bin/env python3
"""Unify per-source tier CSVs into one tcr_native corpus.

Concatenates ``data/tcr_native/unified/{piste,fulllength,tenx,minervina,covidvac}.csv``
(whichever are present), applies cross-source exact dedup on
``(cdr3b_core, cdr3a_core, epitope, mhc_allele_norm, relation)`` preferring the
richest row (full Fv over CDR3, tier A>B>C), and writes ``all_unified.csv`` plus
per-source / per-tier / per-scope counts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import DATA, UNIFIED_COLUMNS, cdr3_core, normalize_sequence

OUT_DIR = DATA / "tcr_native/unified"
SOURCES = ["piste", "fulllength", "tenx", "minervina", "covidvac"]

_TIER_RANK = {"A": 3, "B": 2, "C": 1, "": 0}
_SCOPE_RANK = {"fv": 2, "cdr3": 1, "": 0}


def _dedup_key(row: dict) -> tuple:
    cb = cdr3_core(row.get("cdr3b") or "", has_anchors=False)
    ca = cdr3_core(row.get("cdr3a") or "", has_anchors=False)
    return (cb, ca, normalize_sequence(row.get("epitope_seq")),
            (row.get("mhc_allele_norm") or ""), (row.get("relation") or ""))


def _richness(row: dict) -> tuple:
    return (
        _SCOPE_RANK.get(row.get("sequence_scope", ""), 0),
        _TIER_RANK.get(row.get("tier", ""), 0),
        len(row.get("beta_fv") or "") + len(row.get("alpha_fv") or ""),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "all_unified.csv")
    args = ap.parse_args()

    best: dict[tuple, dict] = {}
    per_source_read = Counter()
    dup_collapsed = 0
    for name in SOURCES:
        p = OUT_DIR / f"{name}.csv"
        if not p.is_file():
            continue
        with p.open(newline="") as handle:
            for row in csv.DictReader(handle):
                per_source_read[name] += 1
                key = _dedup_key(row)
                if key not in best:
                    best[key] = row
                else:
                    dup_collapsed += 1
                    if _richness(row) > _richness(best[key]):
                        best[key] = row

    rows = list(best.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIFIED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in UNIFIED_COLUMNS})

    by_source = Counter(r["source"] for r in rows)
    by_tier = Counter(r["tier"] for r in rows)
    by_scope = Counter(r["sequence_scope"] for r in rows)
    by_relation = Counter(r["relation"] for r in rows)
    src_tier = defaultdict(Counter)
    src_scope = defaultdict(Counter)
    for r in rows:
        src_tier[r["source"]][r["tier"]] += 1
        src_scope[r["source"]][r["sequence_scope"]] += 1

    report = {
        "input_rows_read": dict(per_source_read),
        "duplicates_collapsed": dup_collapsed,
        "final_rows": len(rows),
        "by_source": dict(by_source),
        "by_tier": dict(by_tier),
        "by_scope": dict(by_scope),
        "by_relation": dict(by_relation),
        "by_source_tier": {k: dict(v) for k, v in src_tier.items()},
        "by_source_scope": {k: dict(v) for k, v in src_scope.items()},
        "output": str(args.out),
    }
    (OUT_DIR / "unify_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
