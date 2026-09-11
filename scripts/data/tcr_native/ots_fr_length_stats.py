#!/usr/bin/env python3
"""OTS full-length FR / CDR length histograms (ANARCI regions).

chain1 is beta, chain2 is alpha in ots_paired_clean. FR3 includes the
conserved Cys; CDR3 is the anchor-free loop.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
_DLLM_ROOT = HERE.parents[3]
if str(_DLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DLLM_ROOT))

from common import DATA  # noqa: E402

csv.field_size_limit(2**31 - 1)

OTS_FINAL = DATA / "ots_paired_clean/final"
DEFAULT_OUT = DATA / "ots_paired_clean/fr_length_stats.json"

FRS = ("FR1", "FR2", "FR3", "FR4")
CDRS = ("CDR1", "CDR2", "CDR3")
REGIONS = FRS + CDRS


def _hist_stats(counts: Counter[int]) -> dict:
    n = sum(counts.values())
    if n == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None, "histogram": {}}
    running = 0
    median = None
    mid = (n + 1) // 2
    weighted = 0
    for length in sorted(counts):
        c = counts[length]
        weighted += length * c
        running += c
        if median is None and running >= mid:
            median = length
    return {
        "n": n,
        "min": min(counts),
        "max": max(counts),
        "mean": round(weighted / n, 4),
        "median": median,
        "histogram": {str(k): counts[k] for k in sorted(counts)},
    }


def main() -> None:
    hists: dict[str, Counter[int]] = defaultdict(Counter)
    recon_ok = Counter()
    empty = Counter()
    n_rows = 0
    for split in ("train", "valid", "holdout"):
        path = OTS_FINAL / f"{split}.csv"
        print(f"[read] {path}", flush=True)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                n_rows += 1
                for prefix, fallback in (("chain1", "B"), ("chain2", "A")):
                    kind = (row.get(f"{prefix}_anarci_type") or fallback).strip().upper()
                    label = "beta" if kind == "B" else "alpha"
                    full = (row.get(f"cleaned_{prefix}_seq") or "").strip()
                    parts = {}
                    for region in REGIONS:
                        seq = (row.get(f"{prefix}_{region}") or "").strip()
                        parts[region] = seq
                        if not seq:
                            empty[f"{label}_{region}"] += 1
                        hists[f"{label}_{region}"][len(seq)] += 1
                    fr_sum = sum(len(parts[r]) for r in FRS)
                    cdr_sum = sum(len(parts[r]) for r in CDRS)
                    region_sum = fr_sum + cdr_sum
                    hists[f"{label}_FR_sum"][fr_sum] += 1
                    hists[f"{label}_CDR_sum"][cdr_sum] += 1
                    hists[f"{label}_full"][len(full)] += 1
                    hists[f"{label}_regions_concat"][region_sum] += 1
                    recon = "".join(parts[r] for r in ("FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"))
                    if full and recon == full:
                        recon_ok[f"{label}_exact"] += 1
                    elif full and recon.replace("-", "") == full.replace("-", ""):
                        recon_ok[f"{label}_nogap"] += 1
                    else:
                        recon_ok[f"{label}_mismatch"] += 1
                if n_rows % 500_000 == 0:
                    print(f"    rows={n_rows:,}", flush=True)

    series = {name: _hist_stats(counts) for name, counts in sorted(hists.items())}
    payload = {
        "schema_version": "ots_fr_length_stats.v1",
        "n_rows": n_rows,
        "region_note": (
            "ANARCI IMGT: FR3 includes the conserved Cys; "
            "CDR3 is the loop without C/[FW]; FR4 starts at the J-region F/W."
        ),
        "reconstruct_from_regions": dict(recon_ok),
        "empty_region_counts": dict(empty),
        "series": series,
    }
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "n_rows": n_rows,
        "recon": dict(recon_ok),
        "summary": {
            k: {kk: series[k][kk] for kk in ("n", "min", "median", "mean", "max")}
            for k in series
        },
    }, indent=2), flush=True)
    print(f"WROTE {DEFAULT_OUT}", flush=True)


if __name__ == "__main__":
    main()
