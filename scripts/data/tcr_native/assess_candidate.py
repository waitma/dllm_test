#!/usr/bin/env python3
"""Net-new assessment for a candidate epitope-conditioned TCR source.

Before ingesting a paper's corpus it is worth knowing what it actually adds.
A source can look large and still be worthless: TCRdesign-2025 / UniPMT ships
77,528 rows that collapse to 2,132 unique ``(CDR3b core, epitope)`` pairs, of
which 97.8% are already in our corpus and 66.6% are T4 answer keys -- 14 net
new pairs and zero net new epitopes.

Reports, against the live corpus (tcr_native + tcr_papers + TRAIT) and the T4
reference-binder blocklist:
  * rows read / unique pairs / unique epitopes
  * how many pairs are T4 answer keys (must be dropped)
  * how many pairs we already have
  * NET NEW pairs and epitopes
  * whether the epitopes can be mapped to an allele we hold a 34-mer
    pseudosequence for (decides tier A vs tier C ingestion)

Usage:
  python assess_candidate.py --path FILE --epitope-col Antigen --cdr3b-col CDR3 \
      [--cdr3b-has-anchors] [--allele-col HLA] [--label-col Label] [--sep ,]
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

from common import cdr3_core, normalize_sequence  # noqa: E402

csv.field_size_limit(2**31 - 1)

PROJECT_ROOT = HERE.parents[3]
DATA = PROJECT_ROOT / "data"
T4_BLOCKLIST = DATA / "tcr_native/dataset/t4_refbinder_blocklist.txt"
LIVE = [
    (DATA / "tcr_native/dataset/train.csv", False),
    (DATA / "tcr_papers/dataset/train.csv", False),
    (PROJECT_ROOT / "downstream/trait/step4_final/train.csv", True),
]


def load_live():
    """(pairs, epitopes, epitope->allele counter, allele->34mer pseudo)."""
    pairs: set[str] = set()
    eps: set[str] = set()
    ep2allele: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    allele2pseudo: dict[str, str] = {}
    per_file = {}
    for path, anch in LIVE:
        if not path.is_file():
            per_file[path.name] = 0
            continue
        n = 0
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                ep = normalize_sequence(row.get("epitope_seq") or "")
                core = cdr3_core(row.get("cdr3b") or "", has_anchors=anch)
                allele = (row.get("mhc_allele_norm") or "").strip()
                pseudo = normalize_sequence(row.get("mhc_seq") or "")
                if allele and len(pseudo) == 34:
                    allele2pseudo.setdefault(allele, pseudo)
                if ep and allele:
                    ep2allele[ep][allele] += 1
                if core and ep:
                    pairs.add(f"{core}|{ep}")
                    eps.add(ep)
                    n += 1
        per_file[str(path.relative_to(PROJECT_ROOT))] = n
    return pairs, eps, ep2allele, allele2pseudo, per_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--epitope-col", required=True)
    ap.add_argument("--cdr3b-col", required=True)
    ap.add_argument("--cdr3b-has-anchors", action="store_true",
                    help="set when the CDR3b column is the full C...F junction")
    ap.add_argument("--allele-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--positive-values", default="1,True,true,binding,Binding")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--name", default=None)
    ap.add_argument("--filter-col", default=None,
                    help="e.g. Gene, to keep only beta-chain rows in VDJdb dumps")
    ap.add_argument("--filter-value", default=None)
    ap.add_argument("--header", default=None,
                    help="comma-separated column names for headerless files "
                         "(e.g. TcrDesign-2026 pMHC_TCR_train.tsv)")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print(f"MISSING {path}")
        return
    pos_vals = set(args.positive_values.split(","))

    keys: set[str] = set()
    eps: set[str] = set()
    rows = pos = skipped = 0
    alleles: collections.Counter = collections.Counter()
    # utf-8-sig: several of these dumps ship a BOM, which otherwise renames the
    # first column to "\ufeff<name>" and silently zeroes the whole assessment.
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(
            fh, delimiter=args.sep,
            fieldnames=args.header.split(",") if args.header else None,
        )
        for row in reader:
            rows += 1
            if args.filter_col and str(row.get(args.filter_col)) != args.filter_value:
                skipped += 1
                continue
            ep = normalize_sequence(row.get(args.epitope_col) or "")
            core = cdr3_core(row.get(args.cdr3b_col) or "",
                             has_anchors=args.cdr3b_has_anchors)
            if args.allele_col:
                a = (row.get(args.allele_col) or "").strip()
                if a:
                    alleles[a] += 1
            if not ep or not core:
                skipped += 1
                continue
            keys.add(f"{core}|{ep}")
            eps.add(ep)
            if args.label_col is None or str(row.get(args.label_col)) in pos_vals:
                pos += 1

    t4 = {l.strip() for l in T4_BLOCKLIST.read_text().splitlines() if l.strip()} \
        if T4_BLOCKLIST.is_file() else set()
    live, live_eps, ep2allele, allele2pseudo, per_file = load_live()

    net = keys - t4 - live
    net_eps = eps - live_eps
    name = args.name or path.name

    print(f"=== {name} ===")
    print(f"  path            {path}")
    print(f"  rows            {rows:,}   (skipped {skipped:,}, positive-ish {pos:,})")
    print(f"  unique pairs    {len(keys):,}")
    print(f"  unique epitopes {len(eps):,}")
    if alleles:
        print(f"  allele values   {len(alleles):,}  e.g. {list(alleles)[:3]}")
    print()
    print(f"  live corpus     {len(live):,} pairs / {len(live_eps):,} epitopes")
    for k, v in per_file.items():
        print(f"                    {v:>9,}  {k}")
    print(f"  T4 blocklist    {len(t4):,} keys")
    print()
    if keys:
        print(f"  in T4           {len(keys & t4):>9,}  ({100 * len(keys & t4) / len(keys):5.1f}%)")
        print(f"  already have    {len(keys & live):>9,}  ({100 * len(keys & live) / len(keys):5.1f}%)")
        print(f"  NET NEW pairs   {len(net):>9,}  ({100 * len(net) / len(keys):5.1f}%)")
    print(f"  NET NEW epitopes{len(net_eps):>9,}  of {len(eps):,}")

    mappable = sum(1 for e in eps if e in ep2allele
                   and ep2allele[e].most_common(1)[0][0] in allele2pseudo)
    print(f"  epitopes we can attach OUR 34-mer pseudo to: {mappable:,}/{len(eps):,}"
          f"  -> tier {'A (with MHC)' if mappable == len(eps) else 'A partial / C fallback'}")


if __name__ == "__main__":
    main()
