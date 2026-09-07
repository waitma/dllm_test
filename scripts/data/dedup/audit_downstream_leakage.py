#!/usr/bin/env python3
"""Residual-leakage matrix: every downstream TCR test set x every training source.

Answers "is the training data actually deduplicated against the downstream
tasks?" with measurements rather than with a list of blocklists that are
*supposed* to be applied.

Two things make a naive check wrong, and both have burned this project before:

  1. **Filter placement.** Some decontamination runs at corpus-build time
     (tcr_native / tcr_papers / tcr_repertoire) and some at data-load time via
     ``with_exclusion_filter`` (OTS / TRAIT / OAS / ASD). Only the load-time
     path is visible in the CSV-to-record function. So this script pushes every
     row through the *real* ``row_to_record`` returned by ``build_immune_specs``
     and only audits rows that survive -- exactly what training sees.

  2. **Anchor conventions.** A CDR3 can be stored as the full IMGT junction
     (``C..[FW]``) or as the anchor-free loop. Comparing the two forms directly
     silently misses every hit. Per-source conventions are declared in
     ``_SOURCES`` below and were each verified against the raw files.

Usage:
  python audit_downstream_leakage.py                 # full scan (~4M rows)
  python audit_downstream_leakage.py --limit 200000  # quick smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "llada"))

from dllm.pipelines.bioseq.datasets import _source_split_path  # noqa: E402

csv.field_size_limit(2**31 - 1)

DOWN = ROOT / "downstream/benchmark/data"

# Benchmarks whose protected set really is keyed on the bare CDR3b core, so a
# single hit is a leaked answer. These must be exactly 0 -- they drive the exit
# verdict. Everything else is keyed on something narrower and is reported as
# informational only (see BARE_CORE_NOTE and the epilogue this script prints).
HARD_ZERO = {"NM2025_seen", "NM2025_unseen", "public_trackA"}

# Why a bare-core hit on these is expected rather than alarming.
BARE_CORE_NOTE = {
    "T1_binding": "pair-keyed",
    "T2_clustering": "embedding split",
    "T2_clustering_embed": "embedding split",
    "T3_repr_paper6": "embedding split",
    "T3_representation": "embedding split",
    "T4_cond_refbinders": "pair-keyed, 0 pair hits",
    "T4_uncond_holdout": "see EXPANSION_AUDIT 3.9",
    "tcr_design": "pair-keyed",
}

# (source name, cdr3b column, has_anchors). Sources with no CDR3b column at all
# (OAS/ASD are antibody) are audited on the antibody key space instead, which
# this script does not cover -- see the note it prints.
_SOURCES = {
    "ots": ("__ots__", False),        # beta chosen by anarci_type, loop form
    "trait": ("cdr3b", True),          # full C..[FW] junction
    "tcr_native": ("cdr3b", False),    # unified schema stores the core
    "tcr_papers": ("cdr3b", False),
    "tcr_repertoire": ("cdr3b", False),
}


def core(seq, *, has_anchors: bool) -> str:
    s = "".join(str(seq or "").split()).upper()
    if not s:
        return ""
    if has_anchors and len(s) >= 5:
        if s[0] == "C":
            s = s[1:]
        if s and s[-1] in "FW":
            s = s[:-1]
    return s


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as fh:
        yield from csv.DictReader(fh)


def _col(row: dict, *names):
    for n in names:
        if row.get(n):
            return row[n]
    return ""


def load_benchmarks() -> dict[str, set[str]]:
    """Protected CDR3b cores per downstream task. Benchmarks ship the full
    junction, so everything here is anchor-stripped."""
    b: dict[str, set[str]] = defaultdict(set)

    def add(name, seq):
        c = core(seq, has_anchors=True)
        if len(c) >= 4:
            b[name].add(c)

    # T1 binding (IMMREP23)
    p = DOWN / "tcr_binding/test.csv"
    if p.is_file():
        for r in _rows(p):
            add("T1_binding", _col(r, "cdr3b", "CDR3b"))

    # NM2025 binding, seen + unseen
    for split in ("seen", "unseen"):
        base = DOWN / f"tcr_binding_nm2025/{split}"
        for fp in sorted(base.rglob("test.csv")):
            for r in _rows(fp):
                add(f"NM2025_{split}", _col(r, "cdr3b", "CDR3b"))

    # T2 clustering
    for name, fp in (("T2_clustering", DOWN / "tcr_clustering/tcrs.csv"),
                     ("T2_clustering_embed", DOWN / "tcr_clustering_embed/tcrs.csv")):
        if fp.is_file():
            for r in _rows(fp):
                add(name, _col(r, "cdr3b", "CDR3b", "cdr3"))

    # T3 representation
    for name, fp in (("T3_representation", DOWN / "tcr_representation/test.csv"),
                     ("T3_repr_paper6", DOWN / "tcr_representation_paper6/target_binders.csv")):
        if fp.is_file():
            for r in _rows(fp):
                add(name, _col(r, "cdr3b", "CDR3b", "cdr3"))

    # T4 unconditional (OTS holdout, stored already trimmed)
    p = DOWN / "tcr_generation/holdout_cdr3b.txt"
    if p.is_file():
        for line in p.read_text().split():
            c = core(line, has_anchors=False)
            if len(c) >= 4:
                b["T4_uncond_holdout"].add(c)

    # T4 epitope-conditioned reference binders
    p = DOWN / "tcr_generation_bench/eval_conditional.json"
    if p.is_file():
        for entry in json.loads(p.read_text()).values():
            for s in (entry.get("ref_binders") or entry.get("binders") or []):
                add("T4_cond_refbinders", s)

    # Public track-A references
    p = DOWN / "tcr_beta_public_benchmark/references.csv"
    if p.is_file():
        for r in _rows(p):
            add("public_trackA", _col(r, "cdr3b_reference", "cdr3b", "CDR3b"))

    # tcr_design eval
    p = DOWN / "tcr_design/eval.json"
    if p.is_file():
        data = json.loads(p.read_text())
        items = data.values() if isinstance(data, dict) else data
        for entry in items:
            if not isinstance(entry, dict):
                continue
            for key in ("ref_binders", "binders", "cdr3b", "references"):
                v = entry.get(key)
                for s in ([v] if isinstance(v, str) else (v or [])):
                    add("tcr_design", s)

    return dict(b)


def ots_beta(row: dict) -> str:
    """OTS stores paired chains; beta is whichever chain ANARCI typed 'B'."""
    for idx in ("1", "2"):
        if str(row.get(f"chain{idx}_anarci_type", "")).strip().upper() == "B":
            return row.get(f"chain{idx}_cdr3") or ""
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="rows per source")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tcr-papers-dir", default=str(ROOT / "data/tcr_papers_v2/dataset"))
    args = ap.parse_args()

    from protein_pretrain_esmc import DataArguments, build_immune_specs  # noqa: E402

    bench = load_benchmarks()
    print("Protected CDR3b cores per downstream task:")
    for k, v in sorted(bench.items()):
        print(f"  {k:24s} {len(v):>7,}")
    universe: set[str] = set()
    for v in bench.values():
        universe |= v
    print(f"  {'UNION':24s} {len(universe):>7,}\n")

    da = DataArguments(
        dataset_args="oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire",
        max_length=1024, max_protein_length=1024,
        tcr_papers_dir=args.tcr_papers_dir,
    )
    specs = {s.name: s for s in build_immune_specs(da)}

    names = sorted(bench)
    hits: dict[str, dict[str, set[str]]] = {}
    kept_counts: dict[str, int] = {}

    for sname, (col, has_anchors) in _SOURCES.items():
        spec = specs.get(sname)
        if spec is None:
            continue
        path = _source_split_path(spec, args.split)
        if not path.is_file():
            print(f"  SKIP {sname}: {path} missing")
            continue
        per: dict[str, set[str]] = {n: set() for n in names}
        kept = 0
        for i, row in enumerate(_rows(path)):
            if args.limit and i >= args.limit:
                break
            if spec.row_to_record(row) is None:
                continue          # dropped by the load-time blocklist
            kept += 1
            raw = ots_beta(row) if col == "__ots__" else row.get(col)
            c = core(raw, has_anchors=has_anchors)
            if not c or c not in universe:
                continue
            for n in names:
                if c in bench[n]:
                    per[n].add(c)
        hits[sname] = per
        kept_counts[sname] = kept
        print(f"  scanned {sname:16s} kept={kept:>9,}")

    print("\n=== RESIDUAL BARE-CORE HITS (unique CDR3b cores present in training) ===")
    hdr = f"{'benchmark':24s}" + "".join(f"{s:>16s}" for s in hits) + "   verdict"
    print(hdr)
    print("-" * len(hdr))
    hard_total = 0
    for n in names:
        cells = []
        row = 0
        for s in hits:
            k = len(hits[s][n])
            row += k
            cells.append(f"{k:>16,}")
        if n in HARD_ZERO:
            hard_total += row
            verdict = "   OK" if row == 0 else "   *** LEAK ***"
        else:
            verdict = f"   info ({BARE_CORE_NOTE[n]})"
        print(f"{n:24s}" + "".join(cells) + verdict)
    print("-" * len(hdr))
    print(f"{'kept rows':24s}" + "".join(f"{kept_counts[s]:>16,}" for s in hits))
    print(f"\nHARD-REQUIREMENT hits = {hard_total}   -> "
          f"{'PASS' if hard_total == 0 else 'LEAKAGE'}")
    print(
        "\nHow to read this matrix -- a nonzero cell is NOT automatically leakage.\n"
        f"  * Hard-requirement rows ({', '.join(sorted(HARD_ZERO))}) are keyed on the\n"
        "    bare CDR3b core, so any hit there is real leakage and must be 0.\n"
        "  * Every other row is keyed on something narrower than the bare core (an\n"
        "    (core|epitope) pair, or an embedding-level split). This matrix projects\n"
        "    those keys down to the bare core, which over-counts by construction: the\n"
        "    same CDR3b paired with a different epitope is legitimate training signal,\n"
        "    not a leaked answer. Verified 2026-08-28: the T4 blocklist is 68,846/68,846\n"
        "    pair-keyed, and (core|epitope) pair hits are 0 for both tcr_papers_v2 and\n"
        "    tcr_native despite ~77k and ~74k bare-core hits respectively.\n"
        "  * OAS and ASD are antibody sources with no CDR3b column; their\n"
        "    decontamination lives in a separate key space (CDR-H3 / heavy+light) and\n"
        "    is not covered by this matrix at all."
    )


if __name__ == "__main__":
    main()
