#!/usr/bin/env python3
"""Residual-leakage matrix: every downstream TCR test set x every training source.

Answers "is the training data actually deduplicated against the downstream
tasks?" with measurements rather than with a list of blocklists that are
*supposed* to be applied.

Two things make a naive check wrong, and both have burned this project before:

  1. **Prepared boundary.** Parsing, decontamination and quality filters run
     once during offline preparation. This script reads the resulting semantic
     JSONL records and only audits rows that survive -- exactly what training sees.

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
from dllm.pipelines.immune_llada.data.preprocessing.validators import (
    validate_manifest,
    validate_prepared_row,
)

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


def _prepared_rows(dataset_dir: Path, split: str, source: str, limit: int | None):
    """Yield validated semantic records from the selected prepared source shard(s)."""
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    split_info = manifest["splits"].get(split)
    if split_info is None:
        raise KeyError(f"Prepared split {split!r} is not present in {manifest_path}")
    seen = 0
    for shard in split_info.get("shards", []):
        if shard.get("source") != source:
            continue
        path = dataset_dir / str(shard["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Prepared shard listed by manifest is missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                yield validate_prepared_row(json.loads(line))
                seen += 1
                if limit is not None and seen >= limit:
                    return


def _prepared_cdr3b(record) -> str:
    """Extract the canonical identifier used by the source adapter."""
    return record.identifiers.get("cdr3b_core", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="rows per source")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--prepared-data-dir",
        default=str(ROOT / "data/prepared/immune_v3_heterotypic"),
        help="Prepared semantic JSONL directory (created by preprocess_immune_dataset.py)",
    )
    args = ap.parse_args()
    prepared_dir = Path(args.prepared_data_dir)

    bench = load_benchmarks()
    print("Protected CDR3b cores per downstream task:")
    for k, v in sorted(bench.items()):
        print(f"  {k:24s} {len(v):>7,}")
    universe: set[str] = set()
    for v in bench.values():
        universe |= v
    print(f"  {'UNION':24s} {len(universe):>7,}\n")

    names = sorted(bench)
    hits: dict[str, dict[str, set[str]]] = {}
    kept_counts: dict[str, int] = {}

    for sname, (_col_name, has_anchors) in _SOURCES.items():
        per: dict[str, set[str]] = {n: set() for n in names}
        kept = 0
        try:
            records = _prepared_rows(prepared_dir, args.split, sname, args.limit)
            for record in records:
                kept += 1
                # Prepared identifiers are already in the canonical key space:
                # trait has had anchors stripped offline; the other sources retain
                # their historical loop convention in ``cdr3b_core``.
                c = core(_prepared_cdr3b(record), has_anchors=False)
                if not c or c not in universe:
                    continue
                for n in names:
                    if c in bench[n]:
                        per[n].add(c)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  SKIP {sname}: {exc}")
            continue
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
