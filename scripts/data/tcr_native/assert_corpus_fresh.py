#!/usr/bin/env python3
"""Refuse to train on a decontaminated corpus that has gone stale.

A corpus build report asserting ``PASS: true`` only means "clean against the
blocklists *as they were at build time*". On 2026-08-28 that gap bit us: the
tcr_repertoire corpus was built against a 67,013-key T4 reference-binder
blocklist, the blocklist was rebuilt to 68,846 keys 45 minutes later, and 3 T4
answer-key sequences were left sitting in train.csv with the report still
claiming PASS.

This checks two things per corpus:
  1. the report says PASS
  2. every blocklist the build recorded in ``blocklist_provenance`` still has
     the same content hash as the live file

Corpora whose builder does not yet record provenance are reported as UNVERIFIED
rather than silently passing, so the gap is visible.

Usage:
  python assert_corpus_fresh.py REPORT_JSON [REPORT_JSON ...]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def check(report_path: Path) -> list[str]:
    problems: list[str] = []
    if not report_path.is_file():
        return [f"{report_path}: report missing -- corpus was never built"]
    report = json.loads(report_path.read_text())

    if not report.get("PASS"):
        problems.append(f"{report_path}: PASS is not true")

    prov = report.get("blocklist_provenance")
    if not prov:
        print(f"  UNVERIFIED {report_path.parent.name}: builder records no "
              f"blocklist provenance, staleness cannot be checked")
        return problems

    for name, entry in sorted(prov.items()):
        live = Path(entry["path"])
        if not entry.get("exists", True):
            continue
        if not live.is_file():
            problems.append(f"{report_path}: blocklist {name} disappeared ({live})")
            continue
        sha = hashlib.sha1(live.read_bytes()).hexdigest()[:16]
        if sha != entry.get("sha1"):
            problems.append(
                f"{report_path}: blocklist {name} changed since build "
                f"(built against {entry.get('sha1')}, live is {sha}) -- "
                f"rebuild the corpus"
            )
    return problems


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    problems: list[str] = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        problems.extend(check(path))
        if path.is_file():
            r = json.loads(path.read_text())
            print(f"  {path.parent.parent.name}: PASS={r.get('PASS')} "
                  f"mode={r.get('decontam_mode')}")
    if problems:
        print("\n".join(f"FAIL {p}" for p in problems), file=sys.stderr)
        sys.exit(1)
    print("  corpus freshness OK")


if __name__ == "__main__":
    main()
