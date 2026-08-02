#!/usr/bin/env python3
"""Apply a source's leakage blocklist to its grammar Arrow shard.

``dedup_train_vs_downstream.py`` numbers leaked rows by the extractor's
iteration order. For CSV sources (oas/ots/nanobody) that order is
``csv.DictReader`` order; for Arrow sources it is ``load_from_disk`` order.

Key invariant we exploit + verify: the grammar shard builder iterates each raw
source *in order* and (for these sources) drops zero rows, so the training
Arrow shard row order equals the extractor order. This script therefore filters
the Arrow shard directly (what training actually reads) and HARD-ASSERTS that
the shard length equals the report's ``n_rows``; on mismatch it refuses to run
(indices would be misaligned) and you must rebuild from the deduped CSV.

Output: ``data/bioseq_grammar_v1/<source>/train_dedup`` (original untouched).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DATA = PROJECT_ROOT / "data"
BLOCK_DIR = DATA / "dedup" / "blocklists"
REPORT_DIR = DATA / "dedup" / "reports"
GRAMMAR = DATA / "bioseq_grammar_v1"

SOURCES = ("oas", "nanobody", "ots", "tcr", "tcr_piste", "ppi", "mint_ppi", "mint_actions", "neutralization")


def load_blocked(source: str) -> set[int]:
    path = BLOCK_DIR / f"{source}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no blocklist for {source}: {path} (run dedup first)")
    blocked = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                blocked.add(int(json.loads(line)["row"]))
    return blocked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--force", action="store_true", help="Overwrite existing train_dedup.")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="After building, atomically swap: train -> train_prededup, train_dedup -> train.",
    )
    args = parser.parse_args()

    import shutil

    from datasets import load_from_disk

    shard = GRAMMAR / args.source / "train"
    if not shard.exists():
        raise FileNotFoundError(f"grammar shard missing: {shard}")

    report_path = REPORT_DIR / f"{args.source}.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    ds = load_from_disk(str(shard))
    if "n_rows" in report and int(report["n_rows"]) != len(ds):
        raise SystemExit(
            f"INDEX DESYNC for {args.source}: report n_rows={report['n_rows']} != "
            f"shard rows={len(ds)}. Rebuild shard from the deduped CSV instead of "
            f"filtering by index."
        )

    n_total = len(ds)
    blocked = load_blocked(args.source)
    keep = [i for i in range(n_total) if i not in blocked]
    out = GRAMMAR / args.source / "train_dedup"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")
    if out.exists():
        shutil.rmtree(out)
    ds.select(keep).save_to_disk(str(out), max_shard_size="512MB")
    del ds  # drop the mmap before any rename/promote

    result = {
        "source": args.source,
        "shard": str(shard),
        "output": str(out),
        "n_rows": n_total,
        "blocked": len(blocked),
        "kept": len(keep),
        "dropped": n_total - len(keep),
        "drop_frac": round((n_total - len(keep)) / n_total, 6) if n_total else 0.0,
    }

    if args.promote:
        # Atomic-ish swap so training reads the deduped rows under the normal
        # `train` path. Original is preserved as `train_prededup` (once).
        prededup = GRAMMAR / args.source / "train_prededup"
        if not prededup.exists():
            shard.rename(prededup)
        else:
            shutil.rmtree(shard)  # a prior promote already backed up the raw shard
        out.rename(shard)
        result["promoted"] = True
        result["backup"] = str(prededup)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
