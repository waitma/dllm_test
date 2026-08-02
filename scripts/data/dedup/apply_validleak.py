#!/usr/bin/env python3
"""Drop train rows that reproduce a valid/holdout record (train<->valid leak).

Consumes ``data/dedup/blocklists/validleak_<source>.jsonl`` (produced by
``check_valid_in_train.py``; row indices are into the CURRENT ``train`` shard,
i.e. the downstream-deduped shard that training actually reads) and removes
those rows so the validation set becomes a true held-out set.

Provenance layout after ``--promote``:
    <src>/train_prededup      raw shard (pre any dedup; created by the earlier
                              downstream promote, left untouched here)
    <src>/train_dsonly        the downstream-test-deduped shard (what train was
                              before this step) -- backup, created once
    <src>/train               downstream-test AND valid deduped (training reads)

HARD-ASSERTS the blocklist's recorded train length matches the current shard so
indices cannot be misaligned. valid/holdout shards are never modified.

Run:
    /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
        /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/dedup/apply_validleak.py \
        --source tcr_piste --promote --force
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
GRAMMAR = PROJECT_ROOT / "data/bioseq_grammar_v1"
BLOCK_DIR = PROJECT_ROOT / "data/dedup/blocklists"
REPORT_DIR = PROJECT_ROOT / "data/dedup/reports"


def load_blocked(source: str) -> set[int]:
    path = BLOCK_DIR / f"validleak_{source}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no validleak blocklist for {source}: {path} (run check_valid_in_train first)")
    blocked = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                blocked.add(int(json.loads(line)["row"]))
    return blocked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite existing train_valdedup.")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Swap: train -> train_dsonly (backup, once), filtered -> train.",
    )
    args = parser.parse_args()

    from datasets import load_from_disk

    shard = GRAMMAR / args.source / "train"
    if not shard.exists():
        raise FileNotFoundError(f"grammar shard missing: {shard}")

    report_path = REPORT_DIR / f"valid_in_train_{args.source}.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}

    ds = load_from_disk(str(shard))
    if "n_train_rows" in report and int(report["n_train_rows"]) != len(ds):
        raise SystemExit(
            f"INDEX DESYNC for {args.source}: report n_train_rows={report['n_train_rows']} != "
            f"shard rows={len(ds)}. Re-run check_valid_in_train.py before applying."
        )

    n_total = len(ds)
    blocked = load_blocked(args.source)
    keep = [i for i in range(n_total) if i not in blocked]
    out = GRAMMAR / args.source / "train_valdedup"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")
    if out.exists():
        shutil.rmtree(out)
    ds.select(keep).save_to_disk(str(out), max_shard_size="512MB")
    del ds

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
        dsonly = GRAMMAR / args.source / "train_dsonly"
        if not dsonly.exists():
            shard.rename(dsonly)
        else:
            shutil.rmtree(shard)
        out.rename(shard)
        result["promoted"] = True
        result["backup_dsonly"] = str(dsonly)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
