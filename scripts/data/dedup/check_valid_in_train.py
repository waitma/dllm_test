#!/usr/bin/env python3
"""Check that each source's ``valid`` (and optionally ``holdout``) records do
NOT appear in its (deduped) ``train`` grammar shard.

This validates the reliability of the validation set used for the integrated
ESMC-300M run: a valid record that also lives in train means val-loss measures
memorization, not generalization. The existing leakage tool
(``dedup_train_vs_downstream.py``) only ever handled ``SPLITS = ("train",)``
against the downstream TEST banks; it never checked train<->valid. This script
fills that gap with an exact (whole-record) match.

Key = SHA1 of the sorted, normalized chain sequences of a record
(``records.normalize_sequence``: strip whitespace, upper-case, J->L), so chain
order does not matter (PPI A+B == B+A) and the comparison matches the shard's
own normalization.

Run:
    /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
        /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/dedup/check_valid_in_train.py \
        --sources oas,ots,nanobody,mint_ppi,tcr_piste,ppi,mint_actions,neutralization \
        --splits valid

Writes ``data/dedup/reports/valid_in_train_<src>.json`` per source and prints a
summary table. Also emits the blocked train row indices (rows reproducing a
valid/holdout record) to ``data/dedup/blocklists/validleak_<src>.jsonl`` so they
can be dropped from train with the same index-based promotion flow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
GRAMMAR = PROJECT_ROOT / "data/bioseq_grammar_v1"
REPORT_DIR = PROJECT_ROOT / "data/dedup/reports"
BLOCK_DIR = PROJECT_ROOT / "data/dedup/blocklists"

SOURCES = ("oas", "ots", "nanobody", "mint_ppi", "tcr_piste", "ppi", "mint_actions", "neutralization")


def normalize_sequence(value: object) -> str:
    return "".join(str(value or "").split()).upper().replace("J", "L")


def key_hash(chains: list[str]) -> bytes:
    joined = "\x00".join(sorted(normalize_sequence(c) for c in chains))
    return hashlib.sha1(joined.encode()).digest()


def _load(path: Path):
    from datasets import load_from_disk

    return load_from_disk(str(path))


def build_ref_keys(source: str, splits: list[str], batch_size: int) -> dict[bytes, str]:
    """Union of record keys across the given reference splits (valid/holdout)."""
    ref: dict[bytes, str] = {}
    for split in splits:
        path = GRAMMAR / source / split
        if not path.exists():
            continue
        try:
            ds = _load(path)
        except Exception:
            continue
        for batch in ds.select_columns(["chains"]).iter(batch_size=batch_size):
            for chains in batch["chains"]:
                ref[key_hash(chains)] = split
    return ref


def run_source(source: str, splits: list[str], batch_size: int) -> dict:
    train_path = GRAMMAR / source / "train"
    if not train_path.exists():
        return {"source": source, "error": f"missing train shard: {train_path}"}

    ref = build_ref_keys(source, splits, batch_size)
    if not ref:
        return {"source": source, "note": f"no reference rows for splits={splits}", "n_ref_keys": 0}

    ds = _load(train_path)
    n_train = len(ds)
    blocked_rows: list[int] = []
    hit_keys: set[bytes] = set()
    idx = 0
    for batch in ds.select_columns(["chains"]).iter(batch_size=batch_size):
        for chains in batch["chains"]:
            h = key_hash(chains)
            if h in ref:
                blocked_rows.append(idx)
                hit_keys.add(h)
            idx += 1

    BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    block_path = BLOCK_DIR / f"validleak_{source}.jsonl"
    with block_path.open("w") as fh:
        for r in blocked_rows:
            fh.write(json.dumps({"row": r}) + "\n")

    result = {
        "source": source,
        "splits": splits,
        "n_ref_keys": len(ref),
        "n_train_rows": n_train,
        "train_rows_hitting_ref": len(blocked_rows),
        "ref_keys_found_in_train": len(hit_keys),
        "ref_leak_frac": round(len(hit_keys) / max(len(ref), 1), 6),
        "blocklist": str(block_path),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"valid_in_train_{source}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--splits", default="valid", help="Comma list, e.g. valid or valid,holdout")
    parser.add_argument("--batch-size", type=int, default=20000)
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    rows = []
    for source in sources:
        res = run_source(source, splits, args.batch_size)
        rows.append(res)
        print(json.dumps(res))

    header = f"{'source':<16}{'ref_keys':>12}{'train_rows':>14}{'train_hits':>12}{'ref_leaked':>12}{'leak%':>9}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        if "error" in r or r.get("n_ref_keys", 0) == 0:
            note = r.get("error") or r.get("note", "")
            print(f"{r['source']:<16}{note}")
            continue
        print(
            f"{r['source']:<16}{r['n_ref_keys']:>12,}{r['n_train_rows']:>14,}"
            f"{r['train_rows_hitting_ref']:>12,}{r['ref_keys_found_in_train']:>12,}"
            f"{r['ref_leak_frac'] * 100:>8.3f}%"
        )


if __name__ == "__main__":
    main()
