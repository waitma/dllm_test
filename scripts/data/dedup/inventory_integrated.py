#!/usr/bin/env python3
"""Inventory the integrated grammar mix + effective sampling shares.

Reports, per source and split, the raw row count, the deduped row count (if a
``train_dedup`` / promoted shard + dedup report exist), the assigned mixture
weight, and the resulting sampling share (weight / sum(weight), which is what
``WeightedMixtureDataset`` actually uses -- independent of row count).

Also (with ``--write-manifest``) rewrites ``manifest.json`` to describe the
integrated mix so downstream tooling has an accurate record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
GRAMMAR = PROJECT_ROOT / "data/bioseq_grammar_v1"
REPORT_DIR = PROJECT_ROOT / "data/dedup/reports"

# Integrated mix without MINT pretraining shards (mint_ppi / mint_actions dropped;
# MINT remains a downstream eval suite only). tcr_piste + tcr_pmhc_fulllength cover
# TCR-pMHC; SAbDab2 deferred.
DEFAULT_WEIGHTS = {
    "oas": 3.0,
    "ots": 3.0,
    "nanobody": 2.0,
    "tcr_piste": 1.5,
    "tcr_pmhc_fulllength": 1.5,
    "ppi": 1.0,
    "neutralization": 0.5,
}


def _rows(path: Path) -> int | None:
    if not path.exists():
        return None
    from datasets import load_from_disk

    return len(load_from_disk(str(path)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=",".join(DEFAULT_WEIGHTS))
    parser.add_argument("--weights", default=None, help="Comma list name=w overrides.")
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        for kv in args.weights.split(","):
            k, v = kv.split("=")
            weights[k.strip()] = float(v)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    wsum = sum(weights.get(s, 1.0) for s in sources)

    rows_by_source: dict[str, dict[str, int | None]] = {}
    datasets_manifest = []
    lines = []
    header = f"{'source':<16}{'split':<7}{'raw_rows':>13}{'deduped':>13}{'weight':>8}{'share':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for source in sources:
        rows_by_source[source] = {}
        for split in splits:
            base = GRAMMAR / source / split
            raw = _rows(base)
            # deduped view: promoted (train == deduped, train_prededup holds raw),
            # or a side-by-side train_dedup.
            deduped = None
            if split == "train":
                prededup = GRAMMAR / source / "train_prededup"
                dedup_side = GRAMMAR / source / "train_dedup"
                report = REPORT_DIR / f"{source}.json"
                if prededup.exists():
                    deduped = raw  # already promoted; `train` IS deduped
                    raw = _rows(prededup)
                elif dedup_side.exists():
                    deduped = _rows(dedup_side)
                elif report.is_file():
                    r = json.loads(report.read_text())
                    if raw is not None and "n_leaked_rows" in r:
                        deduped = raw - int(r["n_leaked_rows"])
            rows_by_source[source][split] = deduped if deduped is not None else raw
            w = weights.get(source, 1.0)
            share = w / wsum if split == "train" else 0.0
            raw_s = f"{raw:,}" if raw is not None else "-"
            ded_s = f"{deduped:,}" if deduped is not None else ("=" if raw is not None else "-")
            share_s = f"{share:6.1%}" if split == "train" else ""
            lines.append(f"{source:<16}{split:<7}{raw_s:>13}{ded_s:>13}{w:>8.2f}{share_s:>9}")
            datasets_manifest.append({
                "source": source,
                "split": split,
                "rows": rows_by_source[source][split],
                "raw_rows": raw,
                "weight": w,
                "sampling_share": round(share, 6) if split == "train" else None,
                "path": str(base),
            })

    report_text = "\n".join(lines)
    print(report_text)
    print(f"\nweight sum (train mix) = {wsum:.2f}")

    if args.write_manifest:
        manifest = {
            "format": "bioseq_grammar_v1_semantic_arrow",
            "mix": "integrated_v1_deduped",
            "weights": {s: weights.get(s, 1.0) for s in sources},
            "weight_sum": wsum,
            "datasets": datasets_manifest,
        }
        (GRAMMAR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {GRAMMAR / 'manifest.json'}")


if __name__ == "__main__":
    main()
