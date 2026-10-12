#!/usr/bin/env python
"""Aggregate T3 (TCR representation) results into a single summary.

Reads every ``outputs/tcr_representation/<tag>/fewshot.json`` (main protocol) and
``metrics.json`` (auxiliary 24-way probe), and prints:
- a few-shot AUROC-vs-shots table (macro mean±std over epitopes), and
- the auxiliary probe/kNN table,
then writes a separate report under --output-dir (must be new).

    python scripts/summarize_tcr_representation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from common.reporting import compare_protocols, new_report_dir, write_report_manifest

OUT = BENCH / "outputs" / "tcr_representation"

# Display order + friendly labels (distance/comparison methods first).
ORDER = [
    ("levenshtein", "CDR3-Levenshtein-NN"),
    ("tcrdist", "TCRdist (tcrdist3)"),
    ("sceptr", "SCEPTR"),
    ("tcrbert", "TCR-BERT"),
    ("protbert", "ProtBERT"),
    ("esm2_150m", "ESM2-150M"),
    ("kmer", "k-mer(3)"),
    ("ophiuchus", "Ophiuchus-Ab"),
    # Ours — post-LLaDA headline (grammar:decoder:global whole-record mean-pool).
    ("ours_globalfeat_esmc300m_cmp500k_llada", "Ours esmc300m-cmp500k (post-LLaDA globalfeat)"),
    ("ours_globalfeat_esmc600m_cmp500k_llada", "Ours esmc600m-cmp500k (post-LLaDA globalfeat)"),
    ("ours_globalfeat_esmc300m_mint_llada", "Ours esmc300m-mint (post-LLaDA globalfeat)"),
    # Legacy / old ckpt (resume 前, 仅历史参照).
    ("bioseq_llada_esmc300m_cmp500k_llada", "Ours-BioSeq esmc300m-cmp500k (bioseq-llada, 旧 ckpt)"),
    ("bioseq_llada_esmc300m_mint_llada", "Ours-BioSeq esmc300m-mint (bioseq-llada, 旧 ckpt)"),
    ("bioseq_llada_esmc600m_cmp500k_llada", "Ours-BioSeq esmc600m-cmp500k (bioseq-llada, 旧 ckpt)"),
    ("grammar_tcrpair_esmc300m_cmp500k_llada", "Ours esmc300m-cmp500k (seg-pool, 旧 ckpt)"),
    ("grammar_tcrpair_esmc300m_mint_llada", "Ours esmc300m-mint (seg-pool, 旧 ckpt)"),
    ("grammar_tcrpair_esmc600m_cmp500k_llada", "Ours esmc600m-cmp500k (seg-pool, 旧 ckpt)"),
]
SHOW_SHOTS = [1, 2, 5, 10, 20, 50, 100]


def _load(tag, fname):
    p = OUT / tag / fname
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    destination = new_report_dir(args.output_dir, "t3")
    tags = [t for t, _ in ORDER] + sorted(
        d.name for d in OUT.iterdir()
        if d.is_dir() and d.name not in {t for t, _ in ORDER}
    )
    labels = dict(ORDER)

    summary = {"fewshot": {}, "probe": {}}
    print("\n=== T3 few-shot per-epitope NN AUROC (macro mean over epitopes) ===")
    header = "method".ljust(34) + "".join(f"k={k}".rjust(9) for k in SHOW_SHOTS)
    print(header)
    print("-" * len(header))
    for tag in tags:
        fs = _load(tag, "fewshot.json")
        if not fs:
            continue
        label = labels.get(tag, fs.get("model", tag))
        row = label.ljust(34)
        rec = {}
        for k in SHOW_SHOTS:
            s = fs["auroc_by_shot"].get(str(k)) or fs["auroc_by_shot"].get(k)
            if s and s["mean"] == s["mean"]:  # not NaN
                row += f"{s['mean']:.3f}".rjust(9)
                rec[k] = {"mean": s["mean"], "std": s["std"],
                          "n_epitopes": fs["n_epitopes_by_shot"].get(str(k), fs["n_epitopes_by_shot"].get(k))}
            else:
                row += "-".rjust(9)
        print(row)
        summary["fewshot"][tag] = {"model": fs.get("model", tag), "by_shot": rec,
                                   "seeds": fs.get("seeds")}

    print("\n=== T3 auxiliary 24-way probe / kNN ===")
    print("method".ljust(34) + "probe-AUROC".rjust(12) + "probe-Acc".rjust(11) + "kNN-top1".rjust(10))
    print("-" * 67)
    for tag in tags:
        m = _load(tag, "metrics.json")
        if not m:
            continue
        label = labels.get(tag, m.get("model", tag))
        pa = m.get("probe_auroc", float("nan"))
        print(label.ljust(34) + f"{pa:.3f}".rjust(12)
              + f"{m.get('probe_acc', float('nan')):.3f}".rjust(11)
              + f"{m.get('knn_top1_acc', float('nan')):.3f}".rjust(10))
        summary["probe"][tag] = {"model": m.get("model", tag),
                                 "probe_auroc": pa, "probe_acc": m.get("probe_acc"),
                                 "knn_top1_acc": m.get("knn_top1_acc")}

    for group in ("fewshot", "probe"):
        for result in summary[group].values():
            result.update(compare_protocols(None, None))
    with (destination / "_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    write_report_manifest(destination, [Path(__file__), BENCH / "common" / "reporting.py", *OUT.glob("*/fewshot.json"), *OUT.glob("*/metrics.json")],
                          policy="historical_local_reference; no_strict_ranking")
    print(f"\n-> {destination}/_summary.json")


if __name__ == "__main__":
    main()
