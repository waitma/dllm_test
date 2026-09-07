#!/usr/bin/env python3
"""Score the tcr_peptide vs tcr_pmhc layout diagnostic.

The official T4 metric is exact string match of a generated CDR3b against the
reference binder set, which is brutal for de novo generation -- F1=0 only tells
us "no exact hits" and cannot show whether a change moved the model closer to
the reference distribution.

So alongside the official exact-match precision/recall/F1 this also reports the
nearest-neighbour edit distance to any reference binder and the fraction of
designs within edit distance 1/2/3. If the layout fix helps, those soft numbers
move even while exact F1 stays at 0.

Usage:
  python score_layout_diag.py --eval-json EVAL --designs A.jsonl=no_mhc B.jsonl=with_mhc
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_one(gen: list[str], refs: list[str], k: int) -> dict:
    cands = [s.strip().upper() for s in gen[:k] if s and s.strip()]
    ref_set = {s.strip().upper() for s in refs if s and s.strip()}
    if not cands or not ref_set:
        return {}
    correct = [c for c in cands if c in ref_set]
    precision = len(correct) / len(cands)
    denom = min(k, len(ref_set))
    recall = len(set(correct)) / denom if denom else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    nn = [min(edit_distance(c, r) for r in ref_set) for c in cands]
    return {
        "n_gen": len(cands),
        "n_ref": len(ref_set),
        "exact_hits": len(correct),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "nn_edit_mean": statistics.fmean(nn),
        "nn_edit_min": min(nn),
        "frac_within_1": sum(d <= 1 for d in nn) / len(nn),
        "frac_within_2": sum(d <= 2 for d in nn) / len(nn),
        "frac_within_3": sum(d <= 3 for d in nn) / len(nn),
        "uniq_frac": len(set(cands)) / len(cands),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--designs", nargs="+", required=True,
                    help="PATH=LABEL pairs")
    ap.add_argument("-k", type=int, default=100)
    args = ap.parse_args()

    entries = json.loads(Path(args.eval_json).read_text())

    results: dict[str, dict] = {}
    for item in args.designs:
        path_s, _, label = item.partition("=")
        label = label or Path(path_s).stem
        path = Path(path_s)
        if not path.is_file():
            print(f"MISSING {path}")
            continue
        designs = {}
        layouts = set()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            designs[row["pmhc"]] = row["sequences"]
            if row.get("layout"):
                layouts.add(row["layout"])

        per = {p: score_one(designs.get(p, []), e["ref_binders"], args.k)
               for p, e in entries.items() if designs.get(p)}
        per = {p: v for p, v in per.items() if v}
        if not per:
            print(f"{label}: no scorable pMHC")
            continue

        def m(key):
            return statistics.fmean(v[key] for v in per.values())

        results[label] = {
            "layout": ",".join(sorted(layouts)) or "unknown",
            "n_pmhc": len(per),
            "total_exact_hits": sum(v["exact_hits"] for v in per.values()),
            "precision": m("precision"), "recall": m("recall"), "f1": m("f1"),
            "nn_edit_mean": m("nn_edit_mean"),
            "nn_edit_min": min(v["nn_edit_min"] for v in per.values()),
            "frac_within_1": m("frac_within_1"),
            "frac_within_2": m("frac_within_2"),
            "frac_within_3": m("frac_within_3"),
            "uniq_frac": m("uniq_frac"),
            "_per": per,
        }

    hdr = (f"{'label':12s} {'layout':13s} {'n':>3s} {'exact':>6s} {'f1':>7s} "
           f"{'nnEdit':>7s} {'<=1':>6s} {'<=2':>6s} {'<=3':>6s} {'uniq':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for label, r in results.items():
        print(f"{label:12s} {r['layout']:13s} {r['n_pmhc']:>3d} "
              f"{r['total_exact_hits']:>6d} {r['f1']:>7.4f} "
              f"{r['nn_edit_mean']:>7.3f} {r['frac_within_1']:>6.3f} "
              f"{r['frac_within_2']:>6.3f} {r['frac_within_3']:>6.3f} "
              f"{r['uniq_frac']:>6.3f}")

    if len(results) == 2:
        (la, ra), (lb, rb) = list(results.items())
        print(f"\n{lb} vs {la}:")
        print(f"  exact hits      {ra['total_exact_hits']} -> {rb['total_exact_hits']}")
        print(f"  mean NN edit    {ra['nn_edit_mean']:.3f} -> {rb['nn_edit_mean']:.3f} "
              f"({rb['nn_edit_mean'] - ra['nn_edit_mean']:+.3f}, lower is better)")
        print(f"  frac within 2   {ra['frac_within_2']:.4f} -> {rb['frac_within_2']:.4f} "
              f"({rb['frac_within_2'] - ra['frac_within_2']:+.4f})")
        print(f"  best NN edit    {ra['nn_edit_min']} -> {rb['nn_edit_min']}")


if __name__ == "__main__":
    main()
