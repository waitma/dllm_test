#!/usr/bin/env python
"""Score the OFFICIAL stored TCRT5 / GRATCR / ER-Transformer predictions.

``benchmark_data_w_preds.csv`` ships the exact predictions the TCRT5 paper
produced for 14 pMHC. The paper reserves RVRAYTYSK/HLA-A*03:01 for its separate
simulation and reports the sparse aggregate on the remaining 13 pMHC. Scoring
that 13-pMHC view with our ported metric
suite gives authoritative baseline leaderboard rows without re-generation, and
doubles as the calibration cross-check (see ``scripts/test_tcrt5_metrics.py``).

Run:
    python tcr_generation_bench/score_official.py -k 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from tcr_generation_bench import scoring  # noqa: E402

OUT = BENCH / "outputs" / "tcr_generation_bench" / "setting_B"
METHODS = {
    "tcrt5_official": ("tcrt5", "tcrt5_greedy"),
    "gratcr_official": ("gratcr", "gratcr_greedy"),
    "er_official": ("er", "er_greedy"),
}
RESERVED_SIMULATION_PMHC = scoring.RESERVED_SIMULATION_PMHC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--k", type=int, default=100)
    ap.add_argument("--include-reserved-simulation", action="store_true",
                    help="score all 14 as a labelled control instead of the paper's sparse-13 view")
    args = ap.parse_args()

    eval_all = scoring.load_eval_conditional()
    eval14 = {p: e for p, e in eval_all.items() if e["category"] == "benchmark14"}
    if args.include_reserved_simulation:
        eval_set = eval14
        eval_name = "benchmark14_including_reserved_control"
        out_suffix = "_benchmark14_control"
        alignment = "not_comparable"
    else:
        eval_set = {
            p: {**entry, "category": "paper_sparse13"}
            for p, entry in eval14.items() if p != RESERVED_SIMULATION_PMHC
        }
        if len(eval_set) != 13:
            raise RuntimeError(f"expected paper sparse benchmark n=13, got {len(eval_set)}")
        eval_name = "paper_sparse13"
        out_suffix = ""
        alignment = "exact"
    preds = scoring.load_benchmark_preds()

    for tag, (pred_key, greedy_key) in METHODS.items():
        output_tag = tag + out_suffix
        model_label = tag.replace("_official", "").upper() + " (official preds)"
        model_label += (
            " [14-pMHC control]" if args.include_reserved_simulation
            else " [paper sparse-13]"
        )
        designs = {p: preds[p][pred_key] for p in eval_set if p in preds}
        greedy = {p: preds[p].get(greedy_key, "") for p in eval_set if p in preds}
        scored = scoring.score_conditional(designs, eval_set, k=args.k, greedy_map=greedy)
        result = {"setting": "B", "method": tag, "model": model_label,
                  "eval_set": eval_name, "k": args.k, "source": "benchmark_data_w_preds.csv",
                  "excluded_pmhc": [] if args.include_reserved_simulation else [RESERVED_SIMULATION_PMHC],
                  "baseline_provenance": {
                      "evidence_type": "official_artifact_rescored",
                      "protocol_alignment": alignment,
                      "paper_comparable": alignment == "exact",
                      "citation": "doi:10.1038/s42256-025-01096-6",
                      "source_location": "authors' benchmark_data_w_preds.csv",
                      "notes": "paper Figure 5 sparse aggregate excludes the reserved RVRAYTYSK simulation pMHC",
                  }, **scored}
        out_dir = OUT / output_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        view = "benchmark14" if args.include_reserved_simulation else "paper_sparse13"
        s = scored["summary"][view]
        print(f"{output_tag:28s} n={s.get('n_pmhc_scored')}/{s['n_pmhc']} "
              f"prec={s['precision']:.4f} recall={s['recall']:.4f} f1={s['f1']:.4f} "
              f"d_edit={s['d_edit']:.3f} seqrec={s['seq_recovery']:.4f} bleu={s['char_bleu']:.4f} "
              f"div={s['diversity']:.3f}  -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
