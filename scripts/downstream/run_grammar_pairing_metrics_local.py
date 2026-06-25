"""Evaluate saved grammar light-pairing CSVs with comp_chain generation_eval locally.

Run:
  source activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
  python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_grammar_pairing_metrics_local.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
OUT_DIR = PROJECT_ROOT / "output" / "downstream_generation"
VARIANTS = ("esmc300m", "esmc600m", "no_encoder_38m")
NUM_SEQS = 8

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.grammar.light_chain_pairing import run_comp_chain_eval


def main() -> None:
    summary: dict[str, dict] = {}
    for variant in VARIANTS:
        prefix = f"grammar_v1_{variant}_light_pairing_holdout500_prompt3"
        csv_path = OUT_DIR / f"{prefix}_n8.csv"
        metrics_path = OUT_DIR / f"{prefix}_metrics.json"
        if not csv_path.is_file():
            print(f"skip {variant}: missing {csv_path}")
            continue
        print(f"=== eval {variant} ===", flush=True)
        metrics = run_comp_chain_eval(csv_path, NUM_SEQS)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"saved {metrics_path}", flush=True)
        summary[variant] = {
            k: metrics[k]
            for k in (
                "gen_immunomatch_mean",
                "ref_immunomatch_mean",
                "gen_better_ratio",
                "overall_chain_match_rate",
                "overall_v_gene_match_rate",
                "overall_j_gene_match_rate",
                "diversity_mean",
                "n_heavy_chain_groups",
            )
            if k in metrics
        }
        print(json.dumps(summary[variant], indent=2), flush=True)

    (OUT_DIR / "grammar_v1_pairing_metrics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
