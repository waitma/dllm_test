"""Run ImmunoMatch + ANARCI metrics on saved light-pairing CSVs (no generation).

Usage:
  python scripts/downstream/run_grammar_v2_pairing_metrics_only.py esmc300m_cmp500k_llada
  python scripts/downstream/run_grammar_v2_pairing_metrics_only.py esmc600m_cmp500k_llada
  python scripts/downstream/run_grammar_v2_pairing_metrics_only.py all
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
OUT_DIR = PROJECT_ROOT / "output" / "downstream_generation"
DEFAULT_VARIANTS = ("esmc300m_cmp500k_llada", "esmc600m_cmp500k_llada")
NUM_SEQS = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_variant(variant: str, num_seqs: int = NUM_SEQS) -> dict:
    prefix = f"grammar_v2_{variant}_light_pairing_holdout500_prompt3"
    csv_path = OUT_DIR / f"{prefix}_n{num_seqs}.csv"
    metrics_path = OUT_DIR / f"{prefix}_metrics.json"
    summary_path = OUT_DIR / f"grammar_v2_{variant}_downstream_summary.txt"

    if not csv_path.is_file():
        raise FileNotFoundError(f"missing generation CSV: {csv_path}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as summary:
        summary.write(f"{_utc_now()} pairing metrics only variant={variant} csv={csv_path}\n")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from downstream.grammar.light_chain_pairing import run_comp_chain_eval

    metrics = run_comp_chain_eval(csv_path, num_seqs)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    preview = {
        k: metrics[k]
        for k in sorted(metrics)
        if k.startswith(("gen_", "ref_", "overall_", "diversity"))
    }
    print(json.dumps(preview, indent=2))
    print(f"Saved metrics: {metrics_path}")

    with summary_path.open("a", encoding="utf-8") as summary:
        summary.write(f"{_utc_now()} finished pairing metrics variant={variant}\n")
        for key in ("gen_immunomatch_mean", "ref_immunomatch_mean", "overall_chain_match_rate", "diversity_mean"):
            if key in metrics:
                summary.write(f"{key}={metrics[key]}\n")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute pairing metrics from saved CSVs.")
    parser.add_argument(
        "variant",
        choices=[*DEFAULT_VARIANTS, "all"],
        help="grammar_v2 variant suffix or 'all'",
    )
    parser.add_argument("--num-seqs", type=int, default=NUM_SEQS)
    args = parser.parse_args()

    variants = DEFAULT_VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        print(f"=== pairing metrics {variant} ===", flush=True)
        run_variant(variant, num_seqs=args.num_seqs)
    print(f"{_utc_now()} pairing metrics done", flush=True)


if __name__ == "__main__":
    main()
