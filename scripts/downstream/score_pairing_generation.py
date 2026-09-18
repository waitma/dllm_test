"""Score an existing pairing CSV without regenerating sequences.

Run in the existing protenix_abtcr scoring environment:
  python scripts/downstream/score_pairing_generation.py --csv /abs/run_n8.csv --metrics /abs/run_metrics.json
This reuses the project's ImmunoMatch/ANARCI scorer and preserves the run manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from downstream.grammar.light_chain_pairing import run_comp_chain_eval
from scripts.downstream.pairing_leakage_diagnostic import audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--num-seqs", type=int, default=8)
    args = parser.parse_args()
    if args.metrics.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {args.metrics}")
    manifest_path = args.csv.with_name(args.csv.stem + "_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    metrics = run_comp_chain_eval(args.csv, args.num_seqs)
    metrics["evaluation_protocol"] = {
        key: manifest[key] for key in (
            "protocol_id", "length_condition", "sampler_state", "cfg_scale",
            "checkpoint_path", "checkpoint_sha256", "input_csv_sha256",
            "source_sha256", "max_iter", "sampling_strategy", "seed",
        )
    }
    for key in ("light_prompt_tokens", "cfg_condition", "cfg_formula", "protocol_version", "encoder_state"):
        if key in manifest:
            metrics["evaluation_protocol"][key] = manifest[key]
    metrics["generation_audit"] = audit(args.csv)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.metrics.with_suffix(args.metrics.suffix + ".tmp")
    temporary.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    temporary.replace(args.metrics)
    print(f"Saved pairing metrics: {args.metrics}", flush=True)


if __name__ == "__main__":
    main()
