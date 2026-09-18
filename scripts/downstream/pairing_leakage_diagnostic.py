#!/usr/bin/env python
"""Audit pairing length, reference similarity, and candidate diversity.

Known-reference-length generation is an explicit protocol, not automatically a
leak. High similarity is a review signal; only model-input checks can establish
that hidden reference residues were exposed. This CSV audit does not replace
those checks. The statistics are:

  same_length_frac   fraction of rows where len(gen) == len(ref); expected to be
                     high in the declared reference-length protocol.
  mean_identity      mean per-position identity to the reference over same-length
                     rows. Values >0.9 trigger review, not proof of copying.
  exact_copy_frac    fraction of rows that reproduce the reference verbatim.
  unique_per_heavy   mean number of distinct generated light chains per heavy.
                     1.0 means the sampler collapsed (e.g. deterministic argmax
                     with num_seqs > 1), so n candidates are one draw repeated.

Usage:
  python scripts/downstream/pairing_leakage_diagnostic.py \
    --csv output/downstream_generation/<tag>_n8.csv \
    --out-json output/downstream_generation/<tag>_leakage_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Above these, a run must not be presented as a de-novo pairing capability claim.
SAME_LENGTH_LEAK_THRESHOLD = 0.90
IDENTITY_LEAK_THRESHOLD = 0.90
# At or below this, the sampler produced no real candidate diversity.
COLLAPSE_UNIQUE_THRESHOLD = 1.01


def _identity(gen: str, ref: str) -> float | None:
    if len(gen) != len(ref) or not ref:
        return None
    matches = sum(1 for a, b in zip(gen, ref) if a == b)
    return matches / len(ref)


def audit(csv_path: Path) -> dict:
    frame = pd.read_csv(csv_path)
    gen = frame["gen_l_sequence"].astype(str)
    ref = frame["raw_l_sequence"].astype(str)

    identities = [_identity(a, b) for a, b in zip(gen, ref)]
    same_length = [value is not None for value in identities]
    scored = [value for value in identities if value is not None]
    exact = sum(1 for a, b in zip(gen, ref) if a == b)

    unique_per_heavy = frame.groupby("h_sequence")["gen_l_sequence"].nunique()

    n_rows = len(frame)
    report = {
        "csv": str(csv_path.resolve()),
        "n_rows": int(n_rows),
        "n_heavy_groups": int(unique_per_heavy.size),
        "same_length_frac": float(np.mean(same_length)) if n_rows else 0.0,
        "mean_identity_same_length": float(np.mean(scored)) if scored else None,
        "exact_copy_frac": float(exact / n_rows) if n_rows else 0.0,
        "unique_gen_per_heavy_mean": float(unique_per_heavy.mean()) if unique_per_heavy.size else 0.0,
        "unique_gen_per_heavy_min": int(unique_per_heavy.min()) if unique_per_heavy.size else 0,
        "gen_length_mean": float(gen.str.len().mean()) if n_rows else 0.0,
        "ref_length_mean": float(ref.str.len().mean()) if n_rows else 0.0,
    }

    # Provenance written by light_chain_pairing.py, when present.
    for column in ("light_length_mode", "target_light_length", "ref_light_length"):
        if column not in frame.columns:
            continue
        if column == "light_length_mode":
            report["light_length_mode"] = sorted(set(frame[column].astype(str)))
        else:
            report[f"{column}_mean"] = float(frame[column].mean())
    if {"target_light_length", "ref_light_length"} <= set(frame.columns):
        agree = frame["target_light_length"] == frame["ref_light_length"]
        report["target_equals_ref_length_frac"] = float(agree.mean())

    known_reference_length = report.get("light_length_mode") == ["reference"]
    report["known_reference_length"] = known_reference_length
    identity_leak = (
        report["mean_identity_same_length"] is not None
        and report["mean_identity_same_length"] >= IDENTITY_LEAK_THRESHOLD
    )
    report["length_leak_suspected"] = bool(
        not known_reference_length
        and report["same_length_frac"] >= SAME_LENGTH_LEAK_THRESHOLD and identity_leak
    )
    report["high_identity_review_required"] = bool(identity_leak)
    report["sampler_collapse_suspected"] = bool(
        report["unique_gen_per_heavy_mean"] <= COLLAPSE_UNIQUE_THRESHOLD
    )
    verdicts = []
    if known_reference_length:
        verdicts.append("reference length is an explicit input; not a native-EOS protocol")
    if identity_leak:
        verdicts.append("high reference identity: inspect input-masking tests; similarity alone does not prove copying")
    if report["length_leak_suspected"]:
        verdicts.append("target length leaked -> near-reconstruction, not de-novo pairing")
    if report["sampler_collapse_suspected"]:
        verdicts.append("sampler collapsed -> n candidates are one draw repeated")
    report["verdict"] = verdicts or ["no length leak or sampler collapse detected"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Generation CSV (…_n8.csv).")
    parser.add_argument("--out-json", default="", help="Where to write the report.")
    args = parser.parse_args()

    report = audit(Path(args.csv))
    print(json.dumps(report, indent=2))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
