"""Strict JSON boundary for complete native TCR generation results.

Run: python -m pytest -q scripts/tests/immune_llada/test_tcr_generation_result_io.py
Only undefined conditioning novelty is nullable; other non-finite metrics fail.
The shared scorer and AB JSON writer deliberately remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math

from downstream.benchmark.tcr_generation_bench.scoring import _AGG_KEYS

PROTOCOL = "tcr_generation_metrics_nullable_novelty_v1"
NOVELTY = "novelty_vs_conditioning"
REASON = "no_same_epitope_conditioning_references"


def prepare_metrics_for_json(result, missing_conditioning):
    """Return a copy, preserving every primary metric, member and denominator.

    missing_conditioning must be derived from the actual conditioning bank,
    not inferred from a NaN. This is not a general NaN-to-null conversion.
    """
    out = deepcopy(result)
    missing = set(missing_conditioning)
    if not missing <= set(out["per_pmhc"]):
        raise ValueError("missing-conditioning keys are outside the scored universe")
    unavailable = {}
    for key, item in out["per_pmhc"].items():
        for metric in _AGG_KEYS:
            value = item[metric]
            if metric == NOVELTY and key in missing:
                if not isinstance(value, (int, float)) or not math.isnan(value):
                    raise ValueError(f"{key}: undefined novelty must originate as NaN")
                item[metric] = None
                unavailable[key] = {NOVELTY: REASON}
            elif value is None or not math.isfinite(value):
                raise ValueError(f"unexpected non-finite per-pMHC metric: {key}/{metric}")
    missing_summaries = {}
    for view, item in out["summary"].items():
        for metric in _AGG_KEYS:
            value = item[metric]
            count = item["n_scored_by_metric"][metric]
            if metric == NOVELTY and count == 0 and missing:
                if not isinstance(value, (int, float)) or not math.isnan(value):
                    raise ValueError(f"{view}: zero novelty denominator must originate as NaN")
                item[metric] = None
                missing_summaries[view] = {NOVELTY: REASON}
            elif value is None or not math.isfinite(value) or count <= 0:
                raise ValueError(f"unexpected non-finite/empty aggregate: {view}/{metric}")
    out["metric_availability"] = {
        "protocol": PROTOCOL,
        "per_pmhc_unavailable": unavailable,
        "summary_unavailable": missing_summaries,
        "policy": "undefined optional novelty is null, never zero; primary metrics must be finite",
    }
    json.dumps(out, allow_nan=False)
    return out
