#!/usr/bin/env python
"""Aggregate paper6 (6-pMHC deep few-shot) results + Table SI comparison.

Run: python downstream/benchmark/scripts/summarize_tcr_representation_paper6.py --output-dir <NEW_DIRECTORY>
Reads existing fewshot.json; writes a separate report with explicit comparison limits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
OUT = BENCH / "outputs" / "tcr_representation_paper6"
sys.path.insert(0, str(BENCH))
from common.reporting import compare_protocols, file_identity, new_report_dir, write_report_manifest

PAPER_CSV = BENCH / "outputs" / "external" / "paper_reported_baselines.csv"

TARGETS = ["GILGFVFTL", "NLVPMVATV", "SPRWYFYYL", "TFEYVSQPFLMDLE", "TTDPSFLGRY", "YLQPRTFLL"]
SHORT = ["GIL", "NLV", "SPR", "TFE", "TTD", "YLQ"]
SHOW_SHOTS = [1, 2, 5, 10, 20, 50, 100, 200]

# SCEPTR Cell Systems 2024 Table SI (k=200 per-pMHC AUROC)
PAPER_K200 = {
    "sceptr": [0.911, 0.691, 0.728, 0.976, 0.708, 0.775],
    "tcrdist": [0.904, 0.648, 0.695, 0.970, 0.720, 0.762],
    "levenshtein": [0.872, 0.632, 0.610, 0.964, 0.600, 0.743],
    "tcrbert": [0.876, 0.655, 0.637, 0.966, 0.576, 0.698],
    "esm2_t6_8m": [0.831, 0.652, 0.604, 0.937, 0.579, 0.697],
    "protbert": [0.845, 0.629, 0.575, 0.950, 0.564, 0.669],
}

PAPER_NAME_TO_TAG = {
    "SCEPTR": "sceptr", "TCRdist": "tcrdist",
    "CDR3-Levenshtein": "levenshtein", "TCR-BERT": "tcrbert",
    "ESM2-150M": "esm2_t6_8m",  # legacy transcription label, corrected in reports only
    "ESM2-8M": "esm2_t6_8m", "ProtBERT": "protbert",
}


def load_paper_k200() -> dict[str, list[float]]:
    """Load the central paper transcription, falling back to checked constants."""

    if not PAPER_CSV.is_file():
        return PAPER_K200
    import pandas as pd

    df = pd.read_csv(PAPER_CSV)
    df = df[(df["benchmark"] == "t3_representation") & (df["task"] == "sceptr_six_pmhc_k200") & (df["metric"] == "auroc")]
    loaded = {}
    for method, tag in PAPER_NAME_TO_TAG.items():
        sub = df[df["method"] == method].set_index("split")["value"]
        if all(target in sub for target in SHORT):
            loaded[tag] = [float(sub[target]) for target in SHORT]
    return loaded

ORDER = [
    ("levenshtein", "CDR3-Levenshtein"),
    ("tcrdist", "TCRdist"),
    ("sceptr", "SCEPTR"),
    ("tcrbert", "TCR-BERT"),
    ("protbert", "ProtBERT"),
    ("esm2_150m", "ESM2-150M"),
    ("kmer", "k-mer(3)"),
    # Ours — post-LLaDA headline (grammar:decoder:global whole-record mean-pool).
    ("ours_globalfeat_esmc300m_cmp500k_llada", "Ours esmc300m-cmp500k (post-LLaDA globalfeat)"),
    ("ours_globalfeat_esmc600m_cmp500k_llada", "Ours esmc600m-cmp500k (post-LLaDA globalfeat)"),
    ("ours_globalfeat_esmc300m_mint_llada", "Ours esmc300m-mint (post-LLaDA globalfeat)"),
    # Legacy / old ckpt (resume 前, 仅历史参照).
    ("grammar_tcrpair_seq_esmc300m_cmp500k_llada", "Ours esmc300m-cmp500k (seq-pool, 旧 ckpt)"),
    ("grammar_tcrpair_seq_esmc300m_mint_llada", "Ours esmc300m-mint (seq-pool, 旧 ckpt)"),
    ("grammar_tcrpair_seq_esmc600m_cmp500k_llada", "Ours esmc600m-cmp500k (seq-pool, 旧 ckpt)"),
    ("grammar_tcrpair_esmc300m_cmp500k_llada", "Ours esmc300m-cmp500k (seg-pool, 旧 ckpt)"),
    ("grammar_tcrpair_esmc300m_mint_llada", "Ours esmc300m-mint (seg-pool, 旧 ckpt)"),
    ("grammar_tcrpair_esmc600m_cmp500k_llada", "Ours esmc600m-cmp500k (seg-pool, 旧 ckpt)"),
    # Ours — earlier decoder variant with a constant placeholder peptide "A"
    # (peptide segment dropped); kept for comparison only.
    ("grammar_dec_esmc300m_cmp500k_llada", "Ours-decA esmc300m-cmp500k (旧 ckpt)"),
    ("grammar_dec_esmc300m_mint_llada", "Ours-decA esmc300m-mint (旧 ckpt)"),
    ("grammar_dec_esmc600m_cmp500k_llada", "Ours-decA esmc600m-cmp500k (旧 ckpt)"),
]


def _load(tag: str):
    p = OUT / tag / "fewshot.json"
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def _get(d: dict, k):
    return d.get(str(k)) or d.get(int(k))


def paper_comparison(tag: str, payload: dict) -> dict:
    local = payload.get("comparison_contract") or {}
    # Dataset size is not dataset membership. Historical files do not bind the
    # paper membership, selection or input representation to a verified manifest.
    reference = {"metric": "AUROC", "aggregation": "macro_over_six_pmhc"}
    result = compare_protocols(local, reference)
    reasons = ["Paper/local membership, input and model-selection equivalence unverified"]
    if tag == "esm2_150m":
        result.update(protocol_alignment="mismatched", comparison_allowed=False, rank_eligible=False)
        result["mismatched_fields"].append("model_identity")
        reasons.append("Paper ESM2 is t6/8M; this local run uses 150M")
    if payload.get("n_universe") not in (None, 25816):
        result.update(protocol_alignment="mismatched", comparison_allowed=False, rank_eligible=False)
        result["mismatched_fields"].append("dataset_size")
        reasons.append("Local universe size differs from the 25816-row paper6 reference")
    result["reasons"] = reasons
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New report directory")
    args = parser.parse_args(argv)
    destination = new_report_dir(args.output_dir, "t3_paper6")
    paper_k200 = load_paper_k200()
    tags = [t for t, _ in ORDER] + sorted(
        d.name for d in OUT.iterdir() if d.is_dir() and d.name not in dict(ORDER)
    )
    tags.sort(key=lambda tag: tag.startswith(("ours", "grammar", "immune", "bioseq")))
    inputs = [Path(__file__), BENCH / "common" / "reporting.py", PAPER_CSV]
    summary = {
        "schema_version": 2, "fewshot": {}, "k200_comparison": {},
        "policy": "paper_primary; protocol_unknown_or_mismatched_no_delta_no_rank",
        "paper_reported": {
            "evidence_type": "paper_reported",
            "citation": "arXiv:2406.06397; SCEPTR Table SI",
            "source": file_identity(PAPER_CSV) if PAPER_CSV.is_file() else file_identity(Path(__file__)),
            "esm2_identity": "esm2_t6_8M_UR50D",
            "transcription_correction": "Legacy CSV label ESM2-150M refers to paper t6/8M; source CSV retained",
            "k200": {k: dict(zip(SHORT, v)) for k, v in paper_k200.items()},
        },
    }
    for tag in tags:
        payload = _load(tag)
        if payload is None:
            continue
        path = OUT / tag / "fewshot.json"
        inputs.append(path)
        comparison = paper_comparison(tag, payload)
        summary["fewshot"][tag] = {
            "model": payload.get("model", tag),
            "evidence_type": payload.get("baseline_provenance", {}).get("evidence_type", "unknown"),
            "source": file_identity(path), **comparison,
            "by_shot": payload.get("auroc_by_shot"),
            "seeds": payload.get("seeds"), "n_universe": payload.get("n_universe"),
            "input_fields": payload.get("input_fields"),
            "per_target_binders": payload.get("per_target_binders"),
        }
        per_target = _get(payload.get("per_epitope_by_shot", {}), 200) or {}
        reference_tag = "esm2_t6_8m" if tag == "esm2_150m" else tag
        summary["k200_comparison"][tag] = {
            "local_k200": {short: per_target[t]["mean"] for t, short in zip(TARGETS, SHORT) if t in per_target},
            "reference_tag": reference_tag if reference_tag in paper_k200 else None,
            "paper_k200": dict(zip(SHORT, paper_k200[reference_tag])) if reference_tag in paper_k200 else None,
            **comparison,
        }
    (destination / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report_manifest(destination, inputs, policy=summary["policy"])
    print(f"Report: {destination}; {len(summary['fewshot'])} local runs, no unsupported deltas")
    return destination


if __name__ == "__main__":
    main()
