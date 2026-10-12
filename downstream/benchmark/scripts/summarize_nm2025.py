#!/usr/bin/env python
"""Report T1 paper-primary and separate local diagnostics without changing source results.

Run: python downstream/benchmark/scripts/summarize_nm2025.py --output-dir <NEW_DIRECTORY>
Missing paper cells remain missing. Comparability requires explicit protocol evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
OUT = BENCH / "outputs" / "tcr_binding_nm2025"
PAPER_CSV = BENCH / "outputs" / "external" / "paper_reported_baselines.csv"
TRACK = "cdr3b"

from common.reporting import compare_protocols, new_report_dir, write_report_manifest

from tcr_binding.original_protocol import (  # noqa: E402
    ORIGINAL_BASELINES,
    ORIGINAL_TEST_MANIFEST,
    ORIGINAL_ZIP,
    ORIGINAL_ZIP_SHA256,
    choose_result_source,
)

# Display order + friendly labels + type.
#
# Our-model口径 hierarchy (per PROJ_GUIDE.md "Downstream Feature Extraction,
# post-LLaDA, mandatory" + the 2026-07-05 global whole-feature decision):
#   * HEADLINE = globalfeat: post-LLaDA (grammar:decoder:global:), whole record
#     pushed through ESMC->LLaDA, GLOBAL mean-pool over all residues -> one vector.
#     This is the ONLY primary reported口径.
#   * control-A (post-LLaDA, segment-concat pooling; superseded by global) =
#     ours_postllada_*.
#   * control-B (ESMC-encoder-only, NOT post-LLaDA; kept only as ablation) =
#     ours_biseq_mlp*.
METHODS = [
    ("atmtcr", "ATM-TCR", "official pretrained"),
    ("nettcr", "NetTCR-2.2 (b)", "official pretrained"),
    ("teim", "TEIM", "official pretrained"),
    ("ergo2", "ERGO-II", "official pretrained"),
    ("epitcr_nomhc", "epiTCR (noMHC)", "official RF"),
    ("panpep", "PanPep (zero-shot)", "official pretrained"),
    # Tier-1 official-baseline expansion (2026-07-07): TEINet / TPBTE / ERGO
    # (AE|lstm x vdj|mc). Official code + official weights (Original_model/),
    # cdr3b track, same seen/unseen AS/PS/HS test sets as the 6 above.
    ("teinet_large", "TEINet-large", "official pretrained"),
    ("teinet_small", "TEINet-small", "official pretrained"),
    ("ergo_lstm_vdj", "ERGO-lstm (vdj)", "official pretrained"),
    ("ergo_lstm_mc", "ERGO-lstm (mc)", "official pretrained"),
    ("ergo_ae_vdj", "ERGO-AE (vdj)", "official pretrained"),
    ("ergo_ae_mc", "ERGO-AE (mc)", "official pretrained"),
    ("tpbte_mc", "TPBTE (mc)", "official pretrained"),
    ("tpbte_vdj", "TPBTE (vdj)", "official pretrained"),
    # OUR model — HEADLINE: post-LLaDA global whole-feature (grammar:decoder:global:).
    ("ours_globalfeat_esmc300m_cmp500k", "Ours-BioSeq esmc300m-cmp500k (post-LLaDA globalfeat)", "ours LLaDA-dec globalfeat+MLP"),
    ("ours_globalfeat_esmc600m_cmp500k", "Ours-BioSeq esmc600m-cmp500k (post-LLaDA globalfeat)", "ours LLaDA-dec globalfeat+MLP"),
    ("ours_globalfeat_esmc300m_mint", "Ours-BioSeq esmc300m-mint (post-LLaDA globalfeat)", "ours LLaDA-dec globalfeat+MLP"),
    # OUR model — control-A: post-LLaDA but segment-concat pooling (superseded).
    ("ours_postllada_esmc300m_cmp500k", "Ours-BioSeq esmc300m-cmp500k (post-LLaDA segment, control)", "ours LLaDA-dec segment+MLP"),
    ("ours_postllada_esmc600m_cmp500k", "Ours-BioSeq esmc600m-cmp500k (post-LLaDA segment, control)", "ours LLaDA-dec segment+MLP"),
    ("ours_postllada_esmc300m_mint", "Ours-BioSeq esmc300m-mint (post-LLaDA segment, control)", "ours LLaDA-dec segment+MLP"),
    # OUR model — control-B: ESMC-encoder-only, NOT post-LLaDA (ablation only).
    ("ours_biseq_mlp", "Ours-BioSeq esmc300m-cmp500k (enc-only, control)", "ours ESMC-enc+MLP"),
    ("ours_biseq_mlp_esmc600m", "Ours-BioSeq esmc600m-cmp500k (enc-only, control)", "ours ESMC-enc+MLP"),
    ("ours_biseq_mlp_esmc300m_mint", "Ours-BioSeq esmc300m-mint (enc-only, control)", "ours ESMC-enc+MLP"),
    ("knn", "CDR3-kNN (k=5)", "training-free"),
    ("random", "Random", "floor"),
]
NEG_DIR = {"AS": "{split}", "PS": "{split}_PS", "HS": "{split}_HS"}

PAPER_METHOD = {spec.tag: spec.paper_method for spec in ORIGINAL_BASELINES}
MAIN_LOCAL_CONTROL_TAGS = {"knn", "random"}


def _main_local_context(local_main: pd.DataFrame) -> pd.DataFrame:
    """Keep only Ours and explicit controls beside canonical external rows."""

    is_ours = local_main["type"].astype(str).str.startswith("ours")
    is_control = local_main["tag"].isin(MAIN_LOCAL_CONTROL_TAGS)
    return local_main[is_ours | is_control].copy()


def _load(tag: str, split: str, neg: str) -> dict | None:
    sub = split if neg == "AS" else f"{split}_{neg}"
    p = OUT / tag / TRACK / sub / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _wrapper_available(spec) -> bool:
    if spec.wrapper is None:
        return False
    if spec.variant_tag is not None:
        os.environ["TCRBENCH_BASELINE_TAG"] = spec.variant_tag
    try:
        wrapper = importlib.import_module(f"baselines.wrappers.{spec.wrapper}")
        reporter = getattr(wrapper, "official_checkpoint_paths", None)
        if not callable(reporter):
            return False
        reported = tuple(Path(path) for path in reporter())
        return (
            reported == spec.checkpoint_paths
            and spec.artifact_identity_verified
            and bool(wrapper.is_available())
        )
    except Exception:
        return False


def _original_zip_matches() -> bool:
    return (
        ORIGINAL_ZIP.is_file()
        and _sha256_file(ORIGINAL_ZIP) == ORIGINAL_ZIP_SHA256
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_local_original_split(spec, split: str) -> tuple[dict | None, str]:
    """Validate a stored AS rerun without trusting only its directory name."""

    run_dir = OUT / spec.tag / TRACK / split
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.csv"
    test_path = BENCH / "data" / "tcr_binding_nm2025" / split / TRACK / "AS" / "test.csv"
    if not metrics_path.is_file() or not predictions_path.is_file():
        return None, f"{split}: missing local metrics/predictions"
    if not test_path.is_file():
        return None, f"{split}: missing original.zip-derived test.csv"

    try:
        payload = json.loads(metrics_path.read_text())
        pred = pd.read_csv(predictions_path)
        test = pd.read_csv(test_path).fillna("")
    except Exception as exc:
        return None, f"{split}: unreadable local artifact ({type(exc).__name__})"

    if payload.get("track") != TRACK or payload.get("split") != split:
        return None, f"{split}: metrics track/split mismatch"
    if payload.get("neg_source") != "AS":
        return None, f"{split}: metrics are not the AS original-model protocol"
    if payload.get("model") not in spec.expected_model_names:
        return None, (
            f"{split}: artifact mapping mismatch; model={payload.get('model')!r}, "
            f"expected={spec.expected_model_names!r}"
        )
    if payload.get("method") not in {"official_baseline", "epitcr"}:
        return None, f"{split}: output came from non-official method={payload.get('method')!r}"
    if len(pred) != len(test):
        return None, f"{split}: prediction count {len(pred)} != test count {len(test)}"
    if not {"group", "label", "score"}.issubset(pred.columns):
        return None, f"{split}: predictions missing group/label/score"
    if not pd.to_numeric(pred["score"], errors="coerce").notna().all():
        return None, f"{split}: predictions contain non-numeric/NaN scores"

    expected = (
        pd.DataFrame({
            "group": test["peptide"].astype(str),
            "label": test["label"].astype(int),
        })
        .value_counts(sort=False)
        .sort_index()
    )
    actual = (
        pred.assign(
            group=pred["group"].astype(str),
            label=pred["label"].astype(int),
        )[["group", "label"]]
        .value_counts(sort=False)
        .sort_index()
    )
    if not expected.equals(actual):
        return None, f"{split}: prediction group/label multiset differs from original.zip test"

    protocol = payload.get("baseline_protocol")
    provenance = payload.get("baseline_provenance")
    if not isinstance(protocol, dict) or not isinstance(provenance, dict):
        return None, f"{split}: missing embedded original-protocol metadata"
    if protocol.get("train_csv_loaded") is not False or protocol.get("retrained") is not False:
        return None, f"{split}: protocol used training/retraining"
    if protocol.get("test_artifact_sha256") != ORIGINAL_ZIP_SHA256:
        return None, f"{split}: original.zip checksum mismatch"
    if protocol.get("test_artifact") != str(ORIGINAL_ZIP):
        return None, f"{split}: unexpected test artifact path"
    if protocol.get("normalized_test") != str(test_path):
        return None, f"{split}: normalized test path mismatch"
    if protocol.get("test_manifest") != str(ORIGINAL_TEST_MANIFEST):
        return None, f"{split}: original test manifest path mismatch"
    try:
        manifest = json.loads(ORIGINAL_TEST_MANIFEST.read_text(encoding="utf-8"))
        relative_test = str(test_path.relative_to(BENCH / "data" / "tcr_binding_nm2025"))
        expected_test_sha = manifest["files"][relative_test]["sha256"]
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return None, f"{split}: invalid original test manifest ({type(exc).__name__})"
    actual_test_sha = _sha256_file(test_path)
    if manifest.get("source_artifact_sha256") != ORIGINAL_ZIP_SHA256:
        return None, f"{split}: manifest original.zip checksum mismatch"
    if actual_test_sha != expected_test_sha:
        return None, f"{split}: normalized test differs from extraction manifest"
    if protocol.get("normalized_test_sha256") != actual_test_sha:
        return None, f"{split}: metrics normalized-test checksum mismatch"
    if protocol.get("checkpoint_paths") != [str(path) for path in spec.checkpoint_paths]:
        return None, f"{split}: original checkpoint path mismatch"
    if protocol.get("wrapper") != spec.wrapper:
        return None, f"{split}: official wrapper mismatch"
    if protocol.get("variant_tag") != spec.variant_tag:
        return None, f"{split}: checkpoint variant mismatch"
    if provenance.get("evidence_type") != "official_code_rerun":
        return None, f"{split}: provenance is not official_code_rerun"
    return payload, "embedded_original_protocol"


def _paper_value(paper: pd.DataFrame, method: str, split: str, metric: str):
    hit = paper[
        (paper["method"] == method)
        & (paper["split"] == split)
        & (paper["metric"] == metric)
    ]
    return float(hit.iloc[0]["value"]) if len(hit) == 1 else float("nan")


def _write_original_baseline_summary(paper_long: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Keep every catalogued baseline; paper and local values cannot fill each other."""

    rows = []
    original_zip_ok = _original_zip_matches()
    for spec in ORIGINAL_BASELINES:
        local = {}
        local_statuses = []
        local_errors = []
        run_status_path = OUT / spec.tag / TRACK / "original_run_status.json"
        if run_status_path.is_file():
            try:
                run_status = json.loads(run_status_path.read_text())
                if run_status.get("status") == "paper_fallback" and run_status.get("reason"):
                    local_errors.append(f"last official run: {run_status['reason']}")
            except Exception:
                local_errors.append("unreadable original_run_status.json")
        for split in ("seen", "unseen"):
            payload, status = _validate_local_original_split(spec, split)
            if payload is None:
                local_errors.append(status)
            else:
                local[split] = payload
                local_statuses.append(status)

        runnable = _wrapper_available(spec)
        local_complete = (
            original_zip_ok
            and spec.checkpoint_available
            and runnable
            and set(local) == {"seen", "unseen"}
        )
        if not original_zip_ok:
            local_errors.append("original.zip missing or checksum mismatch")
        paper_method_rows = paper_long[paper_long["method"] == spec.paper_method]
        paper_available = not paper_method_rows.empty
        paper_complete = all(
            pd.notna(_paper_value(paper_long, spec.paper_method, split, metric))
            for split in ("seen", "unseen")
            for metric in ("auprc", "auroc")
        )
        evidence_type, result_label = choose_result_source(
            local_complete=local_complete,
            paper_available=paper_available,
        )
        local_reason = "" if local_complete else "; ".join(local_errors) or "official rerun not verified"
        source_location = "; ".join(sorted(set(paper_method_rows["source_location"].dropna().astype(str))))

        row = {
            "method": spec.display_name,
            "tag": spec.tag,
            "paper_method": spec.paper_method,
            "selected_source": evidence_type,
            "result_label": result_label,
            "evidence_type": evidence_type,
            **compare_protocols(None, None),
            "comparison_reason": "Paper membership/selection equivalence is not established by a local rerun",
            "local_evidence_type": "official_code_rerun" if local_complete else "unavailable",
            "local_source_location": "; ".join(str(OUT / spec.tag / TRACK / split / "metrics.json") for split in local),
            "reproduced_locally": bool(local_complete),
            "retrained": False,
            "local_test_artifact": str(ORIGINAL_ZIP),
            "local_test_artifact_sha256": ORIGINAL_ZIP_SHA256,
            "local_test_manifest": str(ORIGINAL_TEST_MANIFEST),
            "local_checkpoint_paths": "; ".join(str(path) for path in spec.checkpoint_paths),
            "local_checkpoint_available": bool(spec.checkpoint_available),
            "local_checkpoint_aliases_verified": bool(spec.artifact_identity_verified),
            "local_official_runner_available": bool(runnable),
            "local_metadata_status": (
                "embedded_original_protocol"
                if local_complete and set(local_statuses) == {"embedded_original_protocol"}
                else "not_complete"
            ),
            "paper_values_complete": bool(paper_complete),
            "local_unavailable_reason": local_reason,
            "paper_unavailable_reason": "" if paper_complete else "paper cells missing or ambiguous",
            "citation": (
                paper_method_rows.iloc[0]["citation"] if paper_available else ""
            ),
            "source_location": source_location,
        }
        for split in ("seen", "unseen"):
            for metric in ("auprc", "auroc"):
                paper_value = _paper_value(paper_long, spec.paper_method, split, metric)
                row[f"paper_{split}_{metric}"] = paper_value
                row[f"local_{split}_{metric}"] = (
                    float(local[split][f"{metric}_overall"]) if local_complete else float("nan")
                )
                row[f"{split}_{metric}"] = paper_value

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary_original_baselines.csv", index=False)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New report directory; must not already exist")
    args = parser.parse_args(argv)
    destination = new_report_dir(args.output_dir, "t1")
    inputs = [Path(__file__), BENCH / "common" / "reporting.py", BENCH / "tcr_binding" / "original_protocol.py"]
    main_rows = []
    for tag, label, typ in METHODS:
        seen, unseen = _load(tag, "seen", "AS"), _load(tag, "unseen", "AS")
        if seen is None and unseen is None:
            continue
        proofs = [(m or {}).get("baseline_provenance", {}) for m in (seen, unseen)]
        evidence = {p.get("evidence_type", "unknown") for p in proofs}
        row = {"method": label, "tag": tag, "type": typ,
               "evidence_type": evidence.pop() if len(evidence) == 1 else "unknown",
               **compare_protocols(None, None)}
        for split, metrics in (("seen", seen), ("unseen", unseen)):
            if metrics is None:
                continue
            inputs.append(OUT / tag / TRACK / split / "metrics.json")
            for metric, source in (("auprc", "auprc_overall"), ("auroc", "auroc_overall"),
                                   ("auprc_macro", "auprc_macro"), ("auc01", "auc01_macro")):
                row[f"{split}_{metric}"] = metrics.get(source)
        main_rows.append(row)
    local_main = pd.DataFrame(main_rows, columns=None if main_rows else ["method", "tag", "type"])
    # Stable catalog order, baseline/control first and all Ours together at the end.
    local_main["_ours"] = local_main["type"].astype(str).str.startswith("ours")
    local_main = local_main.sort_values("_ours", kind="stable").drop(columns="_ours")
    local_main.to_csv(destination / "summary_local_runs.csv", index=False)

    columns = ["benchmark", "task", "method", "split", "metric", "value", "source_location", "citation"]
    paper = pd.read_csv(PAPER_CSV) if PAPER_CSV.is_file() else pd.DataFrame(columns=columns)
    paper = paper[(paper["benchmark"] == "t1_binding") & (paper["task"] == "original_cdr3b_as")].copy()
    paper.to_csv(destination / "summary_paper_reported.csv", index=False)
    selected = _write_original_baseline_summary(paper, destination)
    selected["type"] = "paper original-model reference"
    main_table = pd.concat([selected, _main_local_context(local_main)], ignore_index=True, sort=False)
    main_table.to_csv(destination / "summary_main.csv", index=False)
    # Side-by-side observations, no unsupported deltas or ranking.
    audit_cols = [c for c in selected if c.startswith(("paper_", "local_"))]
    selected[["tag", "comparison_allowed", "comparison_reason", *audit_cols]].to_csv(
        destination / "baseline_paper_audit.csv", index=False)

    neg_rows = []
    for tag, label, _ in METHODS:
        row = {"method": label, "tag": tag, "comparison_allowed": False}
        for split in ("seen", "unseen"):
            for neg in ("AS", "PS", "HS"):
                metrics = _load(tag, split, neg)
                if metrics is not None:
                    sub = split if neg == "AS" else f"{split}_{neg}"
                    inputs.append(OUT / tag / TRACK / sub / "metrics.json")
                    row[f"{split}_{neg}"] = metrics.get("auprc_overall")
        if len(row) > 3:
            neg_rows.append(row)
    pd.DataFrame(neg_rows).to_csv(destination / "summary_neg_source.csv", index=False)
    inputs += [PAPER_CSV, ORIGINAL_TEST_MANIFEST]
    for spec in ORIGINAL_BASELINES:
        for split in ("seen", "unseen"):
            inputs += [OUT / spec.tag / TRACK / split / name for name in ("metrics.json", "predictions.csv")]
    write_report_manifest(destination, inputs, policy="paper_primary_local_separate; unknown_protocol_no_delta_no_rank")
    print(selected[["tag", "selected_source", "reproduced_locally", "paper_values_complete"]].to_string(index=False))
    print(f"Report: {destination}")
    return destination


if __name__ == "__main__":
    main()
