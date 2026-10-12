"""CPU-only ANARCII evidence for the 80 unresolved observations; no certificates.

Activate pllm; run: python scripts/data/audit_v7_alternative_numbering.py
Uses the isolated official ANARCII wheel/dependencies previously downloaded under
output/immune_data_readiness_20260930/unresolved_coverage/isolated_packages.
No changes to prepared data, thresholds, benchmark records, or installed envs.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/immune_data_readiness_20260930/unresolved_coverage"
BASE = ROOT / "output/immune_data_readiness_20260930/role_resolution"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT / "isolated_packages"))
from scripts.data.audit_v7_observation_roles import read, sha_file, sha_text, summarize_domain, write


def assessment(result, summary, seq_type):
    if result.get("error") or not summary:
        return "failed_numbering"
    if not summary["exact_input_mapping"]:
        return "pending_input_mapping"
    threshold = 24.5 if seq_type == "shark" else 25.0
    if result["score"] <= threshold:
        return "low_score_review"
    if summary["chain_type"] != "H":
        return "unexpected_chain_review"
    if summary["unmapped_prefix_length"] or summary["unmapped_suffix_length"]:
        return "unmapped_residues_review"
    if not summary["regions"]["cdr3"]["length"]:
        return "no_observed_cdr3"
    if not (summary["regions"]["fr3"]["length"] and summary["regions"]["fr4"]["length"]):
        return "partial_cdr3_no_both_flanks"
    return "candidate_observed_cdr3_both_flanks"


def run():
    from anarcii import Anarcii
    import torch

    original = read(BASE / "report.json")
    coverage_path = ROOT / "data/dedup/immune_v7_20260930/benchmark_h3_coverage.json"
    seqs = {sha_text(s): s for s in read(coverage_path)["unresolved"]}
    cases = [c for c in original["cases"] if c["observation_adjudication"] == "pending"]
    controls = [c for c in original["cases"] if c["resolution"] == "observed_partial_h3_exact_native_reference"]
    controls += [c for c in original["cases"] if c["resolution"] == "observed_empty_h3_source_and_numbering_agree"][:5]
    controls += [c for c in original["cases"] if c["resolution"] == "observed_fragment_ends_before_h3"]
    previous = {c["sequence_sha256"]: c["domains"] for c in read(BASE / "numbering.json")["cases"]}
    output = []
    run_times = []
    for seq_type in ("shark", "antibody"):
        selected = [c for c in cases + controls if (c["role"] == "vnar") == (seq_type == "shark")]
        for mode in (("accuracy",) if seq_type == "shark" else ("accuracy", "speed")):
            started = time.monotonic()
            model = Anarcii(seq_type=seq_type, mode=mode, cpu=True, ncpu=4,
                            batch_size=32, low_score_warn=False, verbose=False)
            results = model.number({c["sequence_sha256"]: seqs[c["sequence_sha256"]] for c in selected})
            run_times.append({"seq_type": seq_type, "mode": mode, "count": len(selected),
                              "elapsed_seconds": time.monotonic() - started})
            for case in selected:
                digest = case["sequence_sha256"]
                result = results[digest]
                summary = None
                if result.get("numbering") and result.get("query_start") is not None:
                    summary = summarize_domain(seqs[digest],
                        (result["numbering"], result["query_start"], result["query_end"]),
                        {"chain_type": result["chain_type"]})
                old_positions = {p["raw_index"]: p["imgt"] for d in previous[digest]
                                 if d["exact_input_mapping"] for p in d["positions"]}
                new_positions = {p["raw_index"]: p["imgt"] for p in summary["positions"]} if summary else {}
                intersection = set(old_positions) & set(new_positions)
                output.append({"sequence_sha256": digest, "length": len(seqs[digest]),
                               "population": "target80" if case in cases else "control8",
                               "role": case["role"], "original_resolution": case["resolution"],
                               "seq_type": seq_type, "mode": mode,
                               "score": result.get("score"), "error": result.get("error"),
                               "assessment": assessment(result, summary, seq_type), "domain": summary,
                               "anarci_shared_raw_indices": len(intersection),
                               "anarci_same_imgt_positions": sum(old_positions[i] == new_positions[i] for i in intersection)})
            print(json.dumps(run_times[-1]), flush=True)
    package = OUT / "isolated_packages" / "anarcii"
    package_inputs = {str(p): sha_file(p) for p in package.rglob("*")
                      if p.is_file() and p.suffix in {".py", ".json", ".pt"}}
    report = {"status": "annotation_candidates_not_certificate", "version": "ANARCII-2.0.8",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "source_report_sha256": sha_file(BASE / "report.json"),
              "coverage_sha256": sha_file(coverage_path), "script_sha256": sha_file(__file__),
              "numbering_v1_sha256": sha_file(BASE / "numbering.json"),
              "package_download_sha256": sha_file(OUT / "package_download.json"),
              "package_inputs": package_inputs, "cpu_threads": 4, "torch_version": torch.__version__,
              "parameters": {"ncpu": 4, "cpu": True, "batch_size": 32,
                             "prediction_thresholds": "official defaults, no conditioning",
                             "review_thresholds": "accuracy warning thresholds: shark>24.5, antibody>25"},
              "runs": run_times,
              "counts": dict(Counter((r["population"] + ":" + r["seq_type"] + ":" + r["mode"] + ":" + r["assessment"]) for r in output)),
              "rows": output,
              "limits": ["Predicted numbering is annotation evidence, not experimental ground truth.",
                         "Scores are model output scores, not calibrated probabilities.",
                         "No gene anchors or missing residues added; original benchmark remains immutable.",
                         "Speed and accuracy models are related, not independent experimental evidence."]}
    write(OUT / "anarcii_report.json", report)
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    run()
