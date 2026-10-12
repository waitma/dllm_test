"""Make a read-only, endpoint-level report of existing downstream results.

Run: python scripts/downstream/audit_result_integrity.py --config <CONFIG_JSON> --output-dir <NEW_DIR>
Config lists explicit task/artifact paths and candidate exposure audits. A candidate
training dataset is never treated as checkpoint lineage without recorded hashes.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from downstream.benchmark.common.reporting import compare_protocols, file_identity, new_report_dir, object_sha256, sha256_file, write_report_manifest


@lru_cache(maxsize=None)
def _cached_fingerprint(path: str, size: int, mtime_ns: int) -> dict:
    return file_identity(path)


def fingerprint(path: str) -> dict:
    before = Path(path).stat()
    result = _cached_fingerprint(path, before.st_size, before.st_mtime_ns)
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"input changed during hashing: {path}")
    return result


def exposure_verdict(*, recorded_training_sha: str | None, audit: dict, registry_sha: str,
                     current_training_sha: str, cohort: str | None, scope_version: str) -> dict:
    """No dataset-name or checkpoint-name inference; report only the audited scope."""
    reasons = []
    if not recorded_training_sha:
        reasons.append("checkpoint_training_manifest_binding_missing")
    elif recorded_training_sha != current_training_sha:
        reasons.append("checkpoint_training_manifest_mismatch")
    if audit.get("dataset_manifest_sha256") != current_training_sha:
        reasons.append("audit_training_manifest_mismatch")
    if audit.get("registry_sha256") != registry_sha or audit.get("scope_version") != scope_version:
        reasons.append("audit_registry_or_scope_mismatch")
    if not cohort or cohort not in audit.get("cohorts", []):
        reasons.append("evaluation_cohort_not_covered")
    if audit.get("status") not in {"overlap_found", "review_required", "complete"}:
        reasons.append("audit_not_finalized")
    if reasons:
        return {"status": "unknown", "pretraining_unseen": "unknown", "reasons": reasons}
    hits = [hit for hit in audit.get("unique_exact", []) if hit.get("cohort") == cohort and hit.get("count", 0) > 0]
    return {"status": "exposed" if hits else "no_exact_hits_in_audited_scope",
            "pretraining_unseen": False if hits else "unknown", "exact_hits": hits,
            "pending": audit.get("pending", []), "limitations": audit.get("limitations", []),
            "scope_version": scope_version}


def endpoint_status(task: str, metrics: dict, manifest: dict) -> dict:
    """Completion of one endpoint cannot imply completion of another."""
    if task == "spr":
        probe = metrics.get("probe", {})
        pkd_status = probe.get("status", "missing")
        return {"spr_ranking": metrics.get("status", "unknown"),
                "pkd_regression": "blocked" if pkd_status.startswith("blocked") else "unknown",
                "pkd_reason": pkd_status}
    if task == "gdpa1":
        return {"historical_cv": "complete" if metrics.get("properties") else "missing",
                "independent_generalization": "not_established"}
    if task == "pairing":
        expected = manifest.get("dataset_len", 0) * manifest.get("num_seqs", 0)
        if manifest.get("max_samples"):
            expected = min(manifest.get("dataset_len", 0), manifest["max_samples"]) * manifest.get("num_seqs", 0)
        actual = metrics.get("total_pairs")
        status = "complete" if expected > 0 and actual == expected else "partial" if actual else "missing"
        return {"pairing_metrics": status, "expected_pairs": expected, "observed_pairs": actual,
                "independent_generalization": "not_established"}
    if task == "cdr":
        return {"cdr_aar": metrics.get("status", "aggregate_only"),
                "offline_rescorable": metrics.get("identity_sha256") is not None}
    return {task: metrics.get("status", "artifact_present" if metrics else "missing")}


def _load(path: str | None) -> dict:
    return json.loads(Path(path).read_text()) if path and Path(path).is_file() else {}


def verify_inputs(manifest: dict) -> list[dict]:
    """Verify recorded evaluation data identities when the historical run supplied them."""
    expected = dict(manifest.get("data_sha256", {}))
    for source in manifest.get("source", {}).values():
        if isinstance(source, dict) and source.get("path") and source.get("sha256"):
            expected[source["path"]] = source["sha256"]
    if manifest.get("csv_path") and manifest.get("input_csv_sha256"):
        expected[manifest["csv_path"]] = manifest["input_csv_sha256"]
    result = []
    for path, sha in expected.items():
        if not isinstance(sha, str):
            continue
        if not Path(path).is_absolute():
            # A digest labelled by dataset name is not a file-path binding.
            result.append({"recorded_key": path, "recorded_sha256": sha, "status": "unknown_path"})
            continue
        observed = fingerprint(str(Path(path).resolve())) if Path(path).is_file() else None
        result.append({"path": path, "recorded_sha256": sha, "observed": observed,
                       "status": "missing" if observed is None else "verified" if observed["sha256"] == sha else "mismatched"})
    return result


def audit_run(spec: dict, audits: list[dict]) -> dict:
    metrics = _load(spec.get("metrics"))
    manifest = _load(spec.get("manifest"))
    if spec.get("manifest_task"):
        manifest = {**manifest, **manifest.get("tasks", {}).get(spec["manifest_task"], {}).get("identity", {})}
    manifest = {**manifest, **manifest.get("identity", {})}
    feature = metrics.get("feature_manifest", {})
    artifacts = [fingerprint(str(Path(spec[key]).resolve())) for key in ("metrics", "manifest", "log")
                 if spec.get(key) and Path(spec[key]).is_file()]
    errors, limitations = [], list(spec.get("limitations", []))
    evaluated_inputs = verify_inputs(manifest)
    if any(item["status"] == "mismatched" for item in evaluated_inputs):
        errors.append("evaluation_input_checksum_mismatch")
    hashes = {obj["checkpoint_sha256"] for obj in (metrics, manifest, feature) if obj.get("checkpoint_sha256")}
    if len(hashes) > 1:
        errors.append("metrics_manifest_checkpoint_mismatch")
    recorded_sha = next(iter(hashes)) if len(hashes) == 1 else None
    ckpt_path = manifest.get("checkpoint") or manifest.get("checkpoint_path") or feature.get("checkpoint") or spec.get("checkpoint")
    ckpt = None
    if ckpt_path:
        weights = Path(ckpt_path)
        if weights.is_dir():
            weights /= "model.safetensors"
        if weights.is_file():
            ckpt = fingerprint(str(weights.resolve()))
            if recorded_sha and ckpt["sha256"] != recorded_sha:
                errors.append("checkpoint_file_checksum_mismatch")
        else:
            limitations.append("checkpoint_file_unavailable")
    if not recorded_sha:
        limitations.append("result_checkpoint_hash_missing; configured_path_is_not_historical_proof")
    if metrics.get("run_manifest_sha256") and spec.get("manifest"):
        if sha256_file(spec["manifest"]) != metrics["run_manifest_sha256"]:
            errors.append("run_manifest_checksum_mismatch")
    feature_keys = ("checkpoint_sha256", "pairs_sha256", "input_protocol_sha256", "model_assets_sha256",
                    "feature_dim", "pooling", "input_record")
    feature_bound = False
    if metrics.get("feature_identity") and all(key in manifest for key in feature_keys):
        feature_bound = object_sha256({key: manifest[key] for key in feature_keys}) == metrics["feature_identity"]
        if not feature_bound:
            errors.append("metrics_feature_manifest_identity_mismatch")

    # These are saved claims, not inferred from a path containing v6/v7/clean.
    training_sha = manifest.get("training_manifest_sha256") or feature.get("training_manifest_sha256")
    exposure = []
    for candidate in audits:
        verdict = exposure_verdict(recorded_training_sha=training_sha if recorded_sha and ckpt and not errors else None,
                                   audit=candidate["payload"], registry_sha=candidate["registry"]["sha256"],
                                   current_training_sha=candidate["training_manifest"]["sha256"],
                                   cohort=spec.get("cohort"), scope_version=candidate["scope_version"])
        if candidate.get("errors"):
            verdict = {"status": "unknown", "pretraining_unseen": "unknown", "reasons": candidate["errors"]}
        cohort_inputs = [item for item in candidate.get("cohort_inputs", []) if item.get("cohort") == spec.get("cohort")]
        if not any(item.get("sha256") == source.get("recorded_sha256") and source.get("status") == "verified"
                   for item in cohort_inputs for source in evaluated_inputs):
            verdict = {"status": "unknown", "pretraining_unseen": "unknown",
                       "reasons": [*verdict.get("reasons", []), "evaluation_cohort_hash_binding_missing"]}
        exposure.append({"audit": candidate["name"], "report": candidate["report"], **verdict})
    if spec["task"] == "gdpa1":
        limitations += ["legacy_reference: same CV used for alpha selection and reporting",
                        "label transforms fitted before CV; no independent generalization claim"]
    if spec["task"] == "pairing":
        limitations += manifest.get("known_deviations", [])
        limitations += ["single seed" if manifest.get("seed") is not None else "seed provenance missing",
                        "historical test-set parameter sweep; no independent best-setting claim"]
        if manifest.get("light_length_mode") in {"reference", "auto"}:
            limitations.append("reference-derived target length; length-conditioned generation")
    if spec["task"] == "cdr" and not metrics:
        limitations.append("historical aggregate log only; per-row predictions cannot be reconstructed")
    comparison = compare_protocols(metrics.get("comparison_contract"), spec.get("reference_contract"))
    endpoints = endpoint_status(spec["task"], metrics, manifest) if artifacts else {spec["task"]: "missing"}
    return {"id": spec["id"], "task": spec["task"], "artifacts": artifacts,
            "integrity_status": "invalid" if errors else "observed" if artifacts else "missing",
            "errors": errors, "endpoints": endpoints, "evaluation_inputs": evaluated_inputs,
            "checkpoint": {"observed_file": ckpt, "recorded_sha256": recorded_sha,
                           "binding_status": "verified_metadata_to_weights" if ckpt and recorded_sha == ckpt["sha256"] and not errors else "unknown",
                           "metrics_to_manifest_binding": "hash_verified" if (metrics.get("run_manifest_sha256") or feature_bound) and not errors else "not_cryptographically_bound"},
            "recorded_input_sha256": manifest.get("input_sha256", manifest.get("eval_sha256")),
            "recorded_training_manifest_sha256": training_sha,
            "downstream_unseen": {"claim": spec.get("downstream_unseen", "unknown"),
                                  "scope": "downstream split only; does not establish pretraining-unseen"},
            "pretraining_exposure": exposure,
            "source_generalization_claim": metrics.get("generalization_claim"),
            "reported_protocol": metrics.get("protocol", manifest.get("protocol", manifest.get("protocol_id"))),
            "length_condition": manifest.get("length_condition", manifest.get("light_length_mode")),
            "seed": manifest.get("seed", manifest.get("args", {}).get("seed")),
            "limitations": limitations, **comparison}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    destination = new_report_dir(args.output_dir, "result_integrity")
    inputs = [args.config, Path(__file__), ROOT / "downstream/benchmark/common/reporting.py"]
    audits = []
    for spec in config.get("audits", []):
        refs = {key: file_identity(spec[key]) for key in ("registry", "report", "training_manifest")}
        payload, registry = _load(spec["report"]), _load(spec["registry"])
        errors = []
        if payload.get("registry_sha256") != refs["registry"]["sha256"]:
            errors.append("audit_registry_checksum_mismatch")
        if payload.get("dataset_manifest_sha256") != refs["training_manifest"]["sha256"]:
            errors.append("audit_training_manifest_checksum_mismatch")
        inputs += [Path(spec[key]) for key in refs]
        audits.append({"name": spec["name"], **refs, "payload": payload, "errors": errors,
                       "cohort_inputs": registry.get("inputs", []),
                       "scope_version": registry.get("scope_version"),
                       "known_dataset_exposure": payload.get("status"),
                       "pending": payload.get("pending", [])})
    records = [audit_run(spec, audits) for spec in config["runs"]]
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run IDs in report config")
    inputs += [Path(a["path"]) for record in records for a in record["artifacts"]]
    report = {"schema_version": 1, "policy": "passive_snapshot; endpoint_status_separate; unknown_lineage_never_clean",
              "runs": records, "candidate_dataset_audits": [{k: v for k, v in a.items() if k not in {"payload", "cohort_inputs"}} for a in audits],
              "limitations": ["Candidate dataset exposure audits do not prove which corpus trained a historical checkpoint",
                              "An exact-match audit cannot prove homology-level or family-level absence"]}
    (destination / "integrity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_report_manifest(destination, inputs, policy=report["policy"])
    print(f"Report: {destination}; runs={len(records)}; invalid={sum(bool(r['errors']) for r in records)}")
    return 1 if any(r["errors"] for r in records) or any(a["errors"] for a in audits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
