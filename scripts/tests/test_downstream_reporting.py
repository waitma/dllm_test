"""Synthetic report integrity checks. Run: python -m pytest scripts/tests/test_downstream_reporting.py."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from downstream.benchmark.common.reporting import PROTOCOL_FIELDS, compare_protocols, new_report_dir
from downstream.benchmark.scripts import summarize_nm2025 as t1
from downstream.benchmark.scripts import summarize_tcr_representation_paper6 as t3
from downstream.benchmark.scripts import audit_baseline_provenance as baseline_audit
from downstream.benchmark.tcr_binding.original_protocol import choose_result_source
from scripts.downstream.audit_result_integrity import audit_run, endpoint_status, exposure_verdict


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("paper", [False, True])
def test_paper_is_primary_independent_of_rerun(local, paper):
    assert choose_result_source(local_complete=local, paper_available=paper)[0] == (
        "paper_reported" if paper else "unavailable")


def test_unknown_or_mismatched_protocol_is_not_rankable():
    complete = {field: f"verified-{field}" for field in PROTOCOL_FIELDS}
    assert compare_protocols(complete, complete)["comparison_allowed"]
    for empty in ({}, [], None, ""):
        assert not compare_protocols({field: empty for field in PROTOCOL_FIELDS},
                                     {field: empty for field in PROTOCOL_FIELDS})["comparison_allowed"]
    for field in PROTOCOL_FIELDS:
        missing = {k: v for k, v in complete.items() if k != field}
        assert not compare_protocols(complete, missing)["rank_eligible"]
        different = {**complete, field: "different"}
        assert compare_protocols(complete, different)["protocol_alignment"] == "mismatched"


@pytest.mark.parametrize("paper_present", [True, False])
def test_t1_local_values_never_fill_paper_cells(tmp_path, monkeypatch, paper_present):
    spec = SimpleNamespace(tag="toy", paper_method="Toy", display_name="Toy", wrapper="toy",
                           checkpoint_available=True, artifact_identity_verified=True, checkpoint_paths=())
    monkeypatch.setattr(t1, "ORIGINAL_BASELINES", (spec,))
    monkeypatch.setattr(t1, "OUT", tmp_path)
    monkeypatch.setattr(t1, "_original_zip_matches", lambda: True)
    monkeypatch.setattr(t1, "_wrapper_available", lambda spec: True)
    monkeypatch.setattr(t1, "_validate_local_original_split", lambda spec, split: (
        {"auprc_overall": 0.91, "auroc_overall": 0.99}, "embedded_original_protocol"))
    paper = pd.DataFrame([{"method": "Toy", "split": "seen", "metric": "auprc", "value": 0.7,
                           "source_location": "paper", "citation": "citation"}])
    if not paper_present:
        paper = paper.iloc[:0]
    row = t1._write_original_baseline_summary(paper, tmp_path).iloc[0]
    assert row.local_seen_auprc == 0.91
    assert row.seen_auprc == 0.7 if paper_present else pd.isna(row.seen_auprc)
    assert pd.isna(row.unseen_auprc)
    assert row.source_location != str(tmp_path)
    assert not row.comparison_allowed


def test_t1_missing_paper_file_retains_all_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(t1, "OUT", tmp_path / "raw")
    monkeypatch.setattr(t1, "PAPER_CSV", tmp_path / "absent.csv")
    monkeypatch.setattr(t1, "_original_zip_matches", lambda: False)
    monkeypatch.setattr(t1, "_wrapper_available", lambda spec: False)
    monkeypatch.setattr(t1, "_validate_local_original_split", lambda spec, split: (None, "missing"))
    # Avoid hashing unrelated real checkpoints in a synthetic test.
    monkeypatch.setattr(t1, "ORIGINAL_BASELINES", (SimpleNamespace(
        tag="toy", paper_method="Toy", display_name="Toy", wrapper=None, checkpoint_available=False,
        artifact_identity_verified=False, checkpoint_paths=()),))
    result = t1.main(["--output-dir", str(tmp_path / "report")])
    rows = pd.read_csv(result / "summary_main.csv")
    assert list(rows.tag) == ["toy"]
    assert rows.iloc[0].selected_source == "unavailable"


def test_paper_esm2_identity_is_separate_from_local_150m():
    assert "esm2_t6_8m" in t3.PAPER_K200 and "esm2_150m" not in t3.PAPER_K200
    decision = t3.paper_comparison("esm2_150m", {"n_universe": 25816})
    assert decision["protocol_alignment"] == "mismatched"
    assert not decision["comparison_allowed"]
    assert not t3.paper_comparison("sceptr", {"n_universe": 25816})["comparison_allowed"]
    assert t3.paper_comparison("sceptr", {"n_universe": 7168})["protocol_alignment"] == "mismatched"


def test_paper6_transcription_filters_task(tmp_path, monkeypatch):
    rows = [{"benchmark": "t3_representation", "task": task, "metric": "auroc", "method": "ESM2-150M",
             "split": split, "value": value} for task, value in [("sceptr_six_pmhc_k200", 0.7), ("other", 0.9)]
            for split in t3.SHORT]
    path = tmp_path / "paper.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(t3, "PAPER_CSV", path)
    assert t3.load_paper_k200()["esm2_t6_8m"] == [0.7] * 6


@pytest.mark.parametrize("recorded", [None, "old-training-sha"])
def test_new_corpus_audit_cannot_clean_historical_checkpoint(recorded):
    result = exposure_verdict(recorded_training_sha=recorded, current_training_sha="new-training-sha",
                              registry_sha="registry", scope_version="scope", cohort="toy/test",
                              audit={"dataset_manifest_sha256": "new-training-sha", "registry_sha256": "registry",
                                     "scope_version": "scope", "cohorts": ["toy/test"], "unique_exact": []})
    assert result["status"] == "unknown" and result["pretraining_unseen"] == "unknown"


def test_exposure_scope_does_not_overclaim_unseen():
    kwargs = dict(recorded_training_sha="train", current_training_sha="train", registry_sha="registry",
                  scope_version="scope", cohort="toy/test")
    audit = {"status": "complete", "dataset_manifest_sha256": "train", "registry_sha256": "registry", "scope_version": "scope",
             "cohorts": ["toy/test"], "unique_exact": []}
    assert exposure_verdict(audit=audit, **kwargs)["pretraining_unseen"] == "unknown"
    audit["unique_exact"] = [{"cohort": "toy/test", "role": "synthetic", "count": 1}]
    assert exposure_verdict(audit=audit, **kwargs)["status"] == "exposed"


def test_spr_complete_is_not_pkd_complete():
    result = endpoint_status("spr", {"status": "complete", "probe": {"status": "blocked_unit_and_censoring_no_pkd_fit"}}, {})
    assert result["spr_ranking"] == "complete" and result["pkd_regression"] == "blocked"


def test_partial_pairing_cannot_be_complete():
    manifest = {"dataset_len": 100, "max_samples": 5, "num_seqs": 2}
    assert endpoint_status("pairing", {"total_pairs": 8}, manifest)["pairing_metrics"] == "partial"
    assert endpoint_status("pairing", {"total_pairs": 10}, manifest)["pairing_metrics"] == "complete"


def test_report_directory_never_overwrites(tmp_path):
    path = tmp_path / "report"
    new_report_dir(path, "toy")
    with pytest.raises(FileExistsError):
        new_report_dir(path, "toy")


def test_audit_rejects_local_substitution_and_retains_missing_paper(monkeypatch):
    monkeypatch.setattr(baseline_audit, "ORIGINAL_BASELINES", (SimpleNamespace(tag="toy", paper_method="Toy"),))
    paper = pd.DataFrame(columns=["benchmark", "task", "method", "split", "metric", "value"])
    table = pd.DataFrame([{"tag": "toy", "selected_source": "unavailable", "result_label": "missing",
                           "comparison_allowed": False, "seen_auprc": None, "seen_auroc": None,
                           "unseen_auprc": None, "unseen_auroc": None}])
    assert baseline_audit.validate_t1(table, paper) == []
    table.loc[0, "seen_auprc"] = 0.9
    assert any("missing paper cell was filled" in err for err in baseline_audit.validate_t1(table, paper))


def test_future_paper_registry_preserves_esm2_8m_and_paper_primary():
    from downstream.benchmark.scripts.import_paper_baselines import registry, t3_rows
    methods = {row["method"] for row in t3_rows()}
    assert "ESM2-8M" in methods and "ESM2-150M" not in methods
    assert "Always use paper_reported" in registry()["policy"]


def test_result_cannot_bind_wrong_feature_manifest_or_checkpoint(tmp_path):
    import json
    from downstream.benchmark.common.reporting import object_sha256, sha256_file

    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"toy weights")
    identity = {"checkpoint_sha256": sha256_file(weights), "pairs_sha256": "data", "input_protocol_sha256": "protocol",
                "model_assets_sha256": "assets", "feature_dim": 2, "pooling": "toy", "input_record": "toy"}
    metrics, manifest = tmp_path / "metrics.json", tmp_path / "manifest.json"
    metrics.write_text(json.dumps({"feature_identity": object_sha256(identity)}))
    manifest.write_text(json.dumps({**identity, "checkpoint_path": str(weights)}))
    spec = {"id": "toy", "task": "tcr_binding_retrained", "metrics": str(metrics), "manifest": str(manifest)}
    assert audit_run(spec, [])["checkpoint"]["metrics_to_manifest_binding"] == "hash_verified"
    manifest.write_text(json.dumps({**identity, "checkpoint_path": str(weights), "pairs_sha256": "other-data"}))
    assert "metrics_feature_manifest_identity_mismatch" in audit_run(spec, [])["errors"]
    metrics.write_text(json.dumps({"checkpoint_sha256": "other-checkpoint"}))
    assert "metrics_manifest_checkpoint_mismatch" in audit_run(spec, [])["errors"]
    from scripts.downstream.audit_result_integrity import fingerprint
    original = fingerprint(str(weights))
    weights.write_bytes(b"different toy weights with a different size")
    assert original["sha256"] != fingerprint(str(weights))["sha256"]
