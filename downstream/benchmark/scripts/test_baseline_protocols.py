#!/usr/bin/env python
"""Focused regressions for baseline protocol repairs."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_regression_target_transform_roundtrip():
    from downstream.mint_tasks.finetune_general import (
        fit_regression_target_transform,
        inverse_regression_predictions,
    )

    raw = np.array([-4.5, -1.0, 0.2, 1.7, 8.0, 20.0])
    transformed, transformer = fit_regression_target_transform(raw)
    recovered = inverse_regression_predictions(transformed, transformer)
    np.testing.assert_allclose(recovered, raw, rtol=1e-9, atol=1e-9)


def test_t2_ld_connects_single_indel_but_hd_does_not():
    module = _load_module(
        "import_clustering_baselines",
        BENCH / "scripts" / "import_clustering_baselines.py",
    )
    seqs = ["CASSLGQETQYF", "CASSLQETQYF", "CASSLGQDTQYF"]
    hd = module._distance_components(seqs, "hamming")
    ld = module._distance_components(seqs, "lev")
    component = lambda mapping, seq: mapping.get(seq, seq)
    assert component(hd, seqs[0]) != component(hd, seqs[1])
    assert component(ld, seqs[0]) == component(ld, seqs[1])
    assert component(hd, seqs[0]) == component(hd, seqs[2])


def test_t4_sparse13_aggregation_key():
    from tcr_generation_bench.scoring import _aggregate_conditional

    metric_stub = {
        "precision": 0.5, "recall": 0.5, "f1": 0.5,
        "d_edit": 1.0, "seq_recovery": 0.5, "char_bleu": 0.5,
        "diversity": 0.5, "novelty_vs_conditioning": 0.5,
        "diversity_edit": 1.0, "mean_nn_to_ref": 1.0,
        "category": "paper_sparse13", "bioseq_unseen": False,
    }
    per_pmhc = {"pep1_hla1": dict(metric_stub), "pep2_hla2": dict(metric_stub)}
    result = _aggregate_conditional(per_pmhc)
    assert result["paper_sparse13"]["n_pmhc"] == 2


def test_paper_import_has_all_baseline_families():
    module = _load_module(
        "import_paper_baselines",
        BENCH / "scripts" / "import_paper_baselines.py",
    )
    rows = (
        module.t1_rows() + module.t2_rows() + module.t3_rows()
        + module.mint_rows() + module.mint_flab_rows()
        + module.ophiuchus_rows() + module.nbbench_rows()
    )
    benchmarks = {row["benchmark"] for row in rows}
    assert benchmarks == {
        "t1_binding", "t2_clustering", "t3_representation",
        "mint_generalppi", "flab", "antibody_cdr", "nbbench",
    }
    assert all(row["evidence_type"] == "paper_reported" for row in rows)
    assert all(row["citation"] and row["source_location"] for row in rows)


def test_t1_original_catalog_covers_every_paper_method():
    paper = _load_module(
        "import_paper_baselines_t1_catalog",
        BENCH / "scripts" / "import_paper_baselines.py",
    )
    from tcr_binding.original_protocol import ORIGINAL_BASELINES

    paper_methods = {row["method"] for row in paper.t1_rows()}
    catalog_methods = {spec.paper_method for spec in ORIGINAL_BASELINES}
    assert catalog_methods == paper_methods
    assert len({spec.tag for spec in ORIGINAL_BASELINES}) == len(ORIGINAL_BASELINES)


def test_t1_original_source_selection_is_always_paper_primary():
    from tcr_binding.original_protocol import (
        PAPER_RESULT_LABEL,
        choose_result_source,
    )

    assert choose_result_source(local_complete=True, paper_available=True) == (
        "paper_reported", PAPER_RESULT_LABEL,
    )
    assert choose_result_source(local_complete=False, paper_available=True) == (
        "paper_reported", PAPER_RESULT_LABEL,
    )
    assert choose_result_source(local_complete=True, paper_available=False)[0] == "unavailable"


def test_t1_main_table_excludes_noncanonical_external_reruns():
    module = _load_module(
        "summarize_nm2025_main_filter",
        BENCH / "scripts" / "summarize_nm2025.py",
    )
    local = pd.DataFrame([
        {"tag": "ergo2", "type": "official pretrained"},
        {"tag": "atmtcr", "type": "official pretrained"},
        {"tag": "knn", "type": "training-free"},
        {"tag": "ours", "type": "ours LLaDA-dec globalfeat+MLP"},
    ])
    kept = module._main_local_context(local)
    assert set(kept["tag"]) == {"knn", "ours"}


def test_official_test_loader_does_not_require_train_csv(tmp_path, monkeypatch):
    from tcr_binding import run_nm2025

    test_dir = tmp_path / "seen" / "cdr3b" / "AS"
    test_dir.mkdir(parents=True)
    pd.DataFrame({"peptide": ["PEPTIDE"], "cdr3b": ["CASSF"], "label": [1]}).to_csv(
        test_dir / "test.csv", index=False
    )
    monkeypatch.setattr(run_nm2025, "DATA", tmp_path)
    tests = run_nm2025._load_tests("cdr3b")
    assert list(tests) == [("seen", "AS")]
    assert not (tmp_path / "train" / "cdr3b" / "train.csv").exists()


def test_original_zip_tests_only_preparation_never_opens_train(tmp_path, monkeypatch):
    module = _load_module(
        "prepare_nm2025_tests_only",
        BENCH / "scripts" / "prepare_nm2025_binding.py",
    )
    monkeypatch.setattr(module, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(module, "TEST_MANIFEST", tmp_path / "original_test_manifest.json")
    monkeypatch.setattr(module, "OFFICIAL_REPO", tmp_path / "missing_official_repo")

    report = module.prepare(tests_only=True)
    assert report["mode"] == "tests_only"
    assert (tmp_path / "original_test_manifest.json").is_file()
    assert (tmp_path / "seen" / "cdr3b" / "AS" / "test.csv").is_file()
    assert not (tmp_path / "train").exists()


def test_variant_tag_selects_checkpoint_before_wrapper_import(monkeypatch):
    from tcr_binding.run_official_baseline import _configure_variant

    monkeypatch.delenv("TCRBENCH_BASELINE_TAG", raising=False)
    spec = _configure_variant("teinet", "teinet_small")
    assert spec.variant_tag == "teinet_small"
    assert os.environ["TCRBENCH_BASELINE_TAG"] == "teinet_small"
    with pytest.raises(ValueError, match="belongs to wrapper"):
        _configure_variant("ergo", "teinet_small")


def test_every_official_wrapper_reports_the_catalogued_runtime_artifacts(monkeypatch):
    from tcr_binding.original_protocol import ORIGINAL_BASELINES

    for spec in ORIGINAL_BASELINES:
        if spec.wrapper is None:
            continue
        if spec.variant_tag is not None:
            monkeypatch.setenv("TCRBENCH_BASELINE_TAG", spec.variant_tag)
        wrapper = __import__(f"baselines.wrappers.{spec.wrapper}", fromlist=["*"])
        reported = tuple(Path(path) for path in wrapper.official_checkpoint_paths())
        assert reported == spec.checkpoint_paths, spec.tag
        assert spec.artifact_identity_verified, spec.tag


def test_driver_rejects_wrapper_catalog_artifact_mismatch():
    from tcr_binding.original_protocol import ORIGINAL_BASELINES
    from tcr_binding.run_official_baseline import _checkpoint_paths_for

    class WrongWrapper:
        @staticmethod
        def official_checkpoint_paths():
            return (Path("/wrong/checkpoint.pt"),)

    with pytest.raises(ValueError, match="do not match catalog"):
        _checkpoint_paths_for(WrongWrapper, ORIGINAL_BASELINES[0])


def test_retrained_protocol_resolves_fold_specific_and_shared_tests():
    from tcr_binding.retrained_protocol import resolve_test_member

    fold_test = resolve_test_member(eval_set="seen_test", neg_source="AS", fold=3)
    seen_independent = resolve_test_member(
        eval_set="seen_independent", neg_source="AS", fold=3,
    )
    unseen_independent = resolve_test_member(
        eval_set="unseen_independent", neg_source="AS", fold=3,
    )
    assert fold_test.endswith("/AS/3_1_1test.csv")
    assert seen_independent.endswith("/AS/1_1_1independent_test.csv")
    assert "_only_seen/" in seen_independent
    assert "_only_unseen/" in unseen_independent


def test_retrained_protocol_resolves_fold_train_for_tcrh_preprocessing():
    from tcr_binding.retrained_protocol import resolve_train_member

    member = resolve_train_member(neg_source="AS", fold=4)
    assert "_only_seen/" in member
    assert member.endswith("/AS/4_1_1train.csv")


def test_retrained_model_catalog_has_paper_values_for_every_connected_model():
    from tcr_binding.retrained_protocol import MODEL_SPECS, PAPER_EXPECTED_AS

    assert set(MODEL_SPECS) == set(PAPER_EXPECTED_AS)
    assert {
        "atmtcr", "nettcr", "epitcr", "teim", "tcrh", "teinet",
        "ergo_ae", "ergo_lstm",
    } <= set(MODEL_SPECS)
    assert MODEL_SPECS["ergo_ae"].drop_remainder_batch_size == 50
    assert MODEL_SPECS["ergo_lstm"].drop_remainder_batch_size is None


def test_retrained_prediction_validation_allows_only_declared_tail_drop():
    from dataclasses import replace
    from tcr_binding.retrained_protocol import MODEL_SPECS
    from tcr_binding.run_retrained_baseline import _validate_predictions

    test = pd.DataFrame({
        "Epitope": ["PEP"] * 52,
        "CDR3B": [f"CASS{i}" for i in range(52)],
        "Affinity": [i % 2 for i in range(52)],
    })
    result = test.iloc[:50].rename(columns={"Affinity": "y_true"}).copy()
    result["y_prob"] = 0.5
    _validate_predictions(test, result, MODEL_SPECS["ergo_ae"])
    strict = replace(MODEL_SPECS["ergo_ae"], drop_remainder_batch_size=None)
    with pytest.raises(RuntimeError, match="protocol expects 52"):
        _validate_predictions(test, result, strict)
