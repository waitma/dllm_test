#!/usr/bin/env python
"""Sanity checks for the official-parity binding metric.

Perfect scores -> AUPRC/AUROC == 1; random-ish scores -> ~0.5; and the
per-epitope + all_values structure matches the official ``calculate`` layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import metrics  # noqa: E402


def test_perfect():
    labels = [0, 0, 1, 1, 0, 1]
    scores = [0.01, 0.02, 0.99, 0.98, 0.03, 0.97]
    groups = ["A", "A", "A", "B", "B", "B"]
    rep = metrics.official_binding_report(labels, scores, groups)
    assert abs(rep["all_values"]["prc_auc"] - 1.0) < 1e-9, rep["all_values"]
    assert abs(rep["all_values"]["roc_auc"] - 1.0) < 1e-9
    assert abs(rep["macro"]["prc_auc"] - 1.0) < 1e-9
    assert abs(rep["all_values"]["accuracy"] - 1.0) < 1e-9
    assert rep["n_epitopes"] == 2
    assert set(rep["per_epitope"]) == {"A", "B"}
    print("perfect: prc_auc=%.4f roc_auc=%.4f acc=%.4f mcc=%.4f" % (
        rep["all_values"]["prc_auc"], rep["all_values"]["roc_auc"],
        rep["all_values"]["accuracy"], rep["all_values"]["mcc"]))


def test_random():
    rng = np.random.default_rng(0)
    n = 4000
    labels = rng.integers(0, 2, n)
    scores = rng.random(n)
    groups = rng.choice(list("ABCD"), n)
    rep = metrics.official_binding_report(labels, scores, groups)
    assert 0.4 < rep["all_values"]["prc_auc"] < 0.6, rep["all_values"]["prc_auc"]
    assert 0.45 < rep["all_values"]["roc_auc"] < 0.55, rep["all_values"]["roc_auc"]
    print("random: prc_auc=%.4f roc_auc=%.4f (macro prc=%.4f)" % (
        rep["all_values"]["prc_auc"], rep["all_values"]["roc_auc"],
        rep["macro"]["prc_auc"]))


def test_matches_binding_metrics():
    rng = np.random.default_rng(1)
    n = 1000
    labels = rng.integers(0, 2, n)
    scores = rng.random(n)
    groups = rng.choice(list("ABC"), n)
    rep = metrics.official_binding_report(labels, scores, groups)
    bm = metrics.binding_metrics(labels, scores, groups)
    # macro AUPRC/AUROC across epitopes must agree with legacy binding_metrics.
    assert abs(rep["macro"]["prc_auc"] - bm["macro_auprc"]) < 1e-9
    assert abs(rep["macro"]["roc_auc"] - bm["macro_auroc"]) < 1e-9
    print("parity with binding_metrics: OK")


def test_precrec_pr_auc_differs_from_average_precision_on_ties():
    # Hand-calculated from precrec's default equivalent-tie interpolation.
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.9, 0.2, 0.1]
    paper_auc = metrics.precrec_pr_auc(labels, scores)
    sklearn_ap = metrics._safe_auprc(np.asarray(labels), np.asarray(scores))
    assert 0.0 <= paper_auc <= 1.0
    assert abs(paper_auc - sklearn_ap) > 1e-4


def test_precrec_pr_auc_perfect_is_one():
    assert metrics.precrec_pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_precrec_pr_auc_matches_upstream_numeric_examples():
    # precrec tests/testthat/test_pl5_2_create_prc.R, x_bins=100.
    scores = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    assert abs(metrics.precrec_pr_auc(
        [0, 1, 0, 1, 0, 1], scores, x_bins=100,
    ) - 0.395) < 1e-3
    assert abs(metrics.precrec_pr_auc(
        [1, 1, 0, 1, 0, 0], scores, x_bins=100,
    ) - 0.904) < 1e-3


if __name__ == "__main__":
    test_perfect()
    test_random()
    test_matches_binding_metrics()
    print("ALL PASS")
