#!/usr/bin/env python
"""Regressions for the frozen-BioSeq NM2025 retrained-fold protocol.

Run with::

    source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
    conda activate /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
    pytest -q downstream/benchmark/scripts/test_retrained_ours_protocol.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO))

from tcr_binding import run_retrained_ours as ours  # noqa: E402


def _frame(rows):
    return pd.DataFrame(rows, columns=["Epitope", "CDR3B", "Affinity"])


def _synthetic_protocol():
    folds = {}
    for fold in ours.FOLDS:
        folds[fold] = {
            "train": _frame([
                (f"SEEN{fold}", f"TRAINPOS{fold}", 1),
                (f"SEEN{fold}", f"TRAINNEG{fold}", 0),
            ]),
            "seen_test": _frame([(f"SEEN{fold}", f"TEST{fold}", 1)]),
            "seen_independent": _frame([(f"SEEN{fold}", f"IND{fold}", 0)]),
            "unseen_independent": _frame([(f"NEW{fold}", f"NEWTEST{fold}", 1)]),
        }
    return folds


def test_pair_universe_is_sorted_unique_and_stably_hashed():
    frames = _synthetic_protocol()
    pairs = ours.build_pair_universe(frames)
    assert pairs == sorted(set(pairs))
    assert len(pairs) == 25
    assert ours.pair_universe_sha256(pairs) == ours.pair_universe_sha256(list(pairs))
    assert ours.pair_universe_sha256(pairs) != ours.pair_universe_sha256(pairs[:-1])


def test_overlap_audit_distinguishes_seen_epitope_from_exact_pair_leakage():
    audit = ours.protocol_overlap_audit(_synthetic_protocol())
    for fold in ours.FOLDS:
        tests = audit["folds"][str(fold)]["tests"]
        assert tests["seen_test"]["shared_unique_epitopes"] == 1
        assert tests["seen_test"]["shared_unique_clonotypes"] == 0
        assert tests["seen_test"]["shared_unique_exact_pairs"] == 0
        assert tests["unseen_independent"]["shared_unique_epitopes"] == 0
        assert tests["unseen_independent"]["shared_unique_exact_pairs"] == 0


def test_official_fold_train_has_no_clonotype_pair_or_unseen_epitope_overlap():
    frames, _ = ours.load_protocol("AS")
    audit = ours.protocol_overlap_audit(frames)
    for fold in ours.FOLDS:
        tests = audit["folds"][str(fold)]["tests"]
        for eval_set in ours.EVAL_SETS:
            assert tests[eval_set]["shared_unique_clonotypes"] == 0
            assert tests[eval_set]["shared_unique_exact_pairs"] == 0
        assert tests["unseen_independent"]["shared_unique_epitopes"] == 0


def test_primary_ranking_uses_local_release_values_and_paper_is_separate():
    summary = {
        "tag": "ours",
        "method": "Ours",
        "eval_sets": {
            eval_set: {
                "local": {
                    "auroc_mean": 0.70,
                    "auroc_std_sample": 0.01,
                    "auprc_precrec_mean": 0.60,
                    "auprc_precrec_std_sample": 0.02,
                }
            }
            for eval_set in ours.EVAL_SETS
        },
    }
    comparison = pd.DataFrame([
        {
            "tag": "baseline",
            "method": "Baseline",
            "eval_set": eval_set,
            "local_auroc_mean": 0.65,
            "local_auroc_std": 0.03,
            "local_auprc_mean": 0.55,
            "local_auprc_std": 0.04,
            "paper_auroc_mean": 0.75,
            "paper_auroc_std": 0.05,
            "paper_auprc_mean": 0.80,
            "paper_auprc_std": 0.06,
        }
        for eval_set in ours.EVAL_SETS
    ])
    local = ours._ranking_frame(summary, comparison, source="local_release")
    paper = ours._ranking_frame(summary, comparison, source="paper_reference")
    local_ours = local[local["tag"] == "ours"]
    paper_ours = paper[paper["tag"] == "ours"]
    assert set(local_ours["rank"]) == {1}
    assert set(paper_ours["rank"]) == {2}
    assert not local["mixed_source"].any()
    assert paper["mixed_source"].all()
    assert set(local[local["tag"] == "baseline"]["result_source"]) == {
        "official_retrained_checkpoint_local_rerun"
    }
    assert set(paper[paper["tag"] == "baseline"]["result_source"]) == {
        "paper_reported_retrained"
    }
    wide = ours._wide_comparison(local, tag_order=["baseline", "ours"])
    assert list(wide["tag"]) == ["baseline", "ours"]
    assert wide.loc[wide["tag"] == "ours", "unseen_independent_auprc_rank"].item() == 1
    markdown = ours._comparison_markdown(wide, title="Test", note="Primary")
    assert "| Baseline |" in markdown
    assert "| Ours |" in markdown
    assert "0.6000±0.0200 (#1)" in markdown
    seen = ours._seen_split_comparison(
        wide, eval_set="seen_test", ours_tag="ours",
    )
    assert seen.iloc[-1]["tag"] == "ours"
    assert seen.iloc[:-1]["auprc_mean"].is_monotonic_decreasing
    assert not any("unseen" in column for column in seen.columns)
    seen_markdown = ours._seen_split_markdown(seen, title="Seen test")
    assert "Baselines are sorted by descending AUPRC" in seen_markdown
    assert seen_markdown.index("Baseline") < seen_markdown.index("Ours")


def test_ergo_ae_ranking_note_declares_tail_drop_for_every_eval_set():
    assert "30650/30670" in ours._baseline_n_note("ergo_ae", "seen_test")
    assert "5850/5882" in ours._baseline_n_note("ergo_ae", "seen_independent")
    assert "3150/3162" in ours._baseline_n_note("ergo_ae", "unseen_independent")
    assert ours._baseline_n_note("epitcr", "seen_test") == "all official rows"
