#!/usr/bin/env python
"""Regression tests for the public epitope-conditioned CDR3beta Track A.

Run with::

    /vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python -m pytest -q \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/test_public_beta_benchmark.py

These tests intentionally exercise protocol decisions that could otherwise
silently inflate results: prediction columns are never references, invalid
sequences are never repaired, @K follows raw rank, and TCR-epiDiff cannot enter
the de-novo result table through a reconstruction path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from tcr_generation_bench.public_generators import (  # noqa: E402
    Candidate,
    ReconstructionOnlyError,
    TCREpiDiffGenerator,
    _rank_scored_candidates,
)
from tcr_generation_bench.bioseq_generator import (  # noqa: E402
    BIOSEQ_7L_STEP117000_LABEL,
    BIOSEQ_7L_STEP189000_LABEL,
    DEFAULT_BIOSEQ_7L_STEP117000,
    DEFAULT_BIOSEQ_7L_STEP189000,
    BioSeqTrackAGenerator,
)
from tcr_generation_bench.paired_alpha import (  # noqa: E402
    official_alpha_evaluation_audit,
    select_tcrdesign_generated_betas,
)
from tcr_generation_bench.track_a import (  # noqa: E402
    RVR_TARGET_ID,
    Target,
    best_sequence_recovery,
    canonical_target_id,
    cross_epitope_jaccard,
    edit_distance,
    generation_rows,
    import_official_predictions,
    is_valid_cdr3b,
    load_public_targets,
    migrate_legacy_tcrt5_paper_ranking,
    score_target,
    sequence_recovery,
    validity_reason,
)


def _target(references: tuple[str, ...] = ("CASSLGQETQYF", "CASSIRSSYEQYF")) -> Target:
    return Target(
        target_id="PEPTIDE_HLA-A*01:01",
        source_target_id="PEPTIDE_A*01:01",
        peptide="PEPTIDE",
        mhc_allele="HLA-A*01:01",
        mhc_pseudosequence="A" * 34,
        references=references,
        reference_source="synthetic_test",
        dataset="synthetic_test",
        year="2026",
    )


@pytest.mark.parametrize(
    ("sequence", "reason"),
    [
        ("CASSSSF", "valid"),
        ("CASSSSW", "valid"),
        ("CASSSF", "length_outside_7_24"),
        ("C" + "A" * 23 + "F", "length_outside_7_24"),
        ("AASSSSF", "missing_initial_C"),
        ("CASSSSA", "missing_terminal_F_or_W"),
        ("CASS*SF", "non_standard_amino_acid"),
        ("CASSBSF", "non_standard_amino_acid"),
    ],
)
def test_strict_validity_rules(sequence: str, reason: str):
    assert validity_reason(sequence) == reason
    assert is_valid_cdr3b(sequence) is (reason == "valid")


def test_generation_rows_normalises_case_but_never_repairs_anchors():
    frame = generation_rows("model", _target(), [" asssssf ", "cassssa"])
    assert frame["cdr3b"].tolist() == ["ASSSSSF", "CASSSSA"]
    assert not any(frame["cdr3b"].map(is_valid_cdr3b))


def test_tcrt5_paper_ranking_uses_cumulative_log_likelihood():
    candidates = [
        Candidate("CASSLONGSEQUENCEF", -12.0),
        Candidate("CASSSHORTF", -8.0),
        Candidate("CASSMIDDLEF", -10.0),
    ]
    ranked = _rank_scored_candidates(candidates)
    assert [candidate.cdr3b for candidate in ranked] == [
        "CASSSHORTF", "CASSMIDDLEF", "CASSLONGSEQUENCEF",
    ]


def test_legacy_tcrt5_score_migration_reconstructs_cumulative_ranking():
    target = _target()
    frame = generation_rows(
        "TCRT5",
        target,
        [
            {"cdr3b": "CASSLONGSEQUENCEF", "raw_score": -0.50},
            {"cdr3b": "CASSSHORTF", "raw_score": -0.60},
            {"cdr3b": "CASSMIDDLEF", "raw_score": -0.61},
        ],
        run_id="legacy",
    )
    migrated = migrate_legacy_tcrt5_paper_ranking(frame).sort_values("rank")
    assert migrated["cdr3b"].tolist() == [
        "CASSSHORTF", "CASSMIDDLEF", "CASSLONGSEQUENCEF",
    ]
    assert migrated["run_id"].eq("legacy_paper_loglik_rank").all()


def test_normalised_levenshtein_recovery():
    assert edit_distance("CASSLGQETQYF", "CASSLQETQYF") == 1
    assert sequence_recovery("CASSLGQETQYF", "CASSLQETQYF") == pytest.approx(11 / 12)
    assert best_sequence_recovery("CASSLGQETQYF", ("CASSLQETQYF", "CASAAAAAF")) == pytest.approx(11 / 12)


def test_at_k_uses_raw_rank_and_counts_distinct_hits_only():
    target = _target()
    raw = ["CASSLGQETQYF"] * 2 + ["CASSSSA"] * 98 + ["CASSIRSSYEQYF", "CASSSSF"]
    frame = generation_rows("model", target, raw)
    metrics = score_target(frame, target, num_candidates=1000)
    assert metrics["n_generated"] == 102
    assert metrics["n_valid"] == 4
    assert metrics["n_unique_valid"] == 3
    assert metrics["ValidRate"] == pytest.approx(4 / 1000)
    assert metrics["UniqueRate"] == pytest.approx(3 / 4)
    assert metrics["ExactHit@100"] == 1
    assert metrics["ExactHit@1000"] == 2
    assert metrics["ReferenceRecall@1000"] == 1.0
    assert '"missing_candidate": 898' in metrics["invalid_reasons"]


def test_at_k_does_not_pull_later_rank_across_a_gap():
    target = _target()
    frame = generation_rows("model", target, ["CASSLGQETQYF", "CASSIRSSYEQYF"])
    frame.loc[1, "rank"] = 101
    metrics = score_target(frame, target, num_candidates=1000)
    assert metrics["ExactHit@100"] == 1
    assert metrics["ExactHit@1000"] == 2


def test_cross_target_jaccard_uses_unique_valid_sets():
    left = _target()
    right = Target(**{
        **left.__dict__,
        "target_id": "OTHER_HLA-A*01:01",
        "peptide": "OTHER",
    })
    frame = pd.concat([
        generation_rows("model", left, ["CASSSSF", "CASSSSW", "INVALID"]),
        generation_rows("model", right, ["CASSSSF", "CASSQQF"]),
    ], ignore_index=True)
    result = cross_epitope_jaccard(frame)
    assert len(result) == 1
    assert result.iloc[0]["intersection"] == 1
    assert result.iloc[0]["union"] == 3
    assert result.iloc[0]["Jaccard"] == pytest.approx(1 / 3)


def test_public_benchmark_fields_and_reference_counts():
    targets = load_public_targets()
    assert len(targets) == 14
    rvr = targets[RVR_TARGET_ID]
    assert canonical_target_id("RVRAYTYSK", "A*03:01") == RVR_TARGET_ID
    assert rvr.peptide == "RVRAYTYSK"
    assert rvr.mhc_allele == "HLA-A*03:01"
    assert len(rvr.mhc_pseudosequence) == 34
    assert rvr.n_references == 895
    assert sum(target.n_references for target in targets.values()) == 1312


def test_author_released_predictions_have_complete_schema_and_counts():
    frame = import_official_predictions(("tcrt5", "gratcr"))
    assert set(frame["model"]) == {"TCRT5", "GRATCR"}
    assert frame.groupby(["model", "target_id"]).size().eq(1000).all()
    assert len(frame) == 2 * 14 * 1000
    assert set(frame["source"]) == {
        "TCR-TRANSLATE benchmark_data_w_preds.csv official artifact"
    }
    rvr_references = set(load_public_targets()[RVR_TARGET_ID].references)
    # This guards against accidentally importing the reference_translations
    # list itself under either model label.
    rvr_tcrt5 = set(frame.loc[
        (frame["model"] == "TCRT5") & (frame["target_id"] == RVR_TARGET_ID),
        "cdr3b",
    ])
    assert len(rvr_tcrt5) == 1000
    assert rvr_tcrt5 != rvr_references


def test_tcr_epidiff_is_hard_guarded_as_reconstruction_only():
    generator = TCREpiDiffGenerator()
    ready, reason = generator.check_ready()
    assert not ready
    assert generator.generation_mode == "reconstruction_only"
    assert "original_TCR" in reason
    with pytest.raises(ReconstructionOnlyError, match="excluded from de-novo"):
        generator.generate("RVRAYTYSK", num_candidates=10)


def test_bioseq_track_a_metadata_locks_step117000_epitope_only_protocol():
    generator = BioSeqTrackAGenerator()
    assert generator.name == BIOSEQ_7L_STEP117000_LABEL
    assert generator.checkpoint == DEFAULT_BIOSEQ_7L_STEP117000.resolve()
    assert generator.generation_mode == "de_novo_epitope_conditioned"
    assert generator.conditioning == "epitope_only"
    assert generator.score_semantics == "unscored_seeded_emission_order"
    assert generator.seed == 42
    assert generator.batch_size == 100
    assert generator.max_iter == 32


def test_bioseq_track_a_explicit_checkpoint_uses_manifest_step_label():
    generator = BioSeqTrackAGenerator(DEFAULT_BIOSEQ_7L_STEP189000)
    assert generator.name == BIOSEQ_7L_STEP189000_LABEL
    assert generator.checkpoint_step == 189000
    assert generator.checkpoint == DEFAULT_BIOSEQ_7L_STEP189000.resolve()
    assert generator.checkpoint_manifest()["step"] == 189000


def test_tcrdesign_alpha_conditions_use_generated_betas_not_references():
    target = _target()
    generated = generation_rows(
        "TcrDesign",
        target,
        ["CASSGENAAAF", "CASSGENBBBF", "CASSGENCCCF"],
        generation_mode="de_novo",
        source="official_code_and_weights_runtime",
    )
    distractor = generation_rows(
        "TCRT5",
        target,
        [target.references[0], target.references[1]],
        generation_mode="de_novo",
    )
    selected = select_tcrdesign_generated_betas(
        pd.concat([generated, distractor], ignore_index=True),
        [target.target_id],
        beta_top_k=2,
    )
    assert selected["generated_cdr3b"].tolist() == ["CASSGENAAAF", "CASSGENBBBF"]
    assert selected["beta_rank"].tolist() == [1, 2]
    assert not set(selected["generated_cdr3b"]) & set(target.references)


def test_tcrdesign_official_alpha_audit_keeps_alpha_out_of_ranking():
    audit = official_alpha_evaluation_audit()
    assert audit["alpha_generation_available"] is True
    assert audit["official_alpha_evaluation_available"] is False
    assert audit["included_in_quantitative_benchmark"] is False
    assert audit["decision"] == "qualitative_output_only"
    pairing = audit["released_pair_representation_code"]
    assert pairing["input_csv_columns"] == ["CDR3a", "CDR3b"]
    assert pairing["released_pairing_checkpoint"] is False
