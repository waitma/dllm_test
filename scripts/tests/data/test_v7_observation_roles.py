"""CPU-only audit logic checks: activate pllm; pytest -q this file.

Uses synthetic strings/position metadata only. Does not run ANARCI or alter data.
"""
import copy

import pytest

from scripts.data.audit_v7_observation_roles import (
    adjudicate_short_reference, gap_evidence, observed_scope, parse_embl,
    role_from_labels, short_reference_evidence, summarize_domain,
)


def domain(indices, raw_start=0, prefix=0, suffix=0, chain="H"):
    return {"chain_type": chain, "exact_input_mapping": True,
            "unmapped_prefix_length": prefix, "unmapped_suffix_length": suffix,
            "imgt_min": min(indices), "imgt_max": max(indices),
            "positions": [{"imgt": p, "raw_index": i + raw_start} for i, p in enumerate(indices)],
            "regions": {"cdr3": {"length": sum(105 <= p <= 117 for p in indices),
                                   "raw_indices": [i + raw_start for i, p in enumerate(indices)
                                                   if 105 <= p <= 117]}}}


@pytest.mark.parametrize("end,mode", [(3, "inclusive"), (4, "exclusive")])
def test_mapping_checks_actual_slice_end_convention(end, mode):
    d = summarize_domain("ABCDE", ([((1, " "), "B"), ((2, " "), "-"),
                                    ((3, "A"), "C"), ((4, " "), "D")], 1, end),
                         {"chain_type": "H"})
    assert d["exact_input_mapping"]
    assert d["end_convention_verified"] == mode
    assert d["unmapped_suffix_length"] == 1
    assert [p["raw_index"] for p in d["positions"]] == [1, 2, 3]
    assert "B" not in d["positions"][0].values()


def test_mismatched_numbering_never_invents_indices():
    d = summarize_domain("ABCDE", ([((1, " "), "X")], 0, 0), {"chain_type": "H"})
    assert not d["exact_input_mapping"]
    assert d["positions"][0]["raw_index"] is None
    assert observed_scope([d]) == "pending_numbering_input_mapping"


@pytest.mark.parametrize("label,chain,role", [("3", "L", "light_lambda"),
                                            ("4", "K", "light_kappa"),
                                            ("2", None, "vnar")])
def test_author_roles(label, chain, role):
    domains = [domain([1, 105, 118], chain=chain)] if chain else []
    assert role_from_labels([label, label], domains) == role


def test_role_conflict_cannot_be_waived():
    assert role_from_labels(["3", "4"], []) == "pending_role_conflict"
    assert role_from_labels(["3"], [domain([1], chain="K")]) == "pending_role_numbering_conflict"


def test_fragment_requires_no_unmapped_suffix():
    d = domain([1, 26, 65, 81])
    assert observed_scope([d]) == "observed_fragment_ends_before_h3"
    d["unmapped_suffix_length"] = 1
    assert observed_scope([d]).startswith("pending_")


def test_empty_source_loop_requires_contiguous_flanks_and_full_input_mapping():
    d = domain([1, 103, 118, 128])
    assert observed_scope([d], True) == "observed_empty_h3_source_and_numbering_agree"
    assert observed_scope([d], False).startswith("pending_")
    for key in ("unmapped_prefix_length", "unmapped_suffix_length"):
        altered = copy.deepcopy(d)
        altered[key] = 1
        assert observed_scope([altered], True).startswith("pending_")
    d["positions"][2]["raw_index"] += 1
    assert observed_scope([d], True).startswith("pending_")


def test_no_domain_stays_pending_not_incomplete_by_length_guess():
    assert observed_scope([]) == "pending_no_numbered_domain"
    assert observed_scope([domain([1]), domain([2])]) == "pending_multiple_domains"


def test_native_gap_locations_do_not_assume_source_loop_order():
    row = {"CDR1": "CD", "CDR2": "GH", "CDR3": "AB"}
    evidence = gap_evidence("----EF--", "ABCDEFGH", row)
    assert not evidence["all_three_loops_have_distinct_unique_gap_matches"]
    row = {"CDR1": "DE", "CDR2": "GH", "CDR3": "AB"}
    evidence = gap_evidence("--C--F--", "ABCDEFGH", row)
    assert evidence["visible_residues_match_native"]
    assert evidence["source_loop_gap_matches"]["cdr3"]["matching_gap_intervals"] == [(0, 2)]
    assert evidence["all_three_loops_have_distinct_unique_gap_matches"]
    assert not gap_evidence("--X--F--", "ABCDEFGH", row)["visible_residues_match_native"]


def test_exact_native_prefix_can_show_partial_loop_without_inventing_full_target():
    d = domain([101, 102, 103, 104, 105, 106, 107, 118])
    row = {"CDR1": "A", "CDR2": "C", "CDR3": "EF"}
    evidence = short_reference_evidence("ABCDEF", "-B-D--", row, "ABCDEFGH", [d])
    assert evidence["observed_native_cdr3_length"] == 2
    assert evidence["reference_cdr3_length"] == 3
    assert adjudicate_short_reference(evidence) == "observed_partial_h3_exact_native_reference"
    assert "sequence" not in evidence


def test_unexplained_native_extension_does_not_become_verified_cdr3():
    row = {"CDR1": "C", "CDR2": "D", "CDR3": "EF"}
    evidence = short_reference_evidence("ABCDEF", "AB----", row, "ABCD", [domain([1, 2, 3, 4])])
    assert evidence["extension_matches_source_cdr3"]
    assert adjudicate_short_reference(evidence) == "pending_source_construct_exceeds_native_reference"


def test_embl_reads_only_explicit_translation_and_germline_annotation():
    text = ('DE   partial V segment\nFT   source          1..12\n'
            'FT                   /germline\nFT   CDS             <1..>12\n'
            'FT                   /translation="AB\nFT                   CD"\n'
            'FT   V_segment       <1..>12\n')
    seq, metadata = parse_embl(text)
    assert seq == "ABCD"
    assert metadata["germline_source_explicit"]
    assert "V_segment" in metadata["feature_types"]
    with pytest.raises(ValueError, match="one explicitly supplied"):
        parse_embl("DE   incomplete metadata only\n")


def test_observed_l3_is_typed_by_chain_and_not_heavy_h3():
    d = domain([1, 105, 106, 118], chain="L")
    assert role_from_labels(["3"], [d]) == "light_lambda"
    assert observed_scope([d]) == "observed_numbered_cdr3"
