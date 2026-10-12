"""Run: python -B -m pytest -p no:cacheprovider scripts/tests/immune_llada/test_test_pair_clean_export.py."""
from scripts.data.export_immune_test_pair_clean import select_reasons, protected_test_mask
from scripts.data.export_immune_expanded_clean import DEFAULT_PLAN, read_json


def test_two_independent_chain_hits_do_not_establish_pair():
    # Both chains in the same cohort can match DIFFERENT reference records.
    reasons = {'exact_cdr3a': 1, 'exact_cdr3b': 1,
               'edit80_cdr3a': 1, 'edit80_cdr3b': 1, 'exact_peptide': 1}
    assert select_reasons(['tcr_alpha', 'tcr_beta', 'peptide'], reasons, 1) == {}


def test_pair_requires_test_hit_not_support_hit():
    assert select_reasons(['tcr_alpha', 'tcr_beta'], {'exact_ab': 2, 'exact_cdr3a': 1}, 1) == {}
    assert select_reasons(['tcr_alpha', 'tcr_beta'], {'exact_ab': 3}, 1) == {'exact_ab': 1}


def test_nonpaired_keeps_test_rules_and_filters_support():
    assert select_reasons(['tcr_beta'], {'edit80_cdr3b': 3, 'exact_peptide': 2}, 1) == {'edit80_cdr3b': 1}


def test_scope_includes_actual_paper6_queries_but_not_training():
    cohorts = read_json(DEFAULT_PLAN)['cohort_bit_order']
    mask = protected_test_mask(cohorts)
    for name in ('tcr_representation_paper6/background_pool', 'tcr_representation_paper6/target_binders',
                 'tcr_clustering/tcrs', 'uptcr/majority/test'):
        assert mask & (1 << cohorts.index(name))
    for name in ('tcr_representation/train', 'uptcr/majority/valid', 'uptcr/majority/train'):
        assert not mask & (1 << cohorts.index(name))
