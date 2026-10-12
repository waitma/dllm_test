"""Run in pllm: python -B -m pytest -p no:cacheprovider scripts/tests/immune_llada/test_test_pair_clean_v2.py."""
import pytest

from scripts.data.export_immune_test_pair_clean_v2 import select_reasons
from scripts.data.audit_test_pair_release_v2 import stored_core


def row(alpha=None, beta=None):
    return dict(chain_roles=['tcr_beta', 'tcr_alpha'], chains=['BBB', 'AAA'],
                chain_metadata={str(i): {'synthetic_residue_mask': flags}
                                for i, flags in enumerate((beta, alpha)) if flags is not None})


def test_synthetic_alpha_uses_single_beta_test_filter():
    assert select_reasons(row(alpha=[1, 1, 1]), {'edit80_cdr3b': 3}, 1) == {'edit80_cdr3b': 1}


def test_synthetic_beta_uses_single_alpha_test_filter():
    assert select_reasons(row(beta=[1, 1, 1]), {'exact_cdr3a': 1}, 1) == {'exact_cdr3a': 1}


@pytest.mark.parametrize('record', [row(), row(alpha=[1, 0, 1], beta=[1, 0, 1])])
def test_two_observed_chains_need_same_pair(record):
    assert select_reasons(record, {'exact_cdr3a': 1, 'exact_cdr3b': 1, 'exact_peptide': 1}, 1) == {}
    assert select_reasons(record, {'exact_ab': 3}, 1) == {'exact_ab': 1}


@pytest.mark.parametrize('record', [row(), row(alpha=[1, 1, 1])])
def test_non_test_hits_cannot_remove(record):
    assert select_reasons(record, {'exact_ab': 2, 'edit80_cdr3b': 2}, 1) == {}


@pytest.mark.parametrize('flags', [[1], [1, 2, 1]])
def test_invalid_provenance_fails_closed(flags):
    with pytest.raises(ValueError):
        select_reasons(row(alpha=flags), {'exact_ab': 1}, 1)


def test_empty_partner_is_not_observed():
    record = row()
    record['chains'][1] = ''
    assert select_reasons(record, {'exact_cdr3b': 1}, 1) == {'exact_cdr3b': 1}


def test_audit_existing_identifier_fallback_never_infers_a_missing_core():
    record = row()
    record['identifiers'] = {'cdr3b_core': 'B'}
    assert stored_core(record, 0, 'tcr_beta') == ('B', True)
    assert stored_core(record, 1, 'tcr_alpha') == (None, False)
    record['regions'] = {'0': {'CDR3': 'BB'}}
    assert stored_core(record, 0, 'tcr_beta') == ('BB', False)
