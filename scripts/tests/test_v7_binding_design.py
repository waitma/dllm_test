"""Run after activating pllm/protenix_abtcr: python -m pytest -q FILE.

These checks protect against false pairing and accidental exposure-scope or
layout changes during the new data release, without loading model weights.
"""
from copy import deepcopy

from scripts.data.export_v7_binding_design import audit_keys, reasons, unresolved_tcr


def index():
    return dict(exact={'ab': {'AAA|BBB': 1, 'CCC|DDD': 2}},
                near={'cdr3a': {'AAA': 1}, 'cdr3b': {'DDD': 2}},
                clusters={}, contained={}, t4={'DDD|PEPTIDE': 4})


def test_same_test_record_required_for_paired_tcr():
    keys = dict(paired_tcr=True, cdr3a='AAA', cdr3b='DDD', ab='AAA|DDD')
    assert reasons(keys, index()) == {}
    keys.update(cdr3b='BBB', ab='AAA|BBB')
    assert reasons(keys, index()) == {'exact_test_cdr3_pair': 1}


def test_t4_key_is_checked_even_with_unseen_paired_alpha():
    keys = dict(paired_tcr=True, ab='EEE|DDD', bp='DDD|PEPTIDE')
    assert reasons(keys, index()) == {'exact_t4_beta_epitope': 4}


def test_synthetic_partner_does_not_bypass_unpaired_screen():
    row = dict(chains=['XXX', 'XXADDDXX'], chain_roles=['tcr_alpha', 'tcr_beta'],
               regions={'1': {'CDR3': 'DDD'}},
               chain_metadata={'0': {'synthetic_residue_mask': [1, 1, 1]},
                               '1': {'synthetic_residue_mask': [1, 1, 0, 0, 0, 0, 1, 1]}})
    original = deepcopy(row)
    keys, _ = audit_keys(row, {})
    assert not keys['paired_tcr'] and not keys.get('ab')
    assert reasons(keys, index()) == {'edit80_cdr3b': 2}
    assert row == original


def test_audit_key_repair_never_rewrites_scaffold_or_identifiers():
    row = dict(chains=['CAAFF', 'CBBBFF', 'PEPTIDE'], chain_roles=['tcr_alpha', 'tcr_beta', 'peptide'],
               regions={'1': {'CDR3': 'BBB'}}, metadata={'source': 'tcr_pmhc_fulllength'})
    original = deepcopy(row)
    # B is not a valid observed amino-acid core; use a valid sequence for this test.
    row['chains'][1] = 'CDDDFF'
    row['regions']['1']['CDR3'] = 'DDD'
    original = deepcopy(row)
    keys, issues = audit_keys(row, {('CAAFF', 'CDDDFF', 'PEPTIDE'): {'AAA'}})
    assert keys['ab'] == 'AAA|DDD' and 'upstream_alpha_key_restored' in issues
    assert row == original


def test_train_only_identity_is_allowed_but_test_overlap_is_not():
    train_only = dict(paired_tcr=True, ab='TRAIN|ONLY')
    assert reasons(train_only, index()) == {}
    assert reasons(dict(paired_tcr=True, ab='AAA|BBB'), index())


def test_unresolved_observed_core_is_not_certified_clean():
    assert unresolved_tcr(dict(fv_b='CADDD'))
    assert not unresolved_tcr(dict(fv_b='CADDD', cdr3b='ADDD'))
    assert not unresolved_tcr(dict(heavy='CADDD'))
