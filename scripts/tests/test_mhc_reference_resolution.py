"""Activate pllm; python -B -m pytest scripts/tests/test_mhc_reference_resolution.py -q."""
import hashlib
from pathlib import Path

import pytest

from scripts.data.mhc_reference_resolution import HLAReference, parse_allele


def make_reference(tmp_path, entries):
    path = tmp_path / 'hla.fasta'
    path.write_text(''.join(f'>HLA:ID{i} {allele}\n{sequence}\n'
                            for i, (allele, sequence) in enumerate(entries)))
    return HLAReference.from_fasta(path), path


@pytest.mark.parametrize('name,canonical', [
    ('HLA-A*02:01', 'A*02:01'), ('A02:01', 'A*02:01'), ('a0201', 'A*02:01'),
    ('HLA-DRB104:01', 'DRB1*04:01'), ('DQA1*05:01:01:02N', 'DQA1*05:01:01:02N'),
    ('A*02:101', 'A*02:101'), ('A*02:1000', 'A*02:1000'), ('B27', 'B*27'),
])
def test_preserves_precision_and_suffix(name, canonical):
    assert parse_allele(name).canonical == canonical


@pytest.mark.parametrize('name', ['A020101', 'A02/01', 'A*2:01', 'A*02:01G',
                                  'A*02:01/B*07:02', '', 'BOGUS*02:01'])
def test_rejects_ambiguous_or_unsupported_notation(name):
    with pytest.raises(ValueError):
        parse_allele(name)


def test_coverage_variants_accept_longest_and_preserve_evidence(tmp_path):
    ref, path = make_reference(tmp_path, [('A*02:01:01:01', 'ACDEFGHIK'),
        ('A*02:01:01:02', 'DEFG'), ('A*02:01:02', 'ACDEFGHIK'), ('A*02:01:03', 'FGHIK')])
    result = ref.resolve('HLA-A02:01')
    assert result['accepted'] and result['sequence'] == 'ACDEFGHIK'
    assert result['entry_count'] == 4 and result['unique_sequence_count'] == 3
    assert result['reference_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(c['exact_substring'] for c in result['coverage'])
    fragment = next(c for c in result['coverage'] if c['length'] == 4)
    assert fragment['start_positions_1based'] == [3]
    assert fragment['end_positions_1based_inclusive'] == [6]
    assert ref.resolve('A*02:01:01:02')['sequence'] == 'DEFG'


@pytest.mark.parametrize('entries,status', [
    ([('A*02:01:01', 'ACDEFG'), ('A*02:01:02', 'ACDEYG')], 'conflicting_longest_sequences'),
    ([('A*02:01:01', 'ACDEFGHIK'), ('A*02:01:02', 'DEYG')], 'reference_coverage_conflict'),
    ([('A*02:01:01', 'ACDEFGHIK'), ('A*02:01:02', 'DEXG')], 'invalid_reference_sequence'),
    ([('A*02:01:01', 'ACD*EF')], 'invalid_reference_sequence'),
    ([('A*02:01:01', 'ACD-EF')], 'invalid_reference_sequence'),
    ([('A*02:01:01', '')], 'invalid_reference_sequence'),
])
def test_rejects_conflicting_or_unresolved_amino_acids(tmp_path, entries, status):
    ref, _ = make_reference(tmp_path, entries)
    result = ref.resolve('A02:01')
    assert not result['accepted'] and result['sequence'] is None
    assert result['status'] == status


def test_suffix_and_resolution_cannot_silently_merge_alleles(tmp_path):
    ref, _ = make_reference(tmp_path, [('A*02:01:01:01', 'ACDEFG'),
                                      ('A*02:01:01:02N', 'ACDEYG')])
    assert not ref.resolve('A*02:01')['accepted']
    assert ref.resolve('A*02:01:01:01')['sequence'] == 'ACDEFG'
    assert ref.resolve('A*02:01N')['sequence'] == 'ACDEYG'
    assert ref.resolve('A*02:01:01:01N')['status'] == 'reference_not_found'
    assert ref.resolve('A02')['status'] == 'insufficient_allele_resolution'
    assert ref.resolve('B*07:02')['status'] == 'reference_not_found'


def test_only_formatting_and_terminal_stop_are_normalized(tmp_path):
    ref, _ = make_reference(tmp_path, [('A*02:01:01', 'acde fghik*'),
                                      ('A*02:01:02', 'ACDEFGHIK')])
    result = ref.resolve('A02:01')
    assert result['accepted'] and result['sequence'] == 'ACDEFGHIK'
    assert result['source_entries'][0]['terminal_stop_removed']
    assert result['source_entries'][0]['lowercase_normalized']


def test_actual_a0201_reference_is_coverage_consistent():
    path = Path(__file__).resolve().parents[2] / 'data/tcr_pmhc_fulllength/imgt_hla/hla_prot.fasta'
    if not path.exists():
        pytest.skip('local IMGT/HLA reference is unavailable')
    ref = HLAReference.from_fasta(path)
    result = ref.resolve('A02:01')
    assert result['accepted'] and result['sequence_length'] == 365
    assert result['entry_count'] == 504 and result['unique_sequence_count'] == 10
    assert ref.resolve('B27')['status'] == 'insufficient_allele_resolution'
