"""Activate pllm; USE_TF=0 python -B -m pytest scripts/tests/test_v7_mhc_completion.py -q."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts.data import complete_v7_mhc as builder
from scripts.data.mhc_reference_resolution import HLAReference
from scripts.data.update_mhc_inputs import identity, non_mhc
from dllm.pipelines.immune_llada.data.mhc import PseudoTable


@pytest.fixture
def references(tmp_path):
    path = tmp_path / 'reference.fasta'
    alleles = [('A*02:01:01:01', 'M' + 'A' * 364),
               ('B*07:02:01', 'M' + 'C' * 361),
               ('DQA1*03:01:01', 'M' + 'D' * 253),
               ('DQB1*03:02:12', 'M' + 'E' * 260),
               ('DQA1*05:01:01', 'M' + 'F' * 253),
               ('DQA1*05:01:02', 'M' + 'G' * 253)]
    path.write_text(''.join(f'>ID{i} {allele}\n{seq}\n'
                            for i, (allele, seq) in enumerate(alleles)))
    pseudo_path = tmp_path / 'pseudo.dat'
    pseudo_path.write_text('HLA-A02:01 ' + 'A' * 34 + '\nHLA-B07:02 ' + 'C' * 34 + '\n')
    return HLAReference.from_fasta(path), dict(sequence='I' * 99,
        metadata=dict(mhc_component='b2m', reference_derived=True)), PseudoTable(pseudo_path)


@pytest.fixture
def example():
    row = dict(chains=['A' * 34, 'GILGFVFTL', 'XCASSFX', 'CAVF'],
        chain_roles=['mhc', 'peptide', 'tcr_beta', 'tcr_alpha'],
        task_type='tcr_pmhc', source='trait', split='train', weight=0.75,
        labels={'relation': 'binding'}, identifiers={'pair_key': 'example'},
        metadata={'source': 'example_origin', 'receptor_completion': True},
        regions={'2': {'cdr3': [1, 6]}, '3': {'cdr3': [0, 4]}},
        chain_metadata={'0': {'mhc_representation': 'existing_pseudo34'},
            '2': {'synthetic_residue_mask': [1, 0, 0, 0, 0, 0, 1]},
            '3': {'source_note': 'observed'}})
    candidate = dict(current_shard='train/trait-00000.jsonl', current_row=0,
        source='trait', split='train', old_row_sha256=hashlib.sha256(builder.encoded(row)).hexdigest(),
        identity=identity(row, 'trait'), original_mhc=['A' * 34], allele_names=['A*02:01'],
        evidence=[{'allele': 'A*02:01', 'source_sha256': 'toy'}])
    return row, candidate


def test_explicit_classii_pair_orders_alpha_then_beta_and_keeps_precision(references):
    reference, b2m, _ = references
    result = builder.resolve_context('DQB1*03:02:12/DQA1*03:01:01', reference, b2m)
    assert result['accepted'] and result['mhc_class'] == 'II'
    assert result['normalized_alleles'] == ['DQA1*03:01:01', 'DQB1*03:02:12']
    assert [e['metadata']['mhc_component'] for e in result['entries']] == ['mhc_alpha', 'mhc_beta']
    assert result['entries'][0]['sequence'] == 'M' + 'D' * 253
    assert result['entries'][1]['sequence'] == 'M' + 'E' * 260


@pytest.mark.parametrize('allele,status', [
    ('DQA1*03/DQB1*03:02:12', 'insufficient_allele_resolution'),
    ('DQA1*03:01:01', 'missing_explicit_classII_partner'),
    ('DQA1*05:01/DQB1*03:02:12', 'conflicting_longest_sequences'),
    ('A*02:01/B*07:02', 'invalid_classII_gene_pair'),
])
def test_classii_missing_precision_partner_or_conflicting_reference_is_rejected(references, allele, status):
    reference, b2m, _ = references
    result = builder.resolve_context(allele, reference, b2m)
    assert not result['accepted'] and result['status'] == status


def test_multiple_source_alleles_cannot_pick_first_sequence(references, example):
    reference, b2m, pseudo = references
    _, candidate = example
    candidate['allele_names'] = ['A*02:01', 'B*07:02']
    result = builder.plan_candidate(candidate, reference, b2m, pseudo, {})
    assert not result['accepted'] and result['status'] == 'conflicting_source_allele_sequences'


def test_source_pseudo_must_match_the_allele(references, example):
    reference, b2m, pseudo = references
    _, candidate = example
    candidate['original_mhc'] = ['C' * 34]
    result = builder.plan_candidate(candidate, reference, b2m, pseudo, {})
    assert not result['accepted'] and result['status'] == 'source_pseudo_allele_mismatch'


def test_conflicting_pseudo_reference_is_not_treated_as_missing(references, example, tmp_path):
    reference, b2m, _ = references
    _, candidate = example
    path = tmp_path / 'conflicting_pseudo.dat'
    path.write_text('HLA-A02:01 ' + 'A' * 34 + '\nHLA-A02:01 ' + 'C' * 34 + '\n')
    result = builder.plan_candidate(candidate, reference, b2m, PseudoTable(path), {})
    assert not result['accepted'] and result['status'] == 'source_pseudo_reference_conflict'


def test_checked_replacement_protects_every_non_mhc_field(references, example):
    reference, b2m, pseudo = references
    row, candidate = example
    before = deepcopy(row)
    plan = builder.plan_candidate(candidate, reference, b2m, pseudo, {})
    result = builder.checked_replacement(row, candidate, plan, 'parent_hash', 'ledger_hash')
    assert row == before
    assert non_mhc(result) == non_mhc(row)
    assert result['chain_roles'][:2] == ['mhc', 'mhc']
    assert result['chains'][2:] == row['chains'][1:]
    assert result['regions']['3'] == row['regions']['2']
    assert result['chain_metadata']['3'] == row['chain_metadata']['2']
    assert result['metadata']['mhc_update']['parent_row_sha256'] == candidate['old_row_sha256']


@pytest.mark.parametrize('mutation', ['wrong_identity', 'wrong_split', 'old_mhc'])
def test_checked_replacement_rejects_misanchored_evidence(references, example, mutation):
    reference, b2m, pseudo = references
    row, candidate = example
    plan = builder.plan_candidate(candidate, reference, b2m, pseudo, {})
    if mutation == 'wrong_identity':
        candidate['identity'] = 'incorrect'
    elif mutation == 'wrong_split':
        candidate['split'] = 'valid'
    else:
        candidate['original_mhc'] = ['C' * 34]
    with pytest.raises(AssertionError):
        builder.checked_replacement(row, candidate, plan, 'parent_hash', 'ledger_hash')


def test_process_shard_rejects_stale_raw_row_before_replacement(references, example, tmp_path, monkeypatch):
    reference, b2m, pseudo = references
    row, candidate = example
    plan = builder.plan_candidate(candidate, reference, b2m, pseudo, {})
    parent, stage = tmp_path / 'parent', tmp_path / 'stage'
    path = parent / candidate['current_shard']
    path.parent.mkdir(parents=True)
    path.write_bytes(builder.encoded(row))
    (stage / 'audit').mkdir(parents=True)
    config = builder.PARENT / 'immune_config.json'
    (stage / 'immune_config.json').write_bytes(config.read_bytes())
    monkeypatch.setattr(builder, 'PARENT', parent)
    monkeypatch.setattr(builder, 'STAGE', stage)
    candidate['old_row_sha256'] = 'stale'
    shard = dict(path=candidate['current_shard'], records=1, source='trait', split='train',
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(AssertionError):
        builder.process_shard((shard, {0: (candidate, plan)}, 'parent_hash', 'ledger_hash'))


def test_publish_requires_matching_acceptance_and_renames_stage_only(tmp_path, monkeypatch):
    stage, output = tmp_path / 'release.building', tmp_path / 'release'
    stage.mkdir()
    manifest = stage / 'dataset_manifest.json'
    manifest.write_text('{}\n')
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for name in ('verification.json', 'reader_acceptance.json', 'independent_acceptance.json'):
        (stage / name).write_text(json.dumps(dict(status='passed', dataset_manifest_sha256=manifest_sha)))
    monkeypatch.setattr(builder, 'STAGE', stage)
    monkeypatch.setattr(builder, 'OUTPUT', output)
    audit = stage / 'independent_acceptance.json'
    audit.write_text(json.dumps(dict(status='passed', dataset_manifest_sha256='stale')))
    with pytest.raises(AssertionError):
        builder.publish()
    assert stage.exists() and not output.exists()
    audit.write_text(json.dumps(dict(status='passed', dataset_manifest_sha256=manifest_sha)))
    builder.publish()
    assert output.is_dir() and not stage.exists()
