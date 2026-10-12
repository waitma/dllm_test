"""Artificial fixture contracts; no biological files or GPU jobs.

After activating pllm:
USE_TF=0 python -m pytest scripts/tests/immune_llada/test_expanded_fixtures.py -q
"""
import json
import random

import pytest
import torch

from scripts.architecture_prototypes import expanded_fixtures as fixtures
from dllm.pipelines.immune_llada.data.records import VALID_PROTEIN_CHARS
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig, ImmuneModel


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def rows(fixture):
    return [r.to_dict() for r in (*fixture.train_records, *fixture.valid_records)]


def targets(records):
    return [c.sequence for r in records for c in r.chains if c.role in r.metadata['target_roles']]


def small(**kwargs):
    return fixtures.build_fixture_split(full_capacity=48, fragment_capacity=16, max_length=256, **kwargs)


def test_default_records_are_deterministic_disjoint_and_do_not_consume_global_rng():
    before_py, before_torch = random.getstate(), torch.get_rng_state().clone()
    first, second = fixtures.build_fixture_split(), fixtures.build_fixture_split()
    assert rows(first) == rows(second) and first.manifest == second.manifest
    assert random.getstate() == before_py and torch.equal(torch.get_rng_state(), before_torch)
    assert len(first.train_records) == 128 and len(first.valid_records) == 24
    train, valid = targets(first.train_records), targets(first.valid_records)
    assert not set(train) & set(valid)
    assert len(set(train+valid)) == len(train+valid)
    assert first.manifest['exact_target_sequence_overlap'] == 0
    assert first.manifest['exact_record_overlap'] == 0
    for split in ('train', 'valid'):
        counts = first.manifest['splits'][split]['group_length_counts'].values()
        assert min(counts) > 0 and max(counts)-min(counts) <= 1
    assert fixtures.build_fixture_split(seed=43).manifest['content_sha256'] != first.manifest['content_sha256']


def test_576_72_allocates_every_group_and_length_cell_exactly():
    fixture = fixtures.build_fixture_split(train_size=576, valid_size=72)
    for split, count in [('train', 32), ('valid', 4)]:
        summary = fixture.manifest['splits'][split]
        assert set(summary['group_length_counts'].values()) == {count}
        assert summary['all_group_length_cells_present']
        assert summary['targets_with_residue_distance_gt32'] > 0
    assert fixture.manifest['cross_radius32_coverage'] == dict(possible=True, train=True, valid=True)
    assert not set(targets(fixture.train_records)) & set(targets(fixture.valid_records))


def test_every_conditioned_target_and_length_signature_is_derived_from_context():
    fixture = small(train_size=36, valid_size=18)
    for record in (*fixture.train_records, *fixture.valid_records):
        metadata = record.metadata
        group = metadata['diagnostic_group']
        context = {c.role: c.sequence for c in record.chains if c.role in metadata['condition_roles']}
        if not context:
            continue
        if group == 'tcr_pmhc':
            code = context['peptide'][:3]+context['mhc'][:3]
            markers = [context['peptide'][3], context['mhc'][3]]
        else:
            observed = next(iter(context.values()))
            code, markers = observed[:6], [observed[6]]
        assert set(markers) == {fixtures.ALPHABET[fixtures.LENGTH_BINS.index(metadata['length_bin'])]}
        for chain in record.chains:
            if chain.role in metadata['target_roles']:
                length = fixture.manifest['target_lengths'][metadata['length_bin']]
                motif = code+fixtures.ROLE_MARKERS[chain.role]
                expected = (motif*((length-2+len(motif)-1)//len(motif)))[:length-2]
                assert chain.sequence[1:-1] == expected
                assert len(chain.sequence) == length


def test_boundaries_length_strata_capacity_and_context_roles_are_preserved():
    fixture = small(train_size=18, valid_size=18)
    groups = set()
    lengths = {key: set() for key in fixtures.LENGTH_BINS}
    for record in (*fixture.train_records, *fixture.valid_records):
        groups.add(record.metadata['diagnostic_group'])
        assert record.metadata['synthetic_only'] is True
        assert record.metadata['target_scope'] == 'full'
        for chain in record.chains:
            assert set(chain.sequence) <= VALID_PROTEIN_CHARS
            if chain.role in record.metadata['target_roles']:
                assert chain.sequence[0] == 'C'
                expected_end = 'W' if chain.role in ('antibody_heavy', 'nanobody_vhh') else 'F'
                assert chain.sequence[-1] == expected_end
                assert len(chain.sequence)+1 <= 48
                lengths[record.metadata['length_bin']].add(len(chain.sequence))
    assert groups == set(fixtures.GROUPS)
    assert max(lengths['short']) < min(lengths['medium']) < min(lengths['long'])
    assert min(lengths['long'])-1 > 32
    assert any(set(r.chain_roles) == {'tcr_alpha', 'tcr_beta', 'peptide', 'mhc'} for r in fixture.train_records)


def test_small_capacity_reports_that_radius32_coverage_is_impossible():
    fixture = fixtures.build_fixture_split(train_size=18, valid_size=18,
        full_capacity=24, fragment_capacity=16, max_length=128)
    assert fixture.manifest['cross_radius32_coverage'] == dict(possible=False, train=False, valid=False)
    assert all(len(sequence)+1 <= 24 for sequence in targets(fixture.train_records))
    tiny = small(train_size=3, valid_size=2)
    assert not tiny.manifest['splits']['valid']['all_group_length_cells_present']


def test_full_mask_unconditional_ambiguity_is_real_and_documented():
    fixture = small(train_size=36, valid_size=18)
    rows_same_group = [r for r in fixture.train_records if r.metadata['diagnostic_group'] == 'unconditional_ab']
    a, b = rows_same_group[:2]
    assert a.sequences != b.sequences
    assert not a.metadata['conditional_signature_visible']
    assert a.metadata['unconditional_full_mask_identifiable'] is False
    collator = ImmuneCollator('A4', full_capacity=48, fragment_capacity=16, max_length=256)
    batch_a, batch_b = collator([a]), collator([b])
    for batch in (batch_a, batch_b):
        batch['input_ids'][batch['target_slot_mask'].bool()] = collator.tokenizer.mask_token_id
    features_a, features_b = ImmuneModel.feature_args(batch_a), ImmuneModel.feature_args(batch_b)
    assert all(torch.equal(features_a[name], features_b[name]) for name in features_a)
    assert any('not identifiable' in item for item in fixture.manifest['limitations'])


def test_all_strata_collate_and_a_cpu_model_has_finite_gradient():
    fixture = small(train_size=18, valid_size=18)
    collator = ImmuneCollator('A4', full_capacity=48, fragment_capacity=16, max_length=256)
    for records in (fixture.train_records, fixture.valid_records):
        batch = collator(list(records))
        assert batch['input_ids'].shape[1] <= 256
        supervised_terminal = batch['terminal_mask'] & batch['eligible_mask']
        assert int(supervised_terminal.sum()) == sum(len(r.metadata['target_roles']) for r in records)
        assert not (batch['eligible_mask'] & ~batch['target_slot_mask']).any()
    config = ImmuneConfig(d_model=32, n_layers=2, n_heads=4, mlp_hidden_size=64,
                         full_capacity=48, fragment_capacity=16, max_length=256)
    model = ImmuneModel(config)
    batch = collator(list(fixture.train_records[12:18]))
    result = model(**batch)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_manifest_roundtrip_writes_only_proof_and_is_idempotent(tmp_path):
    expected = fixtures.write_fixture_manifest(tmp_path, train_size=18, valid_size=18,
        full_capacity=48, fragment_capacity=16, max_length=256)
    assert {p.name for p in tmp_path.iterdir()} == {fixtures.MANIFEST_NAME}
    actual = fixtures.load_verified_fixture_manifest(tmp_path)
    assert rows(actual) == rows(expected) and actual.manifest == expected.manifest
    repeated = fixtures.write_fixture_manifest(tmp_path, **expected.manifest['parameters'])
    assert repeated.manifest == expected.manifest
    # An unrelated external records file is never ingested.
    (tmp_path/'records.jsonl').write_text('external arbitrary biological records must not be read')
    assert rows(fixtures.load_verified_fixture_manifest(tmp_path)) == rows(expected)
    with pytest.raises(ValueError, match='Existing fixture differs'):
        fixtures.write_fixture_manifest(tmp_path, seed=99, **{k: v for k, v in expected.manifest['parameters'].items() if k != 'seed'})


@pytest.mark.parametrize('defect', ['schema', 'kind', 'seed', 'source_hash', 'content_hash',
    'record_hash', 'target_hash', 'coverage', 'metadata_field', 'overlap', 'extra_records',
    'extra_parameter', 'typed_counter', 'manifest_hash', 'nan'])
def test_loader_rejects_every_modified_field_or_hash(tmp_path, defect):
    fixtures.write_fixture_manifest(tmp_path, train_size=18, valid_size=18)
    path = tmp_path/fixtures.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    if defect in ('schema', 'kind'):
        manifest[defect] = 'other'
    elif defect == 'seed':
        manifest['parameters']['seed'] += 1
    elif defect == 'source_hash':
        manifest['generator_sha256'] = '0'*64
    elif defect == 'content_hash':
        manifest['content_sha256'] = '0'*64
    elif defect == 'record_hash':
        manifest['splits']['train']['record_sha256'][0] = '0'*64
    elif defect == 'target_hash':
        manifest['splits']['valid']['target_sequence_sha256'][0] = '0'*64
    elif defect == 'coverage':
        manifest['cross_radius32_coverage']['train'] = False
    elif defect == 'metadata_field':
        manifest['metadata_fields']['group'] = 'untrusted'
    elif defect == 'overlap':
        manifest['exact_target_sequence_overlap'] = 1
    elif defect == 'extra_records':
        manifest['records'] = [{'sequence': 'CEXTERNALF'}]
    elif defect == 'extra_parameter':
        manifest['parameters']['records_path'] = '/external/records'
    elif defect == 'typed_counter':
        manifest['exact_target_sequence_overlap'] = False  # False == 0 is insufficient verification.
    elif defect == 'manifest_hash':
        manifest['manifest_sha256'] = '0'*64
    elif defect == 'nan':
        manifest['splits']['train']['target_length_min'] = float('nan')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Invalid synthetic fixture manifest'):
        fixtures.load_verified_fixture_manifest(tmp_path)


@pytest.mark.parametrize('kwargs', [dict(seed=-1), dict(seed=True), dict(train_size=0),
    dict(valid_size=0), dict(train_size=8192), dict(full_capacity=12),
    dict(fragment_capacity=1), dict(max_length=50)])
def test_invalid_sizes_and_impossible_topology_fail_before_writing(tmp_path, kwargs):
    with pytest.raises(ValueError):
        fixtures.write_fixture_manifest(tmp_path/'new', **kwargs)
    assert not (tmp_path/'new').exists()


def test_missing_or_malformed_manifest_rejected(tmp_path):
    with pytest.raises(ValueError, match='Invalid synthetic fixture manifest'):
        fixtures.load_verified_fixture_manifest(tmp_path)
    (tmp_path/fixtures.MANIFEST_NAME).write_text('{incomplete')
    with pytest.raises(ValueError, match='Invalid synthetic fixture manifest'):
        fixtures.load_verified_fixture_manifest(tmp_path)
