"""Pure integrity tests for frozen validation subsets using temporary fake data.

After activating pllm:
  USE_TF=0 python -m pytest scripts/tests/immune_llada/test_validation_selection.py -q
No existing corpus, training run, model, certificate, or benchmark is modified.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from scripts.architecture_prototypes import validation_selection as selection


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + '\n')


def record(source, split, number):
    return dict(schema_version='immune_llada.semantic.v1', chains=['context', 'x'*(number+1), 'z'*(number+2)],
        chain_roles=['antigen', 'tcr_alpha', 'tcr_beta'], source=f'provenance/{source}', split=split,
        task_type='tcr', metadata={'v7_family_group': f'{split}/{source}/{number//2}', 'preserve_unicode': '记录'},
        labels={'label': number}, identifiers={'id': str(number)})


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, rows, blank=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(('\n' if blank else '') + ''.join(json.dumps(row, ensure_ascii=False) + ('\n\n' if blank else '\n') for row in rows))


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path/'prepared'
    items = [('train/a.jsonl', 'a', 'train', [0, 1]),
             ('train/b.jsonl', 'b', 'train', [0]),
             ('train/absent.jsonl', 'absent', 'train', [0]),
             ('valid/a-first.jsonl', 'a', 'valid', [0, 1, 2]),
             ('valid/b-only.jsonl', 'b', 'valid', [0, 1]),
             ('valid/a-second.jsonl', 'a', 'valid', [3, 4, 5, 6, 7])]
    manifest = dict(schema_version='immune_llada.semantic.v1', format='jsonl', dataset_version='invented_test',
        sources=['a', 'b', 'absent'], splits={split: dict(records=0, shards=[]) for split in ('train', 'valid')})
    for relative, source, split, numbers in items:
        path = directory/relative
        write_rows(path, [record(source, split, n) for n in numbers])
        block = manifest['splits'][split]
        block['shards'].append(dict(path=relative, source=source, split=split, records=len(numbers), sha256=selection._sha(path)))
        block['records'] += len(numbers)
    dump(directory/selection.MANIFEST_NAME, manifest)
    f = SimpleNamespace(directory=directory, manifest=manifest, output=tmp_path/'selection.json')
    def update_shard(relative):
        for block in manifest['splits'].values():
            for shard in block['shards']:
                if shard['path'] == relative:
                    shard.update(sha256=selection._sha(directory/relative), records=len(read_rows(directory/relative)))
            block['records'] = sum(shard['records'] for shard in block['shards'])
        dump(directory/selection.MANIFEST_NAME, manifest)
    f.update_shard = update_shard
    return f


def build(dataset, **kwargs):
    return selection.build_validation_selection(dataset.directory, max_records_per_source=3, **kwargs)


def save(dataset, **kwargs):
    return selection.write_validation_selection(dataset.output, dataset.directory, max_records_per_source=3, **kwargs)


def recompute_digest(payload):
    payload['selection_sha256'] = selection._digest({k: v for k, v in payload.items() if k != 'selection_sha256'})


def test_deterministic_exact_source_indices_and_no_global_rng_consumption(dataset):
    before = random.getstate()
    first, second = build(dataset), build(dataset)
    assert random.getstate() == before
    assert first == second
    assert first['data_dir'] == str(dataset.directory.resolve())
    assert first['sources']['a']['source_local_indices'] == random.Random(42).sample(range(8), 3)
    assert first['sources']['b']['source_local_indices'] == random.Random(42).sample(range(2), 2)
    assert first['sources']['absent']['source_local_indices'] == []
    assert first['training_certification'] is False
    assert first['split_reassignment_applied'] is False
    assert first['family_splitting_applied'] is False


def test_source_local_locations_match_real_prepared_loader_and_raw_bytes(dataset):
    from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset
    payload = build(dataset)
    for source in ('a', 'b'):
        loader = PreparedImmuneDataset(dataset.directory, split='valid', source=source)
        for selected in payload['sources'][source]['rows']:
            path = dataset.directory/selected['shard']
            with path.open('rb') as handle:
                handle.seek(selected['byte_offset'])
                raw = handle.readline()
            row = json.loads(raw)
            assert selection._digest(row) == selected['row_content_sha256']
            assert hashlib.sha256(raw).hexdigest() == selected['row_bytes_sha256']
            assert read_rows(path)[selected['shard_record_index']] == row
            actual = loader[selected['source_local_index']]
            assert actual.identifiers == row['identifiers']
            assert actual.metadata == row['metadata']
            assert [chain.sequence for chain in actual.chains] == row['chains']
    assert payload['sources']['a']['rows'][-1]['shard'] == 'valid/a-second.jsonl'


def test_length_stats_are_plain_string_lengths_and_family_counts(dataset):
    payload = build(dataset)
    stats = payload['sources']['a']['available']
    assert stats['records'] == 8 and stats['known_family_count'] == 4
    assert stats['target_chain_lengths'] == dict(count=16, min=1, max=9, mean=5., p50=5, p95=9)
    assert stats['target_total_lengths_per_record'] == dict(count=8, min=3, max=17, mean=10., p50=9, p95=17)
    assert stats['target_chain_lengths_by_role']['tcr_alpha'] == dict(count=8, min=1, max=8, mean=4.5, p50=4, p95=8)
    assert stats['target_chain_lengths_by_role']['antibody_heavy']['count'] == 0
    assert payload['sources']['a']['selected']['known_family_count'] == 2
    assert 'selected_families_below_threshold' in payload['sources']['a']['warnings']
    assert 'available_records_below_threshold' in payload['sources']['absent']['warnings']
    assert payload['sources']['absent']['selected']['target_chain_lengths']['mean'] is None


def test_only_json_written_and_originals_unchanged_idempotent_verification(dataset):
    originals = {p: p.read_bytes() for p in dataset.directory.rglob('*') if p.is_file()}
    expected = save(dataset)
    first_bytes = dataset.output.read_bytes()
    assert save(dataset) == expected
    assert dataset.output.read_bytes() == first_bytes
    actual = selection.load_verified_validation_selection(dataset.output,
        expected_data_dir=dataset.directory, expected_manifest_sha256=expected['manifest_sha256'],
        expected_selection_sha256=expected['selection_sha256'])
    assert actual == expected
    assert {p: p.read_bytes() for p in originals} == originals
    assert not list(dataset.output.parent.glob('.*.tmp'))
    assert {p.name for p in dataset.output.parent.iterdir()} == {'prepared', 'selection.json'}


def test_manifest_change_rejected(dataset):
    save(dataset)
    path = dataset.directory/selection.MANIFEST_NAME
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='manifest SHA256 changed'):
        selection.load_verified_validation_selection(dataset.output)


@pytest.mark.parametrize('changed', ['selected', 'unselected', 'whitespace'])
def test_any_valid_row_or_shard_byte_change_rejected(dataset, changed):
    payload = save(dataset)
    selected = set(payload['sources']['a']['source_local_indices'])
    target = next(i for i in range(8) if (i in selected) == (changed == 'selected'))
    relative = 'valid/a-first.jsonl' if target < 3 else 'valid/a-second.jsonl'
    path = dataset.directory/relative
    if changed == 'whitespace':
        path.write_text(path.read_text() + '\n')
    else:
        value = read_rows(path)
        value[target if target < 3 else target-3]['labels']['label'] = 999
        write_rows(path, value)
    with pytest.raises(ValueError, match='shard SHA256 mismatch'):
        selection.load_verified_validation_selection(dataset.output)


def test_manifest_rebinding_changed_shard_still_rejected(dataset):
    save(dataset)
    path = dataset.directory/'valid/b-only.jsonl'
    path.write_text(path.read_text() + '\n')
    dataset.update_shard('valid/b-only.jsonl')
    with pytest.raises(ValueError, match='manifest SHA256 changed'):
        selection.load_verified_validation_selection(dataset.output)


@pytest.mark.parametrize('field', ['index', 'location', 'byte_offset', 'row_sha', 'family', 'length', 'stats', 'warning', 'extra'])
def test_forged_selection_rejected_even_with_recomputed_digest(dataset, field):
    payload = save(dataset)
    row = payload['sources']['a']['rows'][0]
    if field == 'index':
        payload['sources']['a']['source_local_indices'][0] = 7
    elif field == 'location':
        row['shard_record_index'] = 999
    elif field == 'byte_offset':
        row['byte_offset'] += 1
    elif field == 'row_sha':
        row['row_content_sha256'] = 'a'*64
    elif field == 'family':
        row['family_group'] = 'invented_other_family'
    elif field == 'length':
        row['target_chain_lengths'][0]['length'] += 1
    elif field == 'stats':
        payload['sources']['a']['selected']['records'] += 1
    elif field == 'warning':
        payload['warnings'] = []
    else:
        payload['unrecognized_extra_field'] = True
    recompute_digest(payload)
    dump(dataset.output, payload)
    with pytest.raises(ValueError, match='deterministic regeneration'):
        selection.load_verified_validation_selection(dataset.output)


@pytest.mark.parametrize('expected', ['directory', 'manifest', 'selection'])
def test_optional_external_pins_reject_wrong_selection(dataset, expected):
    save(dataset)
    kwargs = {'directory': {'expected_data_dir': dataset.output.parent/'different'},
              'manifest': {'expected_manifest_sha256': 'a'*64},
              'selection': {'expected_selection_sha256': 'a'*64}}[expected]
    with pytest.raises(ValueError, match='another data|expected'):
        selection.load_verified_validation_selection(dataset.output, **kwargs)


def test_missing_family_and_empty_target_roles_warn_without_imputation(dataset):
    path = dataset.directory/'valid/b-only.jsonl'
    value = read_rows(path)
    value[0]['metadata'].pop('v7_family_group')
    value[1]['chains'], value[1]['chain_roles'] = ['literal_context'], ['antigen']
    write_rows(path, value)
    dataset.update_shard('valid/b-only.jsonl')
    payload = build(dataset)
    source = payload['sources']['b']
    assert source['available']['missing_family_records'] == 1
    assert source['selected']['no_target_records'] == 1
    assert 'selected_family_metadata_missing' in source['warnings']
    assert 'selected_declared_target_roles_absent' in source['warnings']
    assert source['rows'][0]['family_group'] is None


def test_literal_string_length_includes_all_stored_characters(dataset):
    path = dataset.directory/'valid/b-only.jsonl'
    value = read_rows(path)
    value[0]['chains'][1] = ' c?é '
    value[0]['chain_metadata'] = {'1': {'synthetic_residue_mask': [1, 1, 1, 1, 1]}}
    write_rows(path, value)
    dataset.update_shard('valid/b-only.jsonl')
    payload = build(dataset)
    lengths = payload['sources']['b']['rows'][0]['target_chain_lengths']
    assert lengths[0]['length'] == len(' c?é ') == 5


def test_training_shard_bytes_outside_explicit_validation_scope(dataset, monkeypatch):
    expected = save(dataset)
    train = dataset.directory/'train/a.jsonl'
    train.write_text('this change must be handled by the existing training gate')
    assert selection.load_verified_validation_selection(dataset.output) == expected
    assert 'train shard bytes not checked' in expected['verification_scope']


def test_distinct_seed_is_a_distinct_selection_and_does_not_overwrite(dataset):
    payload = save(dataset)
    with pytest.raises(ValueError, match='different selection'):
        save(dataset, seed=7)
    assert selection.load_verified_validation_selection(dataset.output) == payload


@pytest.mark.parametrize('relative', ['../selection.json', '/tmp/outside.jsonl', 'valid/../../outside.jsonl'])
def test_unsafe_manifest_paths_rejected(dataset, relative):
    dataset.manifest['splits']['valid']['shards'][0]['path'] = relative
    dump(dataset.directory/selection.MANIFEST_NAME, dataset.manifest)
    with pytest.raises(ValueError, match='path'):
        build(dataset)


@pytest.mark.parametrize('config', [{'max_records_per_source': 0}, {'max_records_per_source': 257}, {'seed': -1},
                                     {'seed': True}, {'min_source_families': -1}, {'min_source_records': 1.5}])
def test_bad_config_rejected(dataset, config):
    with pytest.raises(ValueError, match='Selection arguments'):
        selection.build_validation_selection(dataset.directory, **config)


def test_refuses_selection_inside_prepared_or_symlink_to_prepared(dataset):
    with pytest.raises(ValueError, match='outside'):
        selection.write_validation_selection(dataset.directory/'selection.json', dataset.directory)
    alias = dataset.output.parent/'alias'
    alias.symlink_to(dataset.directory, target_is_directory=True)
    with pytest.raises(ValueError, match='outside'):
        selection.write_validation_selection(alias/'selection.json', dataset.directory)


def test_unchanged_valid_symlink_is_supported_and_retarget_detected(dataset):
    path = dataset.directory/'valid/b-only.jsonl'
    first = dataset.output.parent/'physical.jsonl'
    second = dataset.output.parent/'same_bytes.jsonl'
    first.write_bytes(path.read_bytes())
    second.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(first)
    payload = save(dataset)
    assert selection.load_verified_validation_selection(dataset.output) == payload
    path.unlink()
    path.symlink_to(second)
    with pytest.raises(ValueError, match='deterministic regeneration'):
        selection.load_verified_validation_selection(dataset.output)


def test_relative_data_dir_is_canonicalized(tmp_path, dataset, monkeypatch):
    monkeypatch.chdir(dataset.directory.parent)
    value = selection.build_validation_selection('prepared')
    assert value['data_dir'] == str(dataset.directory.resolve())


def test_declared_target_roles_match_existing_layout():
    from dllm.pipelines.immune_llada.v3.layout import RECEPTORS
    assert set(selection.TARGET_ROLES) == set(RECEPTORS)
