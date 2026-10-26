"""Activate pllm/protenix_abtcr; python -B -m pytest THIS_FILE -q."""
import json

import pytest

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.data.preprocessing.validators import SCHEMA_VERSION
from scripts.research.oas_pretraining_data import describe_selection, load_datasets


@pytest.fixture
def prepared(tmp_path):
    splits = {}
    for split, count in (('train', 3), ('valid', 5)):
        # OAS rows can have a different provenance name from their shard group.
        row = BioSeqRecord([BioSeqChain('CAVF', 'antibody_heavy')],
                           'antibody', 'oas_paired', split).to_dict()
        path = tmp_path / f'{split}.jsonl'
        path.write_text((json.dumps({**row, 'schema_version': SCHEMA_VERSION}) + '\n') * count)
        splits[split] = {'records': count + 9, 'shards': [
            {'path': path.name, 'source': 'oas', 'records': count},
            # This file deliberately does not exist: it must never be opened.
            {'path': 'excluded-tcr.jsonl', 'source': 'ots', 'records': 9}]}
    (tmp_path / 'dataset_manifest.json').write_text(json.dumps({
        'schema_version': SCHEMA_VERSION, 'format': 'jsonl', 'splits': splits}))
    return tmp_path


def test_only_oas_is_loaded_in_both_splits(prepared):
    selection = describe_selection(prepared)
    train, valid, indices = load_datasets(prepared, 2, selection)
    assert len(train) == 3 and set(valid) == {'oas'} and len(valid['oas']) == 2
    assert train[0].source == 'oas_paired' and train[0].source_group == 'oas'
    assert all(row.source_group == 'oas' for row in valid['oas'])
    assert indices == load_datasets(prepared, 2, selection)[2]


def test_changed_selection_and_wrong_row_counts_fail(prepared):
    selection = describe_selection(prepared)
    wrong = {**selection, 'train_sources': ['oas', 'ots']}
    with pytest.raises(ValueError, match='selection differs'):
        load_datasets(prepared, 2, wrong)
    with (prepared / 'train.jsonl').open('a') as stream:
        stream.write((prepared / 'train.jsonl').read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='count differs'):
        load_datasets(prepared, 2, selection)


def test_missing_oas_does_not_fall_back_to_other_sources(prepared):
    path = prepared / 'dataset_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['splits']['valid']['shards'] = manifest['splits']['valid']['shards'][1:]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='No nonempty OAS'):
        describe_selection(prepared)
