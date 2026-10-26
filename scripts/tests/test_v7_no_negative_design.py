"""Activate pllm/protenix_abtcr; run python -m pytest -q FILE.

Regression tests for retaining unlabeled repertoire without relabeling or
changing V5's synthetic scaffold during negative-sample filtering.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data import export_v7_no_negative_design as export


@pytest.mark.parametrize('labels,excluded', [
    ({'relation': 'binding'}, False),
    ({'relation': 'nonbinding'}, True),
    ({'relation': 'unknown'}, False),
    ({}, False),
])
def test_only_explicit_negative_is_excluded(labels, excluded):
    row = {'labels': labels}
    before = deepcopy(row)
    assert export.excluded_by_label(row) is excluded
    assert row == before


def test_unrecognized_label_fails_instead_of_silently_dropping_repertoire():
    with pytest.raises(AssertionError):
        export.excluded_by_label({'labels': {'relation': 'unexpected'}})


def test_saved_candidate_selection_preserves_unlabeled_scaffold(tmp_path, monkeypatch):
    source, work = tmp_path / 'source', tmp_path / 'work'
    source.mkdir()
    monkeypatch.setattr(export, 'SOURCE', source)
    monkeypatch.setattr(export, 'RUN', work)
    monkeypatch.setattr(export, 'upstream_lookup', lambda: {})
    scaffold = dict(chains=['XXX', 'XXADDDXX'], chain_roles=['tcr_alpha', 'tcr_beta'],
                    regions={'1': {'CDR3': 'DDD'}},
                    chain_metadata={'0': {'synthetic_residue_mask': [1, 1, 1]},
                                    '1': {'synthetic_residue_mask': [1, 1, 0, 0, 0, 0, 1, 1]}})
    rows = []
    for relation in ('binding', 'nonbinding', 'unknown', None):
        row = deepcopy(scaffold)
        if relation is not None:
            row['labels'] = {'relation': relation}
        rows.append(row)
    raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    (source / 'sample.jsonl').write_bytes(raw)
    shard = dict(path='sample.jsonl', records=4, split='train', source='test',
                 source_sha256=export.sha(source / 'sample.jsonl'))
    receipt = export.candidate_shard(shard)
    assert receipt['candidates'] == 3
    assert np.load(receipt['state_path']).tolist() == [0, 1, 0, 0]
    candidates = export.read_pickle(Path(receipt['key_path']))['candidates']
    assert [i for i, _ in candidates] == [0, 2, 3]
    assert all(keys['cdr3b'] == 'DDD' and not keys['paired_tcr'] for _, keys in candidates)
    assert (source / 'sample.jsonl').read_bytes() == raw
    assert export.candidate_shard(shard) == receipt
