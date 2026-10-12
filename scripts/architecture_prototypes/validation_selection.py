"""Freeze and verify source-local validation indices without changing a dataset.

Activate pllm, then run from the repository root:
  python scripts/architecture_prototypes/validation_selection.py create --data-dir /prepared/version --selection /outside/prepared/selection.json
  python scripts/architecture_prototypes/validation_selection.py verify --selection /outside/prepared/selection.json

Only a small selection JSON is written. Every valid shard, including unselected
rows, is hashed on creation and verification. Train shard bytes remain the
responsibility of the existing training-data gate. This is an integrity manifest,
not an audit certificate. Lengths are literal len(serialized_chain_string), with
no normalization, biological annotation, matching, or split reassignment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import tempfile

SCHEMA = 'immune.validation_selection.v1'
# Declared receptor roles from v3/layout.py; no model/layout import is required.
TARGET_ROLES = ('antibody_heavy', 'antibody_light', 'nanobody_vhh', 'tcr_alpha', 'tcr_beta')
MANIFEST_NAME = 'dataset_manifest.json'
CONFIG_KEYS = {'seed', 'max_records_per_source', 'min_source_records', 'min_source_families'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _is_count(value):
    return type(value) is int and value >= 0


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _relative(value):
    _require(isinstance(value, str) and value, 'Missing shard path')
    path = PurePosixPath(value)
    _require(not path.is_absolute() and '..' not in path.parts and str(path) == value,
             'Unsafe or noncanonical shard path')
    _require(path.suffix == '.jsonl', 'Expected JSONL shard path')
    return value


def _shards_and_sources(manifest):
    _require(manifest.get('schema_version') == 'immune_llada.semantic.v1' and manifest.get('format') == 'jsonl',
             'Unsupported prepared manifest schema')
    splits = manifest.get('splits')
    _require(isinstance(splits, dict) and 'valid' in splits, 'Missing valid split')
    declared = manifest.get('sources', [])
    _require(isinstance(declared, list) and all(isinstance(s, str) and s for s in declared), 'Malformed manifest sources')
    sources, seen, valid = set(declared), set(), []
    for split, block in splits.items():
        _require(isinstance(block, dict) and _is_count(block.get('records')) and isinstance(block.get('shards'), list),
                 'Malformed split metadata')
        count = 0
        for shard in block['shards']:
            relative = _relative(shard['path'])
            _require(relative not in seen and PurePosixPath(relative).parts[0] == split, 'Duplicate/wrong-split shard')
            _require(shard.get('split') == split and _is_count(shard.get('records')), 'Malformed shard split/count')
            source = shard.get('source')
            _require(isinstance(source, str) and source and _valid_sha(shard.get('sha256')), 'Missing shard source or SHA256')
            sources.add(source)
            seen.add(relative)
            count += shard['records']
            if split == 'valid':
                valid.append(shard)
        _require(count == block['records'], 'Manifest split count mismatch')
    return valid, sorted(sources)


def _summary(histogram):
    count = sum(histogram.values())
    if not count:
        return dict(count=0, min=None, max=None, mean=None, p50=None, p95=None)
    def quantile(fraction):
        threshold, cumulative = max(1, math.ceil(count * fraction)), 0
        for length in sorted(histogram):
            cumulative += histogram[length]
            if cumulative >= threshold:
                return length
    return dict(count=count, min=min(histogram), max=max(histogram),
        mean=sum(length*n for length, n in histogram.items())/count,
        p50=quantile(.5), p95=quantile(.95))


class _Stats:
    def __init__(self):
        self.records = self.missing_family_records = 0
        self.families = set()
        self.lengths, self.record_lengths = Counter(), Counter()
        self.by_role = {role: Counter() for role in TARGET_ROLES}
        self.no_target_records = 0

    def add(self, family, lengths):
        self.records += 1
        if family is None:
            self.missing_family_records += 1
        else:
            self.families.add(family)
        total = 0
        for item in lengths:
            self.lengths[item['length']] += 1
            self.by_role[item['role']][item['length']] += 1
            total += item['length']
        self.record_lengths[total] += 1
        self.no_target_records += int(not lengths)

    def value(self):
        return dict(records=self.records, known_family_count=len(self.families),
            missing_family_records=self.missing_family_records, no_target_records=self.no_target_records,
            target_chain_lengths=_summary(self.lengths), target_total_lengths_per_record=_summary(self.record_lengths),
            target_chain_lengths_by_role={role: _summary(self.by_role[role]) for role in TARGET_ROLES})


def _row_metadata(row):
    _require(isinstance(row, dict) and isinstance(row.get('chains'), list) and isinstance(row.get('chain_roles'), list),
             'Malformed prepared row chains')
    chains, roles = row['chains'], row['chain_roles']
    _require(chains and len(chains) == len(roles) and all(isinstance(s, str) for s in chains+roles),
             'Malformed prepared chain strings/roles')
    _require(row.get('split') in (None, 'valid'), 'Row is not in the declared valid split')
    metadata = row.get('metadata', {})
    _require(isinstance(metadata, dict), 'Malformed row metadata')
    family = metadata.get('v7_family_group')
    _require(family is None or isinstance(family, str), 'Malformed family group')
    family = family or None
    lengths = [dict(chain_index=i, role=role, length=len(sequence))
               for i, (sequence, role) in enumerate(zip(chains, roles)) if role in TARGET_ROLES]
    return family, lengths


def build_validation_selection(data_dir, *, seed=42, max_records_per_source=256,
                               min_source_records=100, min_source_families=20):
    """Reproduce PreparedImmuneDataset(source=...) nonblank source-local indices."""
    config = dict(seed=seed, max_records_per_source=max_records_per_source,
                  min_source_records=min_source_records, min_source_families=min_source_families)
    _require(all(_is_count(value) for value in config.values()) and 1 <= max_records_per_source <= 256,
             'Selection arguments must be nonnegative integers, with a record cap in 1..256')
    data_dir = Path(data_dir).resolve()
    manifest_path = data_dir / MANIFEST_NAME
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    valid_shards, source_names = _shards_and_sources(manifest)
    counts = Counter()
    for shard in valid_shards:
        counts[shard['source']] += shard['records']
    indices = {source: random.Random(seed).sample(range(counts[source]), min(counts[source], max_records_per_source))
               for source in source_names}
    wanted = {source: set(values) for source, values in indices.items()}
    selected = {source: {} for source in source_names}
    source_offsets = Counter()
    available_stats, selected_stats = {s: _Stats() for s in source_names}, {s: _Stats() for s in source_names}
    bindings = []
    for shard in valid_shards:
        source, relative = shard['source'], shard['path']
        path = data_dir / relative
        resolved = path.resolve()
        total, digest = 0, hashlib.sha256()
        with path.open('rb') as handle:
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                digest.update(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw)
                family, lengths = _row_metadata(row)
                available_stats[source].add(family, lengths)
                source_index = source_offsets[source] + total
                if source_index in wanted[source]:
                    selected_stats[source].add(family, lengths)
                    selected[source][source_index] = dict(source_local_index=source_index,
                        shard=relative, shard_record_index=total, byte_offset=offset,
                        row_content_sha256=_digest(row), row_bytes_sha256=hashlib.sha256(raw).hexdigest(),
                        family_group=family, target_chain_lengths=lengths)
                total += 1
        _require(total == shard['records'], 'Valid shard record count changed: ' + relative)
        _require(digest.hexdigest() == shard['sha256'], 'Valid shard SHA256 mismatch: ' + relative)
        _require(path.resolve() == resolved, 'Valid shard symlink target changed while reading')
        source_offsets[source] += total
        bindings.append(dict(path=relative, resolved_path=str(resolved), source=source, split='valid',
                             records=total, sha256=digest.hexdigest()))
    _require(_sha(manifest_path) == manifest_sha, 'Manifest changed while selecting')
    sources, warnings = {}, []
    for source in source_names:
        _require(source_offsets[source] == counts[source] and set(selected[source]) == wanted[source],
                 'Selection/source-local indexing mismatch')
        available, chosen = available_stats[source].value(), selected_stats[source].value()
        source_warnings = []
        for name, stats in (('available', available), ('selected', chosen)):
            if stats['records'] < min_source_records:
                source_warnings.append(name + '_records_below_threshold')
            if stats['known_family_count'] < min_source_families:
                source_warnings.append(name + '_families_below_threshold')
            if stats['missing_family_records']:
                source_warnings.append(name + '_family_metadata_missing')
            if stats['no_target_records']:
                source_warnings.append(name + '_declared_target_roles_absent')
        sources[source] = dict(source_local_indices=indices[source],
            rows=[selected[source][index] for index in indices[source]], available=available, selected=chosen,
            warnings=source_warnings)
        warnings.extend(dict(source=source, code=code) for code in source_warnings)
    result = dict(schema=SCHEMA, kind='validation_subset_integrity_only', training_certification=False,
        data_dir=str(data_dir), manifest_sha256=manifest_sha, implementation_sha256=_sha(Path(__file__)),
        dataset_version=manifest.get('dataset_version'), split='valid', config=config,
        sampling='independent random.Random(seed).sample(range(source_records), min(cap, source_records)) per source',
        source_indexing='zero-based nonblank records; source-filtered manifest shard order',
        row_content_hash='SHA256 of sorted compact UTF-8 JSON, no normalization; raw-line SHA also retained',
        verification_scope='entire manifest and all valid shard bytes, including unselected rows; train shard bytes not checked',
        target_roles=list(TARGET_ROLES), length_definition='len(serialized chain string); no residue normalization or biological annotation',
        quantile_method='nearest_rank', split_reassignment_applied=False, family_splitting_applied=False,
        thresholds_are_advisory=True, valid_shards=bindings, sources=sources, warnings=warnings)
    result['selection_sha256'] = _digest(result)
    return result


def _check_envelope(payload):
    _require(isinstance(payload, dict) and payload.get('schema') == SCHEMA and
             payload.get('kind') == 'validation_subset_integrity_only', 'Unsupported selection schema/kind')
    _require(payload.get('training_certification') is False, 'Selection cannot be a training certificate')
    _require(isinstance(payload.get('config'), dict) and set(payload['config']) == CONFIG_KEYS, 'Malformed selection config')
    _require(isinstance(payload.get('data_dir'), str) and Path(payload['data_dir']).is_absolute() and
             str(Path(payload['data_dir']).resolve()) == payload['data_dir'], 'Selection data_dir must be canonical and absolute')
    without_digest = {key: value for key, value in payload.items() if key != 'selection_sha256'}
    _require(payload.get('selection_sha256') == _digest(without_digest), 'Selection content SHA256 mismatch')


def load_verified_validation_selection(path, *, expected_data_dir=None, expected_manifest_sha256=None,
                                       expected_selection_sha256=None):
    """Regenerate and compare every field; hashes in JSON alone are not trusted."""
    payload = _read(path)
    _check_envelope(payload)
    if expected_data_dir is not None:
        _require(payload['data_dir'] == str(Path(expected_data_dir).resolve()), 'Selection points to another data directory')
    if expected_manifest_sha256 is not None:
        _require(payload['manifest_sha256'] == expected_manifest_sha256, 'Selection manifest differs from expected SHA256')
    if expected_selection_sha256 is not None:
        _require(payload['selection_sha256'] == expected_selection_sha256, 'Selection differs from expected selection SHA256')
    current_manifest = Path(payload['data_dir']) / MANIFEST_NAME
    _require(_sha(current_manifest) == payload['manifest_sha256'], 'Dataset manifest SHA256 changed')
    regenerated = build_validation_selection(payload['data_dir'], **payload['config'])
    _require(payload == regenerated, 'Selection contents/locations/bindings/stats differ from deterministic regeneration')
    return regenerated


def write_validation_selection(path, data_dir, **kwargs):
    """Publish one JSON outside prepared data; identical reruns are idempotent."""
    path, data_dir = Path(path).absolute(), Path(data_dir).resolve()
    _require(not path.resolve().is_relative_to(data_dir), 'Selection output must be outside the prepared dataset')
    payload = build_validation_selection(data_dir, **kwargs)
    if path.exists() or path.is_symlink():
        _require(path.is_file() and not path.is_symlink(), 'Selection destination is not a regular file')
        current = load_verified_validation_selection(path, expected_data_dir=data_dir)
        _require(current == payload, 'Refusing to replace a different selection')
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(_canonical(payload) + b'\n')
        try:
            os.link(name, path)
        except FileExistsError:
            _require(path.is_file() and not path.is_symlink() and _read(path) == payload,
                     'A different selection appeared while publishing')
    finally:
        Path(name).unlink()
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('create', 'verify'))
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-records-per-source', type=int, default=256)
    parser.add_argument('--min-source-records', type=int, default=100)
    parser.add_argument('--min-source-families', type=int, default=20)
    args = parser.parse_args()
    if args.mode == 'create':
        if args.data_dir is None:
            parser.error('create requires --data-dir')
        result = write_validation_selection(args.selection, args.data_dir, seed=args.seed,
            max_records_per_source=args.max_records_per_source, min_source_records=args.min_source_records,
            min_source_families=args.min_source_families)
    else:
        result = load_verified_validation_selection(args.selection, expected_data_dir=args.data_dir)
    print(json.dumps(dict(selection_sha256=result['selection_sha256'], training_certification=False,
        records={s: value['selected']['records'] for s, value in result['sources'].items()}, warnings=result['warnings']),
        ensure_ascii=False))


if __name__ == '__main__':
    main()
