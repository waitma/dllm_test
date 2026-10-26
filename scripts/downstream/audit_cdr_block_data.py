"""Activate pllm; python -B scripts/downstream/audit_cdr_block_data.py --out ABS.

Independently verify frozen OAS rows, source coordinates, prior/selection exclusion,
components and checkpoint-bound training hash receipts without rescanning train.
No model is loaded and no prediction scores are read.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def normalized(value):
    return value.replace('J', 'L')


def heavy_keys(record):
    heavy, h3, clusters = set(), set(), set()
    for index, role in enumerate(record['chain_roles']):
        if role == 'antibody_heavy':
            heavy.add(normalized(record['chains'][index]))
            regions = record.get('regions', {}).get(str(index), {})
            metadata = record.get('chain_metadata', {}).get(str(index), {})
            if regions.get('CDR3'):
                h3.add(normalized(regions['CDR3']))
            if metadata.get('H_cluster_id'):
                clusters.add(metadata['H_cluster_id'])
    if heavy and record.get('identifiers', {}).get('heavy_h3_core'):
        h3.add(normalized(record['identifiers']['heavy_h3_core']))
    return heavy, h3, clusters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    plan, rows = read(out / 'data_plan.json'), read(out / 'rows.json')
    assert sha(out / 'rows.json') == plan['rows_sha256']
    assert sha(out / 'prepare_source.py') == plan['source_sha256']
    for path, expected in plan['source_data_sha256'].items():
        assert sha(path) == expected, path
    provenance = {k: read(Path(v) / 'run_provenance.json')
                  for k, v in plan['checkpoints'].items()}
    first = next(iter(provenance.values()))
    data = Path(first['data_dir'])
    manifest = read(data / 'dataset_manifest.json')
    for value in provenance.values():
        assert value['data_dir'] == str(data)
        assert value['parent_metadata']['dataset_manifest.json'] == sha(data / 'dataset_manifest.json')
        assert value['train_records'] == manifest['splits']['train']['records']
    # Bind the already recorded scan to exactly the immutable training bytes.
    binding_path = Path(first['source_root']).parent / 'data_binding.json'
    binding = read(binding_path)
    full_path = Path(binding['audit']['path'])
    assert sha(full_path) == binding['audit']['sha256']
    full = read(full_path)
    assert full['status'] == 'passed' and full['data_dir'] == str(data)
    for value in provenance.values():
        receipt = value['audit_execution']
        assert receipt['audit_sha256'] == sha(full_path)
        assert receipt['data_binding_sha256'] == sha(binding_path)
        assert receipt['status'] == 'passed' and receipt['content_hashes_verified']
    original_shards = {r['path']: r for r in full['shard_results']}
    train_paths = {str(data / r['path']) for r in manifest['splits']['train']['shards']}
    assert set(plan['train_shard_sha256']) == train_paths
    for path, digest in plan['train_shard_sha256'].items():
        assert original_shards[path]['sha256'] == digest, path
    assert sum(original_shards[p]['records'] for p in train_paths) == plan['train_scan']['all_rows']

    banned = [set(), set(), set()]
    previous_rows = 0
    for path in plan['source_data_sha256']:
        if path.endswith('/rows.json'):
            previous = read(path)
            previous_rows += len(previous)
            for row in previous:
                banned[0].add(normalized(row['gold_h']))
                if row['mode'] == 'cdrh3':
                    banned[1].add(normalized(''.join(row['gold_h'][p] for p in row['mask_h'])))
    selection_counts = {}
    source_records = {}
    for source in first['validation_indices']:
        selected = set().union(*(set(p['validation_indices'][source]) for p in provenance.values()))
        offset = 0
        paths = [data / r['path'] for r in manifest['splits']['valid']['shards'] if r['source'] == source]
        assert paths == sorted(paths), 'Preparation assumed lexicographic per-source shard order'
        for path in paths:
            values = [json.loads(line) for line in path.open()]
            for index, record in enumerate(values):
                if offset + index in selected:
                    for destination, valueset in zip(banned, heavy_keys(record)):
                        destination.update(valueset)
            if source == 'oas':
                source_records[str(path)] = values
            offset += len(values)
        assert selected and max(selected) < offset
        selection_counts[source] = len(selected)

    counts = Counter(row['split'] + ':' + row['mode'] for row in rows)
    assert dict(counts) == plan['counts']
    assert len({r['id'] for r in rows}) == len(rows)
    by_pair = defaultdict(list)
    h3_by_key = {}
    for row in rows:
        raw = source_records[row['source']][row['row']]
        assert raw['chain_roles'] == ['antibody_heavy', 'antibody_light']
        heavy, light = map(normalized, raw['chains'])
        assert heavy == row['heavy'] == row['gold_h']
        assert light == row['light'] == row['gold_l']
        assert len(heavy) <= 148 and len(light) <= 126
        assert set(heavy + light) <= set('ACDEFGHIKLMNPQRSTVWY')
        assert row['dataset'] == 'oas_new' and row['fold'] == 0 and not row['mask_l']
        labels = raw['chain_metadata']['0']['h_region_labels']
        assert len(labels) == len(heavy)
        assert row['mask_h'] == [i for i, label in enumerate(labels) if label == row['mode'][-1]]
        cdr = ''.join(heavy[i] for i in row['mask_h'])
        assert cdr == normalized(raw['regions']['0']['CDR' + row['mode'][-1]])
        parts = [raw['regions']['0'][region] for region in ('FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4')]
        assert normalized(''.join(parts)) == heavy
        assert row['h_cluster'] == raw['chain_metadata']['0']['H_cluster_id']
        assert row['key'] == hashlib.sha256((heavy + '|' + light).encode()).hexdigest()
        assert row['id'] == 'oas_new:' + row['mode'] + ':' + row['key']
        assert all(not (observed & excluded) for observed, excluded in zip(heavy_keys(raw), banned))
        by_pair[row['key']].append(row)
        h3_by_key[row['key']] = normalized(raw['regions']['0']['CDR3'])
    for group in by_pair.values():
        assert {r['mode'] for r in group} == {'cdrh1', 'cdrh2', 'cdrh3'} and len(group) == 3
        assert len({(r['group'], r['split'], r['row']) for r in group}) == 1

    # Reconstruct components using an independent pair-key disjoint-set forest.
    parent = {key: key for key in by_pair}
    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    seen = {}
    for key, values in by_pair.items():
        row = values[0]
        for marker in [('heavy', row['heavy']), ('h3', h3_by_key[key]), ('cluster', row['h_cluster'])]:
            if marker in seen:
                parent[root(key)] = root(seen[marker])
            else:
                seen[marker] = key
    components = defaultdict(list)
    for key in by_pair:
        components[root(key)].append(key)
    groups = {}
    for keys in components.values():
        group = hashlib.sha256('|'.join(sorted(keys)).encode()).hexdigest()
        groups[group] = keys
        assert all(row['group'] == group for key in keys for row in by_pair[key])
    assert len(groups) == plan['groups']
    confirm_groups = set(sorted(groups)[:2048])
    assert all(row['split'] == ('confirm' if row['group'] in confirm_groups else 'reserve') for row in rows)
    split_keys = {split: [set(), set(), set()] for split in ('confirm', 'reserve')}
    for key, values in by_pair.items():
        row = values[0]
        for dest, value in zip(split_keys[row['split']], (row['heavy'], h3_by_key[key], row['h_cluster'])):
            dest.add(value)
    assert all(not (a & b) for a, b in zip(split_keys['confirm'], split_keys['reserve']))
    source_count = sum(map(len, source_records.values()))
    selected_or_prior_exclusions = sum(any(a & b for a, b in zip(heavy_keys(r), banned))
                                      for values in source_records.values() for r in values)
    assert selected_or_prior_exclusions == plan['exclusions']['previous_experiments_or_checkpoint_selection']
    assert source_count - selected_or_prior_exclusions - len(by_pair) == plan['exclusions']['training_exact_heavy_h3_or_cluster']
    report = dict(
        status='passed_scoped_checks', created_utc=datetime.now(timezone.utc).isoformat(),
        audit_source_sha256=sha(__file__), data_plan_sha256=sha(out / 'data_plan.json'),
        rows_sha256=sha(out / 'rows.json'), rows=len(rows), pairs=len(by_pair),
        counts=dict(counts), groups=len(groups), confirm_groups=len(confirm_groups),
        reserve_groups=len(groups)-len(confirm_groups), source_rows=source_count,
        h3_min_length=min(map(len, h3_by_key.values())),
        prior_rows_checked=previous_rows, checkpoint_selection_counts=selection_counts,
        prior_or_selection_exclusions=selected_or_prior_exclusions,
        raw_coordinates_and_regions_all_exact=True, components_independently_reconstructed=True,
        heavy_h3_supplied_cluster_cross_split_overlap=0,
        training_hash_binding=dict(checked_shards=len(train_paths),
            bound_records=sum(original_shards[p]['records'] for p in train_paths),
            audit_path=str(full_path), audit_sha256=sha(full_path),
            all_scan_hashes_match_original_checkpoint_training_audit=True,
            training_records_reparsed_in_this_audit=False),
        limitations=[
            'Train exclusion algorithm was reviewed and its shard hashes match the checkpoint-bound original audit; exclusion key extraction was not independently rerun over all training records.',
            'Supplied cluster labels, exact heavy and H3 keys do not establish universal sequence-homology or wider pretraining decontamination.',
            'This is a supplementary OAS validation cohort, not the official SAbDab benchmark.',
            'Freshness concerns the recorded prior experiments and checkpoint-selection indices; unrelated unrecorded use cannot be established from these artifacts.',
            'Runner must use adapter masking before model input and must not consume reserve rows for scoring or selection.'
        ])
    (out / 'data_independent_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
