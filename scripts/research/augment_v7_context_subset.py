"""Append existing supervised context/binding positives to an exploratory subset.

Activate conda pllm, then run this absolute script with --base-dir /absolute/base
--parent-dir /absolute/prepared --output-dir /absolute/new_subset
--additional-records 20000 --seed 20261003. Use --self-test for stdlib fixtures.
Copies existing rows; never annotates sequences, invents labels, or starts jobs.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.sample_v7_architecture_pilot import digest, quotas, stamp, write_json, sample

CONTEXT = {'antigen', 'peptide', 'mhc'}


def augment(base, parent, output, additional=20000, seed=20261003):
    paths = [Path(path) for path in (base, parent, output)]
    if not all(path.is_absolute() for path in paths):
        raise ValueError('Use absolute directory paths')
    base, parent, output = [path.resolve() for path in paths]
    if any(output == path or path in output.parents or output in path.parents for path in (base, parent)):
        raise ValueError('Output must be separate from base and parent')
    if not 1 <= additional <= 20000:
        raise ValueError('Additional record count must be 1..20000')
    metadata, observed_stamps = {}, {}

    def read_metadata(path):
        before = stamp(path)
        raw = path.read_bytes()
        if stamp(path) != before:
            raise ValueError('Metadata changed while reading')
        metadata[path] = raw
        observed_stamps[path] = before
        return json.loads(raw)

    base_manifest_path = base / 'sample_manifest.json'
    original = read_metadata(base_manifest_path)
    if original.get('schema') != 'v7.exploratory_uncertified.v1' or original.get('formal_training_ready') is not False:
        raise ValueError('Base must be an explicitly exploratory subset')
    parent_path = parent / 'dataset_manifest.json'
    parent_manifest = read_metadata(parent_path)
    if (Path(original['parent']['manifest']['path']).resolve() != parent_path
            or digest(metadata[parent_path]) != original['parent']['manifest']['sha256']):
        raise ValueError('Base refers to another parent version')
    for item in original['parent']['certificates'].values():
        path = Path(item['path'])
        read_metadata(path)
        if digest(metadata[path]) != item['sha256']:
            raise ValueError('Existing parent certificate changed')

    base_bytes, base_lines = {}, {}
    for split in ('train', 'valid'):
        item = original['files'][split]
        path = Path(item['path']).resolve()
        if path.parent != base:
            raise ValueError('Base split path escapes the subset')
        observed_stamps[path] = stamp(path)
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        if (digest(raw) != item['sha256'] or len(lines) != item['records']
                or any(not line.strip() or not line.endswith(b'\n') for line in lines)):
            raise ValueError('Base split hash/count/newlines differ')
        base_bytes[split], base_lines[split] = raw, lines
    provenance_path = Path(original['provenance']['path']).resolve()
    if provenance_path.parent != base:
        raise ValueError('Base provenance path escapes the subset')
    observed_stamps[provenance_path] = stamp(provenance_path)
    provenance_raw = provenance_path.read_bytes()
    if digest(provenance_raw) != original['provenance']['sha256'] or not provenance_raw.endswith(b'\n'):
        raise ValueError('Base provenance hash/newline differs')
    base_origins, base_hashes, covered = set(), set(), set()
    for line in provenance_raw.splitlines():
        item = json.loads(line)
        key = (item['split'], item['output_line'])
        if key in covered or digest(base_lines[key[0]][key[1] - 1]) != item['row_sha256']:
            raise ValueError('Base provenance does not match its output rows')
        covered.add(key)
        base_origins.add((str(Path(item['source_shard']).resolve()), item['source_record']))
        base_hashes.add(item['row_sha256'])
    if len(covered) != sum(map(len, base_lines.values())):
        raise ValueError('Base provenance coverage differs')
    shards = parent_manifest['splits']['train']['shards']
    if sum(s['records'] for s in shards) != parent_manifest['splits']['train']['records']:
        raise ValueError('Parent train counts differ')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Never overwrite an earlier sample or failure receipt.
    inventory, excluded, eligible, reservoirs, generators = Counter(), Counter(), Counter(), {}, {}
    seen_hashes, seen_paths, inspected = set(), set(), []
    try:
        for shard in sorted(shards, key=lambda item: item['path']):
            path = (parent / shard['path']).resolve()
            if parent not in path.parents or path in seen_paths or shard.get('split', 'train') != 'train':
                raise ValueError('Invalid/duplicate parent training shard')
            seen_paths.add(path)
            before = stamp(path)
            observed_stamps[path] = before
            source = shard['source']
            rng = generators.setdefault(source, random.Random(int(digest(json.dumps([seed, source]).encode()), 16)))
            reservoir = reservoirs.setdefault(source, [])
            hasher, count = hashlib.sha256(), 0
            with path.open('rb') as handle:
                for physical_line, raw in enumerate(handle, 1):
                    hasher.update(raw)
                    if not raw.strip():
                        continue
                    count += 1
                    row = json.loads(raw)
                    if not CONTEXT.intersection(row.get('chain_roles', [])):
                        continue
                    relation = row.get('labels', {}).get('relation')
                    supervised = row.get('metadata', {}).get('relation_supervised') is True
                    label = relation if relation in ('binding', 'nonbinding') else 'other_or_missing'
                    inventory[f'{source}/{label}/supervised={supervised}'] += 1
                    if relation != 'binding' or not supervised:
                        continue
                    row_sha = digest(raw)
                    if (str(path), count) in base_origins or row_sha in base_hashes:
                        excluded['base_train_or_valid_overlap'] += 1
                        continue
                    if row_sha in seen_hashes:
                        excluded['duplicate_candidate_raw_bytes'] += 1
                        continue
                    if row.get('split') not in (None, 'train') or not raw.endswith(b'\n'):
                        raise ValueError('Candidate split or newline differs')
                    seen_hashes.add(row_sha)
                    eligible[source] += 1
                    candidate = (str(path), count, physical_line, row_sha, raw)
                    if len(reservoir) < additional:
                        reservoir.append(candidate)
                    else:
                        slot = rng.randrange(eligible[source])
                        if slot < additional:
                            reservoir[slot] = candidate
            if count != shard['records'] or hasher.hexdigest() != shard['sha256'] or stamp(path) != before:
                raise ValueError('Parent shard changed or count/hash differs: ' + str(path))
            inspected.append(dict(path=str(path), records=count, sha256=hasher.hexdigest()))
            print(json.dumps({'scanned_shards': len(inspected), 'eligible_unique_context_rows': sum(eligible.values())}), flush=True)
        if sum(eligible.values()) < additional:
            raise ValueError(f'Only {sum(eligible.values())} eligible unique rows; need {additional}')
        allocation = quotas(dict(eligible), additional)
        selected = {}
        for source, number in allocation.items():
            rng = random.Random(int(digest(json.dumps([seed, source, 'final_selection']).encode()), 16))
            selected[source] = sorted(rng.sample(reservoirs[source], number), key=lambda item: item[:2])
        files = json.loads(json.dumps(original['files']))
        for split in ('train', 'valid'):
            destination = output / f'{split}.jsonl'
            destination.write_bytes(base_bytes[split])
            files[split]['path'] = str(destination)
        output_provenance = output / 'provenance.jsonl'
        output_provenance.write_bytes(provenance_raw)
        total = len(base_lines['train'])
        with (output / 'train.jsonl').open('ab') as train, output_provenance.open('ab') as provenance:
            for source in sorted(selected):
                first = total + 1
                for origin, record, line, row_sha, raw in selected[source]:
                    total += 1
                    train.write(raw)
                    entry = dict(split='train', output_line=total, source_group=source, source_shard=origin,
                                 source_record=record, source_line=line, row_sha256=row_sha)
                    provenance.write((json.dumps(entry, sort_keys=True) + '\n').encode())
                files['train']['source_counts'][source] = files['train']['source_counts'].get(source, 0) + len(selected[source])
                files['train']['source_ranges'].append(dict(source_group=source, start_line=first, end_line=total))
        files['train'].update(records=total, sha256=digest((output / 'train.jsonl').read_bytes()))
        for path, before in observed_stamps.items():
            if stamp(path) != before:
                raise ValueError('Input changed during augmentation: ' + str(path))
        for path, raw in metadata.items():
            if path.read_bytes() != raw:
                raise ValueError('Metadata changed during augmentation')
            prefix = 'base_' if path == base_manifest_path else 'parent_'
            (output / (prefix + path.name)).write_bytes(raw)
        result = dict(schema=original['schema'], status='exploratory_uncertified', formal_training_ready=False,
            seed=seed, parent=original['parent'], files=files,
            provenance=dict(path=str(output_provenance), sha256=digest(output_provenance.read_bytes())),
            augmentation=dict(base_manifest_path=str(base_manifest_path), base_manifest_sha256=digest(metadata[base_manifest_path]),
                base_train_rows=len(base_lines['train']), added_records=additional, added_source_counts=allocation,
                added_relation_counts={'binding': additional}, parent_context_inventory=dict(inventory),
                eligible_unique_by_source=dict(eligible), excluded=dict(excluded), inspected_shards=inspected,
                method='per-source min-one proportional quotas; uniform reservoir; binding positives only',
                duplicates_definition='same parent shard/record or identical original row bytes; not sequence clustering'),
            limitations=original['limitations'] + ['Added existing binding positives only; no binary discrimination claim.'])
        write_json(output / 'sample_manifest.json', result)
        return result
    except BaseException as error:
        write_json(output / 'FAILURE.json', dict(status='failed', error=repr(error), context_inventory=dict(inventory),
            eligible_unique_by_source=dict(eligible), inspected_shards=inspected, excluded=dict(excluded)))
        raise


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        parent = root / 'parent'
        parent.mkdir()
        manifest = {'splits': {}}
        for split in ('train', 'valid'):
            shards = []
            for source in ('a', 'b'):
                rows = [dict(chains=['ACD', 'CDE'], chain_roles=['antigen', 'antibody_heavy'], source=source,
                    split=split, identity=i, labels={'relation': 'binding' if i % 3 else 'nonbinding'},
                    metadata={'relation_supervised': True}) for i in range(12)]
                raw = b''.join((json.dumps(row) + '\n').encode() for row in rows)
                path = parent / f'{split}_{source}.jsonl'
                path.write_bytes(raw)
                shards.append(dict(path=path.name, split=split, source=source, records=len(rows), sha256=digest(raw)))
            manifest['splits'][split] = dict(records=24, shards=shards)
        write_json(parent / 'dataset_manifest.json', manifest)
        sample(parent, root / 'base', 4, 2, 42)
        left = augment(root / 'base', parent, root / 'left', 4, 43)
        right = augment(root / 'base', parent, root / 'right', 4, 43)
        assert left['files']['train']['sha256'] == right['files']['train']['sha256']
        base_raw = (root / 'base/train.jsonl').read_bytes()
        final_raw = (root / 'left/train.jsonl').read_bytes()
        assert final_raw.startswith(base_raw) and left['files']['train']['records'] == 8
        assert (root / 'left/valid.jsonl').read_bytes() == (root / 'base/valid.jsonl').read_bytes()
        base_hashes = {digest(line) for split in ('train', 'valid') for line in (root / 'base' / f'{split}.jsonl').read_bytes().splitlines(keepends=True)}
        extra = final_raw[len(base_raw):].splitlines(keepends=True)
        assert len(set(map(digest, extra))) == 4 and not base_hashes.intersection(map(digest, extra))
        assert all(json.loads(line)['labels']['relation'] == 'binding' for line in extra)
        damaged = parent / 'train_a.jsonl'
        damaged.write_bytes(damaged.read_bytes().replace(b'"identity": 0', b'"identity": 9'))
        try:
            augment(root / 'base', parent, root / 'changed', 4, 43)
        except ValueError:
            assert (root / 'changed/FAILURE.json').exists()
            assert not (root / 'changed/sample_manifest.json').exists()
        else:
            raise AssertionError('Changed parent shard accepted')
    return dict(status='passed', deterministic=True, base_prefix_and_valid_preserved=True,
                positive_only_no_overlap=True, changed_parent_rejected=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-dir', 'parent-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--additional-records', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    if any(getattr(args, name) is None for name in ('base_dir', 'parent_dir', 'output_dir')):
        parser.error('Require --base-dir, --parent-dir and --output-dir')
    result = augment(args.base_dir, args.parent_dir, args.output_dir, args.additional_records, args.seed)
    print(json.dumps({'status': result['status'], 'train_rows': result['files']['train']['records'],
                      'added': result['augmentation']['added_source_counts']}))


if __name__ == '__main__':
    main()
