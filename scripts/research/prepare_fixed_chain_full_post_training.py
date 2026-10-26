"""Index all admitted V5 training rows with the pilot's exact-receptor exclusion.

Activate pllm; python -m scripts.research.prepare_fixed_chain_full_post_training
Writes row indices, never changes source JSONL or reads reserved confirmation files.
Only receptor identity is inspected in original validation shards for split exclusion.
"""
from array import array
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.research.prepare_fixed_chain_exploration import receptor_key

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/fixed_chain_exploration_20261014'


def main():
    pilot_path = RUN / 'data/manifest.json'
    pilot = json.loads(pilot_path.read_text())
    manifest_path = Path(pilot['parent_manifest'])
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pilot['parent_manifest_sha256']:
        raise ValueError('Original data manifest changed')
    manifest = json.loads(raw)
    data = manifest_path.parent
    out = RUN / 'full_data_index'
    out.mkdir(exist_ok=False)
    expected = pilot['parent_shard_hashes']
    blocked, hashes, kept, counts = set(), {}, defaultdict(lambda: array('Q')), defaultdict(lambda: [0, 0])

    def rows(shard):
        path = data / shard['path']
        digest, count = hashlib.sha256(), 0
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                if line.strip():
                    count += 1
                    yield json.loads(line)
        if digest.hexdigest() != expected[str(path)] or count != shard['records']:
            raise ValueError('Source shard hash/count changed: ' + str(path))
        hashes[str(path)] = digest.hexdigest()

    for shard in manifest['splits']['valid']['shards']:
        for row in rows(shard):
            blocked.add(receptor_key(row))
    print('validation receptor identities', len(blocked), flush=True)
    for shard in manifest['splits']['train']['shards']:
        source = shard['source']
        for row in rows(shard):
            index = counts[source][0]
            counts[source][0] += 1
            if receptor_key(row) in blocked:
                counts[source][1] += 1
            else:
                kept[source].append(index)
        print(shard['path'], counts[source], flush=True)
    if set(hashes) != set(expected):
        raise ValueError('Incomplete original shard coverage')
    files = {}
    for source, indices in sorted(kept.items()):
        if not indices:
            raise ValueError('A source became empty')
        path = out / f'{source}.npy'
        np.save(path, np.asarray(indices, dtype=np.uint64), allow_pickle=False)
        files[source] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'original_rows': counts[source][0], 'excluded_rows': counts[source][1],
                         'retained_rows': len(indices)}
    report = {'schema': 'fixed_chain.full_post_training_index.v1', 'status': 'ready',
              'at_utc': datetime.now(timezone.utc).isoformat(), 'parent_manifest': str(manifest_path),
              'parent_manifest_sha256': pilot['parent_manifest_sha256'], 'shard_hashes': hashes,
              'pilot_manifest_sha256': hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
              'files': files, 'validation_receptor_identities': len(blocked),
              'original_train_rows': sum(x[0] for x in counts.values()),
              'excluded_exact_validation_overlap': sum(x[1] for x in counts.values()),
              'retained_train_rows': sum(len(x) for x in kept.values()),
              'scope': 'Same exact rendered-receptor-tuple exclusion as pilot; no new homology or parent decontamination claim',
              'confirmation_file_opened': False, 'benchmark_rows_opened': False}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'original_train_rows',
          'excluded_exact_validation_overlap', 'retained_train_rows')}), flush=True)


if __name__ == '__main__':
    main()
