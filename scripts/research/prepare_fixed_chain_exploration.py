"""Freeze a source-balanced pilot from existing V5 train/valid splits.

Run after activating pllm: python -m scripts.research.prepare_fixed_chain_exploration
No benchmark rows are read. Confirmation rows stay separate from screening.
"""
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/prepared/immune_v5_receptor_completion'
OUT = ROOT / 'output/fixed_chain_exploration_20261014/data'


def receptor_key(row):
    pairs = sorted((role, seq) for role, seq in zip(row['chain_roles'], row['chains'])
                   if role in ('antibody_heavy', 'antibody_light', 'tcr_beta', 'tcr_alpha'))
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((DATA / 'dataset_manifest.json').read_text())
    rng = random.Random(20261014)
    selected = {'train': [], 'screen': [], 'confirm': []}
    blocked, seen_valid, hashes, counts = set(), set(), {}, {}
    # Partition by receptor identity, so repeated receptors cannot cross the two views.
    for shard in manifest['splits']['valid']['shards']:
        path = DATA / shard['path']
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        pools = {'screen': [], 'confirm': []}
        for line in raw.splitlines():
            row = json.loads(line)
            key = receptor_key(row)
            blocked.add(key)
            if key in seen_valid:
                continue
            seen_valid.add(key)
            name = 'screen' if int(key[:8], 16) % 2 == 0 else 'confirm'
            pools[name].append(row)
        for name, rows in pools.items():
            rng.shuffle(rows)
            chosen = rows[:128]
            selected[name].extend(chosen)
            counts[f'{name}/{shard["source"]}'] = len(chosen)
    # Reservoir over raw lines avoids parsing all 7.6M records; scan and hash every shard.
    for source in manifest['sources']:
        pool, n = [], 0
        for shard in manifest['splits']['train']['shards']:
            if shard['source'] != source:
                continue
            path = DATA / shard['path']
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for line in handle:
                    digest.update(line)
                    n += 1
                    if len(pool) < 16384:
                        pool.append(line)
                    else:
                        j = rng.randrange(n)
                        if j < len(pool):
                            pool[j] = line
            hashes[str(path)] = digest.hexdigest()
        rng.shuffle(pool)
        kept = []
        for line in pool:
            row = json.loads(line)
            if receptor_key(row) not in blocked:
                kept.append(row)
            if len(kept) == 8192:
                break
        if len(kept) < 128:
            raise ValueError(f'Too few independent training rows: {source}')
        selected['train'].extend(kept)
        counts[f'train/{source}'] = len(kept)
        print(source, len(kept), flush=True)
    files = {}
    for name, rows in selected.items():
        path = OUT / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
        files[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'records': len(rows)}
    keys = {name: {receptor_key(r) for r in rows} for name, rows in selected.items()}
    assert not keys['train'] & (keys['screen'] | keys['confirm'])
    assert not keys['screen'] & keys['confirm']
    report = dict(schema='fixed_chain.posttraining.pilot.v1', seed=20261014,
                  parent_manifest=str(DATA / 'dataset_manifest.json'),
                  parent_manifest_sha256=hashlib.sha256((DATA / 'dataset_manifest.json').read_bytes()).hexdigest(),
                  files=files, source_counts=counts, parent_shard_hashes=hashes,
                  scope='Exploratory V5; exact receptor-tuple split exclusion only. No new homology/benchmark decontamination claim.',
                  benchmark_used_for_training=False, confirmation_used_for_selection=False)
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
