"""Build and verify the 167/135-slot V7 release without changing its parent.

Activate pllm, then run from the repository root:
  USE_TF=0 python -B scripts/data/resize_v7_design_slots.py --workers 8
  USE_TF=0 python -B scripts/data/check_v7_no_negative_design_reader.py --data-dir \
    /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_20261015

Overlong records are quarantined whole, never truncated. Every retained byte,
original split, multiplicity, label and synthetic scaffold is preserved.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.rebuild_v7_from_v5 import sha, write

PARENT = ROOT / 'data/prepared/immune_v7_from_v5_no_negative_design_20261015'
OUTPUT = ROOT / 'data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_20261015'
CAPACITIES = {'antibody_heavy': 167, 'tcr_beta': 167,
              'antibody_light': 135, 'tcr_alpha': 135}


def overflows(row):
    assert len(row['chains']) == len(row['chain_roles'])
    return [dict(chain_index=i, role=role, residues=len(seq), slots=CAPACITIES[role])
            for i, (seq, role) in enumerate(zip(row['chains'], row['chain_roles']))
            if role in CAPACITIES and len(seq) + 1 > CAPACITIES[role]]


def process_shard(args):
    parent, stage, shard = args
    import numpy as np
    target = stage / shard['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    selection = np.ones(shard['records'], dtype=np.uint8)
    counts, labels, excluded = Counter(), Counter(), []
    digest = hashlib.sha256()
    with (parent / shard['path']).open('rb') as source, target.open('wb') as out:
        for i, raw in enumerate(source):
            digest.update(raw)
            row = json.loads(raw)
            failures = overflows(row)
            counts['input'] += 1
            if failures:
                excluded.append(dict(parent_shard=shard['path'], parent_row=i,
                                     source=shard['source'], split=shard['split'],
                                     failures=failures, record=row))
                counts['overflow'] += 1
            else:
                selection[i] = 0
                out.write(raw)
                counts['retained'] += 1
                relation = row.get('labels', {}).get('relation', 'absent')
                assert relation in {'binding', 'unknown', 'absent'}
                labels[relation] += 1
    assert counts['input'] == shard['records'] and digest.hexdigest() == shard['sha256']
    selection_path = 'selection/' + shard['path'].replace('/', '__') + '.npy'
    np.save(stage / selection_path, selection, allow_pickle=False)
    # Independently rescan every input record and compare retained output bytes.
    emitted = 0
    with (parent / shard['path']).open('rb') as source, target.open('rb') as out:
        for i, raw in enumerate(source):
            row = json.loads(raw)
            fits = all(len(s) < CAPACITIES[r] for s, r in zip(row['chains'], row['chain_roles'])
                       if r in CAPACITIES)
            assert (selection[i] == 0) == fits
            if fits:
                assert out.readline() == raw
                emitted += 1
        assert not out.read(1)
    assert emitted == counts['retained']
    return dict(path=shard['path'], source=shard['source'], split=shard['split'],
                records=emitted, sha256=sha(target), parent_sha256=digest.hexdigest(),
                selection_path=selection_path, selection_sha256=sha(stage / selection_path),
                counts=dict(counts), labels=dict(labels), excluded=excluded)


def main(workers):
    manifest_path = PARENT / 'dataset_manifest.json'
    parent = json.loads(manifest_path.read_text())
    verification = json.loads((PARENT / 'verification.json').read_text())
    acceptance = json.loads((PARENT / 'reader_acceptance.json').read_text())
    assert verification['status'] == acceptance['status'] == 'passed'
    assert verification['dataset_manifest_sha256'] == acceptance['dataset_manifest_sha256'] == sha(manifest_path)
    stage = OUTPUT.with_name(OUTPUT.name + '.building')
    assert not OUTPUT.exists() and not stage.exists(), 'Refusing to overwrite existing release/build'
    (stage / 'selection').mkdir(parents=True)
    config = json.loads((ROOT / 'configs/models/v5_fixed_chain_f011_75m_20261012/immune_config.json').read_text())
    config.update(first_chain_capacity=167, second_chain_capacity=135, full_capacity=304)
    write(stage / 'immune_config.json', config)
    shards = [s for spec in parent['splits'].values() for s in spec['shards']]
    results, excluded, totals, labels = [], [], Counter(), Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(process_shard, [(PARENT, stage, s) for s in shards]):
            excluded.extend(result.pop('excluded'))
            totals.update(result['counts'])
            labels.update(result['labels'])
            results.append(result)
            print(json.dumps(dict(shard=result['path'], **result['counts'])), flush=True)
    assert totals['input'] == verification['retained']
    assert totals['input'] == totals['retained'] + totals['overflow']
    with (stage / 'capacity_exclusions.jsonl').open('w') as handle:
        for row in excluded:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = dict(parent)
    manifest.update(created_at=datetime.now(timezone.utc).isoformat(), dataset_version=OUTPUT.name)
    manifest['splits'] = {}
    for split in parent['splits']:
        items = [s for s in results if s['split'] == split and s['records']]
        sources = Counter()
        for s in items:
            sources[s['source']] += s['records']
        manifest['splits'][split] = dict(records=sum(s['records'] for s in items), shards=items,
            sources={name: dict(records=n) for name, n in sources.items()})
    manifest['reader_contract'] = dict(parent['reader_contract'], first_chain_capacity=167,
        second_chain_capacity=135, full_capacity=304, config_path='immune_config.json',
        config_sha256=sha(stage / 'immune_config.json'))
    manifest['capacity_adjustment'] = dict(parent=str(PARENT), parent_manifest_sha256=sha(manifest_path),
        parent_verification_sha256=sha(PARENT / 'verification.json'),
        policy='quarantine whole record if any receptor sequence plus EOS exceeds its slot capacity; no truncation',
        counts=dict(totals), labels=dict(labels), selection_codes={'0': 'retained', '1': 'capacity_overflow'},
        exclusions_sha256=sha(stage / 'capacity_exclusions.jsonl'), code_sha256=sha(Path(__file__)))
    # Parent rebuild counts describe the upstream cleaning, not this release's total.
    manifest['parent_rebuild'] = manifest.pop('rebuild')
    write(stage / 'dataset_manifest.json', manifest)
    write(stage / 'cleaning_report.json', dict(status='verified', totals=dict(totals), labels=dict(labels),
        by_shard=results, parent=str(PARENT), excluded_records=excluded))
    write(stage / 'verification.json', dict(status='passed', retained=totals['retained'],
        nonbinding=0, capacity_overflow=0, quarantined=totals['overflow'],
        source_to_output_bytes='all retained records identical; selection independently recomputed',
        test_isolation='subset of hash-verified parent; no new clustering or changed test scope',
        dataset_manifest_sha256=sha(stage / 'dataset_manifest.json'),
        cleaning_report_sha256=sha(stage / 'cleaning_report.json')))
    (stage / 'README.md').write_text(
        '# V7 no-negative design: restored 167/135 slots\n\n'
        'Load immune_config.json with FixedChainConfig and FixedChainCollator. '
        'H/beta slots=167, L/alpha slots=135, including EOS; full_capacity=304.\n\n'
        'Overlong records are quarantined in capacity_exclusions.jsonl. Retained JSONL bytes, '
        'split assignments, multiplicities, labels and synthetic masks are unchanged.\n\n'
        f'Parent release: {PARENT}\n\nReader acceptance is recorded separately in reader_acceptance.json.\n')
    stage.rename(OUTPUT)
    print(json.dumps(dict(output=str(OUTPUT), totals=dict(totals), labels=dict(labels))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    main(parser.parse_args().workers)
