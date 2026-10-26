"""Build a versioned test-clean corpus while retaining V5 scaffold coordinates.

Activate pllm, then the existing protenix_abtcr environment. From repository root:
  USE_TF=0 python -B scripts/data/rebuild_v7_from_v5.py inventory --workers 4

The inventory is read-only with respect to the source corpus. A completion
receipt binds every original shard, observed sequence and synthetic mask.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'data/prepared/immune_v5_receptor_completion'
WORK = ROOT / 'output/v7_from_v5_scaffold_20261015'
ROLES = ('heavy', 'light', 'h3', 'cdr3a', 'cdr3b', 'peptide', 'antigen')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for b in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def keys_for_row(row):
    """Read observed regions only; never turn an all-synthetic chain into a partner.

    The fallback motif is the existing source-adapter extractor. It is recorded
    separately from explicit region/identifier evidence and is not new numbering.
    """
    from dllm.pipelines.immune_llada.data.sources import _h3_core_from_chain
    out, issues, observed_roles = {}, [], set()
    for i, (role, seq) in enumerate(zip(row['chain_roles'], row['chains'])):
        meta = row.get('chain_metadata', {}).get(str(i), {})
        mask = meta.get('synthetic_residue_mask')
        if mask is not None:
            if len(mask) != len(seq) or any(x not in (0, 1) for x in mask):
                raise ValueError('Invalid synthetic residue mask')
            observed = ''.join(c for c, flag in zip(seq, mask) if not flag)
        else:
            observed = seq
        if not observed:
            continue
        observed_roles.add(role)
        region = row.get('regions', {}).get(str(i), {})
        ident = row.get('identifiers', {})
        if role in ('tcr_alpha', 'tcr_beta'):
            key = 'cdr3a' if role == 'tcr_alpha' else 'cdr3b'
            core = region.get('CDR3') or ident.get(key + '_core', '')
            scope = meta.get('sequence_scope', 'full')
            if not core and scope in ('core', 'cdr3', 'cdr3_core'):
                core = observed
            if not core and scope in ('junction', 'cdr3_junction'):
                start = 1 if observed.startswith('C') else 0
                end = len(observed) - int(observed.endswith(('F', 'W')))
                core = observed[start:end]
            if not core:
                core = _h3_core_from_chain(observed)
                issues.append('motif_' + key if core else 'missing_' + key)
            if core and set(core) <= set('ACDEFGHIKLMNPQRSTVWY'):
                out[key] = core
            else:
                issues.append('unresolved_' + key)
            out['fv_a' if role == 'tcr_alpha' else 'fv_b'] = observed
        elif role == 'antibody_heavy':
            out['heavy'] = observed
            core = region.get('CDR3') or ident.get('heavy_h3_core', '')
            if not core:
                core = _h3_core_from_chain(observed)
                issues.append('motif_h3' if core else 'missing_h3')
            if core:
                out['h3'] = core
        elif role == 'antibody_light':
            out['light'] = observed
        elif role in ('peptide', 'antigen'):
            out[role] = observed
    out['paired_tcr'] = {'tcr_alpha', 'tcr_beta'} <= observed_roles
    if out.get('cdr3a') and out.get('cdr3b'):
        out['ab'] = out['cdr3a'] + '|' + out['cdr3b']
    if out.get('cdr3b') and out.get('peptide'):
        out['bp'] = out['cdr3b'] + '|' + out['peptide']
    return out, issues


def inventory_shard(args):
    source, work, shard = args
    path, dest = source / shard['path'], work / 'inventory' / (shard['path'].replace('/', '__'))
    receipt = dest.with_suffix('.json')
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['source_sha256'] != sha(path) or previous['keys_sha256'] != sha(dest.with_suffix('.pkl')):
            raise ValueError('Stale inventory shard: ' + str(path))
        return previous
    values = {r: set() for r in ROLES}
    stats, labels, lengths, h = Counter(), Counter(), Counter(), hashlib.sha256()
    with path.open('rb') as handle:
        for raw in handle:
            h.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            assert row['split'] == shard['split']
            assert len(row['chains']) == len(row['chain_roles'])
            keys, issues = keys_for_row(row)
            stats.update(issues)
            stats['records'] += 1
            stats['paired_tcr'] += int(keys['paired_tcr'])
            for role in ROLES:
                if keys.get(role):
                    values[role].add(keys[role])
            if keys.get('cdr3b'):
                lengths[len(keys['cdr3b'])] += 1
            labels[row.get('labels', {}).get('relation', 'absent')] += 1
            for i, seq in enumerate(row['chains']):
                mask = row.get('chain_metadata', {}).get(str(i), {}).get('synthetic_residue_mask', [])
                stats['synthetic_positions'] += sum(mask)
                stats['synthetic_chains'] += int(bool(mask) and any(mask))
    assert stats['records'] == shard['records'], path
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.with_suffix('.pkl').open('wb') as handle:
        pickle.dump(values, handle, protocol=5)
    result = dict(**shard, source_sha256=h.hexdigest(), bytes=path.stat().st_size,
                  counts=dict(stats), labels=dict(labels), beta_core_lengths=dict(lengths),
                  keys_path=str(dest.with_suffix('.pkl')), keys_sha256=sha(dest.with_suffix('.pkl')))
    write(receipt, result)
    return result


def inventory(source, work, workers):
    work.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'dataset_manifest.json').read_text())
    binding = dict(source=str(source), manifest_sha256=sha(source / 'dataset_manifest.json'),
                   script_sha256=sha(Path(__file__)))
    bound = work / 'inventory_binding.json'
    if bound.exists():
        assert json.loads(bound.read_text()) == binding, 'Inventory implementation/source changed'
    else:
        write(bound, binding)
    shards = [s for block in manifest['splits'].values() for s in block['shards']]
    receipts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(inventory_shard, (source, work, s)) for s in shards]
        for future in as_completed(futures):
            r = future.result()
            receipts.append(r)
            print(json.dumps({'stage': 'inventory', 'completed': len(receipts), 'total': len(shards),
                              'shard': r['path'], 'records': r['records']}), flush=True)
    receipts.sort(key=lambda x: x['path'])
    total = sum(r['records'] for r in receipts)
    assert total == sum(block['records'] for block in manifest['splits'].values())
    write(work / 'inventory.json', dict(**binding, created_at=datetime.now(timezone.utc).isoformat(),
          status='complete', records=total, shards=receipts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inventory'])
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    inventory(args.source.resolve(), args.work.resolve(), args.workers)
