"""Collect source-proven HLA names for current V7 rows without changing data.

Activate conda pllm, then run from the repository root:
  USE_TF=0 python -B scripts/data/collect_v7_mhc_alleles.py
Output identities use the shared observed-residue matcher, and retain every
source match (including empty or conflicting HLA names) for downstream review.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dllm.pipelines.immune_llada.data.sources import row_to_record
from scripts.data.update_mhc_inputs import identity, sha, write_json

GROUPS = {'trait', 'tcr_native', 'tcr_papers'}
PARENT = ROOT / 'data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_20261015'
OUT = ROOT / 'output/mhc_completion_20261015'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--shard-prefix', default='', help='Restrict to shard basenames, e.g. unbind_ for V8')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.parent / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source_report_path = ROOT / 'data/mhc_sequence_v1/source_report.json'
    source_report = json.loads(source_report_path.read_text())
    result = {
        'purpose': 'Exact observed-input identity join; no HLA name inference or sequence selection',
        'parent_manifest': str(manifest_path), 'parent_manifest_sha256': sha(manifest_path),
        'source_report': str(source_report_path), 'source_report_sha256': sha(source_report_path),
        'script_sha256': sha(__file__), 'source_files': {}, 'parent_shards': {},
        'shard_prefix': args.shard_prefix,
    }
    current, wanted = [], set()
    current_stats = Counter()
    for split, block in manifest['splits'].items():
        for spec in block['shards']:
            group = spec['source']
            if group not in GROUPS:
                continue
            if not Path(spec['path']).name.startswith(args.shard_prefix):
                continue
            path = args.parent / spec['path']
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for index, raw in enumerate(handle):
                    digest.update(raw)
                    row = json.loads(raw)
                    key = identity(row, group)
                    mhc_indices = [i for i, role in enumerate(row['chain_roles']) if role == 'mhc']
                    current.append(dict(current_shard=spec['path'], current_row=index,
                        split=split, source=group, old_row_sha256=hashlib.sha256(raw).hexdigest(),
                        identity=key, original_mhc=[row['chains'][i] for i in mhc_indices],
                        old_mhc_chains=mhc_indices))
                    wanted.add(key)
                    current_stats[f'{split}/{group}/rows'] += 1
                    current_stats[f'{split}/{group}/mhc_{len(mhc_indices)}'] += 1
            if digest.hexdigest() != spec['sha256']:
                raise ValueError(f'Parent shard hash mismatch: {path}')
            result['parent_shards'][str(path)] = digest.hexdigest()
    print('current rows', len(current), 'unique identities', len(wanted), flush=True)
    lookup = defaultdict(list)
    raw_stats = Counter()
    raw_alleles = defaultdict(Counter)
    csv.field_size_limit(2**31 - 1)
    for label, info in source_report['files'].items():
        group, raw_split = label.split('/')
        if group not in GROUPS:
            continue
        before = sha(info['input'])
        if before != info['input_sha256']:
            raise ValueError(f'Source hash mismatch: {info["input"]}')
        result['source_files'][info['input']] = before
        with open(info['input']) as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                raw_stats[f'{label}/rows'] += 1
                allele = row.get('mhc_allele_norm', '')
                raw_alleles[group][allele] += 1
                record = row_to_record(group, row, split=raw_split)
                if record is None:
                    raw_stats[f'{label}/invalid'] += 1
                    continue
                key = identity(record.to_dict(), group)
                if key not in wanted:
                    continue
                lookup[key].append(dict(source_csv=info['input'], source_sha256=before,
                    record_id=row.get('record_id', ''), raw_split=raw_split, raw_row=index,
                    allele=allele, source=row.get('source', '')))
                raw_stats[f'{label}/matched_rows'] += 1
        if sha(info['input']) != before:
            raise ValueError(f'Source changed during scan: {info["input"]}')
        print(label, 'matched raw rows', raw_stats[f'{label}/matched_rows'], flush=True)
    matched_stats, mhc_alleles = Counter(), defaultdict(Counter)
    path = args.output / 'allele_candidates.jsonl'
    with path.open('w') as handle:
        for row in current:
            evidence = lookup.get(row['identity'], [])
            row['allele_names'] = sorted({e['allele'] for e in evidence})
            row['evidence'] = evidence
            status = ('unmatched' if not evidence else 'empty_allele' if row['allele_names'] == ['']
                      else 'unique_allele' if len(row['allele_names']) == 1 else 'multiple_allele_names')
            prefix = f'{row["split"]}/{row["source"]}'
            matched_stats[f'{prefix}/{status}'] += 1
            if not row['original_mhc']:
                matched_stats[f'{prefix}/no_original_mhc/{status}'] += 1
            for allele in row['allele_names']:
                mhc_alleles[row['source']][allele] += 1
            handle.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False) + '\n')
    result.update(current_counts=dict(current_stats), raw_counts=dict(raw_stats),
        matched_counts=dict(matched_stats), current_alleles={k:dict(v) for k,v in mhc_alleles.items()},
        raw_alleles={k:dict(v) for k,v in raw_alleles.items()},
        output=str(path), output_sha256=sha(path), rows=len(current),
        unique_current_identities=len(wanted), matched_current_identities=len(lookup))
    write_json(args.output / 'allele_candidates_audit.json', result)
    print(json.dumps(result['matched_counts'], indent=2), flush=True)
    print('wrote', path, 'rows', len(current), flush=True)


if __name__ == '__main__':
    main()
