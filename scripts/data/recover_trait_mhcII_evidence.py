"""Recover explicit TRAIT MHC-II allele pairs without changing training rows.

Activate pllm, then run USE_TF=0 python -B scripts/data/recover_trait_mhcII_evidence.py.
Requires collect_v7_mhc_alleles.py output. Every matching pre-deduplication source
row must agree, and a retained step3 row must corroborate the original row ID.
The output proves allele-pair provenance only; reference resolution is separate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dllm.pipelines.immune_llada.data.sources import row_to_record
from scripts.data.rebuild_v7_from_v5 import sha, write
from scripts.data.update_mhc_inputs import identity

CURRENT = ROOT / 'data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_20261015'
OUT = ROOT / 'output/mhc_completion_20261015'
BASE = OUT / 'allele_candidates.jsonl'
SOURCES = {
    'step2': ROOT / 'downstream/trait/step2/trait_hla.parquet',
    'step3': ROOT / 'downstream/trait/step3/trait_dedup.parquet',
}


def pair_allele(value):
    """Normalize harmless spelling only, preserving all allele fields."""
    value = str(value).strip().upper().replace('HLA-', '')
    value = value.replace('DRA1*', 'DRA*')
    return 'HLA-' + value if value else ''


def main(current=CURRENT, output_dir=OUT):
    base_path = output_dir / 'allele_candidates.jsonl'
    binding = {str(p): sha(p) for p in (base_path, *SOURCES.values(), current / 'dataset_manifest.json')}
    candidates = {}
    for line in base_path.open():
        item = json.loads(line)
        if item['source'] == 'trait' and any(re.search(r'D[RPQ][AB]', a) for a in item['allele_names']):
            key = item['identity']
            if key in candidates:
                raise ValueError('Multiple current rows have the same trait identity')
            candidates[key] = item

    matching = {stage: defaultdict(list) for stage in SOURCES}
    for stage, path in SOURCES.items():
        frame = pd.read_parquet(path).fillna('')
        for pos, row in enumerate(frame.to_dict('records')):
            record = row_to_record('trait', row, split='train')
            if record is None:
                continue
            key = identity(record.to_dict(), 'trait')
            if key in candidates:
                matching[stage][key].append((pos, row))

    counts = Counter()
    output = output_dir / 'trait_classII_candidates.jsonl'
    rejected = []
    with output.open('w') as handle:
        for key, base in candidates.items():
            counts['examined'] += 1
            source_rows = matching['step2'].get(key, [])
            retained_rows = matching['step3'].get(key, [])
            variants = {(pair_allele(r['mhc_allele']), pair_allele(r['mhc_b']),
                         r['mhc_class'], r['species']) for _, r in source_rows}
            reason = ''
            if not source_rows or not retained_rows:
                reason = 'missing_source_or_retained_identity'
            elif len(variants) != 1:
                reason = 'multiple_source_allele_pairs_or_species'
            else:
                a, b, cls, species = next(iter(variants))
                if not a or not b or cls != 'MHCII' or species != 'HomoSapiens':
                    reason = 'missing_pair_or_not_human_classII'
                elif not any(a.startswith('HLA-' + alpha) and b.startswith('HLA-' + beta)
                             for alpha, beta in [('DRA*', 'DRB'), ('DQA1*', 'DQB1*'), ('DPA1*', 'DPB1*')]):
                    reason = 'noncanonical_alpha_beta_locus_pair'
            source_by_id = {r['row_id']: r for _, r in source_rows}
            if not reason and any(source_by_id.get(r['row_id']) != r for _, r in retained_rows):
                raise AssertionError('Step3 retained row differs from its step2 row ID')
            if reason:
                counts['rejected/' + reason] += 1
                rejected.append(dict(identity=key, current_shard=base['current_shard'],
                                     current_row=base['current_row'], reason=reason))
                continue
            item = deepcopy(base)
            item['previous_allele_names'] = item['allele_names']
            item['allele_names'] = [a + '/' + b]
            item['candidate_type'] = 'explicit_trait_classII_pair'
            item['allele_pair_source'] = 'all_matching_pre_deduplication_rows_agree'
            for stage in ('step2', 'step3'):
                for pos, raw in matching[stage][key]:
                    item['evidence'].append(dict(
                        source_parquet=str(SOURCES[stage]), source_sha256=binding[str(SOURCES[stage])],
                        raw_row=pos, row_id=int(raw['row_id']), source=raw['source'],
                        species=raw['species'], mhc_class=raw['mhc_class'],
                        allele_alpha_original=raw['mhc_allele'], allele_beta_original=raw['mhc_b'],
                        allele= a + '/' + b,
                        identity=key,
                        observed_pair={k: raw[k] for k in ('cdr3a', 'cdr3b', 'epitope_seq', 'relation', 'mhc_seq')},
                    ))
            handle.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')
            counts['candidates/' + item['split']] += 1
            counts['pair/' + a + '/' + b] += 1

    # Independently bind every candidate to the unchanged actual parent row.
    by_shard = defaultdict(dict)
    for item in candidates.values():
        by_shard[item['current_shard']][item['current_row']] = item
    for shard, selected in by_shard.items():
        with (current / shard).open('rb') as handle:
            for index, raw in enumerate(handle):
                if index not in selected:
                    continue
                item = selected[index]
                assert hashlib.sha256(raw).hexdigest() == item['old_row_sha256']
                row = json.loads(raw)
                assert row['split'] == item['split']
                assert identity(row, 'trait') == item['identity']
    for path, digest in binding.items():
        assert sha(path) == digest, 'Source changed while recovering evidence'
    report = dict(status='complete', counts=dict(counts), rejected=rejected,
                  decision='Allele-pair evidence only; no sequence inference or training-row mutation',
                  inputs=binding, output=str(output), output_sha256=sha(output),
                  script_sha256=sha(Path(__file__)))
    write(output_dir / 'trait_classII_candidates_audit.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, default=CURRENT)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    main(args.current, args.output)
