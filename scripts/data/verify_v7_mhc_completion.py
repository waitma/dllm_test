"""Independently verify a completed V7 release before publication.

Activate conda pllm, then run from the repository root:
  USE_TF=0 python -B scripts/data/verify_v7_mhc_completion.py --workers 4
Pass --data-dir to inspect a published release instead of the staging directory.
No builder resolution/acceptance helpers or model weights are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.rebuild_v7_from_v5 import sha, write
from scripts.data.update_mhc_inputs import identity

PARENT = ROOT / 'data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_20261015'
STAGE = ROOT / 'data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_mhc_completed_20261015.building'
GROUPS = {'trait', 'tcr_native', 'tcr_papers'}
AA = set('ACDEFGHIKLMNPQRSTVWY')
GENES = ('DPA1', 'DPB1', 'DQA1', 'DQB1', 'DRB1', 'DRB3', 'DRB4', 'DRB5', 'DRA', 'A', 'B', 'C', 'E')
PAIR_GENES = {('DPA1', 'DPB1'), ('DQA1', 'DQB1'), ('DRA', 'DRB1'),
              ('DRA', 'DRB3'), ('DRA', 'DRB4'), ('DRA', 'DRB5')}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()


def allele_parts(name):
    """Independent parser for accepted explicit multi-field allele spellings."""
    name = name.strip().upper().removeprefix('HLA-')
    match = re.fullmatch('(' + '|'.join(GENES) + r')\*?(\d{2,}(?::\d{2,}){1,3})([NLSCAQ]?)', name)
    if match is None:
        raise AssertionError(f'Not an explicit supported multi-field HLA allele: {name}')
    gene, numbers, suffix = match.groups()
    assert suffix not in {'N', 'Q'}, f'Null/questionable expression: {name}'
    return gene, tuple(numbers.split(':')), suffix


class IndependentReference:
    """Re-read all matching FASTA proteins and require one covering longest form."""

    def __init__(self, path):
        self.path, self.digest = Path(path), sha(path)
        self.entries = defaultdict(list)
        self.cache = {}
        current, parts = None, []

        def store_entry():
            if current is None:
                return
            match = re.fullmatch(r'([^*]+)\*(\d+(?::\d+)*)([A-Z]?)', current)
            assert match, current
            gene, numbers, suffix = match.groups()
            protein = ''.join(parts).upper()
            if protein.endswith('*'):
                protein = protein[:-1]
            fields = tuple(numbers.split(':'))
            for n in range(2, len(fields) + 1):
                self.entries[(gene, fields[:n])].append((protein, suffix))

        for line in self.path.read_text().splitlines():
            if line.startswith('>'):
                store_entry()
                current, parts = line.split()[1], []
            else:
                parts.append(''.join(line.split()))
        store_entry()

    def resolve(self, name):
        if name in self.cache:
            return self.cache[name]
        gene, fields, suffix = allele_parts(name)
        entries = [p for p, s in self.entries[(gene, fields)] if not suffix or suffix == s]
        assert entries, f'No reference entries for {name}'
        assert all(p and not set(p) - AA for p in entries), name
        maximum = max(map(len, entries))
        longest = {p for p in entries if len(p) == maximum}
        assert len(longest) == 1, f'Multiple longest proteins for {name}'
        sequence = next(iter(longest))
        assert all(p in sequence for p in entries), f'Uncovered/conflicting reference for {name}'
        assert sequence.startswith('M'), name
        bounds = (330, 400) if gene in {'A', 'B', 'C', 'E'} else (230, 310)
        assert bounds[0] <= len(sequence) <= bounds[1], (name, len(sequence))
        canonical = gene + '*' + ':'.join(fields) + suffix
        result = dict(sequence=sequence, canonical=canonical, gene=gene,
                      matching_entries=len(entries), unique_proteins=len(set(entries)))
        self.cache[name] = result
        return result

    def context(self, name, b2m):
        parts = [p.strip() for p in re.split(r'__|[+/;,|]', name) if p.strip()]
        values = [self.resolve(p) for p in parts]
        genes = tuple(v['gene'] for v in values)
        if len(values) == 1 and genes[0] in {'A', 'B', 'C', 'E'}:
            return 'I', [values[0]['sequence'], b2m], [values[0]['canonical']], ['mhc_heavy', 'b2m']
        if genes not in PAIR_GENES and genes[::-1] in PAIR_GENES:
            values.reverse()
            genes = genes[::-1]
        assert genes in PAIR_GENES, f'Incomplete/invalid explicit Class II pair: {name}'
        return 'II', [v['sequence'] for v in values], [v['canonical'] for v in values], ['mhc_alpha', 'mhc_beta']


def assert_non_mhc_unchanged(old, new):
    """Compare all record fields without the builder's invariant projection."""
    mutable = {'chains', 'chain_roles', 'regions', 'chain_metadata', 'metadata', 'task_type'}
    assert {k: v for k, v in old.items() if k not in mutable} == {
        k: v for k, v in new.items() if k not in mutable}
    assert {k: v for k, v in old.get('metadata', {}).items() if k != 'mhc_update'} == {
        k: v for k, v in new.get('metadata', {}).items() if k != 'mhc_update'}
    old_non, new_non = [], []
    for row, target in ((old, old_non), (new, new_non)):
        assert len(row['chains']) == len(row['chain_roles'])
        for field in ('regions', 'chain_metadata'):
            assert all(k.isdigit() and int(k) < len(row['chains']) for k in row.get(field, {}))
        for i, (role, sequence) in enumerate(zip(row['chain_roles'], row['chains'])):
            if role != 'mhc':
                target.append((role, sequence, str(i) in row.get('regions', {}),
                               row.get('regions', {}).get(str(i)),
                               str(i) in row.get('chain_metadata', {}),
                               row.get('chain_metadata', {}).get(str(i))))
    assert old_non == new_non, 'Non-MHC chains/annotations/masks differ'
    assert new['task_type'] == 'tcr_pmhc'
    assert new['chain_roles'][:2] == ['mhc', 'mhc']
    assert new['chain_roles'].count('mhc') == 2
    assert all(str(i) not in new.get('regions', {}) for i in (0, 1))


def check_source_evidence(candidates, pins):
    """Re-read actual raw rows referenced by every provenance entry."""
    from dllm.pipelines.immune_llada.data.sources import row_to_record
    csv.field_size_limit(2**31 - 1)
    requests = defaultdict(lambda: defaultdict(list))
    stats = Counter()
    for candidate in candidates:
        assert candidate['evidence'], 'Every current source identity should have raw evidence'
        csv_names, pair_names = set(), set()
        for evidence in candidate['evidence']:
            path = evidence.get('source_csv') or evidence.get('source_parquet')
            assert path in pins and pins[path] == evidence['source_sha256']
            requests[path][evidence['raw_row']].append((candidate, evidence))
            (csv_names if 'source_csv' in evidence else pair_names).add(evidence['allele'])
        assert sorted(csv_names) == sorted(candidate.get('base_allele_names', candidate['allele_names']))
        if 'base_allele_names' in candidate:
            assert pair_names and sorted(pair_names) == sorted(candidate['allele_names'])
        else:
            assert not pair_names
    for path, selected in requests.items():
        assert sha(path) == pins[path]
        seen = set()
        if Path(path).suffix == '.parquet':
            import pandas as pd
            rows = pd.read_parquet(path).fillna('').to_dict('records')
            iterator = enumerate(rows)
            handle = None
        else:
            handle = open(path)
            iterator = enumerate(csv.DictReader(handle))
        try:
            for index, row in iterator:
                if index not in selected:
                    continue
                seen.add(index)
                for candidate, evidence in selected[index]:
                    record = row_to_record(candidate['source'], row, split=evidence.get('raw_split', 'train'))
                    assert record is not None and identity(record.to_dict(), candidate['source']) == candidate['identity']
                    assert row.get('source', '') == evidence['source']
                    if 'source_csv' in evidence:
                        assert row.get('record_id', '') == evidence['record_id']
                        assert row.get('mhc_allele_norm', '') == evidence['allele']
                        assert Path(path).stem == evidence['raw_split']
                    else:
                        assert int(row['row_id']) == evidence['row_id']
                        assert row['species'] == evidence['species'] == 'HomoSapiens'
                        assert row['mhc_class'] == evidence['mhc_class'] == 'MHCII'
                        assert row['mhc_allele'] == evidence['allele_alpha_original']
                        assert row['mhc_b'] == evidence['allele_beta_original']
                        assert evidence['observed_pair'] == {k: row[k] for k in evidence['observed_pair']}
                        normalize = lambda x: 'HLA-' + str(x).strip().upper().replace('HLA-', '').replace('DRA1*', 'DRA*')
                        assert evidence['allele'] == normalize(row['mhc_allele']) + '/' + normalize(row['mhc_b'])
                    stats['evidence_entries_checked'] += 1
        finally:
            if handle is not None:
                handle.close()
        assert seen == set(selected), path
        assert sha(path) == pins[path]
        stats['source_files_checked'] += 1
    return dict(stats)


def verify_shard(arguments):
    parent_dir, output_dir, parent_spec, spec, candidates, context = arguments
    source, destination = Path(parent_dir) / spec['path'], Path(output_dir) / spec['path']
    reference = IndependentReference(context['fasta']) if candidates else None
    old_digest, new_digest = hashlib.sha256(), hashlib.sha256()
    counts, samples, seen = Counter(), {}, set()
    with source.open('rb') as left, destination.open('rb') as right:
        for index, (raw_old, raw_new) in enumerate(zip_longest(left, right)):
            assert raw_old is not None and raw_new is not None, f'Row count differs in {spec["path"]}'
            assert raw_old.strip() and raw_new.strip(), 'Blank prepared row'
            old_digest.update(raw_old)
            new_digest.update(raw_new)
            counts['records'] += 1
            candidate = candidates.get(index)
            if candidate is not None:
                seen.add(index)
                assert candidate['current_shard'] == spec['path']
                assert candidate['current_row'] == index and candidate['split'] == spec['split']
                assert candidate['source'] == spec['source']
                assert hashlib.sha256(raw_old).hexdigest() == candidate['old_row_sha256']
                old = json.loads(raw_old)
                assert identity(old, spec['source']) == candidate['identity']
                assert candidate['original_mhc'] == [s for r, s in zip(old['chain_roles'], old['chains']) if r == 'mhc']
                counts['source_anchored'] += 1
            if raw_old == raw_new:
                counts['byte_identical'] += 1
                continue
            assert candidate is not None, 'Changed row has no source-anchored candidate'
            new = json.loads(raw_new)
            assert_non_mhc_unchanged(old, new)
            update = new['metadata']['mhc_update']
            assert update['status'] == 'reference_context_completed'
            assert update['parent_manifest_sha256'] == context['parent_manifest_sha256']
            assert update['parent_shard'] == spec['path'] and update['parent_row'] == index
            assert update['parent_row_sha256'] == hashlib.sha256(raw_old).hexdigest()
            assert update['candidate_ledger_sha256'] == context['candidate_sha256']
            assert update['evidence_sha256'] == hashlib.sha256(encoded(candidate['evidence'])).hexdigest()
            assert update['allele_candidates'] == candidate['allele_names']
            assert update['original_mhc'] == candidate['original_mhc']
            assert Path(update['evidence_ledger']).name == 'allele_candidates.jsonl'
            contexts = [reference.context(name, context['b2m']) for name in candidate['allele_names']]
            assert contexts
            assert len({(cls, tuple(sequences)) for cls, sequences, _, _ in contexts}) == 1
            mhc_class, sequences, normalized, components = contexts[0]
            assert new['chains'][:2] == sequences
            assert update['mhc_class'] == mhc_class
            assert update['normalized_alleles'] == sorted({a for _, _, names, _ in contexts for a in names})
            for i, component in enumerate(components):
                metadata = new['chain_metadata'][str(i)]
                assert metadata['mhc_component'] == component
                assert metadata['reference_derived'] is True
                assert metadata['mhc_representation'] == 'reference_sequence'
                if component == 'b2m':
                    assert metadata['mhc_origin'] == 'UniProt:P61769'
                    assert metadata['reference_range_1based'] == [21, 119]
                    assert metadata['sequence_scope'] == 'mature'
                    assert metadata['reference_sha256'] == context['b2m_sha256']
                    assert len(sequences[i]) == 99
                else:
                    assert metadata['mhc_origin'] == 'IPD-IMGT/HLA'
                    assert metadata['reference_sha256'] == reference.digest
                    assert metadata['mhc_class'] == mhc_class
                    assert metadata['sequence_scope'] == 'reference_longest'
                    assert metadata['mhc_allele'] == normalized[i]
            assert old['split'] == new['split'] == spec['split']
            counts['changed'] += 1
            counts['class_' + mhc_class] += 1
            counts['added_to_missing_mhc'] += not candidate['original_mhc']
            sample_key = (spec['source'], spec['split'], tuple(update['normalized_alleles']))
            samples.setdefault(sample_key, dict(shard=spec['path'], row=index,
                source=spec['source'], split=spec['split'], normalized_alleles=update['normalized_alleles'],
                mhc_class=mhc_class, output_row_sha256=hashlib.sha256(raw_new).hexdigest()))
    assert seen == set(candidates)
    assert counts['records'] == parent_spec['records'] == spec['records']
    assert old_digest.hexdigest() == parent_spec['sha256']
    assert new_digest.hexdigest() == spec['sha256']
    assert spec['parent_sha256'] == parent_spec['sha256']
    assert counts['changed'] == spec['mhc_completed']
    return dict(path=spec['path'], split=spec['split'], source=spec['source'], counts=dict(counts),
        parent_sha256=old_digest.hexdigest(), output_sha256=new_digest.hexdigest(), samples=list(samples.values()))


def check_reader(parent_dir, data_dir, manifest, samples):
    import torch
    torch.set_num_threads(1)
    from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset
    from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
    from dllm.pipelines.immune_llada.fixed_chain import FixedChainConfig, FixedChainCollator
    config = FixedChainConfig(**json.loads((data_dir / 'immune_config.json').read_text()))
    assert (config.first_chain_capacity, config.second_chain_capacity, config.max_length) == (167, 135, 2048)
    collator = FixedChainCollator(config)
    grouped, unique = defaultdict(list), {}
    for sample in samples:
        key = (sample['split'], sample['source'], tuple(sample['normalized_alleles']))
        unique.setdefault(key, sample)
    for sample in unique.values():
        grouped[(sample['split'], sample['source'])].append(sample)
    checked = []
    for (split, source), selected in grouped.items():
        old_dataset = PreparedImmuneDataset(parent_dir, split=split, source=source)
        new_dataset = PreparedImmuneDataset(data_dir, split=split, source=source)
        offsets, count = {}, 0
        for shard in manifest['splits'][split]['shards']:
            if shard['source'] == source:
                offsets[shard['path']] = count
                count += shard['records']
        assert len(old_dataset) == len(new_dataset) == count
        for sample in selected:
            index = offsets[sample['shard']] + sample['row']
            old_record, new_record = old_dataset[index], new_dataset[index]
            before, after = collator([old_record]), collator([new_record])
            assert new_record.source_group == source
            for role, capacity in (('tcr_beta', 167), ('tcr_alpha', 135)):
                mask_old = before['role_ids'] == CHAIN_ROLE_TO_ID[role]
                mask_new = after['role_ids'] == CHAIN_ROLE_TO_ID[role]
                expected_capacity = capacity if any(chain.role == role for chain in old_record.chains) else 0
                assert int(mask_old.sum()) == int(mask_new.sum()) == expected_capacity
                for feature in ('input_ids', 'labels', 'position_ids', 'target_slot_mask',
                                'eligible_mask', 'residue_mask', 'terminal_mask'):
                    assert torch.equal(before[feature][mask_old], after[feature][mask_new]), feature
            mhc = after['role_ids'] == CHAIN_ROLE_TO_ID['mhc']
            assert len(after['chain_ids'][mhc].unique()) == 2
            assert not after['eligible_mask'][mhc].any() and not after['target_slot_mask'][mhc].any()
            assert after['labels'][mhc].eq(-100).all()
            tokens = int(after['attention_mask'].sum())
            assert tokens <= config.max_length
            checked.append(dict(**sample, dataset_index=index, tokens=tokens))
    return checked


def verify(data_dir, parent_dir, workers):
    manifest_path = data_dir / 'dataset_manifest.json'
    manifest_sha = sha(manifest_path)
    parent_manifest_path = parent_dir / 'dataset_manifest.json'
    parent_sha = sha(parent_manifest_path)
    manifest, parent = json.loads(manifest_path.read_text()), json.loads(parent_manifest_path.read_text())
    completion = manifest['mhc_completion']
    assert completion['parent_manifest_sha256'] == parent_sha
    assert Path(completion['parent']).resolve() == parent_dir.resolve()
    assert (parent_dir / 'immune_config.json').read_bytes() == (data_dir / 'immune_config.json').read_bytes()
    assert manifest['reader_contract'] == parent['reader_contract']
    pins = completion['sources']
    for path, digest in pins.items():
        assert sha(path) == digest, path
    fasta = data_dir / 'provenance/hla_prot.fasta'
    b2m_path = data_dir / 'provenance/b2m_P61769.json'
    assert sha(fasta) == pins[str(ROOT / 'data/tcr_pmhc_fulllength/imgt_hla/hla_prot.fasta')]
    assert sha(b2m_path) == pins[str(ROOT / 'data/reference/mhc_completion_20261015/b2m_P61769.json')]
    b2m_entry = json.loads(b2m_path.read_text())
    assert b2m_entry['primaryAccession'] == 'P61769' and b2m_entry['organism']['taxonId'] == 9606
    features = [f for f in b2m_entry['features'] if f['type'] == 'Chain'
                and f.get('description') == 'Beta-2-microglobulin']
    assert len(features) == 1
    assert features[0]['location']['start']['value'] == 21 and features[0]['location']['end']['value'] == 119
    b2m = b2m_entry['sequence']['value'][20:119]
    assert len(b2m) == 99 and not set(b2m) - AA
    ledger_path = data_dir / 'provenance/allele_candidates.jsonl'
    ledger_sha = sha(ledger_path)
    assert ledger_sha == completion['candidate_ledger_sha256']
    candidates = [json.loads(line) for line in ledger_path.open()]
    by_shard = defaultdict(dict)
    for item in candidates:
        assert item['current_row'] not in by_shard[item['current_shard']]
        by_shard[item['current_shard']][item['current_row']] = item
    evidence_stats = check_source_evidence(candidates, pins)
    print('actual source evidence checked', json.dumps(evidence_stats), flush=True)
    context = dict(fasta=str(fasta), b2m=b2m, b2m_sha256=sha(b2m_path),
                   candidate_sha256=ledger_sha, parent_manifest_sha256=parent_sha)
    jobs, split_counts = [], defaultdict(Counter)
    assert list(manifest['splits']) == list(parent['splits'])
    for split, block in manifest['splits'].items():
        previous = parent['splits'][split]
        assert block['records'] == previous['records']
        assert [s['path'] for s in block['shards']] == [s['path'] for s in previous['shards']]
        for old_spec, spec in zip(previous['shards'], block['shards']):
            assert old_spec['source'] == spec['source'] and old_spec['split'] == spec['split'] == split
            jobs.append((str(parent_dir), str(data_dir), old_spec, spec, by_shard.pop(spec['path'], {}), context))
    assert not by_shard
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(verify_shard, jobs))
    totals, samples = Counter(), []
    for result in results:
        totals.update(result['counts'])
        split_counts[result['split']].update(result['counts'])
        samples.extend(result.pop('samples'))
    for split, counts in split_counts.items():
        assert counts['records'] == manifest['splits'][split]['records']
        for key in ('records', 'changed', 'class_I', 'class_II', 'added_to_missing_mhc'):
            assert counts[key] == completion['splits'][split].get(key, 0), (split, key)
    for key in ('records', 'changed', 'class_I', 'class_II', 'added_to_missing_mhc'):
        assert totals[key] == completion['counts'].get(key, 0), key
    assert totals['source_anchored'] == len(candidates)
    assert totals['byte_identical'] + totals['changed'] == totals['records']
    print('all shards independently checked', json.dumps(dict(totals)), flush=True)
    reader_samples = check_reader(parent_dir, data_dir, manifest, samples)
    assert sha(manifest_path) == manifest_sha and sha(parent_manifest_path) == parent_sha
    return dict(status='passed', dataset_manifest_sha256=manifest_sha,
        parent_manifest_sha256=parent_sha, code_sha256=sha(__file__), counts=dict(totals),
        splits={k: dict(v) for k, v in split_counts.items()}, evidence=evidence_stats,
        checks=dict(all_shards_hashes_counts=True, every_row_compared=True, unchanged_bytes_identical=True,
            all_changed_non_mhc_fields_identical=True, actual_source_rows_rechecked=True,
            every_changed_hla_independently_resolved_from_pinned_fasta=True,
            b2m_from_uniprot_annotated_mature_chain=True),
        shards=results, reader_samples=reader_samples,
        reader_sample_count=len(reader_samples), max_sampled_tokens=max(s['tokens'] for s in reader_samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=STAGE)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    assert args.workers > 0 and (args.data_dir / 'dataset_manifest.json').is_file()
    receipt = args.data_dir / 'independent_acceptance.json'
    try:
        report = verify(args.data_dir, args.parent, args.workers)
    except BaseException as error:
        write(receipt, dict(status='failed', error_type=type(error).__name__, error=str(error)))
        raise
    write(receipt, report)
    print(json.dumps(dict(status='passed', counts=report['counts'], reader_samples=report['reader_sample_count'])), flush=True)


if __name__ == '__main__':
    main()
