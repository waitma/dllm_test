"""Audit local dual-MHC candidates against current V7; does not change training data.

Activate pllm, then USE_TF=0 python -B scripts/data/explore_dual_mhc.py.
Writes evidence and candidate context replacements under output/dual_mhc_exploration_20261015.
"""
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.rebuild_v7_from_v5 import sha, write
from scripts.data.update_mhc_inputs import identity, replace_mhc
from dllm.pipelines.immune_llada.data.sources import row_to_record
from dllm.pipelines.immune_llada.fixed_chain import FixedChainConfig, FixedChainCollator

OUT = ROOT / 'output/dual_mhc_exploration_20261015'
CURRENT = ROOT / 'data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_20261015'


def main():
    OUT.mkdir(exist_ok=True)
    result = {'status': 'exploration_only', 'files': {}, 'historical': {}}
    # Count actual historical dual-MHC rows, not the broader long-MHC report.
    for name in ('immune_v6_mhc_sequence_v1', 'immune_v7_mhc_sequence_v1'):
        folder = ROOT / 'data/prepared' / name
        manifest = json.loads((folder / 'dataset_manifest.json').read_text())
        counts = Counter()
        for split, spec in manifest['splits'].items():
            for shard in spec['shards']:
                if shard['source'] not in {'trait', 'tcr_native', 'tcr_papers'}:
                    continue
                with (folder / shard['path']).open() as f:
                    for line in f:
                        row = json.loads(line)
                        indices = [i for i, r in enumerate(row['chain_roles']) if r == 'mhc']
                        counts[f'{split}/mhc_chains_{len(indices)}'] += 1
                        if len(indices) > 1:
                            components = [row.get('chain_metadata', {}).get(str(i), {}).get('mhc_component') for i in indices]
                            counts['components/' + str(components)] += 1
        result['historical'][name] = dict(counts)
    lookup = defaultdict(dict)
    source_report_path = ROOT / 'data/mhc_sequence_v1/source_report.json'
    report = json.loads(source_report_path.read_text())
    counts = Counter()
    for label, info in report['files'].items():
        group, split = label.split('/')
        if group != 'tcr_native':
            continue
        assert sha(info['input']) == info['input_sha256']
        assert sha(info['output']) == info['output_sha256']
        result['files'][info['input']] = info['input_sha256']
        result['files'][info['output']] = info['output_sha256']
        with open(info['input']) as left, open(info['output']) as right:
            reader = csv.DictReader(right)
            for old in csv.DictReader(left):
                new = next(reader)
                assert old['record_id'] == new['record_id']
                entries = json.loads(new['mhc_chains_json'])
                if len(entries) != 2:
                    continue
                counts[f'{split}/raw_dual'] += 1
                row = row_to_record(group, old, split=split)
                if row is None:
                    counts['invalid_raw'] += 1
                    continue
                key = identity(row.to_dict(), group)
                proposal = json.dumps(entries, sort_keys=True)
                lookup[key].setdefault(proposal, []).append(dict(record_id=old['record_id'],
                    raw_split=split, resolution=json.loads(new['mhc_resolution_json'])))
            assert next(reader, None) is None
    result['raw_dual'] = dict(counts)
    # Exact allele/full-sequence audit: never accept a first/prefix representative
    # if different protein sequences are possible for the supplied allele.
    fasta = ROOT / 'data/tcr_pmhc_fulllength/imgt_hla/hla_prot.fasta'
    refs, name, parts = [], None, []
    with fasta.open() as handle:
        for line in handle:
            if line.startswith('>'):
                if name:
                    refs.append((name, ''.join(parts)))
                name, parts = line.split()[1], []
            else:
                parts.append(line.strip())
        if name:
            refs.append((name, ''.join(parts)))
    allele_sequences = defaultdict(set)
    for name, seq in refs:
        fields = name.split(':')
        for n in range(1, len(fields) + 1):
            allele_sequences[':'.join(fields[:n]).replace('*', '')].add(seq)
    result['files'][str(fasta)] = sha(fasta)
    manifest = json.loads((CURRENT / 'dataset_manifest.json').read_text())
    config = FixedChainConfig(**json.loads((CURRENT / 'immune_config.json').read_text()))
    collator = FixedChainCollator(config)
    counts, lengths, examples = Counter(), [], []
    output = (OUT / 'local_context_candidates.jsonl').open('w')
    for split, spec in manifest['splits'].items():
        for shard in spec['shards']:
            if shard['source'] != 'tcr_native':
                continue
            assert sha(CURRENT / shard['path']) == shard['sha256']
            with (CURRENT / shard['path']).open() as f:
                for i, line in enumerate(f):
                    row = json.loads(line)
                    proposals = lookup.get(identity(row, 'tcr_native'), {})
                    if not proposals:
                        counts[f'{split}/no_dual_source_match'] += 1
                        continue
                    if len(proposals) != 1:
                        counts[f'{split}/ambiguous_context'] += 1
                        continue
                    proposal, evidence = next(iter(proposals.items()))
                    entries = json.loads(proposal)
                    counts[f'{split}/unique_context_match'] += 1
                    alleles = {e['resolution'].get('allele', '').replace('HLA-', '').strip() for e in evidence}
                    statuses = []
                    for allele in alleles:
                        possible = allele_sequences.get(allele.replace('*', ''), set())
                        statuses.append('exact_unique_sequence' if possible == {entries[0]['sequence']} else
                                        'ambiguous_allele_sequences' if len(possible) > 1 else 'unresolved_or_mismatch')
                    approved = bool(statuses) and all(s == 'exact_unique_sequence' for s in statuses)
                    status = 'exact_unique_sequence' if approved else '+'.join(sorted(set(statuses)))
                    counts[f'{split}/allele/{status}'] += 1
                    updated = replace_mhc(row, entries, dict(status='exploratory_context_replacement', evidence=evidence))
                    batch = collator([updated])
                    length = int(batch['attention_mask'].sum())
                    lengths.append(length)
                    from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
                    mhc_positions = batch['role_ids'] == CHAIN_ROLE_TO_ID['mhc']
                    assert not batch['eligible_mask'][mhc_positions].any()
                    assert not batch['target_slot_mask'][mhc_positions].any()
                    assert len(batch['chain_ids'][mhc_positions].unique()) == 2
                    item = dict(current_shard=shard['path'], current_row=i, split=split,
                        allele_status=status, allele_names=sorted(alleles), token_length=length,
                        ready_for_context_replacement=approved, record=updated)
                    output.write(json.dumps(item, ensure_ascii=False) + '\n')
                    if len(examples) < 3:
                        examples.append({k: v for k, v in item.items() if k != 'record'})
    output.close()
    result['current_matches'] = dict(counts)
    result['collator'] = dict(checked=len(lengths), max_tokens=max(lengths, default=0),
        slots=[167,135], mhc_chains_distinct=True, mhc_loss_excluded=True)
    result['examples'] = examples
    result['files'][str(CURRENT / 'dataset_manifest.json')] = sha(CURRENT / 'dataset_manifest.json')
    result['files'][str(OUT / 'local_context_candidates.jsonl')] = sha(OUT / 'local_context_candidates.jsonl')
    result['files'][str(Path(__file__))] = sha(Path(__file__))
    write(OUT / 'local_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
