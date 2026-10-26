"""Inspect the supplied TCRvdb CSV without changing raw or prepared data.

Activate pllm; USE_TF=0 python -B scripts/data/inspect_tcrvdb.py.
Writes a reproducible inventory and exact-overlap audit under output/tcrvdb_inspection_20261016.
Training overlap covers current V7 trait/tcr_native/tcr_papers, not all repertoire.
Test checks use exact CDR-pair and T4 beta/peptide keys from the frozen V7 scope.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.mhc_reference_resolution import HLAReference
from scripts.data.rebuild_v7_from_v5 import keys_for_row
from dllm.pipelines.immune_llada.data.mhc import PseudoTable

INPUT = ROOT / 'data/tcrvdb/ttd_gil_ylq_nlv_glc_01-10-2026_TCRvdb.csv'
CURRENT = ROOT / 'data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_mhc_completed_20261015'
OUT = ROOT / 'output/tcrvdb_inspection_20261016'
GROUPS = {'trait', 'tcr_native', 'tcr_papers'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def label(row):
    values = [int(row[k]) for k in ('pos', 'neg', 'ambiguous')]
    assert sum(values) == 1 and set(values) <= {0, 1}
    return ('positive', 'negative', 'ambiguous')[values.index(1)]


def cdr_core(seq):
    return seq[int(seq.startswith('C')):len(seq) - int(seq.endswith(('F', 'W')))]


def main():
    before = sha(INPUT)
    with INPUT.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns, rows = reader.fieldnames, list(reader)
    assert all(None not in r and set(r) == set(columns) for r in rows)
    report = dict(input=str(INPUT), input_sha256=before, rows=len(rows), columns=columns,
        labels=dict(Counter(label(r) for r in rows)),
        missing={k: sum(not r[k].strip() for r in rows) for k in columns},
        source_values=dict(Counter(r['data_origin'] for r in rows)),
        vdjdb_score_counts=dict(Counter(r['vdjdb_score'] for r in rows)),
        supplied_train_valid_test_column=False)
    report['epitopes'] = []
    for peptide in sorted({r['epitope_aa'] for r in rows}):
        selected = [r for r in rows if r['epitope_aa'] == peptide]
        report['epitopes'].append(dict(peptide=peptide, hla=sorted({r['hla_long'] for r in selected}),
            records=len(selected), labels=dict(Counter(label(r) for r in selected))))
    report['sequence_fields'] = {}
    for key in ('cdr3_alpha_aa', 'cdr3_beta_aa', 'full_seq_alpha_aa', 'full_seq_beta_aa', 'epitope_aa'):
        lengths = sorted(len(r[key]) for r in rows)
        report['sequence_fields'][key] = dict(min=min(lengths), max=max(lengths),
            median=lengths[len(lengths) // 2],
            invalid_rows=sum(bool(set(r[key]) - set('ACDEFGHIKLMNPQRSTVWY')) for r in rows))
    report['cdr3_not_contained'] = {c: sum(r['cdr3_' + c + '_aa'] not in r['full_seq_' + c + '_aa']
        for r in rows) for c in ('alpha', 'beta')}
    report['slot_overflow'] = {c: sum(len(r['full_seq_' + c + '_aa']) + 1 > cap for r in rows)
                               for c, cap in (('alpha', 135), ('beta', 167))}
    assert len({r['tcr_id'] for r in rows}) == len(rows)
    report['duplicates'], duplicate_groups = {}, []
    for name, keys in (
        ('full_pair', ('full_seq_alpha_aa', 'full_seq_beta_aa')),
        ('cdr3_pair', ('cdr3_alpha_aa', 'cdr3_beta_aa')),
        ('interaction', ('full_seq_alpha_aa', 'full_seq_beta_aa', 'epitope_aa', 'hla_long'))):
        grouped = defaultdict(list)
        for row in rows:
            grouped[tuple(row[k] for k in keys)].append(row)
        report['duplicates'][name] = dict(unique=len(grouped),
            duplicate_excess_rows=sum(len(rs) - 1 for rs in grouped.values()),
            groups_with_multiple_labels=sum(len({label(r) for r in rs}) > 1 for rs in grouped.values()))
        if name == 'cdr3_pair':
            duplicate_groups = [dict(cdr3_pair=key, rows=rs) for key, rs in grouped.items() if len(rs) > 1]
    fasta = ROOT / 'data/tcr_pmhc_fulllength/imgt_hla/hla_prot.fasta'
    ref = HLAReference.from_fasta(fasta)
    resolutions = {a: ref.resolve(a) for a in sorted({r['hla_long'] for r in rows})}
    report['hla_reference_resolution'] = {a: {k: r.get(k) for k in
        ('accepted', 'status', 'entry_count', 'sequence_length', 'invalid_symbols')}
        for a, r in resolutions.items()}
    pseudo = PseudoTable(ROOT / 'data/mhc_sequence_v1/reference/MHC_pseudo.dat')
    manifest_path = CURRENT / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    lookup, cdr_lookup = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        lookup[(row['full_seq_alpha_aa'], row['full_seq_beta_aa'])].append(i)
        cdr_lookup[cdr_core(row['cdr3_alpha_aa']) + '|' + cdr_core(row['cdr3_beta_aa'])].append(i)
    matches, cdr_matches, checked = [], [], Counter()
    pins = {}
    for split, block in manifest['splits'].items():
        for shard in block['shards']:
            if shard['source'] not in GROUPS:
                continue
            path = CURRENT / shard['path']
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for index, raw in enumerate(f):
                    digest.update(raw)
                    current = json.loads(raw)
                    checked[split + '/' + shard['source']] += 1
                    keys, issues = keys_for_row(current)
                    for i in cdr_lookup.get(keys.get('ab'), []):
                        cdr_matches.append(dict(incoming_row=i, incoming_id=rows[i]['tcr_id'],
                            incoming_label=label(rows[i]), split=split, source=shard['source'],
                            current_shard=shard['path'], current_row=index,
                            current_relation=current.get('labels', {}).get('relation'),
                            same_peptide=keys.get('peptide') == rows[i]['epitope_aa'],
                            key_extraction_issues=issues))
                    components = defaultdict(list)
                    for role, seq in zip(current['chain_roles'], current['chains']):
                        components[role].append(seq)
                    if len(components['tcr_alpha']) != 1 or len(components['tcr_beta']) != 1:
                        continue
                    key = (components['tcr_alpha'][0], components['tcr_beta'][0])
                    for i in lookup.get(key, []):
                        incoming = rows[i]
                        same_peptide = components['peptide'] == [incoming['epitope_aa']]
                        expected, _ = pseudo.lookup(incoming['hla_long'])
                        same_mhc = bool(expected) and components['mhc'] == [expected]
                        reference = resolutions[incoming['hla_long']]
                        if reference['accepted'] and len(components['mhc']) == 2:
                            same_mhc = components['mhc'][0] == reference['sequence']
                        matches.append(dict(incoming_row=i, incoming_id=incoming['tcr_id'],
                            incoming_label=label(incoming), split=split, source=shard['source'],
                            current_shard=shard['path'], current_row=index,
                            current_relation=current.get('labels', {}).get('relation'),
                            same_peptide=same_peptide, same_hla_context=same_mhc))
            assert digest.hexdigest() == shard['sha256']
            pins[str(path)] = digest.hexdigest()
    overlap = {}
    for split in ('train', 'valid'):
        for level in ('full_pair', 'pair_peptide', 'pair_peptide_hla'):
            matched = {m['incoming_row'] for m in matches if m['split'] == split
                and (level == 'full_pair' or m['same_peptide'])
                and (level != 'pair_peptide_hla' or m['same_hla_context'])}
            overlap[split + '/' + level] = dict(incoming_records=len(matched),
                labels=dict(Counter(label(rows[i]) for i in matched)))
        for level in ('cdr3_pair', 'cdr3_pair_peptide'):
            matched = {m['incoming_row'] for m in cdr_matches if m['split'] == split
                       and (level == 'cdr3_pair' or m['same_peptide'])}
            overlap[split + '/' + level] = dict(incoming_records=len(matched),
                labels=dict(Counter(label(rows[i]) for i in matched)))
    report['current_overlap'] = dict(scope='Exact supplied alpha/beta chain strings and existing V7 CDR-core extraction; only current trait/tcr_native/tcr_papers shards. Not a full repertoire or homology audit. CDR-pair hits do not assert full receptor identity.',
        current=str(CURRENT), manifest_sha256=sha(manifest_path), scanned=dict(checked), results=overlap,
        source_shard_sha256=pins)
    banks_path = ROOT / 'output/v7_from_v5_scaffold_20261015/test_scope_v2/banks.json'
    registry_path = banks_path.with_name('registry.json')
    registry = json.loads(registry_path.read_text())
    assert registry['banks_sha256'] == sha(banks_path)
    banks = json.loads(banks_path.read_text())
    pairs = {x for bank in banks.values() for x in bank.get('ab', [])}
    pair_cohorts = defaultdict(list)
    for name, bank in banks.items():
        for pair in bank.get('ab', []):
            pair_cohorts[pair].append(name)
    t4 = {x for name, bank in banks.items() if name.startswith('t4/') for x in bank.get('bp', [])}
    test_hits = []
    for i, row in enumerate(rows):
        reasons = []
        if cdr_core(row['cdr3_alpha_aa']) + '|' + cdr_core(row['cdr3_beta_aa']) in pairs:
            reasons.append('exact_frozen_test_cdr3_pair')
        if cdr_core(row['cdr3_beta_aa']) + '|' + row['epitope_aa'] in t4:
            reasons.append('exact_frozen_t4_beta_peptide')
        if reasons:
            pair = cdr_core(row['cdr3_alpha_aa']) + '|' + cdr_core(row['cdr3_beta_aa'])
            test_hits.append(dict(incoming_row=i, incoming_id=row['tcr_id'], label=label(row), reasons=reasons,
                                  pair_cohorts=sorted(pair_cohorts.get(pair, []))))
    report['frozen_test_key_check'] = dict(path=str(banks_path), sha256=sha(banks_path),
        registry_path=str(registry_path), registry_sha256=sha(registry_path),
        scope='Only exact CDR3 pair and T4 beta-core/peptide keys from the frozen V7 test scope; not complete homology decontamination.',
        hit_records=len(test_hits), labels=dict(Counter(r['label'] for r in test_hits)),
        reason_counts=dict(Counter(reason for r in test_hits for reason in r['reasons'])),
        overlapping_cohort_counts=dict(Counter(name for r in test_hits for name in r['pair_cohorts'])))
    report['reference_sha256'] = {str(fasta): sha(fasta), str(Path(pseudo.path)): pseudo.sha256}
    report['inspection_code_sha256'] = sha(Path(__file__))
    assert sha(INPUT) == before
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in [('inventory.json', report), ('duplicate_cdr3_groups.json', duplicate_groups),
                        ('current_exact_matches.json', matches), ('current_cdr3_matches.json', cdr_matches),
                        ('frozen_test_key_hits.json', test_hits)]:
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('rows', 'labels', 'epitopes', 'hla_reference_resolution')},
                     ensure_ascii=False, indent=2))
    print(json.dumps(overlap, ensure_ascii=False, indent=2))
    print(json.dumps(report['frozen_test_key_check'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
