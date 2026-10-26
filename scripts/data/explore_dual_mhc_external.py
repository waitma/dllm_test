"""Activate pllm; python -B scripts/data/explore_dual_mhc_external.py.

Audit downloaded TCR3d table and 1FYT mmCIF plus local VDJdb; no training edits.
Requires raw/tcr3d_complexes.tsv and raw/1FYT.cif under the exploration output.
"""
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.rebuild_v7_from_v5 import sha, write
OUT = ROOT / 'output/dual_mhc_exploration_20261015'


def cif_loop(text, first_tag):
    """Read one known loop; handle CIF semicolon text and quoted descriptions."""
    block = next(b for b in text.split('\n#') if first_tag in b)
    tags, tokens, multiline = [], [], None
    for line in block.splitlines():
        if line.startswith(';'):
            if multiline is None:
                multiline = [line[1:]]
            else:
                tokens.append('\n'.join(multiline))
                multiline = None
        elif multiline is not None:
            multiline.append(line)
        elif line.startswith('_'):
            tags.append(line.strip())
        elif line.strip() and line.strip() != 'loop_':
            tokens.extend(shlex.split(line))
    assert multiline is None and len(tokens) % len(tags) == 0
    return [dict(zip(tags, tokens[i:i+len(tags)])) for i in range(0, len(tokens), len(tags))]


def main():
    table = OUT / 'raw/tcr3d_complexes.tsv'
    rows = list(csv.DictReader(table.open(), delimiter='\t'))
    ii = [r for r in rows if r['TCR_complex'] == 'CLASSII']
    with (OUT / 'tcr3d_classII_candidates.jsonl').open('w') as f:
        for r in ii:
            f.write(json.dumps(r) + '\n')
    cif_path = OUT / 'raw/1FYT.cif'
    text = cif_path.read_text()
    descriptions = {r['_entity.id']: r['_entity.pdbx_description'] for r in cif_loop(text, '_entity.id ')}
    entities = [dict(entity=r['_entity_poly.entity_id'],
        description=descriptions[r['_entity_poly.entity_id']], chains=r['_entity_poly.pdbx_strand_id'],
        sequence=''.join(r['_entity_poly.pdbx_seq_one_letter_code_can'].split()))
        for r in cif_loop(text, '_entity_poly.entity_id ')]
    for e in entities:
        e['length'] = len(e['sequence'])
    vdj = ROOT / 'data/tcr/vdjdb_2026_06_03/vdjdb-2026-06-03/vdjdb.txt'
    counts, names, complexes = Counter(), Counter(), defaultdict(list)
    with vdj.open() as f, (OUT / 'vdjdb_classII_candidates.jsonl').open('w') as out:
        for r in csv.DictReader(f, delimiter='\t'):
            counts['all_rows'] += 1
            if r['mhc.class'] != 'MHCII':
                continue
            counts['classII_rows'] += 1
            counts['species/' + r['species']] += 1
            names[(r['mhc.a'], r['mhc.b'])] += 1
            if r['mhc.a'] and r['mhc.b']:
                counts['both_mhc_fields'] += 1
            if r['complex.id'] != '0':
                complexes[r['complex.id']].append(r)
            out.write(json.dumps(r) + '\n')
    counts['paired_complex_ids'] = sum({'TRA', 'TRB'} <= {r['gene'] for r in group} for group in complexes.values())
    # Strict, human paired-TCR shortlist with both MHC reference sequences.
    fasta = ROOT / 'data/tcr_pmhc_fulllength/imgt_hla/hla_prot.fasta'
    reference = defaultdict(set)
    name, parts = None, []
    def add_reference(name, parts):
        fields = name.split(':')
        for n in range(1, len(fields) + 1):
            reference[':'.join(fields[:n])].add(''.join(parts))
    for line in fasta.read_text().splitlines():
        if line.startswith('>'):
            if name:
                add_reference(name, parts)
            name, parts = line.split()[1], []
        else:
            parts.append(line.strip())
    if name:
        add_reference(name, parts)
    banks_path = ROOT / 'output/v7_from_v5_scaffold_20261015/test_scope_v2/banks.json'
    banks = json.loads(banks_path.read_text())
    pairs = {x for b in banks.values() for x in b.get('ab', [])}
    t4 = {x for name, b in banks.items() if name.startswith('t4/') for x in b.get('bp', [])}
    def core(s):
        return s[int(s.startswith('C')):len(s)-int(s.endswith(('F','W')))]
    structural_counts = Counter()
    structural_checks = []
    for r in ii:
        a, b, pep = r['CDR3_alpha'], r['CDR3_beta'], r['Epitope']
        hits = []
        if core(a)+'|'+core(b) in pairs:
            hits.append('exact_test_cdr3_pair')
        if core(b)+'|'+pep in t4:
            hits.append('exact_t4_beta_epitope')
        structural_counts[('human/' if r['TCR_organism']=='Human' else 'other/') +
                          ('test_overlap' if hits else 'no_exact_hit')] += 1
        structural_checks.append(dict(pdb=r['PDB_ID'], organism=r['TCR_organism'], hits=hits))
    write(OUT / 'tcr3d_classII_exact_scope_check.json', structural_checks)
    shortlist, seen, paired_counts = [], set(), Counter()
    for cid, group in complexes.items():
        human = [r for r in group if r['species'] == 'HomoSapiens']
        a = {r['cdr3'] for r in human if r['gene'] == 'TRA'}
        b = {r['cdr3'] for r in human if r['gene'] == 'TRB'}
        if not a or not b:
            continue
        paired_counts['human_paired_complex_ids'] += 1
        contexts = {(r['mhc.a'],r['mhc.b'],r['antigen.epitope']) for r in human}
        if len(a) != 1 or len(b) != 1 or len(contexts) != 1:
            paired_counts['ambiguous_group'] += 1
            continue
        alpha, beta = next(iter(a)), next(iter(b))
        ma, mb, pep = next(iter(contexts))
        key = (alpha,beta,ma,mb,pep)
        if key in seen:
            paired_counts['duplicate_pair_context'] += 1
            continue
        seen.add(key)
        paired_counts['unique_pair_context'] += 1
        sa, sb = reference.get(ma.replace('HLA-', ''), set()), reference.get(mb.replace('HLA-', ''), set())
        if len(sa) != 1 or len(sb) != 1:
            paired_counts['missing_or_ambiguous_mhc_reference'] += 1
            continue
        paired_counts['both_mhc_sequences_unique'] += 1
        overlaps = []
        if core(alpha)+'|'+core(beta) in pairs:
            overlaps.append('exact_test_cdr3_pair')
        if core(beta)+'|'+pep in t4:
            overlaps.append('exact_t4_beta_epitope')
        paired_counts['test_overlap' if overlaps else 'no_exact_pair_or_t4_hit'] += 1
        shortlist.append(dict(complex_id=cid,cdr3_alpha=alpha,cdr3_beta=beta,epitope=pep,
            mhc_alpha_allele=ma,mhc_beta_allele=mb,mhc_alpha=next(iter(sa)),mhc_beta=next(iter(sb)),
            test_overlap=overlaps, references=sorted({r['reference.id'] for r in human}),
            status='exploratory; source validation, scaffold construction and final integration audit pending'))
    with (OUT / 'vdjdb_human_paired_mhcII_shortlist.jsonl').open('w') as out:
        for item in shortlist:
            out.write(json.dumps(item) + '\n')
    result = dict(tcr3d_counts=dict(Counter(r['TCR_complex'] for r in rows)),
        classII_organisms=dict(Counter(r['TCR_organism'] for r in ii)),
        classII_exact_test_scope=dict(structural_counts),
        classII_unique_tcr_epitope_mhc=len({(r['CDR3_alpha'],r['CDR3_beta'],r['Epitope'],r['MHC_allele']) for r in ii}),
        example_1FYT=entities, vdjdb=dict(counts), vdjdb_human_paired_shortlist=dict(paired_counts),
        vdjdb_mhc_pairs=[dict(alpha=a,beta=b,rows=n) for (a,b),n in names.most_common()],
        files={str(p):sha(p) for p in (table,cif_path,vdj,fasta,banks_path,Path(__file__))},
        limitations=['Counts are source rows/structures, not accepted training examples.',
                    'New external receptor pairs require current test-scope exclusion and sequence provenance checks.',
                    'PDB TCR ectodomains require validated variable-domain extraction; never truncate to slot length.',
                    'VDJdb MHC names require exact unique reference mapping for both chains.'])
    write(OUT / 'external_audit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'files','vdjdb_mhc_pairs','example_1FYT'}},indent=2))
    print([(e['chains'],e['description'],e['length']) for e in entities])


if __name__ == '__main__':
    main()
