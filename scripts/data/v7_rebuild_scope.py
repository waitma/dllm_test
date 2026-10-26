"""Freeze current evaluation inputs for a V5-preserving corpus rebuild.

Activate pllm and protenix_abtcr; from repository root run:
  USE_TF=0 python -B scripts/data/v7_rebuild_scope.py

Only held-out/query records enter the banks. Reading a training partition to
validate a split does not add its records to a protected bank. No input is edited.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.rebuild_v7_from_v5 import WORK, sha, write
from scripts.data.audit_tcr_exposure import benchmark_keys, input_specs
from dllm.pipelines.immune_llada.data.sources import _h3_core_from_chain

SCOPE = 'v5_scaffold_current_tests_binding_v2'
SCOPE_DIR = WORK / 'test_scope_v2'
AA = set('ACDEFGHIKLMNPQRSTVWY')


def normalize(value):
    return str(value or '').strip().upper().replace('-', '').replace('J', 'L')


def freeze():
    if (SCOPE_DIR / 'registry.json').exists():
        return verify()
    banks, inputs, cohorts, issues, members = {}, {}, {}, Counter(), []

    def bind(path):
        path = Path(path).resolve()
        digest = sha(path)
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest
        return path

    def cohort(name, rows, selection):
        assert name not in cohorts
        cohorts[name] = dict(rows=rows, selection=selection)
        banks[name] = defaultdict(set)
        return banks[name]

    def ab(bank, heavy, light, core=None, antigen=None):
        h, l = normalize(heavy), normalize(light)
        if h:
            bank['heavy'].add(h)
            loop = normalize(core) or _h3_core_from_chain(h)
            if loop:
                bank['h3'].add(loop)
            else:
                issues['ab_inputs_without_h3_key'] += 1
        if l:
            bank['light'].add(l)
        if h and l:
            bank['pair'].add(h + '|' + l)
        if antigen:
            bank['antigen'].add(normalize(antigen))

    # Current T1 suite runs AS negatives. Retired IMMREP T1 and T2B are excluded.
    for name, path, kind, usage in input_specs():
        if name.startswith('uptcr/') and not name.endswith('/test'):
            continue
        if name in {'tcr_representation/train', 'tcr_clustering_embed/tcrs', 'tcr_binding/test'}:
            continue
        if name.startswith('nm2025/'):
            continue
        bank = cohort(name, 0, usage)
        with bind(path).open() as handle:
            for row in csv.DictReader(handle):
                keys, warnings = benchmark_keys(row, kind)
                issues.update(warnings)
                for role, value in keys.items():
                    if value:
                        bank[role].add(value)
                cohorts[name]['rows'] += 1

    # The active T1 entrypoint reads retrain.zip, including each fold's test,
    # not just the older standalone independent-test CSVs. Protect their union.
    from downstream.benchmark.tcr_binding.retrained_protocol import load_official_test, RETRAIN_DATA_ZIP
    bind(RETRAIN_DATA_ZIP)
    for track in ('cdr3b', 'cdr3ab'):
        for eval_set, folds in (('seen_test', range(1, 6)), ('seen_independent', (1,)), ('unseen_independent', (1,))):
            for fold in folds:
                frame, payload, meta = load_official_test(eval_set=eval_set, neg_source='AS', fold=fold, track=track)
                name = f'nm2025_official/{track}/{eval_set}/fold{fold}'
                bank = cohort(name, len(frame), 'actual retrain.zip AS test member; training members excluded')
                members.append(dict(cohort=name, **meta))
                for row in frame.to_dict('records'):
                    mapped = dict(cdr3b=row['CDR3B'], cdr3a=row.get('CDR3A'),
                                  peptide=row['Epitope'], tcra=row.get('LongA'), tcrb=row.get('LongB'))
                    keys, warnings = benchmark_keys(mapped, 'generic')
                    issues.update(warnings)
                    for role, value in keys.items():
                        if value:
                            bank[role].add(value)

    t4 = ROOT / 'downstream/benchmark/data/tcr_generation_bench/eval_conditional.json'
    payload = json.loads(bind(t4).read_text())
    for category, expected in (('benchmark14', 14), ('held20', 20)):
        entries = [r for r in payload.values() if r['category'] == category]
        assert len(entries) == expected
        bank = cohort('t4/' + category, 0, 'ref_binders only; train_binders excluded')
        for entry in entries:
            for seq in entry['ref_binders']:
                # Preserve even invalid historical references in the exact bank;
                # do not impute the '*' residue or silently discard a test row.
                core = seq.strip().upper().replace('-', '').replace('.', '')
                if len(core) >= 5:
                    if core.startswith('C'):
                        core = core[1:]
                    if core.endswith(('F', 'W')):
                        core = core[:-1]
                assert core
                if not set(core) <= AA:
                    issues['t4_noncanonical_reference'] += 1
                bank['bp'].add(core + '|' + entry['epitope'].strip().upper())
                cohorts['t4/' + category]['rows'] += 1

    base = ROOT / 'data/downstream/cdr_infilling'
    for mode in ('cdrh1', 'cdrh2', 'cdrh3'):
        name = 'ab_cdr/kong/' + mode
        bank = cohort(name, 0, 'test.json, all ten folds')
        for fold in range(10):
            path = bind(base / 'sabdab_kong' / mode / f'fold_{fold}/test.json')
            for line in path.read_text().splitlines():
                row = json.loads(line)
                ab(bank, row['heavy_chain_seq'], row['light_chain_seq'], row.get('cdrh3_seq'))
                cohorts[name]['rows'] += 1
    raw = base / 'sab23h2/SAb-23-H2-Ab'
    ids = bind(raw / 'prot_ids.txt').read_text().splitlines()
    assert len(ids) == 60
    bank = cohort('ab_cdr/sab23h2', len(ids), '60 native paired references, all six CDR tasks')
    for name in ids:
        native = bind(raw / f'fasta.files.native/{name}.fasta').read_text().splitlines()
        ab(bank, native[1], native[3])

    import pandas as pd
    pairing = json.loads(bind(ROOT / 'output/v7_pairing_adapter_20261006/benchmark_inventory_v2.json').read_text())['benchmark']
    path = bind(pairing['path'])
    assert inputs[str(path)] == pairing['sha256']
    frame = pd.read_csv(path, keep_default_na=False)
    assert len(frame) == 500
    bank = cohort('ab_pairing', len(frame), 'actual pinned 500-row generation benchmark')
    for row in frame.itertuples():
        ab(bank, row.h_sequence, row.l_sequence)

    from downstream.grammar.ab_probes import prepare_task, m396_splits
    from downstream.grammar.ab_fv_input import adapt_pairs, MANIFEST
    from downstream.ophiuchus_eval.developability import ALL_COLS
    bind(MANIFEST)
    for task in ('specificity', 'gdp_a1', 'm396'):
        frame, pairs, paths, folds = prepare_task(task)
        for path in paths:
            bind(path)
        if task == 'specificity':
            selected = sorted({int(i) for _, test in folds for i in test})
            assert selected == list(range(len(frame)))
            chosen = [pairs[i] for i in selected]
            rule = 'union of five official held-out folds'
        elif task == 'gdp_a1':
            selected = [i for i, valid in enumerate(frame[list(ALL_COLS)].notna().any(axis=1)) if valid]
            chosen = [pairs[i] for i in selected]
            rule = 'union of held-out CV folds, rows eligible for at least one reported property'
        else:
            # Both existing row-reference and grouped task views appear in artifacts.
            groups = [h + '|' + l for h, l in pairs[1:]]
            selected = sorted({int(i) for _, _, _, test in m396_splits(frame, groups, 2023) for i in test})
            chosen = [pairs[i + 1] for i in selected]
            rule = 'union of reported row-reference and sequence-grouped test splits, seed2023; WT is common input'
        adapted, _ = adapt_pairs(task, chosen)
        bank = cohort('ab_repr/' + task, len(chosen), rule)
        for raw_pair, fv_pair in zip(chosen, adapted):
            ab(bank, *raw_pair)
            if fv_pair != raw_pair:
                ab(bank, *fv_pair)
        if task == 'm396':
            wt, _ = adapt_pairs(task, [pairs[0]])
            ab(bank, *pairs[0])
            ab(bank, *wt[0])

    from downstream.benchmark.ab_antigen.abbibench_scope import select_rows, ASSAYS
    base = ROOT / 'data/downstream/ab_antigen_sequence_v2_raw_rows'
    for task in ('binding', 'spr', 'abbibench'):
        with bind(base / f'{task}.jsonl').open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if task == 'abbibench':
            rows = select_rows(rows)
            assert {r['assay'] for r in rows} == set(ASSAYS)
        bank = cohort('ab_ag/' + task, len(rows), 'current zero-shot evaluation rows; AbBiBench official 16 only')
        for row in rows:
            ab(bank, row['heavy'], row['light'])
            if row.get('wt_heavy'):
                ab(bank, row['wt_heavy'], row.get('wt_light'))
    path = ROOT / 'data/downstream/chimera_h3_epitope_group_v1/imgt.jsonl'
    with bind(path).open() as handle:
        rows = [r for r in map(json.loads, handle) if r['split'] == 'test']
    assert len(rows) == 292
    bank = cohort('ab_ag/chimera_h3', len(rows), 'official epitope_group test only')
    for row in rows:
        ab(bank, row['heavy'], row['light'], row['native_h3'], row['antigen'])

    # Existing exact upstream columns can repair audit keys without changing JSONL.
    upstream = bind(ROOT / 'data/tcr_native/unified/fulllength.csv')
    assert inputs[str(upstream)] == '727d0e7a8f71836d4294bb7abf5671f0e3424f8b5863625367f2694ac6e95228'
    code = [Path(__file__), ROOT / 'scripts/data/rebuild_v7_from_v5.py',
            ROOT / 'scripts/data/audit_tcr_exposure.py', ROOT / 'downstream/grammar/ab_probes.py',
            ROOT / 'downstream/benchmark/tcr_binding/retrained_protocol.py',
            ROOT / 'downstream/benchmark/tcr_binding/run_retrained_ours.py',
            ROOT / 'downstream/grammar/ab_fv_input.py',
            ROOT / 'downstream/ophiuchus_eval/affinity_m396.py',
            ROOT / 'downstream/benchmark/ab_antigen/abbibench_scope.py',
            ROOT / 'dllm/pipelines/immune_llada/data/sources.py']
    for path in code:
        bind(path)
    bank_path = SCOPE_DIR / 'banks.json'
    write(bank_path, {name: {role: sorted(values) for role, values in b.items()} for name, b in banks.items()})
    for name, bank in banks.items():
        cohorts[name]['unique_keys'] = {role: len(values) for role, values in bank.items()}
    registry = dict(scope=SCOPE, inputs=inputs, archive_test_members=members, cohorts=cohorts, issues=dict(issues),
                    banks_sha256=sha(bank_path), upstream=str(upstream),
                    selection='only actual held-out/query inputs; original evaluation rows and labels unchanged',
                    excluded=['downstream train-only', 'UpTCR valid', 'T2B', 'retired IMMREP T1',
                              'T1 unused HS/PS views', 'T4 train_binders', 'extra AbBiBench assay',
                              'historical whole-chain CHIMERA', 'VHH/FLAb/S2ALM/BioMap tasks'],
                    rules=dict(paired_tcr='same-row exact CDR3alpha+CDR3beta; all-synthetic partner is absent',
                               unpaired_tcr='exact FV; global normalized Levenshtein >=0.8 for observed cores/peptide; test epitope containment in antigen',
                               t4='exact beta_core+epitope; applies to paired and unpaired TCR',
                               antibody='exact heavy/light/pair; MMseqs joint clusters heavy95/H3core80, bidirectional coverage0.8',
                               antigen='CHIMERA test antigen MMseqs70 clusters, bidirectional coverage0.8'),
                    limitations=['MMseqs cluster membership is operational, not exhaustive pairwise homology.',
                                 'H3 keys use explicit source fields or existing conserved-motif extraction; no new antibody numbering.',
                                 'Paired TCR near-similarity is outside the agreed exact-pair rule.',
                                 'No source-protein accession or inferred MHC identity protection.'])
    write(SCOPE_DIR / 'registry.json', registry)
    print(json.dumps(dict(stage='scope', cohorts=len(cohorts), files=len(inputs),
                          rows=sum(x['rows'] for x in cohorts.values()), issues=dict(issues))), flush=True)
    return registry


def verify():
    registry = json.loads((SCOPE_DIR / 'registry.json').read_text())
    assert registry['scope'] == SCOPE
    assert sha(SCOPE_DIR / 'banks.json') == registry['banks_sha256']
    for path, digest in registry['inputs'].items():
        assert sha(path) == digest, path
    return registry


if __name__ == '__main__':
    freeze()
