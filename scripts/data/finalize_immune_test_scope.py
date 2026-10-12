"""Restore existing identifiers, screen declared exact keys, then independently verify.

Activate pllm and run:
  python -B scripts/data/finalize_immune_test_scope.py build --dataset /absolute/new/dataset
  python -B scripts/data/finalize_immune_test_scope.py verify --dataset /absolute/new/dataset

No new sequence annotation, similarity search, or generation. Uses existing
upstream columns and hash-bound prior exclusion evidence. Certificates describe
only the declared operational scope and explicitly record user exclusions.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.immune_test_scope_policy import SCOPE, WAIVERS, T4, required_test_paths
from scripts.data.audit_test_pair_release import observed_roles
from scripts.data.audit_test_pair_release_v2 import stored_core

PARENT = ROOT / 'data/prepared/immune_v7_test_pair_exact_20261002_r2'
PARENT_SHA = '77b91a4b3f2a67f2bc1491589a1850392141f33c68ebb30ec21c94ff6c9a3108'
UPSTREAM = ROOT / 'data/tcr_native/unified/fulllength.csv'
UPSTREAM_SHA = '727d0e7a8f71836d4294bb7abf5671f0e3424f8b5863625367f2694ac6e95228'
WORK = ROOT / 'output/test_scope_finalization_20261002'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')


def inputs():
    assert sha(PARENT / 'dataset_manifest.json') == PARENT_SHA
    assert sha(UPSTREAM) == UPSTREAM_SHA
    old_report = read(PARENT / 'cleaning_report.json')
    bindings = dict(old_report['bindings'])
    # Parent construction, independent selection scan and semantic fix evidence.
    for p in [PARENT / n for n in ('dataset_manifest.json', 'cleaning_report.json', 'selection_verification.json')]:
        bindings[str(p)] = sha(p)
    for n in ('fix_acceptance.json', 'audit_report.json'):
        p = ROOT / 'output/test_pair_release_rebuild_r2_20261002' / n
        bindings[str(p)] = sha(p)
    fix = read(ROOT / 'output/test_pair_release_rebuild_r2_20261002/fix_acceptance.json')
    assert fix['manifest_sha256'] == PARENT_SHA and fix['status'] == 'observed_partner_bug_fix_passed'
    waiver = ROOT / 'output/test_pair_release_rebuild_r2_20261002/user_scope_amendment.json'
    assert set(read(waiver)['excluded_from_required_audit']) == set(WAIVERS)
    legacy = ROOT / 'data/prepared/immune_v7_clean_r2/audit_certificate.json'
    cert = read(legacy)
    assert cert['pending'] == ['unresolved_benchmark_h3']
    assert cert['forbidden_overlap_rows'] == cert['family_split_overlap'] == 0
    legacy_registry = ROOT / 'data/dedup/immune_v7_20260930/registry.json'
    assert sha(legacy_registry) == cert['registry_sha256']
    assert sha(ROOT / 'data/dedup/immune_v7_20260930/effective_banks.json') == cert['effective_banks_sha256']
    for p, h in read(legacy_registry)['inputs'].items():
        bindings[p] = h
    paths = [UPSTREAM, waiver, legacy, legacy_registry,
             ROOT / 'data/dedup/immune_v7_20260930/effective_banks.json',
             ROOT / 'data/tcr_native/dataset/t4_refbinder_blocklist.txt',
             ROOT / 'scripts/data/immune_test_scope_policy.py',
             ROOT / 'scripts/data/audit_test_pair_release.py',
             ROOT / 'scripts/data/audit_test_pair_release_v2.py']
    paths += required_test_paths()
    for p in paths:
        bindings[str(p)] = sha(p)
    for p, h in bindings.items():
        assert sha(p) == h, p
    # Verify the actual r2 shards, not only the much older inherited input graph.
    for block in read(PARENT / 'dataset_manifest.json')['splits'].values():
        for s in block['shards']:
            bindings[str(PARENT / s['path'])] = s['sha256']
    registry = read(ROOT / 'data/dedup/tcr_exposure_20260930_r2/registry.json')
    banks_path = ROOT / 'data/dedup/tcr_exposure_20260930_r2/banks.json'
    assert sha(banks_path) == registry['banks_sha256']
    banks = read(banks_path)
    selected_paths = set(map(str, required_test_paths()))
    pairs = set()
    for item in registry['inputs']:
        if item['path'] in selected_paths:
            pairs.update(banks[item['cohort']]['ab'])
    lookup = defaultdict(list)
    with UPSTREAM.open() as f:
        for r in csv.DictReader(f):
            lookup[(r['alpha_fv'], r['beta_fv'], r['epitope_seq'])].append(r)
    # Same endpoint convention as the existing T4 blocklist; no new inference.
    t4_keys, t4_stats = set(), []
    for i, p in enumerate(T4):
        payload, keys, categories = read(p), set(), Counter()
        for name, entry in payload.items():
            ep = (name if i else entry['epitope']).strip().upper()
            categories[entry.get('category', '?')] += 1
            for raw in entry.get('ref_binders', []):
                core = raw.strip().upper().replace('-', '').replace('.', '')
                if len(core) >= 5:
                    if core.startswith('C'): core = core[1:]
                    if core.endswith(('F', 'W')): core = core[:-1]
                if core: keys.add(core + '|' + ep)
        t4_keys.update(keys)
        t4_stats.append(dict(path=str(p), sha256=sha(p), entries=len(payload),
                             categories=categories, unique_test_keys=len(keys)))
    blocklist = set((ROOT / 'data/tcr_native/dataset/t4_refbinder_blocklist.txt').read_text().splitlines())
    assert not t4_keys - blocklist, 'Current T4 test references missing from historical protection'
    return bindings, pairs, lookup, t4_keys, t4_stats


def repair(row, lookup):
    """Join identical existing fields; reject ambiguous or absent evidence."""
    if row.get('metadata', {}).get('source') != 'tcr_pmhc_fulllength':
        return row, []
    roles = observed_roles(row)
    if not {'tcr_alpha', 'tcr_beta'} <= roles:
        return row, []
    indices = {role: i for i, role in enumerate(row['chain_roles'])}
    a, _ = stored_core(row, indices['tcr_alpha'], 'tcr_alpha')
    if a:
        return row, []
    chains = dict(zip(row['chain_roles'], row['chains']))
    matches = lookup.get((chains['tcr_alpha'], chains['tcr_beta'], chains['peptide']), [])
    values = {r['cdr3a'] for r in matches if r['cdr3a']}
    assert len(values) == 1, 'Cannot restore alpha identifier uniquely'
    result = deepcopy(row)
    result.setdefault('identifiers', {})['cdr3a_core'] = next(iter(values))
    return result, sorted({r['record_id'] for r in matches})


def conflicts(row, pairs, t4_keys):
    cores = {}
    for i, role in enumerate(row['chain_roles']):
        if role in ('tcr_alpha', 'tcr_beta'):
            value, _ = stored_core(row, i, role)
            if value: cores[role] = value
    hits = []
    paired = {'tcr_alpha', 'tcr_beta'} <= observed_roles(row)
    if paired:
        assert len(cores) == 2, 'Unresolved observed-pair core fields'
        if cores['tcr_alpha'] + '|' + cores['tcr_beta'] in pairs:
            hits.append('exact_test_cdr3_pair')
    ep = row.get('identifiers', {}).get('epitope')
    if not ep:
        ep = next((s for role, s in zip(row['chain_roles'], row['chains']) if role == 'peptide'), '')
    beta = cores.get('tcr_beta') or row.get('identifiers', {}).get('cdr3b_core')
    if beta and ep and beta + '|' + ep in t4_keys:
        hits.append('exact_t4_beta_epitope_test_key')
    return hits


def build(dataset):
    assert not dataset.exists()
    bindings, pairs, lookup, t4, t4_stats = inputs()
    parent = read(PARENT / 'dataset_manifest.json')
    stage = Path(tempfile.mkdtemp(prefix='.' + dataset.name + '.', dir=dataset.parent))
    counts, shards = defaultdict(Counter), []
    with (stage / 'repair_ledger.jsonl').open('w') as ledger:
        for split, block in parent['splits'].items():
            for s in block['shards']:
                target = stage / s['path']; target.parent.mkdir(exist_ok=True)
                h, oh, n, kept = hashlib.sha256(), hashlib.sha256(), 0, 0
                c = counts[split + '/' + s['source']]
                with (PARENT / s['path']).open('rb') as f, target.open('wb') as g:
                    for raw in f:
                        h.update(raw)
                        if not raw.strip(): continue
                        row = json.loads(raw)
                        updated, ids = repair(row, lookup)
                        hits = conflicts(updated, pairs, t4) if s['source'].startswith('tcr') or s['source'] in ('ots', 'trait') else []
                        c['input'] += 1
                        if ids: c['repaired'] += 1
                        if hits: c['removed'] += 1
                        else:
                            data = (json.dumps(updated, ensure_ascii=False, separators=(',', ':')) + '\n').encode() if ids else raw
                            g.write(data); oh.update(data); kept += 1; c['retained'] += 1
                        if ids or hits:
                            ledger.write(json.dumps(dict(shard=s['path'], source_row=n,
                                output_row=None if hits else kept-1, upstream_record_ids=ids, reasons=hits))+'\n')
                        n += 1
                assert n == s['records'] and h.hexdigest() == s['sha256']
                if kept:
                    shards.append(dict(path=s['path'], source=s['source'], split=split, records=kept, sha256=oh.hexdigest()))
                else: target.unlink()
                print(json.dumps(dict(shard=s['path'], counts=c)), flush=True)
    totals = sum(counts.values(), Counter()); assert totals['repaired'] == 8765
    m = deepcopy(parent)
    m.update(dataset_version=dataset.name, created_at=datetime.now(timezone.utc).isoformat(), training_ready=False,
        splits={sp: dict(records=sum(s['records'] for s in shards if s['split']==sp), shards=[s for s in shards if s['split']==sp]) for sp in parent['splits']},
        decontamination=dict(status='audit_pending', scope_version=SCOPE, pending=['independent_final_verification']))
    # Older transform metadata describes the ancestor, not the current row numbering.
    for key in ('test_pair_cleanup', 'mhc_update'):
        if key in m: m.setdefault('inherited_transforms', {})[key] = m.pop(key)
    m['scope_finalization'] = dict(scope_version=SCOPE, parent=str(PARENT), parent_manifest_sha256=PARENT_SHA,
        user_excluded_audit_items=WAIVERS, upstream=str(UPSTREAM), upstream_sha256=UPSTREAM_SHA,
        repair_ledger_sha256=sha(stage/'repair_ledger.jsonl'), script_sha256=sha(Path(__file__)),
        t4_rule='reference beta-core plus epitope exact; support-only train_binders excluded',
        paired_rule='two observed chains, same-test-record exact CDR3alpha/CDR3beta',
        limitations=['No new paired near-similarity or biological novelty claim.',
                     'Legacy exclusion rules inherited; user-waived tasks are not certified.'])
    dump(stage/'dataset_manifest.json', m)
    dump(stage/'construction_report.json', dict(counts=counts, totals=totals, bindings=bindings, t4_reference_coverage=t4_stats,
        t4_missing_from_historical_blocklist=0, manifest_sha256=sha(stage/'dataset_manifest.json')))
    assert not dataset.exists(); os.rename(stage, dataset)
    print(json.dumps(dict(dataset=str(dataset), totals=totals)), flush=True)


def verify(dataset):
    """Independent source/output correspondence and complete declared-key audit."""
    manifest = read(dataset/'dataset_manifest.json')
    assert manifest['scope_finalization']['scope_version'] == SCOPE
    assert manifest['scope_finalization']['script_sha256'] == sha(Path(__file__))
    report = read(dataset/'construction_report.json')
    assert report['manifest_sha256'] == sha(dataset/'dataset_manifest.json')
    bindings, pairs, lookup, t4, t4_stats = inputs()
    assert bindings == report['bindings']
    outputs = {s['path']:s for b in manifest['splits'].values() for s in b['shards']}
    ledger = {}
    assert sha(dataset/'repair_ledger.jsonl') == manifest['scope_finalization']['repair_ledger_sha256']
    with (dataset/'repair_ledger.jsonl').open() as f:
        for line in f:
            item=json.loads(line); key=(item['shard'],item['source_row']); assert key not in ledger; ledger[key]=item
    families, totals = {}, Counter()
    parent = read(PARENT/'dataset_manifest.json')
    for split, block in parent['splits'].items():
        for s in block['shards']:
            out=(dataset/s['path']).open('rb') if s['path'] in outputs else None
            h, oh, n, kept = hashlib.sha256(), hashlib.sha256(), 0, 0
            with (PARENT/s['path']).open('rb') as f:
                for raw in f:
                    h.update(raw)
                    if not raw.strip(): continue
                    row=json.loads(raw); event=ledger.pop((s['path'],n),None)
                    # Independently derive required repair from absence of existing alpha fields.
                    obs=observed_roles(row); ix={r:i for i,r in enumerate(row['chain_roles'])}
                    need=(row.get('metadata',{}).get('source')=='tcr_pmhc_fulllength' and
                          {'tcr_alpha','tcr_beta'}<=obs and not stored_core(row,ix['tcr_alpha'],'tcr_alpha')[0])
                    expected=deepcopy(row) if need else row
                    if need:
                        chain=dict(zip(row['chain_roles'],row['chains']))
                        refs=lookup[(chain['tcr_alpha'],chain['tcr_beta'],chain['peptide'])]
                        values={r['cdr3a'] for r in refs if r['cdr3a']}; assert len(values)==1
                        expected.setdefault('identifiers',{})['cdr3a_core']=next(iter(values))
                        assert event and event['upstream_record_ids']==sorted({r['record_id'] for r in refs})
                        totals['repaired']+=1
                    hit=conflicts(expected,pairs,t4) if s['source'].startswith('tcr') or s['source'] in ('ots','trait') else []
                    if hit:
                        assert event and event['reasons']==hit and event['output_row'] is None
                        totals['removed']+=1
                    else:
                        assert out is not None
                        actual=out.readline(); assert actual and json.loads(actual)==expected
                        if not need: assert actual==raw
                        if event: assert event['output_row']==kept and event['reasons']==[]
                        oh.update(actual); kept+=1; totals['retained']+=1
                        family=expected['metadata']['v7_family_group']; assert family
                        assert families.setdefault(family,split)==split
                        if {'tcr_alpha','tcr_beta'}<=obs: totals['observed_pairs_checked']+=1
                    n+=1
            assert n==s['records'] and h.hexdigest()==s['sha256']
            if out:
                assert not out.read(1); out.close()
                assert kept==outputs[s['path']]['records'] and oh.hexdigest()==outputs[s['path']]['sha256']
            else: assert kept==0
            print(json.dumps(dict(verified=s['path'], retained=kept)),flush=True)
    assert not ledger and totals['repaired']==8765
    assert totals['retained']==report['totals']['retained'] and totals['removed']==report['totals'].get('removed',0)
    assert all(b['records'] and all(s['records'] for s in b['shards']) for b in manifest['splits'].values())
    result=dict(status='passed', scope_version=SCOPE, totals=totals, forbidden_overlap_rows=0,
        family_split_overlap=0, unresolved_observed_pair_fields=0, t4_reference_coverage=t4_stats,
        t4_missing_from_historical_blocklist=0, user_excluded_audit_items=WAIVERS,
        inherited_nonpaired_filter_evidence=bindings,
        limitations=manifest['scope_finalization']['limitations'], script_sha256=sha(Path(__file__)))
    # Declare completion only after both actual passes; bind certificates to the final manifest.
    manifest['training_ready']=True
    manifest['decontamination'].update(status='passed',pending=[],training_ready=True)
    dump(dataset/'dataset_manifest.json',manifest)
    final_sha=sha(dataset/'dataset_manifest.json'); result['dataset_manifest_sha256']=final_sha
    dump(dataset/'final_verification.json',result)
    evidence={str(dataset/n):sha(dataset/n) for n in ('construction_report.json','repair_ledger.jsonl','final_verification.json')}
    evidence.update(bindings)
    common=dict(status='passed',pending=[],scope_version=SCOPE,forbidden_overlap_rows=0,
        family_split_overlap=0,user_excluded_audit_items=WAIVERS,evidence_files=evidence,
        scope='Operational test-only exact observed-pair policy, inherited nonpaired filters and T4 reference keys; explicit user exclusions.')
    dump(dataset/'audit_certificate.json',dict(common,manifest_sha256=final_sha,
        rows={s:b['records'] for s,b in manifest['splits'].items()}))
    dump(dataset/'downstream_scope_certificate.json',dict(common,dataset_manifest_sha256=final_sha,
        benchmark_inputs={str(p):sha(p) for p in required_test_paths()}))
    print(json.dumps(dict(status='passed',dataset=str(dataset),manifest_sha256=final_sha,totals=totals)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['build','verify'])
    p.add_argument('--dataset',type=Path,required=True);args=p.parse_args()
    (build if args.mode=='build' else verify)(args.dataset.absolute())
