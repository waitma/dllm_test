"""Publish V7 plus re-filtered original V5 nonbinding rows as an immutable V8.

Activate pllm, then run from /project_c20250601/251105016/project/dllm_test:
  USE_TF=0 python -B -m scripts.data.prepare_immune_v8 --stage all --workers 8
Stages candidates/match/export are restartable separately. Existing V7 shards
are hash-checked and linked unchanged; only explicit V5 negatives are restored.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from scripts.data import export_v7_no_negative_design as filters
from scripts.data.export_v7_binding_design import (
    audit_keys, upstream_lookup, unresolved_tcr, bank_index, reasons, save_pickle, read_pickle)
from scripts.data.rebuild_v7_from_v5 import SOURCE, WORK as V7_WORK, sha, write
from scripts.data.resize_v7_design_slots import overflows
from scripts.data.v7_rebuild_scope import SCOPE_DIR, verify as verify_scope
from scripts.research.bert_mlm_data import receptor_key

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_mhc_completed_20261015'
OUTPUT = ROOT / 'data/prepared/immune_v8_unbinding_slots167_135_20261026'
WORK = ROOT / 'output/immune_v8_unbinding_20261026'
PROBE_INDEX = ROOT / 'output/bert_mlm_v7_300k_20261016/data/index_test_clean.json'


def parent_gate(parent=PARENT):
    manifest = json.loads((parent / 'dataset_manifest.json').read_text())
    digest = sha(parent / 'dataset_manifest.json')
    for name in ('verification.json', 'independent_acceptance.json', 'reader_acceptance.json'):
        check = json.loads((parent / name).read_text())
        if check['status'] != 'passed' or check['dataset_manifest_sha256'] != digest:
            raise ValueError('V7 acceptance differs: ' + name)
    contract = manifest['reader_contract']
    if sha(parent / contract['config_path']) != contract['config_sha256']:
        raise ValueError('V7 config changed')
    reader = json.loads((parent / 'reader_acceptance.json').read_text())
    if reader['collator_sha256'] != sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/layout.py'):
        raise ValueError('V7 reader changed')
    return manifest


def held_probe_keys():
    index = json.loads(PROBE_INDEX.read_text())
    keys = set()
    for name in ('probe_train', 'probe_valid'):
        item = index['files'][name]
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen diagnostic probe changed')
        rows = [json.loads(line) for line in Path(item['path']).read_text().splitlines()]
        if len(rows) != item['records']:
            raise ValueError('Frozen probe count changed')
        if name == 'probe_valid':
            keys.update(row['receptor_key'] for row in rows)
    return keys


def collect_shard(shard):
    lookup = upstream_lookup()
    values, near = defaultdict(set), defaultdict(set)
    rows, counts = [], Counter()
    digest = hashlib.sha256()
    with (SOURCE / shard['path']).open('rb') as stream:
        for i, raw in enumerate(stream):
            digest.update(raw)
            row = json.loads(raw)
            counts['input'] += 1
            if row.get('labels', {}).get('relation') != 'nonbinding':
                continue
            counts['nonbinding'] += 1
            keys, _ = audit_keys(row, lookup)
            if unresolved_tcr(keys):
                counts['unresolved_tcr'] += 1
                continue
            rows.append((i, keys))
            for role, value in keys.items():
                if isinstance(value, str) and value:
                    values[role].add(value)
                    if not keys['paired_tcr'] and role in ('peptide', 'cdr3a', 'cdr3b'):
                        near[role].add(value)
    if counts['input'] != shard['records'] or digest.hexdigest() != shard['source_sha256']:
        raise ValueError('V5 source changed: ' + shard['path'])
    path = WORK / 'candidates' / (shard['path'].replace('/', '__') + '.pkl')
    save_pickle(path, dict(rows=rows, values=dict(values), near=dict(near)))
    return dict(shard=shard, counts=dict(counts), cache=str(path), sha256=sha(path))


def candidates(workers):
    parent_gate()
    verify_scope()
    held_probe_keys()
    inventory_path = V7_WORK / 'inventory.json'
    inventory = json.loads(inventory_path.read_text())
    if inventory['manifest_sha256'] != sha(SOURCE / 'dataset_manifest.json'):
        raise ValueError('V5 manifest changed')
    shards = [s for s in inventory['shards'] if s['labels'].get('nonbinding', 0)]
    WORK.mkdir(parents=True, exist_ok=True)
    if (WORK / 'candidates.json').exists():
        raise FileExistsError('Candidate receipt already exists; resume with --stage match/export')
    values, near, totals, results = defaultdict(set), defaultdict(set), Counter(), []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for item in pool.map(collect_shard, shards):
            data = read_pickle(Path(item['cache']))
            for key, seqs in data['values'].items():
                values[key].update(seqs)
            for key, seqs in data['near'].items():
                near[key].update(seqs)
            totals.update(item['counts'])
            results.append(item)
            print(json.dumps(dict(stage='candidates', shard=item['shard']['path'], **item['counts'])), flush=True)
    save_pickle(WORK / 'candidate_keys.pkl', dict(values=dict(values), near=dict(near)))
    write(WORK / 'candidates.json', dict(status='complete', shards=results, counts=dict(totals),
        keys_sha256=sha(WORK / 'candidate_keys.pkl'), registry_sha256=sha(SCOPE_DIR / 'registry.json'),
        parent_manifest_sha256=sha(PARENT / 'dataset_manifest.json'),
        source_manifest_sha256=sha(SOURCE / 'dataset_manifest.json'),
        probe_index_sha256=sha(PROBE_INDEX), inventory_sha256=sha(inventory_path)))


def verify_candidates():
    report = json.loads((WORK / 'candidates.json').read_text())
    for key, path in (('keys_sha256', WORK / 'candidate_keys.pkl'),
                      ('registry_sha256', SCOPE_DIR / 'registry.json'),
                      ('parent_manifest_sha256', PARENT / 'dataset_manifest.json'),
                      ('source_manifest_sha256', SOURCE / 'dataset_manifest.json'),
                      ('probe_index_sha256', PROBE_INDEX),
                      ('inventory_sha256', V7_WORK / 'inventory.json')):
        if sha(path) != report[key]:
            raise ValueError('V8 bound input changed: ' + str(path))
    verify_scope()
    return report


def match(workers):
    verify_candidates()
    # Reuse the existing exact/near/cluster algorithms in a separate cache root.
    filters.RUN = WORK
    data = read_pickle(WORK / 'candidate_keys.pkl')
    names, exact, t4 = bank_index()
    near, clusters, contained = {}, {}, {}
    for role in ('peptide', 'cdr3a', 'cdr3b'):
        sequences = data['near'].get(role, set())
        near[role] = filters.near_matches(role, sequences, exact.get(role, {}), workers) if sequences else {}
    for role, threshold in (('heavy', .95), ('h3', .8), ('antigen', .7)):
        sequences = data['values'].get(role, set())
        clusters[role] = filters.clustered_matches(role, sequences, exact.get(role, {}), threshold, workers) if sequences else {}
    for antigen in data['values'].get('antigen', ()):
        mask = 0
        for peptide, bits in exact.get('peptide', {}).items():
            if peptide in antigen:
                mask |= bits
        if mask:
            contained[antigen] = mask
    save_pickle(WORK / 'match_index.pkl', dict(names=names, exact=exact, t4=t4,
                near=near, clusters=clusters, contained=contained))
    write(WORK / 'matching.json', dict(status='complete', index_sha256=sha(WORK / 'match_index.pkl'),
        candidates_sha256=sha(WORK / 'candidates.json'), registry_sha256=sha(SCOPE_DIR / 'registry.json'),
        policy='V7 protected banks/thresholds; fresh negative-candidate + reference clustering'))


def init_export():
    global _INDEX, _LOOKUP, _HELD, _COLLATOR
    from dllm.pipelines.immune_llada.fixed_chain.unbinding_mlm import UnbindingMLMCollator
    from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
    _INDEX, _LOOKUP, _HELD = read_pickle(WORK / 'match_index.pkl'), upstream_lookup(), held_probe_keys()
    _COLLATOR = UnbindingMLMCollator(FixedChainConfig(**json.loads((PARENT / 'immune_config.json').read_text())))


def rejection(row, split):
    keys, _ = audit_keys(row, _LOOKUP)
    if unresolved_tcr(keys):
        return 'unresolved_tcr'
    if reasons(keys, _INDEX):
        return 'test_overlap'
    if overflows(row):
        return 'capacity_overflow'
    if split == 'train' and receptor_key(row) in _HELD:
        return 'probe_valid_overlap'
    return None


def export_shard(job):
    from dllm.pipelines.immune_llada.data.preprocessing.validators import validate_prepared_row
    stage, item = job
    shard = item['shard']
    if sha(item['cache']) != item['sha256']:
        raise ValueError('Candidate cache changed')
    relative = shard['split'] + '/unbind_' + Path(shard['path']).name
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    ledger_path = stage / 'audit' / (target.stem + '_' + shard['split'] + '.jsonl')
    counts, digest = Counter(), hashlib.sha256()
    with (SOURCE / shard['path']).open('rb') as source, target.open('wb') as out, ledger_path.open('w') as ledger:
        for i, raw in enumerate(source):
            digest.update(raw)
            counts['input'] += 1
            row = json.loads(raw)
            if row.get('labels', {}).get('relation') != 'nonbinding':
                continue
            counts['nonbinding'] += 1
            reason = rejection(row, shard['split'])
            if reason is None:
                validate_prepared_row(row)
                if row['split'] != shard['split']:
                    raise ValueError('Original split differs')
                # The actual reader is authoritative for max length and grammar.
                batch = _COLLATOR([row])
                if not batch['eligible_mask'].any() or (batch['eligible_mask'] & ~batch['explicit_relation_mask']).any():
                    raise ValueError('Negative row does not have relation-only targets')
                out.write(raw)
                counts['restored'] += 1
            else:
                counts[reason] += 1
            ledger.write(json.dumps(dict(source_row=i, decision=reason or 'restored',
                                         row_sha256=hashlib.sha256(raw).hexdigest())) + '\n')
    if digest.hexdigest() != shard['source_sha256'] or counts['input'] != shard['records']:
        raise ValueError('V5 source changed during export')
    # Independent source rescan: recompute decisions and compare every emitted byte.
    verified = 0
    with (SOURCE / shard['path']).open('rb') as source, target.open('rb') as out:
        for raw in source:
            row = json.loads(raw)
            if row.get('labels', {}).get('relation') == 'nonbinding' and rejection(row, shard['split']) is None:
                if out.readline() != raw:
                    raise ValueError('Negative export bytes/selection differ')
                verified += 1
        if out.read(1) or verified != counts['restored']:
            raise ValueError('Negative export record count differs')
    return dict(path=relative, split=shard['split'], source=shard['source'],
                records=verified, sha256=sha(target), counts=dict(counts),
                original_path=str(SOURCE / shard['path']), original_sha256=digest.hexdigest(),
                audit_path=str(ledger_path.relative_to(stage)), audit_sha256=sha(ledger_path))


def export(workers):
    candidates = verify_candidates()
    matching = json.loads((WORK / 'matching.json').read_text())
    if (matching['candidates_sha256'] != sha(WORK / 'candidates.json')
            or matching['index_sha256'] != sha(WORK / 'match_index.pkl')):
        raise ValueError('Negative matching cache changed')
    parent = parent_gate()
    stage = OUTPUT.with_name(OUTPUT.name + '.building')
    if OUTPUT.exists() or stage.exists():
        raise FileExistsError('Refusing to overwrite V8 release/build')
    (stage / 'audit').mkdir(parents=True)
    shutil.copyfile(PARENT / 'immune_config.json', stage / 'immune_config.json')
    results = []
    for split, block in parent['splits'].items():
        for shard in block['shards']:
            source = PARENT / shard['path']
            if sha(source) != shard['sha256']:
                raise ValueError('V7 shard changed: ' + str(source))
            target = stage / shard['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
            results.append({k: shard[k] for k in ('path', 'source', 'records', 'sha256')} | dict(split=split))
    totals = Counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_export) as pool:
        for item in pool.map(export_shard, [(stage, s) for s in candidates['shards']]):
            results.append(item)
            totals.update(item['counts'])
            print(json.dumps(dict(stage='export', shard=item['path'], **item['counts'])), flush=True)
    if totals['nonbinding'] != candidates['counts']['nonbinding'] or not totals['restored']:
        raise ValueError('Negative restoration coverage/count differs')
    manifest = {k: deepcopy(parent[k]) for k in ('format', 'schema_version', 'reader_contract')}
    manifest.update(dataset_version=OUTPUT.name, created_at=datetime.now(timezone.utc).isoformat(), splits={})
    for split in parent['splits']:
        items = [s for s in results if s['split'] == split and s['records']]
        by_source = Counter()
        for item in items:
            by_source[item['source']] += item['records']
        manifest['splits'][split] = dict(records=sum(s['records'] for s in items), shards=items,
                                        sources={k: dict(records=v) for k, v in by_source.items()})
    manifest['sources'] = sorted({s['source'] for s in results if s['records']})
    manifest['v8_restoration'] = dict(parent=str(PARENT), parent_manifest_sha256=sha(PARENT / 'dataset_manifest.json'),
        source=str(SOURCE), source_manifest_sha256=sha(SOURCE / 'dataset_manifest.json'),
        counts=dict(totals), work=str(WORK), matching_sha256=sha(WORK / 'matching.json'),
        registry_sha256=sha(SCOPE_DIR / 'registry.json'), probe_index=str(PROBE_INDEX),
        probe_index_sha256=sha(PROBE_INDEX),
        policy='V7 retained byte-identically; restore only explicit V5 nonbinding with original splits/multiplicity; '
               'rerun test filtering, capacity and probe-valid exclusion; no sequence/label synthesis',
        mhc='Existing V7 completed contexts preserved; restored negatives retain original source MHC context')
    write(stage / 'dataset_manifest.json', manifest)
    evidence = dict(status='passed', dataset_manifest_sha256=sha(stage / 'dataset_manifest.json'),
        parent_records={s: b['records'] for s, b in parent['splits'].items()},
        records={s: b['records'] for s, b in manifest['splits'].items()}, restored_nonbinding=totals['restored'],
        negative_selection_independently_recomputed=True, all_restored_bytes_identical=True,
        all_restored_rows_actual_collator=True, config_sha256=sha(stage / 'immune_config.json'),
        collator_sha256=sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/layout.py'),
        objective_sha256=sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/unbinding_mlm.py'))
    write(stage / 'verification.json', evidence)
    write(stage / 'reader_acceptance.json', evidence)
    write(stage / 'cleaning_report.json', dict(status='passed', totals=dict(totals), shards=results))
    stage.rename(OUTPUT)
    write(WORK / 'release.json', dict(**evidence, output=str(OUTPUT)))
    print(json.dumps(dict(stage='published', output=str(OUTPUT), **evidence)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('candidates', 'match', 'export', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    for name, operation in (('candidates', candidates), ('match', match), ('export', export)):
        if args.stage in ('all', name):
            operation(args.workers)
