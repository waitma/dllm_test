"""Plan or export a new, explicitly uncertified strict-union MHC candidate.

Activate pllm, then run from the repository root:
  python scripts/data/export_immune_expanded_clean.py --mode plan --report /tmp/expanded-plan.json
  python scripts/data/export_immune_expanded_clean.py --mode verify --report /tmp/expanded-verified.json
  python scripts/data/export_immune_expanded_clean.py --mode export --output-dir /new/candidate --report /tmp/export.json

Plan reads bound small metadata only. Verify/export stream all row hits and both
prepared versions and verify their SHA256 bindings; neither re-runs a sequence
similarity search. Export preserves each retained MHC row byte-for-byte and its
original split. Whole-family validation proposals are advisory, never applied.
No mode writes an audit certificate or declares a corpus ready for training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MHC = ROOT / 'data/prepared/immune_v7_mhc_sequence_v1'
DEFAULT_PLAN = ROOT / 'data/dedup/tcr_exposure_20260930_r2/cleanup_plan.json'
MHC_SHA = 'e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94'
PLAN_SHA = '7df1a147b4263835c686c74c5969808faad8ccb5609d1caf4385057a64f9b13f'
POLICY = 'strict_union_all_declared_exact_and_edit80_v1'
SCOPE = 'tcr_all_support_query_uptcr_v1'
INDEXING = 'zero-based nonblank record number within named shard'
REASONS = frozenset('exact_' + name for name in
    ('cdr3a', 'cdr3b', 'fva', 'fvb', 'peptide', 'ab', 'ap', 'bp', 'abp', 'bpmhc', 'epitope_in_antigen')) | frozenset(
    ('edit80_cdr3a', 'edit80_cdr3b', 'edit80_peptide'))
PENDING = ['role_specific_annotation_and_near_neighbor_review_pending',
           'independent_parent_and_expanded_scope_rescans_required',
           'new_manifest_requires_two_independent_audit_certificates',
           'upstream_parent_antigen_provenance_unresolved']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def relative_path(value):
    require(isinstance(value, str) and value, 'Empty shard path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and str(path) == value,
            'Unsafe or noncanonical shard path: ' + value)
    require(len(path.parts) == 2 and path.parts[0] in ('train', 'valid') and path.suffix == '.jsonl',
            'Expected split/shard.jsonl path: ' + value)
    return value


def manifest_shards(manifest):
    require(manifest.get('schema_version') == 'immune_llada.semantic.v1' and
            manifest.get('format') == 'jsonl', 'Unsupported prepared manifest')
    require(set(manifest.get('splits', {})) == {'train', 'valid'}, 'Expected train and valid splits')
    entries, seen = [], set()
    for split, block in manifest['splits'].items():
        total = 0
        for shard in block['shards']:
            rel = relative_path(shard['path'])
            require(rel not in seen and rel.startswith(split + '/'), 'Duplicate/wrong split shard')
            require(shard.get('split') == split and isinstance(shard.get('source'), str) and shard['source'],
                    'Invalid manifest shard source/split')
            require(integer(shard['records']), 'Invalid shard count')
            require(isinstance(shard.get('sha256'), str) and len(shard['sha256']) == 64, 'Missing shard SHA')
            entries.append(shard)
            seen.add(rel)
            total += shard['records']
        require(integer(block['records']) and total == block['records'], 'Manifest split count mismatch')
    return entries


@dataclass
class Evidence:
    mhc_dir: Path
    parent_dir: Path
    manifest: dict
    parent_manifest: dict
    plan: dict
    audit: dict
    shards: list
    parent_shards: dict
    bindings: dict
    verified: set
    hit_path: Path

    def bind(self, path, expected, *, verify=False):
        path = Path(path).resolve()
        require(isinstance(expected, str) and len(expected) == 64 and
                all(c in '0123456789abcdef' for c in expected), 'Invalid SHA256 binding')
        previous = self.bindings.setdefault(str(path), expected)
        require(previous == expected, 'Conflicting SHA256 bindings: ' + str(path))
        if verify and str(path) not in self.verified:
            require(sha(path) == expected, 'SHA256 mismatch: ' + str(path))
            self.verified.add(str(path))
        return path


def load_evidence(mhc_dir, cleanup_plan, *, mhc_manifest_sha256, cleanup_plan_sha256):
    """Check the binding graph; large bound content remains pending until scan."""
    mhc_dir, cleanup_plan = Path(mhc_dir).resolve(), Path(cleanup_plan).resolve()
    # Read only after checking the caller's pinned entry points.
    require(sha(cleanup_plan) == cleanup_plan_sha256, 'Cleanup plan SHA256 mismatch')
    require(sha(mhc_dir / 'dataset_manifest.json') == mhc_manifest_sha256, 'MHC manifest SHA256 mismatch')
    plan, manifest = read_json(cleanup_plan), read_json(mhc_dir / 'dataset_manifest.json')
    require(plan.get('status') == 'not_applied' and plan.get('row_indexing') == INDEXING,
            'Unsupported cleanup plan state or indexing')
    require(plan.get('rule') == 'union of exact core/FV/pair, edit80 core/peptide and exact epitope-in-antigen reasons; preserve official benchmark rows',
            'Cleanup plan is not the supported strict-union rule')
    update = manifest['mhc_update']
    require(update.get('policy') == 'sequence_first_mhc_v1', 'Wrong MHC policy')
    parent = Path(update['parent']).resolve()
    require(parent == Path(plan['source_dataset']).resolve(), 'Plan and MHC parents differ')
    require(update['parent_manifest_sha256'] == plan['source_manifest_sha256'], 'Parent SHA bindings differ')
    evidence = Evidence(mhc_dir, parent, manifest, {}, plan, {}, [], {}, {}, set(), Path(plan['row_hits']).resolve())
    evidence.bind(cleanup_plan, cleanup_plan_sha256, verify=True)
    evidence.bind(mhc_dir / 'dataset_manifest.json', mhc_manifest_sha256, verify=True)
    evidence.bind(parent / 'dataset_manifest.json', plan['source_manifest_sha256'], verify=True)
    evidence.parent_manifest = read_json(parent / 'dataset_manifest.json')
    report_path = evidence.bind(plan['audit_report'], plan['audit_report_sha256'], verify=True)
    audit = evidence.audit = read_json(report_path)
    require(audit.get('scope_version') == SCOPE and audit.get('status') in ('overlap_found', 'review_required'),
            'Unsupported expanded audit scope/status')
    require(audit['dataset_manifest_sha256'] == plan['source_manifest_sha256'], 'Audit belongs to another parent')
    require(audit['row_hits_sha256'] == plan['row_hits_sha256'], 'Row-hit SHA bindings differ')
    cohorts = plan['cohort_bit_order']
    require(cohorts == audit['cohorts'] and cohorts and len(cohorts) == len(set(cohorts)), 'Cohort bit order mismatch')
    evidence.bind(evidence.hit_path, plan['row_hits_sha256'])
    registry_path = evidence.bind(report_path.parent.parent / 'registry.json', audit['registry_sha256'], verify=True)
    registry = read_json(registry_path)
    require(registry.get('scope_version') == SCOPE and
            set(item['cohort'] for item in registry['inputs']) == set(cohorts), 'Registry cohorts differ')
    for item in registry['inputs']:
        evidence.bind(item['path'], item['sha256'])
    evidence.bind(registry_path.parent / 'banks.json', registry['banks_sha256'])
    evidence.bind(ROOT / 'scripts/data/audit_tcr_exposure.py', registry['script_sha256'])
    source_report_path = evidence.bind(update['source_report'], update['source_report_sha256'], verify=True)
    source_report = read_json(source_report_path)
    require(source_report['policy'] == update['policy'] and
            source_report['reference']['sha256'] == update['reference_sha256'], 'MHC reference binding mismatch')
    for key in ('reference', 'full_source'):
        evidence.bind(source_report[key]['path'], source_report[key]['sha256'])
    for item in source_report['files'].values():
        for key in ('input', 'output'):
            evidence.bind(item[key], item[key + '_sha256'])
    evidence.shards = manifest_shards(manifest)
    parent_shards = manifest_shards(evidence.parent_manifest)
    require([(s['path'], s['source'], s['split'], s['records']) for s in evidence.shards] ==
            [(s['path'], s['source'], s['split'], s['records']) for s in parent_shards],
            'MHC shard membership/order/counts differ from parent')
    evidence.parent_shards = {s['path']: s for s in parent_shards}
    audit_shards = {str(Path(item['path']).resolve()): item for item in audit['shards']}
    require(len(audit_shards) == len(parent_shards) == len(audit['shards']), 'Audit shard inventory mismatch')
    for current, old in zip(evidence.shards, parent_shards):
        parent_path = (parent / old['path']).resolve()
        item = audit_shards.pop(str(parent_path), None)
        require(item is not None and item['sha256'] == old['sha256'] and item['rows'] == old['records'],
                'Audit/parent shard binding mismatch: ' + old['path'])
        require(current.get('parent_sha256', old['sha256']) == old['sha256'], 'MHC parent shard SHA mismatch')
        evidence.bind(parent_path, old['sha256'])
        evidence.bind(mhc_dir / current['path'], current['sha256'])
    return evidence


def counts_from_audit(evidence):
    totals, result = evidence.audit['totals'], {}
    for shard in evidence.shards:
        key = shard['split'] + '/' + shard['source']
        result.setdefault(key, {'input_rows': 0})['input_rows'] += shard['records']
    for key, value in result.items():
        removed = totals.get(key + '/flagged_rows', 0)
        require(totals.get(key + '/rows') == value['input_rows'] and integer(removed) and
                removed <= value['input_rows'], 'Audit source/split counts mismatch: ' + key)
        value.update(removed_rows=removed, retained_rows=value['input_rows'] - removed)
    return result


def plan_report(evidence, min_valid_rows=100):
    require(integer(min_valid_rows, 1), 'min_valid_rows must be positive')
    counts = counts_from_audit(evidence)
    scarce = {key.split('/', 1)[1]: value['retained_rows'] for key, value in counts.items()
              if key.startswith('valid/') and value['retained_rows'] < min_valid_rows}
    return dict(schema='immune.expanded_candidate_export.v1', mode='plan',
        status='estimated_not_row_verified', policy=POLICY, training_ready=False, pending=list(PENDING),
        role_review_unresolved_observations_at_implementation=80,
        counts=counts, total_retained_rows=sum(v['retained_rows'] for v in counts.values()),
        validation=dict(min_valid_rows=min_valid_rows, threshold_is_advisory=True, scarce_sources=scarce,
            family_feasibility='requires_verified_row_scan', split_reassignment_applied=False),
        cohort_bit_order=evidence.plan['cohort_bit_order'], row_indexing=INDEXING,
        bindings=dict(evidence.bindings), verified_binding_paths=sorted(evidence.verified),
        pending_binding_paths=sorted(set(evidence.bindings) - evidence.verified),
        note='Plan reuses bound prior audit totals; it does not verify row hits, shard bytes, or family membership.')


class HashedRows:
    def __init__(self, path):
        self.path, self.digest, self.rows = Path(path), hashlib.sha256(), 0

    def __iter__(self):
        with self.path.open('rb') as handle:
            for line in handle:
                self.digest.update(line)
                if line.strip():
                    self.rows += 1
                    yield line, json.loads(line)

    def check(self, expected, count=None):
        require(self.digest.hexdigest() == expected, 'SHA256 mismatch: ' + str(self.path))
        if count is not None:
            require(self.rows == count, 'Row count mismatch: ' + str(self.path))


def non_mhc_projection(row):
    """Same invariant as update_mhc_inputs.non_mhc, plus unknown top-level fields."""
    require(isinstance(row.get('chains'), list) and isinstance(row.get('chain_roles'), list) and
            len(row['chains']) == len(row['chain_roles']), 'Malformed chains')
    result = {k: v for k, v in row.items() if k not in ('chains', 'chain_roles', 'regions', 'chain_metadata', 'metadata', 'task_type')}
    result['metadata'] = {k: v for k, v in row.get('metadata', {}).items() if k != 'mhc_update'}
    result['non_mhc'] = [[role, seq, row.get('regions', {}).get(str(i), {}),
                          row.get('chain_metadata', {}).get(str(i), {})]
                         for i, (role, seq) in enumerate(zip(row['chain_roles'], row['chains'])) if role != 'mhc']
    return result


def checked_hits(evidence, reader):
    order = {shard['path']: (i, shard['records']) for i, shard in enumerate(evidence.shards)}
    previous = (-1, -1)
    for _, hit in reader:
        require(set(hit) == {'shard', 'row', 'reasons'} and hit['shard'] in order, 'Malformed/unknown row hit')
        index, count = order[hit['shard']]
        require(integer(hit['row']) and hit['row'] < count, 'Row-hit index out of range')
        key = (index, hit['row'])
        require(key > previous, 'Duplicate or out-of-order row hit')
        previous = key
        reasons = hit['reasons']
        require(isinstance(reasons, dict) and reasons and set(reasons) <= REASONS, 'Unknown/empty removal reasons')
        require(all(integer(mask, 1) and mask < 2 ** len(evidence.plan['cohort_bit_order']) for mask in reasons.values()),
                'Invalid cohort bitmask')
        yield key, hit


def family_report(db, counts, min_valid_rows):
    """Construct an advisory solution using complete families across all sources."""
    require(db.execute('SELECT family FROM families GROUP BY family HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone() is None,
            'Original family crosses train/valid; cannot inherit a clean split')
    sources = sorted({key.split('/', 1)[1] for key in counts})
    train = {s: counts.get('train/' + s, {}).get('retained_rows', 0) for s in sources}
    valid = {s: counts.get('valid/' + s, {}).get('retained_rows', 0) for s in sources}
    details = {}
    for source in sources:
        ntrain, nvalid = (db.execute('SELECT COUNT(*) FROM families WHERE source=? AND split=? AND retained>0',
                                    (source, split)).fetchone()[0] for split in ('train', 'valid'))
        details[source] = dict(retained_train_rows=train[source], retained_valid_rows=valid[source],
            retained_train_families=ntrain, retained_valid_families=nvalid,
            deficit=max(0, min_valid_rows - valid[source]))
    moves, moved = [], set()
    # Deterministic constructive example; failure is not an infeasibility proof.
    for source in sorted(sources, key=lambda s: (train[s], s)):
        if valid[source] >= min_valid_rows:
            continue
        rows = db.execute('SELECT family,retained FROM families WHERE source=? AND split="train" AND retained>0 ORDER BY retained,family', (source,))
        for family, _ in rows:
            if family in moved:
                continue
            group = dict(db.execute('SELECT source,retained FROM families WHERE family=? AND split="train" AND retained>0', (family,)))
            if any(train[s] - n < 1 for s, n in group.items()):
                continue
            moves.append(dict(family_group=family, source_rows=group, total_rows=sum(group.values())))
            moved.add(family)
            for s, n in group.items():
                train[s] -= n
                valid[s] += n
            if valid[source] >= min_valid_rows:
                break
    feasible = all(valid[s] >= min_valid_rows and train[s] >= 1 for s in sources)
    return dict(min_valid_rows=min_valid_rows, threshold_is_advisory=True, sources=details,
        family_split_overlap=0, split_reassignment_applied=False,
        feasibility='constructive_whole_family_example_found' if feasible else 'no_constructive_solution_not_a_proof_of_infeasibility',
        proposed_whole_family_moves=moves, hypothetical_counts={s: dict(train=train[s], valid=valid[s]) for s in sources},
        note='Advisory only. Moving a family means all its retained rows in every source; export preserves original splits. Counts alone do not establish useful validation diversity.')


def verify_or_export(evidence, *, min_valid_rows=100, output_dir=None, scratch_dir=None, progress=None):
    """Full bound scan, optionally publish new filtered shards plus row mapping."""
    report = plan_report(evidence, min_valid_rows)
    output_dir = Path(output_dir).absolute() if output_dir is not None else None
    if output_dir is not None:
        require(not output_dir.exists() and not output_dir.is_symlink(), 'Candidate directory already exists')
        protected = (evidence.mhc_dir, evidence.parent_dir)
        require(all(not output_dir.resolve().is_relative_to(path) and not path.is_relative_to(output_dir.resolve()) for path in protected),
                'Candidate must be separate from both original datasets')
        require(output_dir.parent.is_dir(), 'Candidate parent directory must already exist')
    dataset_paths = {str((base / s['path']).resolve()) for base in (evidence.parent_dir, evidence.mhc_dir) for s in evidence.shards}
    # Validate non-row dependencies, including benchmark and MHC provenance. Row
    # files are hashed during the scan itself instead of reading them twice.
    for path, expected in evidence.bindings.items():
        if path not in dataset_paths and path != str(evidence.hit_path):
            evidence.bind(path, expected, verify=True)
    with tempfile.TemporaryDirectory(prefix='immune-expanded-families-', dir=scratch_dir) as temporary:
        db = sqlite3.connect(str(Path(temporary) / 'families.sqlite'))
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('CREATE TABLE families(family TEXT, split TEXT, source TEXT, input INTEGER, retained INTEGER, PRIMARY KEY(family,split,source)) WITHOUT ROWID')
        stage = Path(tempfile.mkdtemp(prefix='.' + output_dir.name + '.staging-', dir=output_dir.parent)) if output_dir else None
        ledger = (stage / 'row_mapping.jsonl').open('w') if stage else None
        staged_shards, actual, family_buffer = [], {}, Counter()
        reason_counts, cohorts = Counter(), evidence.plan['cohort_bit_order']
        reader = HashedRows(evidence.hit_path)
        hits = checked_hits(evidence, reader)
        try:
            next_hit = next(hits, None)
            from dllm.pipelines.immune_llada.data.preprocessing.validators import validate_prepared_row
            for shard_index, shard in enumerate(evidence.shards):
                relative = shard['path']
                source, split = shard['source'], shard['split']
                key = split + '/' + source
                values = actual.setdefault(key, dict(input_rows=0, removed_rows=0, retained_rows=0, exact_reason_rows=0, near_only_rows=0))
                old = evidence.parent_shards[relative]
                old_reader, new_reader = HashedRows(evidence.parent_dir / relative), HashedRows(evidence.mhc_dir / relative)
                same = old_reader.path.resolve() == new_reader.path.resolve()
                pairs = ((item, item) for item in new_reader) if same else zip_longest(old_reader, new_reader)
                out_path = stage / relative if stage else None
                if out_path:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                output = out_path.open('wb') if out_path else None
                output_hash, retained = hashlib.sha256(), 0
                try:
                    for index, pair in enumerate(pairs):
                        before, after = pair
                        require(before is not None and after is not None, 'MHC row membership changed: ' + relative)
                        raw, row = after
                        require(non_mhc_projection(before[1]) == non_mhc_projection(row),
                                f'Non-MHC row/order changed: {relative}:{index}')
                        require(row.get('split') == split, 'Record split differs from manifest')
                        if row.get('task_type') != before[1].get('task_type'):
                            require(row.get('task_type') == ('tcr_pmhc' if 'mhc' in row['chain_roles'] else 'tcr_epitope') and
                                    before[1].get('task_type') in ('tcr_pmhc', 'tcr_epitope'), 'Unexpected task_type change')
                        validate_prepared_row(row)
                        family = row.get('metadata', {}).get('v7_family_group')
                        require(isinstance(family, str) and family, 'Missing v7_family_group')
                        hit = next_hit[1] if next_hit is not None and next_hit[0] == (shard_index, index) else None
                        if hit is not None:
                            next_hit = next(hits, None)
                        values['input_rows'] += 1
                        family_buffer[(family, split, source, 'input')] += 1
                        if hit is None:
                            new_index = retained
                            retained += 1
                            values['retained_rows'] += 1
                            family_buffer[(family, split, source, 'retained')] += 1
                            output_hash.update(raw if raw.endswith(b'\n') else raw + b'\n')
                            if output:
                                output.write(raw if raw.endswith(b'\n') else raw + b'\n')
                        else:
                            new_index = None
                            values['removed_rows'] += 1
                            category = 'exact_reason_rows' if any(r.startswith('exact_') for r in hit['reasons']) else 'near_only_rows'
                            values[category] += 1
                            reason_counts.update(hit['reasons'].keys())
                        if ledger:
                            mapping = dict(old=dict(shard=relative, row=index), new=dict(shard=relative, row=new_index) if hit is None else None,
                                source_group=source, split=split, family_group=family, reasons=hit['reasons'] if hit else {})
                            ledger.write(json.dumps(mapping, ensure_ascii=False, separators=(',', ':')) + '\n')
                        if len(family_buffer) >= 10000:
                            flush_families(db, family_buffer)
                    new_reader.check(shard['sha256'], shard['records'])
                    if same:
                        new_reader.check(old['sha256'], old['records'])
                    else:
                        old_reader.check(old['sha256'], old['records'])
                    evidence.verified.update((str(old_reader.path.resolve()), str(new_reader.path.resolve())))
                finally:
                    if output:
                        output.close()
                if retained:
                    staged_shards.append(dict(path=relative, source=source, split=split, records=retained,
                        sha256=output_hash.hexdigest(), input_shard_sha256=shard['sha256'], parent_shard_sha256=old['sha256']))
                elif out_path:
                    out_path.unlink()
                if progress:
                    progress(dict(shard=relative, scanned_rows=shard['records'], retained_rows=retained,
                                  shards_finished=shard_index + 1, shards_total=len(evidence.shards)))
            require(next_hit is None, 'Unconsumed row hits')
            reader.check(evidence.plan['row_hits_sha256'])
            evidence.verified.add(str(evidence.hit_path))
            flush_families(db, family_buffer)
            for key, expected in report['counts'].items():
                require(all(actual[key][name] == value for name, value in expected.items()), 'Scanned removal counts differ from audit: ' + key)
            report.update(mode='export' if stage else 'verify', status='candidate_exported_audit_pending' if stage else 'verified_selection_audit_pending',
                counts=actual, removed_reason_counts=dict(reason_counts), validation=family_report(db, actual, min_valid_rows),
                pending_binding_paths=[], verified_binding_paths=sorted(evidence.verified),
                note='All selected input bindings and row/order invariants verified; no new similarity audit or training certificate was performed.')
            require(set(evidence.bindings) == evidence.verified, 'Unverified input bindings remain')
            # Recheck the small entry points so a concurrent manifest/plan edit
            # cannot be hidden by already-open shard readers.
            for path in (evidence.mhc_dir / 'dataset_manifest.json', evidence.parent_dir / 'dataset_manifest.json', Path(evidence.plan['audit_report'])):
                require(sha(path) == evidence.bindings[str(path.resolve())], 'Input metadata changed during scan')
            if stage:
                ledger.close()
                ledger = None
                manifest = deepcopy(evidence.manifest)
                manifest.update(dataset_version=output_dir.name, created_at=datetime.now(timezone.utc).isoformat(),
                    training_ready=False,
                    splits={split: dict(records=sum(s['records'] for s in staged_shards if s['split'] == split),
                        shards=[s for s in staged_shards if s['split'] == split]) for split in evidence.manifest['splits']},
                    decontamination=dict(status='audit_pending', training_ready=False, pending=list(PENDING)),
                    expanded_cleanup=dict(policy=POLICY, input_mhc_directory=str(evidence.mhc_dir),
                        input_mhc_manifest_sha256=evidence.bindings[str((evidence.mhc_dir / 'dataset_manifest.json').resolve())],
                        parent_directory=str(evidence.parent_dir), parent_manifest_sha256=evidence.plan['source_manifest_sha256'],
                        audit_report_sha256=evidence.plan['audit_report_sha256'], row_hits_sha256=evidence.plan['row_hits_sha256'],
                        cohort_bit_order=cohorts, row_indexing=INDEXING, split_reassignment_applied=False,
                        row_mapping_sha256=sha(stage / 'row_mapping.jsonl'), exporter_sha256=sha(Path(__file__))))
                write_json(stage / 'dataset_manifest.json', manifest)
                report.update(output_directory=str(output_dir), manifest_sha256=sha(stage / 'dataset_manifest.json'),
                              row_mapping_sha256=manifest['expanded_cleanup']['row_mapping_sha256'])
                write_json(stage / 'cleaning_report.json', report)
                # An exclusive sibling lock prevents two cooperating exporters
                # from racing; publish never merges into an existing directory.
                lock = output_dir.parent / ('.' + output_dir.name + '.publish.lock')
                with lock.open('x'):
                    try:
                        require(not output_dir.exists() and not output_dir.is_symlink(), 'Candidate directory appeared during scan')
                        os.rename(stage, output_dir)
                        stage = None
                    finally:
                        lock.unlink()
            return report
        finally:
            if ledger:
                ledger.close()
            db.close()
            if stage:
                shutil.rmtree(stage)


def flush_families(db, buffer):
    groups = {(family, split, source) for family, split, source, _ in buffer}
    db.executemany('INSERT INTO families VALUES(?,?,?,?,?) ON CONFLICT(family,split,source) DO UPDATE SET input=input+excluded.input,retained=retained+excluded.retained',
        [(family, split, source, buffer[(family, split, source, 'input')], buffer[(family, split, source, 'retained')])
         for family, split, source in groups])
    buffer.clear()
    db.commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan', 'verify', 'export'), default='plan')
    parser.add_argument('--mhc-dir', type=Path, default=DEFAULT_MHC)
    parser.add_argument('--cleanup-plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--mhc-manifest-sha256', default=MHC_SHA)
    parser.add_argument('--cleanup-plan-sha256', default=PLAN_SHA)
    parser.add_argument('--policy', choices=(POLICY,), default=POLICY)
    parser.add_argument('--min-valid-rows', type=int, default=100)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--scratch-dir', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if (args.mode == 'export') != (args.output_dir is not None):
        parser.error('--output-dir is required only for mode export')
    require(not args.report.exists() and not args.report.is_symlink(), 'Report already exists')
    evidence = load_evidence(args.mhc_dir, args.cleanup_plan,
        mhc_manifest_sha256=args.mhc_manifest_sha256, cleanup_plan_sha256=args.cleanup_plan_sha256)
    require(not any(args.report.resolve().is_relative_to(path) for path in (evidence.parent_dir, evidence.mhc_dir)),
            'Report must not modify an original dataset')
    report = plan_report(evidence, args.min_valid_rows) if args.mode == 'plan' else verify_or_export(
        evidence, min_valid_rows=args.min_valid_rows, output_dir=args.output_dir, scratch_dir=args.scratch_dir,
        progress=lambda value: print(json.dumps(value), file=sys.stderr, flush=True))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], training_ready=False, report=str(args.report),
                         counts=report['counts'], validation=report['validation']), ensure_ascii=False))


if __name__ == '__main__':
    main()
