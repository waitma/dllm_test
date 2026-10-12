"""Audit the release using existing fields, masks and frozen evidence only.

Activate pllm; run from the repository root:
  python -B scripts/data/audit_test_pair_release.py --output /absolute/new/report.json

No new biological annotation, similarity search, or training certification.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/prepared/immune_v7_test_pair_exact_20261002'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def observed_roles(row):
    result = set()
    for i, (role, seq) in enumerate(zip(row['chain_roles'], row['chains'])):
        mask = row.get('chain_metadata', {}).get(str(i), {}).get('synthetic_residue_mask')
        if mask is not None:
            if len(mask) != len(seq) or any(x not in (0, 1) for x in mask):
                raise ValueError('Malformed synthetic residue mask')
            if not any(x == 0 for x in mask):
                continue
        if seq:
            result.add(role)
    return result


def audit(output):
    assert not output.exists()
    m = read(DATA / 'dataset_manifest.json')
    export = read(DATA / 'cleaning_report.json')
    selection = read(DATA / 'selection_verification.json')
    manifest_sha = sha(DATA / 'dataset_manifest.json')
    assert manifest_sha == export['manifest_sha256'] == selection['manifest_sha256']
    for path, digest in export['bindings'].items():
        assert sha(path) == digest, path
    cfg = m['test_pair_cleanup']
    assert sha(DATA / 'removed_rows.jsonl') == cfg['removed_rows_sha256']
    names = cfg['cohort_bit_order']
    mask = cfg['test_cohort_mask']
    registry_path = ROOT / 'data/dedup/tcr_exposure_20260930_r2/registry.json'
    registry = read(registry_path)
    banks_path = registry_path.parent / 'banks.json'
    assert sha(banks_path) == registry['banks_sha256']
    banks = read(banks_path)
    pair_bank = set()
    for i, name in enumerate(names):
        if mask & (1 << i):
            pair_bank.update(banks[name]['ab'])
    issues = {x['cohort']: x['field_issues'] for x in registry['inputs']
              if mask & (1 << names.index(x['cohort'])) and x['field_issues']}
    removed = defaultdict(set)
    with (DATA / 'removed_rows.jsonl').open() as f:
        for line in f:
            x = json.loads(line)
            removed[x['shard']].add(x['row'])
    valid_families = set()
    for s in m['splits']['valid']['shards']:
        with (DATA / s['path']).open() as f:
            for line in f:
                if line.strip():
                    family = json.loads(line).get('metadata', {}).get('v7_family_group')
                    if family:
                        valid_families.add(family)
    counters = defaultdict(Counter)
    examples, overlap_families, files = [], set(), []
    plan = read(ROOT / 'data/dedup/tcr_exposure_20260930_r2/cleanup_plan.json')
    old_shards = read(Path(cfg['source']) / 'dataset_manifest.json')
    order = {s['path']: i for i, s in enumerate(
        s for b in old_shards['splits'].values() for s in b['shards'])}
    with Path(plan['row_hits']).open() as hf:
        hits = map(json.loads, hf)
        hit = next(hits, None)
        for split, block in m['splits'].items():
            for shard in block['shards']:
                rel = shard['path']
                c = counters[split + '/' + shard['source']]
                digest, n, original = hashlib.sha256(), 0, 0
                with (DATA / rel).open('rb') as f:
                    for raw in f:
                        digest.update(raw)
                        if not raw.strip():
                            continue
                        while original in removed[rel]:
                            original += 1
                        key = (order[rel], original)
                        while hit is not None and (order[hit['shard']], hit['row']) < key:
                            hit = next(hits, None)
                        reasons = hit['reasons'] if hit is not None and (order[hit['shard']], hit['row']) == key else {}
                        row = json.loads(raw)
                        assert row['split'] == split
                        c['rows'] += 1
                        family = row.get('metadata', {}).get('v7_family_group')
                        if not family:
                            c['missing_family_rows'] += 1
                        elif split == 'train' and family in valid_families:
                            c['family_cross_split_rows'] += 1
                            overlap_families.add(family)
                        declared = {'tcr_alpha', 'tcr_beta'} <= set(row['chain_roles'])
                        observed = observed_roles(row)
                        real_pair = {'tcr_alpha', 'tcr_beta'} <= observed
                        if declared:
                            c['declared_pair_rows'] += 1
                            c['observed_pair_rows' if real_pair else 'not_observed_pair_rows'] += 1
                            if not real_pair:
                                protected = {k: v & mask for k, v in reasons.items() if v & mask}
                                if protected:
                                    c['nonpair_retained_with_test_hits'] += 1
                                    if any(k.startswith('exact_') for k in protected):
                                        c['nonpair_retained_with_exact_test_hits'] += 1
                                    if len(examples) < 20:
                                        examples.append(dict(shard=rel, new_row=n, source_row=original,
                                                             observed_roles=sorted(observed), reasons=protected))
                            else:
                                cores = {}
                                for i, role in enumerate(row['chain_roles']):
                                    if role in ('tcr_alpha', 'tcr_beta'):
                                        value = row.get('regions', {}).get(str(i), {}).get('CDR3')
                                        if value:
                                            cores[role] = value
                                if len(cores) != 2:
                                    c['observed_pair_missing_explicit_cdr3_fields'] += 1
                                else:
                                    c['explicit_pair_checked_rows'] += 1
                                    if cores['tcr_alpha'] + '|' + cores['tcr_beta'] in pair_bank:
                                        c['explicit_pair_test_overlap_rows'] += 1
                        n += 1
                        original += 1
                assert n == shard['records'] and digest.hexdigest() == shard['sha256']
                files.append(dict(path=rel, rows=n, sha256=digest.hexdigest()))
                print(json.dumps(dict(shard=rel, counts=dict(c))), flush=True)
    totals = sum(counters.values(), Counter())
    result = dict(status='not_passed', created_at=datetime.now(timezone.utc).isoformat(),
        dataset=str(DATA), dataset_manifest_sha256=manifest_sha, totals=totals,
        counts=counters, verified_shards=files, verified_inherited_bindings=len(export['bindings']),
        family_cross_split_groups=len(overlap_families), field_issues_in_test_cohorts=issues,
        examples_without_sequences=examples, pair_bank_unique=len(pair_bank),
        blockers=['synthetic_placeholder_roles_misclassified_as_observed_pairs',
                  'inherited_H3_and_current_benchmark_coverage_incomplete',
                  'training_gate_still_requires_superseded_all_support_query_scope'],
        interpretation='Existing explicit CDR3 fields only; no inference for missing fields. Existing test hit masks quantify nonpair impact; no new similarity search.',
        script_sha256=sha(Path(__file__)), training_ready=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], totals=totals)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    audit(p.parse_args().output)
