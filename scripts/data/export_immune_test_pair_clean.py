"""Rebuild from the MHC parent using test-only, exact CDR3-pair exclusions.

Activate conda pllm; run from the repository root:
  python -B scripts/data/export_immune_test_pair_clean.py --output-dir /absolute/new/dataset

Reuses hash-bound, existing row-hit evidence; performs no new annotation or
similarity search. Paired TCR records require exact_ab from one test pair.
Nonpaired records retain the previous matching rules restricted to test cohorts.
An output-byte verification pass precedes publication. This is not a new
sequence audit, a full-chain uniqueness claim, or a training certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from itertools import zip_longest
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data import export_immune_expanded_clean as old

POLICY = 'test_only_exact_cdr3_pair_nonpaired_legacy_v1'
EXCLUDED = frozenset({
    'tcr_representation/train', 'uptcr/few-shot/train', 'uptcr/few-shot/valid',
    'uptcr/majority/train', 'uptcr/majority/valid',
    'uptcr/zero-shot/train', 'uptcr/zero-shot/valid',
})
PENDING = [
    'inherited_v7_unresolved_benchmark_h3',
    'independent_audit_of_new_manifest_and_current_benchmark_coverage_required',
    'paired_near_similarity_not_screened_by_this_exact_pair_policy',
    'inherited_annotation_and_source_provenance_limitations',
]


def protected_test_mask(cohorts):
    old.require(EXCLUDED <= set(cohorts), 'Expected non-test cohorts missing')
    old.require(len(cohorts) == 24, 'Unexpected registry; review test roles explicitly')
    return sum(1 << i for i, name in enumerate(cohorts) if name not in EXCLUDED)


def select_reasons(roles, reasons, mask):
    """An OR of separate chain matches is never evidence of the same pair."""
    paired = {'tcr_alpha', 'tcr_beta'} <= set(roles)
    return {name: value & mask for name, value in reasons.items()
            if value & mask and (not paired or name == 'exact_ab')}


def rebuild(output):
    output = Path(output).absolute()
    old.require(not output.exists(), 'Output already exists; use a new version')
    old.require(output.parent.is_dir(), 'Output parent must exist')
    evidence = old.load_evidence(old.DEFAULT_MHC, old.DEFAULT_PLAN,
        mhc_manifest_sha256=old.MHC_SHA, cleanup_plan_sha256=old.PLAN_SHA)
    for protected in (evidence.mhc_dir, evidence.parent_dir):
        old.require(not output.is_relative_to(protected) and not protected.is_relative_to(output),
                    'Output overlaps source dataset')
    cohorts = evidence.plan['cohort_bit_order']
    mask = protected_test_mask(cohorts)
    dataset_paths = {str((base / s['path']).resolve())
                     for base in (evidence.mhc_dir, evidence.parent_dir) for s in evidence.shards}
    for path, digest in list(evidence.bindings.items()):
        if path not in dataset_paths and path != str(evidence.hit_path):
            evidence.bind(path, digest, verify=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.staging-', dir=output.parent))
    print(json.dumps({'stage': str(stage), 'policy': POLICY}), flush=True)
    reader = old.HashedRows(evidence.hit_path)
    hits = old.checked_hits(evidence, reader)
    next_hit = next(hits, None)
    counts, shards = {}, []
    from dllm.pipelines.immune_llada.data.preprocessing.validators import validate_prepared_row
    with (stage / 'removed_rows.jsonl').open('w') as removed:
        for shard_index, shard in enumerate(evidence.shards):
            rel = shard['path']
            before_reader = old.HashedRows(evidence.parent_dir / rel)
            source_reader = old.HashedRows(evidence.mhc_dir / rel)
            same = before_reader.path.resolve() == source_reader.path.resolve()
            pairs = ((x, x) for x in source_reader) if same else zip_longest(before_reader, source_reader)
            target = stage / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            values = counts.setdefault(shard['split'] + '/' + shard['source'], Counter())
            retained = 0
            import hashlib
            digest = hashlib.sha256()
            with target.open('wb') as out:
                for row_index, pair in enumerate(pairs):
                    before, after = pair
                    old.require(before is not None and after is not None, 'Parent membership mismatch')
                    raw, row = after
                    old.require(old.non_mhc_projection(before[1]) == old.non_mhc_projection(row),
                                'Parent/MHC row order or content mismatch')
                    old.require(row['split'] == shard['split'], 'Split mismatch')
                    validate_prepared_row(row)
                    hit = next_hit[1] if next_hit is not None and next_hit[0] == (shard_index, row_index) else None
                    if hit is not None:
                        next_hit = next(hits, None)
                    reasons = select_reasons(row['chain_roles'], hit['reasons'] if hit else {}, mask)
                    paired = {'tcr_alpha', 'tcr_beta'} <= set(row['chain_roles'])
                    values['input_rows'] += 1
                    if paired:
                        values['paired_input_rows'] += 1
                    if hit:
                        values['old_removed_rows'] += 1
                    if reasons:
                        values['removed_rows'] += 1
                        values['paired_removed_rows' if paired else 'nonpaired_removed_rows'] += 1
                        removed.write(json.dumps(dict(shard=rel, row=row_index, reasons=reasons,
                                                      paired_tcr=paired), separators=(',', ':')) + '\n')
                    else:
                        values['retained_rows'] += 1
                        if hit:
                            values['restored_from_old_candidate_rows'] += 1
                        retained += 1
                        old.require(raw.endswith(b'\n'), 'Source record lacks newline')
                        out.write(raw)
                        digest.update(raw)
            source_reader.check(shard['sha256'], shard['records'])
            parent = evidence.parent_shards[rel]
            (source_reader if same else before_reader).check(parent['sha256'], parent['records'])
            evidence.verified.update((str(before_reader.path.resolve()), str(source_reader.path.resolve())))
            if retained:
                shards.append(dict(path=rel, split=shard['split'], source=shard['source'],
                                   records=retained, sha256=digest.hexdigest(),
                                   input_shard_sha256=shard['sha256']))
            else:
                target.unlink()
            print(json.dumps(dict(shard=rel, input=shard['records'], retained=retained,
                                  finished=shard_index + 1, total=len(evidence.shards))), flush=True)
    old.require(next_hit is None, 'Unconsumed row hits')
    reader.check(evidence.plan['row_hits_sha256'])
    evidence.verified.add(str(evidence.hit_path))
    old.require(set(evidence.bindings) == evidence.verified, 'Unverified bindings remain')
    # Independent I/O pass: verify published bytes and record counts, not merely
    # the checksums produced by the write loop. No mutation of source splits.
    for shard in shards:
        import hashlib
        h, n = hashlib.sha256(), 0
        with (stage / shard['path']).open('rb') as f:
            for line in f:
                h.update(line)
                n += bool(line.strip())
        old.require(h.hexdigest() == shard['sha256'] and n == shard['records'], 'Output verification failed')
    for path in (evidence.mhc_dir / 'dataset_manifest.json', evidence.parent_dir / 'dataset_manifest.json',
                 Path(evidence.plan['audit_report'])):
        old.require(old.sha(path) == evidence.bindings[str(path.resolve())], 'Input metadata changed')
    manifest = deepcopy(evidence.manifest)
    manifest.update(dataset_version=output.name, created_at=datetime.now(timezone.utc).isoformat(),
        training_ready=False, decontamination=dict(status='audit_pending', training_ready=False,
                                                  policy=POLICY, pending=PENDING),
        splits={split: dict(records=sum(s['records'] for s in shards if s['split'] == split),
                            shards=[s for s in shards if s['split'] == split])
                for split in evidence.manifest['splits']})
    manifest['test_pair_cleanup'] = dict(policy=POLICY, source=str(evidence.mhc_dir),
        source_manifest_sha256=old.MHC_SHA, old_plan_sha256=old.PLAN_SHA,
        old_row_hits_sha256=evidence.plan['row_hits_sha256'], cohort_bit_order=cohorts,
        test_cohort_mask=mask, excluded_cohorts=sorted(EXCLUDED),
        pair_definition='same-test-record exact CDR3alpha and CDR3beta; not full-chain identity',
        nonpaired_policy='existing match reasons restricted to test cohorts',
        removed_rows_sha256=old.sha(stage / 'removed_rows.jsonl'),
        row_indexing=old.INDEXING, retained_mapping='source nonblank row order minus removed_rows',
        exporter_sha256=old.sha(Path(__file__)), split_reassignment_applied=False)
    old.write_json(stage / 'dataset_manifest.json', manifest)
    totals = sum(counts.values(), Counter())
    old.require(totals['input_rows'] == totals['removed_rows'] + totals['retained_rows'], 'Count mismatch')
    old.require(totals['old_removed_rows'] == totals['removed_rows'] + totals['restored_from_old_candidate_rows'],
                'Restoration accounting mismatch')
    report = dict(status='exported_bytes_verified_audit_pending', policy=POLICY, training_ready=False,
        pending=PENDING, counts=counts, totals=totals, output_directory=str(output),
        manifest_sha256=old.sha(stage / 'dataset_manifest.json'),
        bindings=evidence.bindings, verified_binding_paths=sorted(evidence.verified),
        output_readback_verified=True, source_rows_preserved_byte_for_byte=True,
        limitations=['Base V7 exclusions inherited; this export restores only the later expanded exclusions.',
                     'Existing cohort coverage and field-normalization limitations remain; no new sequence audit.'])
    old.write_json(stage / 'cleaning_report.json', report)
    lock = output.parent / ('.' + output.name + '.publish.lock')
    with lock.open('x'):
        try:
            old.require(not output.exists(), 'Output appeared during export')
            os.rename(stage, output)
        finally:
            lock.unlink()
    print(json.dumps(dict(output=str(output), totals=totals, status=report['status'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.output_dir)
