"""Verify the new export as a byte-preserving subsequence of the bound source.

Activate conda pllm, then run:
  python -B scripts/data/verify_immune_test_pair_clean_v2.py --dataset /absolute/dataset

Independently derives expected decisions from old row-hit masks and source role
fields. No sequence annotation or similarity computation is performed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(dataset):
    manifest_path = dataset / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    policy = manifest['test_pair_cleanup']
    assert policy['policy'] == 'test_only_observed_cdr3_pair_nonpaired_legacy_v2'
    source = Path(policy['source'])
    assert digest(source / 'dataset_manifest.json') == policy['source_manifest_sha256']
    parent = json.loads((source / 'dataset_manifest.json').read_text())
    root = Path(__file__).resolve().parents[2]
    assert digest(root / 'scripts/data/audit_test_pair_release.py') == policy['observation_policy_sha256']
    plan_path = root / 'data/dedup/tcr_exposure_20260930_r2/cleanup_plan.json'
    assert digest(plan_path) == policy['old_plan_sha256']
    plan = json.loads(plan_path.read_text())
    assert policy['cohort_bit_order'] == plan['cohort_bit_order']
    included = {i for i, name in enumerate(plan['cohort_bit_order'])
                if name.rsplit('/', 1)[-1] not in ('train', 'valid')}
    mask = sum(2 ** i for i in included)
    assert mask == policy['test_cohort_mask']
    hits_path = Path(plan['row_hits'])
    assert digest(hits_path) == policy['old_row_hits_sha256'] == plan['row_hits_sha256']
    ledger_path = dataset / 'removed_rows.jsonl'
    assert digest(ledger_path) == policy['removed_rows_sha256']
    outputs = {s['path']: s for b in manifest['splits'].values() for s in b['shards']}
    counts = Counter()
    by_source = {}
    with hits_path.open() as hf, ledger_path.open() as lf:
        hits = map(json.loads, hf)
        ledger = map(json.loads, lf)
        hit, removal = next(hits, None), next(ledger, None)
        for split, block in parent['splits'].items():
            for shard in block['shards']:
                relative = shard['path']
                output = (dataset / relative).open('rb') if relative in outputs else None
                h, out_hash, n, kept = hashlib.sha256(), hashlib.sha256(), 0, 0
                bucket = by_source.setdefault(split + '/' + shard['source'], Counter())
                try:
                    with (source / relative).open('rb') as f:
                        for raw in f:
                            h.update(raw)
                            if not raw.strip():
                                continue
                            reasons, paired = {}, False
                            if hit is not None and (hit['shard'], hit['row']) == (relative, n):
                                row = json.loads(raw)
                                observed = set()
                                for ci, (role, seq) in enumerate(zip(row['chain_roles'], row['chains'])):
                                    flags = row.get('chain_metadata', {}).get(str(ci), {}).get('synthetic_residue_mask')
                                    if flags is not None:
                                        assert len(flags) == len(seq) and all(x in (0, 1) for x in flags)
                                    if seq and (flags is None or 0 in flags):
                                        observed.add(role)
                                paired = 'tcr_alpha' in observed and 'tcr_beta' in observed
                                for reason, bitmask in hit['reasons'].items():
                                    protected = sum(2 ** i for i in included if bitmask & 2 ** i)
                                    if protected and (not paired or reason == 'exact_ab'):
                                        reasons[reason] = protected
                                hit = next(hits, None)
                            if reasons:
                                assert removal is not None
                                assert (removal['shard'], removal['row']) == (relative, n)
                                assert removal['reasons'] == reasons and removal['paired_tcr'] == paired
                                removal = next(ledger, None)
                                bucket['removed_rows'] += 1
                            else:
                                assert output is not None and output.readline() == raw, (relative, n)
                                out_hash.update(raw)
                                kept += 1
                                bucket['retained_rows'] += 1
                            n += 1
                            bucket['input_rows'] += 1
                    assert n == shard['records'] and h.hexdigest() == shard['sha256']
                    if output is not None:
                        assert output.read(1) == b''
                        assert outputs[relative]['records'] == kept
                        assert outputs[relative]['sha256'] == out_hash.hexdigest()
                    else:
                        assert kept == 0
                finally:
                    if output is not None:
                        output.close()
                counts['input_rows'] += n
                counts['retained_rows'] += kept
                counts['removed_rows'] += n - kept
                print(json.dumps({'verified': relative, 'retained': kept}), flush=True)
        assert hit is None and removal is None
    assert counts['retained_rows'] == sum(b['records'] for b in manifest['splits'].values())
    report = json.loads((dataset / 'cleaning_report.json').read_text())
    for key, value in by_source.items():
        for metric, number in value.items():
            assert report['counts'][key][metric] == number
    result = dict(status='passed_existing_evidence_selection_and_byte_subsequence_verification',
        counts=counts, by_source=by_source, manifest_sha256=digest(manifest_path),
        verifier_sha256=digest(Path(__file__)),
        new_sequence_audit=False, training_certificate=False)
    with (dataset / 'selection_verification.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    verify(parser.parse_args().dataset)
