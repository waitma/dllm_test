"""Reject stale foundation-training audit scopes.

Used by immune_pretrain_v3.assert_ready; test with:
python -m pytest scripts/tests/immune_llada/test_tcr_scope_gate.py -q
"""
import json
from pathlib import Path

from scripts.data.audit_tcr_exposure import SCOPE, input_specs
from scripts.data.prepare_immune_v7 import sha
from scripts.data.immune_test_scope_policy import SCOPE as TEST_SCOPE, WAIVERS, required_test_paths


def require_current_scope(directory):
    """A passed legacy audit cannot silently cover newly introduced benchmarks."""
    path = directory / "downstream_scope_certificate.json"
    if not path.is_file():
        raise ValueError("Current downstream scope audit is missing")
    proof = json.loads(path.read_text())
    manifest_path = directory / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    declared = manifest.get('scope_finalization', {}).get('scope_version')
    expected_scope = TEST_SCOPE if declared == TEST_SCOPE else SCOPE
    if proof.get("scope_version") != expected_scope or proof.get("status") != "passed" or proof.get("pending"):
        raise ValueError("Current downstream scope audit is incomplete")
    if proof.get("dataset_manifest_sha256") != sha(directory / "dataset_manifest.json"):
        raise ValueError("Downstream scope audit belongs to another dataset")
    if proof.get("forbidden_overlap_rows") != 0:
        raise ValueError("Downstream scope audit has forbidden overlaps")
    if expected_scope == TEST_SCOPE:
        if (manifest.get('training_ready') is not True or
            manifest.get('decontamination', {}).get('status') != 'passed' or
            manifest.get('decontamination', {}).get('pending') or
            set(proof.get('user_excluded_audit_items', [])) != set(WAIVERS) or
            set(manifest['scope_finalization'].get('user_excluded_audit_items', [])) != set(WAIVERS)):
            raise ValueError('Test-only manifest scope or readiness differs from its audit')
        required = {str(p): sha(p) for p in required_test_paths()}
        scan_path = directory / 'final_verification.json'
        if str(scan_path) not in proof.get('evidence_files', {}):
            raise ValueError('Test-only audit is missing its final verification')
        scan = json.loads(scan_path.read_text())
        if (scan.get('status') != 'passed' or scan.get('scope_version') != TEST_SCOPE or
            scan.get('dataset_manifest_sha256') != sha(manifest_path) or
            scan.get('unresolved_observed_pair_fields') != 0 or
            scan.get('forbidden_overlap_rows') != 0 or scan.get('family_split_overlap') != 0 or
            scan.get('t4_missing_from_historical_blocklist') != 0):
            raise ValueError('Test-only final verification is incomplete')
    else:
        required = {str(p.resolve()): sha(p) for _, p, _, _ in input_specs()}
    if any(proof.get("benchmark_inputs", {}).get(p) != digest for p, digest in required.items()):
        raise ValueError("Downstream scope inputs changed or a required evaluation pool is missing")
    evidence = proof.get("evidence_files", {})
    if not evidence:
        raise ValueError("Downstream scope audit has no verifiable evidence")
    for p, digest in evidence.items():
        if not Path(p).is_absolute() or sha(p) != digest:
            raise ValueError("Downstream scope evidence changed")
    return proof
