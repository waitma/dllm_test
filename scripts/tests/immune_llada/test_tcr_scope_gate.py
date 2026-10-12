"""Run: python -m pytest scripts/tests/immune_llada/test_tcr_scope_gate.py -q."""
import json

import pytest

from scripts.data import tcr_scope_gate as gate


def test_new_support_pool_cannot_reuse_old_passed_certificate(tmp_path, monkeypatch):
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text("{}")
    support = tmp_path / "support.csv"
    support.write_text("test input")
    evidence = tmp_path / "independent_rescan.json"
    evidence.write_text('{"overlap":0}')
    monkeypatch.setattr(gate, "input_specs", lambda: [("support", support, "core", "support")])
    with pytest.raises(ValueError, match="missing"):
        gate.require_current_scope(tmp_path)
    proof = dict(status="passed", scope_version=gate.SCOPE, pending=[], forbidden_overlap_rows=0,
        dataset_manifest_sha256=gate.sha(manifest), benchmark_inputs={str(support): gate.sha(support)},
        evidence_files={str(evidence): gate.sha(evidence)})
    cert = tmp_path / "downstream_scope_certificate.json"
    cert.write_text(json.dumps(proof))
    assert gate.require_current_scope(tmp_path) == proof
    support.write_text("different support pool")
    with pytest.raises(ValueError, match="inputs changed"):
        gate.require_current_scope(tmp_path)
    proof["benchmark_inputs"][str(support)] = gate.sha(support)
    cert.write_text(json.dumps(proof))
    evidence.write_text('{"overlap":1}')
    with pytest.raises(ValueError, match="evidence changed"):
        gate.require_current_scope(tmp_path)
