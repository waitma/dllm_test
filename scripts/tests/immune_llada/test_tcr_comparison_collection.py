"""Run: python -m pytest -q scripts/tests/immune_llada/test_tcr_comparison_collection.py."""
from copy import deepcopy
import json

import pytest

from scripts.downstream import collect_tcr_49000_comparison as c


def config():
    return {'head_config': {'seed': 1}, 'folds_requested': [1, 2, 3, 4, 5],
            'eval_sets_requested': ['seen_test'], 'feature_manifest': {
                'pairs_sha256': 'pairs', 'data_archive_md5': 'archive',
                'input_protocol': {'recognition': 'mask', 'code_sha256': {'adapter': 'new'}}}}


def test_code_drift_is_disclosed_not_claimed_weight_only():
    new, old = config(), config()
    old['feature_manifest']['input_protocol']['code_sha256']['adapter'] = 'old'
    result = c.protocol_comparison(new, old)
    assert result['input_semantics_equal']
    assert result['changed_code_files'] == ['adapter']
    assert not result['strict_weight_only_causal_comparison']


@pytest.mark.parametrize('field', ['pairs_sha256', 'data_archive_md5'])
def test_data_identity_mismatch_rejected(field):
    new, old = config(), config()
    old['feature_manifest'][field] = 'different'
    with pytest.raises(ValueError, match='protocol mismatch'):
        c.protocol_comparison(new, old)


def test_input_or_head_change_rejected():
    new = config()
    old = deepcopy(new)
    old['feature_manifest']['input_protocol']['recognition'] = 'binding'
    with pytest.raises(ValueError):
        c.protocol_comparison(new, old)
    old = config()
    old['head_config']['seed'] = 2
    with pytest.raises(ValueError):
        c.protocol_comparison(new, old)


def test_only_prediction_score_may_differ():
    rows = [{'row_id': '1', 'label': '0', 'cdr3a': 'ABC', 'score': '.2'}]
    other = deepcopy(rows)
    other[0]['score'] = '.9'
    assert c.prediction_identity(rows) == c.prediction_identity(other)
    for field in ('label', 'row_id', 'cdr3a'):
        changed = deepcopy(other)
        changed[0][field] = 'changed'
        assert c.prediction_identity(rows) != c.prediction_identity(changed)


def test_incomplete_report_cannot_pass_final_gate(monkeypatch):
    def read(_, path):
        if path.name == 'monitor_plan_v3.json':
            return {'checkpoint_sha256': 'ckpt', 'job_keys': ['t4-held20'], 'audit_groups': []}
        return {'global_step': 49000, 'checkpoint_sha256': 'ckpt', 'tasks': {}, 'complete': False}
    monkeypatch.setattr(c.Sources, 'json', read)
    with pytest.raises(RuntimeError, match='unaudited tasks'):
        c.collect(require_complete=True)


def test_workbook_drift_rejected(monkeypatch):
    monkeypatch.setattr(c, 'sha', lambda _: 'different')
    monkeypatch.setattr(c.Sources, 'record', lambda _, path: path)
    with pytest.raises(AssertionError, match='workbook drift'):
        c.paper_binding(c.Sources())


@pytest.mark.parametrize('status', ['attention_required', 'all_tasks_executed_and_audits_passed'])
def test_waiter_obeys_monitor_terminal_status(tmp_path, monkeypatch, status):
    monkeypatch.setattr(c, 'RUN', tmp_path)
    monkeypatch.setattr(c, 'sha', lambda _: 'fixed')
    path = tmp_path / 'monitor_v3/status.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'status': status, 'plan_sha256': 'fixed',
                               'monitor_sha256': 'fixed', 'audited': {}}))
    output = tmp_path / 'result.json'
    if status == 'attention_required':
        with pytest.raises(RuntimeError, match='needs attention'):
            c.wait_for_complete(output)
        assert json.loads(output.with_suffix('.status.json').read_text())['status'] == 'attention_required'
    else:
        state, status_path = c.wait_for_complete(output)
        assert state['monitor_status'] == status
        assert status_path.exists()
    assert not output.exists(), 'waiting alone must never create a claimed final result'


def test_duplicate_waiter_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'sha', lambda _: 'fixed')
    output = tmp_path / 'result.json'
    output.with_suffix('.status.json').write_text('reserved')
    with pytest.raises(FileExistsError):
        c.wait_for_complete(output)
    assert output.with_suffix('.status.json').read_text() == 'reserved'
