"""Activate pllm; python -m pytest -q scripts/tests/immune_llada/test_typed_diagnostic_monitor_integration.py.

Actual adapter -> filesystem -> supervisor/monitor, with fabricated CUDA child
JSON. This verifies cross-module reporting only; it does not execute any GPU.
"""
import json

import pytest
import torch

from scripts import immune_diagnostic_reporting as reporting
from scripts import immune_research_supervision as supervision
from scripts import monitor_immune_research_pool as monitor
from scripts.tests.immune_llada.test_pool_attention_diagnostic import adapter, mock_workers, envelope


@pytest.mark.parametrize('last,completed,all_passed', [
    ('passed', True, True), ('numerical_mismatch', True, False), ('backend_mismatch', False, False)])
def test_adapter_report_reaches_typed_supervision(tmp_path, monkeypatch, last, completed, all_passed):
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        pytest.fail('A reporting test attempted GPU execution')
    for name in ('is_available', 'set_device', 'synchronize', 'empty_cache', 'get_device_properties'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(adapter.probe, 'run_cuda_case', forbidden)
    mock_workers(monkeypatch, ['passed']*3+[last])
    pool = tmp_path/'pool'
    for name in ('pending', 'running', 'finished', 'results', 'workers', 'logs', 'runs', 'profiles'):
        (pool/name).mkdir(parents=True)
    output = pool/'runs/job'
    report = adapter.run_diagnostic(envelope(), output_dir=output, profile_dir=pool/'profiles'/adapter.VARIANT)
    assert reporting.COMPLETED_STATUSES == adapter.COMPLETED_STATUSES
    assert report['execution_completed'] is completed and report['all_cases_passed'] is all_passed
    adapter.probe.write_json(pool/'finished/job.json', envelope())
    adapter.probe.write_json(pool/'results/job.json', dict(state='succeeded' if completed else 'failed', job=envelope()))
    before = {path: path.read_bytes() for path in output.rglob('*') if path.is_file()}
    record = supervision.update(pool, tmp_path/'supervision.md')['records']['job']
    assert record['alerts'] == ([] if completed else ['diagnostic_execution_failed'])
    assert record['source_status'] == 'matched' and record['source_files'] == 3
    assert record['step'] is None and record['profile_status'] is None
    assert record['case_statuses'] == report['case_statuses']
    snapshot = monitor.snapshot(pool)
    rendered = monitor.render(snapshot, 30)
    assert ('4/4 cases' if completed else '3/4 cases') in rendered
    assert 'global_step' not in json.dumps(record['completion'])
    assert all(path.read_bytes() == value for path, value in before.items())
