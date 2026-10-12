"""Read-only monitor checks: pytest -q scripts/tests/immune_llada/test_research_pool_monitor.py."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import monitor_immune_research_pool as monitor


def make_pool(tmp_path):
    pool = tmp_path / 'pool'
    for name in ('pending', 'running', 'results', 'workers', 'logs', 'profiles', 'runs'):
        (pool / name).mkdir(parents=True)
    monitor.write_json(pool / 'pool.json', {'schema_version': 1})
    return pool


def test_corrupt_artifacts_preserve_results_and_escape_markdown(tmp_path):
    pool = make_pool(tmp_path)
    job_id = 'r00_测试'
    monitor.write_json(pool / 'results' / (job_id + '.json'), dict(job_id=job_id, state='succeeded', job=dict(variant='r00', mode='profile-synthetic', steps=20)))
    monitor.write_json(pool / 'runs' / job_id / 'summary.json', dict(initial=dict(loss=9), final=dict(loss=1), trace=[dict(step=20, aa_accuracy=.75)], biological_quality_claim=False))
    monitor.write_json(pool / 'gpu_status.json', dict(time=time.time(), hostname='<host>|测试', gpus=[dict(index=0, utilization_gpu=5, memory_used_mib=200, memory_total_mib=80000)]))
    state = monitor.snapshot(pool)
    document = monitor.render(state, 30)
    assert '20/20' in document and 'aa\\_accuracy' in document
    assert '&lt;host&gt;&#124;测试' in document
    assert '低于 20%' in document
    (pool / 'runs' / job_id / 'summary.json').write_text('{invalid')
    (pool / 'results' / (job_id + '.json')).unlink()
    (pool / 'workers' / 'bad.json').write_bytes(b'\xff')
    updated = monitor.snapshot(pool, state)
    assert updated['records'][job_id]['summary']['final']['loss'] == 1
    assert len(updated['errors']) == 2
    assert '读取错误' in monitor.render(updated, 30)


def test_latest_training_state_and_stale_telemetry(tmp_path):
    pool = make_pool(tmp_path)
    monitor.write_json(pool / 'running' / 'job.json', dict(variant='r00', mode='train', steps=1000))
    monitor.write_json(pool / 'runs' / 'job' / 'checkpoint-100' / 'trainer_state.json', dict(global_step=100, log_history=[dict(step=100, eval_loss=2.1)]))
    monitor.write_json(pool / 'runs' / 'job' / 'checkpoint-200' / 'trainer_state.json', dict(global_step=200, log_history=[dict(step=200, eval_loss=1.9)]))
    monitor.write_json(pool / 'gpu_status.json', dict(time=time.time()-400, gpus=[dict(index=0, utilization_gpu=0)]))
    state = monitor.snapshot(pool)
    assert monitor.progress(state['records']['job']) == 200
    document = monitor.render(state, 30)
    assert '200/1000' in document and 'eval\\_loss' in document
    assert '过期或无时间戳' in document and '低于 20%' not in document


def test_artificial_train_kind_and_latest_mask_mean_are_explicit(tmp_path):
    pool = make_pool(tmp_path)
    monitor.write_json(pool/'running/job.json', dict(variant='r20', mode='train', steps=500))
    monitor.write_json(pool/'runs/job/summary.json', dict(kind='synthetic_generalization_diagnostic',
        dataset_kind='deterministic_artificial_sequences', biological_quality_claim=False,
        initial=dict(loss=9, mask_mean=dict(loss=8)),
        trace=[dict(step=25, loss=4, mask_mean=dict(loss=3),
                    strata={'condition/long': {'mask_mean': {'loss': 2}}})]))
    document = monitor.render(monitor.snapshot(pool), 30)
    assert 'synthetic\\_generalization\\_diagnostic' in document
    assert 'mode=train 不等同于真实语料训练' in document
    assert '| 25/500 | 8 | 3 |' in document
    assert '逐次全局记录' in document
    assert 'strata.condition/long' not in document


def test_lock_heartbeat_stop_and_no_queue_execution(tmp_path):
    pool = make_pool(tmp_path)
    marker = tmp_path / 'should-not-exist'
    monitor.write_json(pool / 'pending' / 'untrusted.json', dict(command=f'touch {marker}', variant='测试'))
    doc = tmp_path / 'live.md'
    argv = [sys.executable, str(Path(monitor.__file__)), '--pool-dir', str(pool), '--document', str(doc), '--interval', '.1']
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic()+5
        while not (pool / 'monitor' / 'heartbeat.json').exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert process.poll() is None
        heartbeat = monitor.read_json(pool / 'monitor' / 'heartbeat.json')
        assert heartbeat['state'] == 'watching' and heartbeat['pid'] == process.pid
        duplicate = subprocess.run(argv + ['--once'], capture_output=True, timeout=5)
        assert duplicate.returncode != 0 and b'lock' in duplicate.stderr
        assert not marker.exists() and (pool / 'pending' / 'untrusted.json').exists()
        (pool / 'monitor' / 'STOP').touch()
        process.wait(timeout=5)
        assert process.returncode == 0
        assert monitor.read_json(pool / 'monitor' / 'heartbeat.json')['state'] == 'stopped'
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_platform_read_only_query_rate_limit_and_timeout(tmp_path, monkeypatch):
    pool = make_pool(tmp_path)
    (pool / 'monitor').mkdir()
    task_id = 't-20261001033327-fmnql'
    monitor.write_json(pool / 'task_info.json', dict(task_id=task_id))
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        assert argv[-6:] == ['ml_task', 'get', '-i', task_id, '-o', 'json']
        assert kwargs['timeout'] == 20
        return subprocess.CompletedProcess(argv, 0, stdout='升级提示\n'+json.dumps([dict(JobId=task_id, Status='Running')]))
    monkeypatch.setattr(monitor.subprocess, 'run', fake_run)
    monitor.platform_query(pool)
    first = monitor.read_json(pool / 'platform_status.json')
    assert first['status'] == 'Running' and first['observed_at']
    monitor.platform_query(pool)
    assert len(calls) == 1
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 20)
    monkeypatch.setattr(monitor.subprocess, 'run', timeout)
    monitor.platform_query(pool, interval=0)
    failed = monitor.read_json(pool / 'platform_status.json')
    assert failed['status'] == 'Running' and failed['observed_at'] == first['observed_at']
    assert 'timed out' in failed['query_error']
    monitor.write_json(pool / 'task_info.json', dict(task_id='x; touch unsafe'))
    monitor.platform_query(pool, interval=0)
    assert monitor.read_json(pool / 'platform_status.json') == failed
