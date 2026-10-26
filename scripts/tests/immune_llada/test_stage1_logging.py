"""Activate pllm; python -B -m pytest THIS_FILE -q. No GPU or teacher imports.

Check timed aggregation, diagnostic visibility and actual child exit propagation.
"""
import json
import subprocess
import sys

import pytest

from scripts.research.run_stage1_checks import ConsoleSummary, execute


def test_teacher_records_are_aggregated_at_interval_and_final():
    now, output = [0.], []
    log = ConsoleSummary(60, emit=output.append, clock=lambda: now[0])
    record = json.dumps(dict(status="cached", request="abc", pairs=1024))
    for _ in range(500):
        log.feed(record)
    assert output == []
    now[0] = 60.
    log.flush()
    assert json.loads(output[0])["cached_records"] == 500
    now[0] = 61.
    log.feed(record)
    assert len(output) == 1
    log.flush(final=True)
    assert len(output) == 2 and json.loads(output[-1])["cached_records"] == 501
    log.flush(final=True)
    assert len(output) == 2


def test_hashes_and_progress_bars_are_quiet_but_errors_are_immediate():
    output = []
    log = ConsoleSummary(60, emit=output.append)
    log.feed('/project/weights.safetensors: OK\n')
    log.feed('Loading weights:  80%|████| 8/10\n')
    assert output == []
    messages = ['RuntimeError: CUDA out of memory', 'Traceback (most recent call last):',
                '{"status":"failed","error":"bad hash"}', '/project/weights.safetensors: FAILED',
                'Unknown diagnostic: investigate']
    for message in messages:
        log.feed(message)
    assert output == messages
    log.flush(final=True)
    assert json.loads(output[-1])["hash_checks"] == 1


def test_metrics_keep_latest_values_without_per_step_console_output():
    output = []
    log = ConsoleSummary(60, emit=output.append)
    log.feed('{"step":10,"loss":1.2,"mlm":1.1,"kd_coverage":0.2}')
    log.feed('{"step":20,"loss":1.0,"mlm":0.9,"kd_coverage":0.3}')
    assert output == []
    log.flush(final=True)
    assert json.loads(output[-1])["training"]["step"] == 20


def test_child_failure_retains_error_and_nonzero_exit(capsys):
    with pytest.raises(subprocess.CalledProcessError) as failure:
        execute([sys.executable, '-c', "import sys;print('RuntimeError: test failure',file=sys.stderr);sys.exit(7)"])
    assert failure.value.returncode == 7
    assert 'RuntimeError: test failure' in capsys.readouterr().out


@pytest.mark.parametrize('interval', [0, -1, float('nan')])
def test_invalid_interval_is_rejected(interval):
    with pytest.raises(ValueError):
        ConsoleSummary(interval)
