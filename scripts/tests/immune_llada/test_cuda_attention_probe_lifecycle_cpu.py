"""Real CPU process/descendant cleanup tests; no CUDA case or dataset is opened.

Activate pllm, then:
USE_TF=0 python -m pytest -q scripts/tests/immune_llada/test_cuda_attention_probe_lifecycle_cpu.py
The artificial workers only record PIDs and sleep. No kill/wait call is mocked.
"""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from scripts.architecture_prototypes import cuda_attention_probe as probe


pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux subreaper contract')

GRANDCHILD = '''
import os, signal, sys, time
from pathlib import Path
if sys.argv[2] == 'ignore':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(60)
'''

CHILD = '''
import json, os, signal, subprocess, sys, time
from pathlib import Path
path, mode, policy, grandcode = sys.argv[1:]
if policy == 'ignore':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
marker = Path(path+'.grand')
grand = subprocess.Popen([sys.executable, '-c', grandcode, str(marker), policy])
until = time.monotonic()+5
while not marker.exists():
    if time.monotonic() >= until:
        raise RuntimeError('Artificial grandchild did not start')
    time.sleep(.005)
Path(path).write_text(json.dumps(dict(child=os.getpid(), grandchild=grand.pid,
    pgid=os.getpgrp(), sid=os.getsid(0))))
if mode.startswith('exit'):
    sys.exit(int(mode[4:]))
time.sleep(60)
'''

DRIVER = '''
import ctypes, json, signal, sys
from pathlib import Path
from scripts.architecture_prototypes import cuda_attention_probe as probe
request = json.loads(Path(sys.argv[1]).read_text())
before = ctypes.c_int()
ctypes.CDLL(None).prctl(37, ctypes.byref(before), 0, 0, 0)
handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
try:
    with Path(request['log']).open('w') as log:
        probe.run_case_process(request['command'], stdout=log, timeout=20)
except probe.ProbeInterrupted as error:
    after = ctypes.c_int()
    ctypes.CDLL(None).prctl(37, ctypes.byref(after), 0, 0, 0)
    Path(request['done']).write_text(json.dumps(dict(signum=error.signum,
        subreaper_restored=before.value == after.value,
        signals_restored=all(signal.getsignal(k) == v for k,v in handlers.items()))))
    sys.exit(128+error.signum)
'''


def subreaper_state():
    value = ctypes.c_int()
    assert ctypes.CDLL(None).prctl(37, ctypes.byref(value), 0, 0, 0) == 0
    return value.value


def wait_json(path, timeout=8):
    until = time.monotonic()+timeout
    while time.monotonic() < until:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(.01)
    raise AssertionError(f'CPU process did not produce {path}')


def assert_reaped(info):
    for field in ('child', 'grandchild'):
        assert not Path(f'/proc/{info[field]}').exists(), f'{field} was not reaped: {info}'
    with pytest.raises(ProcessLookupError):
        os.killpg(info['pgid'], 0)


def fallback_cleanup(path):
    """Keep a failed regression from leaving live artificial sleepers behind."""
    if not path.exists():
        return
    info = json.loads(path.read_text())
    try:
        os.killpg(info['pgid'], signal.SIGKILL)
    except ProcessLookupError:
        return
    until = time.monotonic()+2
    while time.monotonic() < until:
        try:
            child, _ = os.waitpid(-info['pgid'], os.WNOHANG)
        except ChildProcessError:
            break
        if not child:
            time.sleep(.01)


@pytest.fixture
def family(tmp_path):
    path = tmp_path/'pids.json'
    before = subreaper_state()
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    yield path
    fallback_cleanup(path)
    assert subreaper_state() == before
    assert all(signal.getsignal(k) == v for k,v in handlers.items())


def command(path, mode='hang', policy='ignore'):
    return [sys.executable, '-u', '-c', CHILD, str(path), mode, policy, GRANDCHILD]


@pytest.mark.parametrize('code', [0, 7])
def test_finished_case_reaps_grandchild_even_after_leader_exits(family, code):
    with family.with_suffix('.log').open('w') as log:
        result = probe.run_case_process(command(family, f'exit{code}'), stdout=log, timeout=8)
    info = wait_json(family)
    assert result.returncode == code
    assert info['sid'] == os.getsid(0), 'Case must stay in the existing session'
    assert info['pgid'] == info['child'] and info['pgid'] != os.getpgrp()
    assert_reaped(info)


@pytest.mark.parametrize('policy', ['default', 'ignore'])
def test_real_timeout_terminates_and_reaps_entire_case_group(family, policy):
    started = time.monotonic()
    with family.with_suffix('.log').open('w') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            probe.run_case_process(command(family, policy=policy), stdout=log, timeout=3)
    assert time.monotonic()-started < 3.5
    assert_reaped(wait_json(family))


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt, SystemExit])
def test_exception_after_real_spawn_still_reaps_children(family, monkeypatch, error_type):
    def injected(pid, deadline):
        info = wait_json(family)
        assert info['child'] == pid
        raise error_type('injected CPU test exception after real child/grandchild launch')
    # Only the fault is injected; spawn, signals, group termination and wait are real.
    monkeypatch.setattr(probe, 'wait_case_leader', injected)
    with family.with_suffix('.log').open('w') as log:
        with pytest.raises(error_type):
            probe.run_case_process(command(family), stdout=log, timeout=8)
    assert_reaped(wait_json(family))


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT])
def test_parent_signal_cleans_ignoring_descendants_before_exit(family, signum):
    request = family.with_suffix('.request.json')
    done = family.with_suffix('.done.json')
    request.write_text(json.dumps(dict(command=command(family), done=str(done),
                                      log=str(family.with_suffix('.child.log')))))
    with family.with_suffix('.driver.log').open('w') as log:
        parent = subprocess.Popen([sys.executable, '-u', '-c', DRIVER, str(request)],
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            info = wait_json(family, timeout=30)
            started = time.monotonic()
            parent.send_signal(signum)
            assert parent.wait(timeout=8) == 128+signum
            assert time.monotonic()-started < 8  # Existing pool grace is 10 seconds.
            proof = wait_json(done)
            assert proof == dict(signum=signum, subreaper_restored=True, signals_restored=True)
            assert_reaped(info)
        finally:
            if parent.poll() is None:
                parent.kill()
            parent.wait(timeout=5)


def test_parent_signal_during_cleanup_is_deferred_not_lost(family, monkeypatch):
    original = probe.terminate_case_group
    def signal_then_cleanup(pgid, deadline):
        os.kill(os.getpid(), signal.SIGTERM)
        return original(pgid, deadline)
    monkeypatch.setattr(probe, 'terminate_case_group', signal_then_cleanup)
    with family.with_suffix('.log').open('w') as log:
        with pytest.raises(probe.ProbeInterrupted) as caught:
            probe.run_case_process(command(family, 'exit0'), stdout=log, timeout=8)
    assert caught.value.signum == signal.SIGTERM
    assert_reaped(wait_json(family))


def test_cleanup_failure_aborts_plan_before_next_case(tmp_path, monkeypatch):
    calls = []
    def failure(*args, **kwargs):
        calls.append(args)
        raise probe.CaseCleanupError('Deliberate schema test: no process was launched')
    monkeypatch.setattr(probe, 'run_case_process', failure)
    with pytest.raises(probe.CaseCleanupError):
        probe.execute_plan([probe.ProbeCase(backend='math'), probe.ProbeCase()], tmp_path/'plan')
    assert len(calls) == 1


def test_import_alone_does_not_change_subreaper_or_signal_handlers():
    code = '''
import ctypes, signal
def state():
    value=ctypes.c_int()
    assert ctypes.CDLL(None).prctl(37,ctypes.byref(value),0,0,0)==0
    return value.value,signal.getsignal(signal.SIGTERM),signal.getsignal(signal.SIGINT)
before=state()
from scripts.architecture_prototypes import cuda_attention_probe
assert state()==before
'''
    subprocess.run([sys.executable, '-B', '-c', code], check=True, timeout=30)
