"""CPU orchestration checks: pytest -q scripts/tests/immune_llada/test_research_pool.py."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from scripts import immune_research_pool as pool_module


@pytest.fixture
def pool(tmp_path):
    root = tmp_path/'pool'
    subprocess.run([sys.executable, str(Path(pool_module.__file__)), 'init', '--pool-dir', str(root)], check=True)
    return root


def test_six_workers_claim_each_job_once(pool):
    def consume():
        found = []
        while True:
            path = pool_module.claim(pool)
            if path is None:
                return found
            found.append(path.name)
    with ThreadPoolExecutor(max_workers=6) as executor:
        claims = sum(list(executor.map(lambda _: consume(), range(6))), [])
    assert len(claims) == len(set(claims)) == 12
    assert not list((pool/'pending').glob('*.json'))


def test_child_is_single_gpu_without_distributed_rank(monkeypatch):
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE', 'ACCELERATE_USE_CPU'):
        monkeypatch.setenv(key, '6')
    env = pool_module.child_environment('GPU-example-uuid')
    assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-example-uuid'
    assert not any(key in env for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE', 'ACCELERATE_USE_CPU'))


def test_job_validation_and_stop(pool):
    for job in [dict(variant='r00_a4_control', mode='train', steps=1000),
                dict(variant='r00_a4_control', mode='profile-synthetic', steps=1000),
                dict(variant='r00_a4_control', mode='profile-synthetic', steps=20, command='anything')]:
        with pytest.raises(ValueError):
            pool_module.enqueue(pool, job)
    job = dict(variant='r00_a4_control', mode='train', steps=1000, data_dir='/data/certified', resume_from_checkpoint='/run/checkpoint-500')
    argv = pool_module.command(pool, 'test', job)
    assert argv[argv.index('--resume-from-checkpoint')+1] == '/run/checkpoint-500'
    (pool/'STOP').touch()
    with pytest.raises(ValueError):
        pool_module.enqueue(pool, job)


def test_retention_protects_latest_best_and_pending_resume(pool):
    run = pool/'runs'/'training'
    for step in range(100, 601, 100):
        checkpoint = run/f'checkpoint-{step}'
        checkpoint.mkdir(parents=True)
        (checkpoint/'trainer_state.json').write_text(json.dumps(dict(best_model_checkpoint=str(run/'checkpoint-200'))))
    pool_module.enqueue(pool, dict(variant='r00_a4_control', mode='train', steps=1000,
                                  data_dir='/data/clean', resume_from_checkpoint=str(run/'checkpoint-300')))
    removed = pool_module.prune_checkpoints(pool, run)
    assert {Path(path).name for path in removed} == {'checkpoint-100', 'checkpoint-400'}
    assert {p.name for p in run.glob('checkpoint-*')} == {'checkpoint-200', 'checkpoint-300', 'checkpoint-500', 'checkpoint-600'}


def test_persistent_worker_can_receive_later_jobs(pool, monkeypatch):
    for path in (pool/'pending').glob('*.json'):
        path.unlink()
    stop = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pool_module.worker, pool, '0', stop, time.monotonic()+20, 0, 1)
        time.sleep(.05)
        assert not future.done()
        stop.set()
        assert future.result(timeout=2) == 0


def test_workers_execute_isolated_children_and_report_failure(pool, monkeypatch):
    def fake_command(root, job_id, job):
        destination = root/'runs'/job_id
        code = ('import json,os,pathlib,sys; '
                f'p=pathlib.Path({str(destination)!r}); p.mkdir(); '
                'p.joinpath("COMPLETE.json").write_text(json.dumps(dict(device=os.environ["CUDA_VISIBLE_DEVICES"]))); '
                f'sys.exit({3 if job["variant"] == "r00_a4_control" else 0})')
        return [sys.executable, '-c', code]
    monkeypatch.setattr(pool_module, 'command', fake_command)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(pool_module.worker, pool, str(device), threading.Event(),
                                   time.monotonic()+30, .01, 10) for device in range(6)]
        codes = [future.result() for future in futures]
    records = [json.loads(path.read_text()) for path in (pool/'results').glob('*.json')]
    assert len(records) == 12
    assert sum(r['state'] == 'failed' for r in records) == 1
    assert sum(codes) == 1
    assert not list((pool/'running').glob('*.json'))
    for record in records:
        proof = json.loads((pool/'runs'/record['job_id']/'COMPLETE.json').read_text())
        assert proof['device'] == record['device']


def test_stop_drains_without_claiming_and_timeout_marks_failure(pool, monkeypatch):
    (pool/'STOP').touch()
    assert pool_module.worker(pool, '0', threading.Event(), time.monotonic()+5, .01, 1) == 0
    assert len(list((pool/'pending').glob('*.json'))) == 12
    (pool/'STOP').unlink()
    monkeypatch.setattr(pool_module, 'command', lambda *_: [sys.executable, '-c', 'import time; time.sleep(60)'])
    assert pool_module.worker(pool, '0', threading.Event(), time.monotonic()+.1, .01, .1) == 1
    records = [json.loads(path.read_text()) for path in (pool/'results').glob('*.json')]
    assert len(records) == 1 and records[0]['state'] == 'failed'
    assert 'time limit' in records[0]['error']
