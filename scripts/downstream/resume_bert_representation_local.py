"""Resume the authorized best48828 evaluation locally with its original protocol.

Activate pllm/protenix_abtcr and the original evaluation environment, then run:
python -B SCRIPT --run ABSOLUTE_RUN
Requires local_20261018/manifest.json and confirmed terminal cloud jobs. Successful
tasks are skipped; interrupted attempts are archived, and every retry runs once.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def process_start(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def binding_cache(folder, plan):
    """Validate every cached row against the pinned source before using its score."""
    provenance = json.loads((folder / 'provenance.json').read_text())
    identity = provenance['native_identity']
    assert identity['model_sha256'] == plan['model_sha256']
    assert identity['checkpoint'] == plan['checkpoint']
    assert provenance['task'] == 'binding' and provenance['assay'] == ''
    assert provenance['pll'] == 'single_residue_mask_HL_mean_log_probability'
    source = provenance['input_path']
    assert sha(source) == provenance['input_sha256'] == plan['data_sha256'][source]
    rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    saved = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
    assert 0 < len(saved) <= len(rows)
    cache = {}
    for original, result in zip(rows, saved):
        assert {k: v for k, v in result.items() if k not in ('score', 'features')} == original
        assert math.isfinite(result['score'])
        assert len(result['features']) == 2304 and all(map(math.isfinite, result['features']))
        key = tuple(original[k] for k in ('antigen', 'heavy', 'light'))
        value = (result['score'], result['features'])
        assert key not in cache or cache[key] == value
        cache[key] = value
    return cache, identity, dict(saved_rows=len(saved), unique_pairs=len(cache), total_rows=len(rows))


def suite_module(run):
    spec = importlib.util.spec_from_file_location('pinned_bert_suite', run / 'run_suite.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(run, task, manifest, suite):
    if task == 'ag-binding':
        cache, identity, evidence = binding_cache(
            run / 'local_20261018' / 'prior_attempts' / task, suite.verify(run))
        from downstream import v5_native_generation_eval as antigen
        original = antigen.score_pair

        def cached_score(backbone, ag, heavy, light, batch_size=8, features=True):
            # Only reuse full, verified binding scores; all new scoring stays upstream.
            assert backbone.identity == identity
            key = (ag, heavy, light)
            if features and key in cache:
                return cache[key]
            return original(backbone, ag, heavy, light, batch_size=batch_size, features=features)

        antigen.score_pair = cached_score
        print('Validated binding cache:', evidence, flush=True)
    suite.worker(run, task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--task')
    args = parser.parse_args()
    run = args.run.resolve()
    local = run / 'local_20261018'
    manifest = json.loads((local / 'manifest.json').read_text())
    assert sha(__file__) == manifest['runner_sha256']
    assert sha(run / 'plan.json') == manifest['plan_sha256']
    assert sha(local / 'platform_terminal.json') == manifest['platform_sha256']
    terminal = json.loads((local / 'platform_terminal.json').read_text())
    assert len(terminal) == 9 and all(x['Status'] in ('Killed', 'Failed', 'Success') for x in terminal)
    suite = suite_module(run)
    suite.verify(run)
    if args.task:
        assert args.task in manifest['task_order']
        worker(run, args.task, manifest, suite)
        return

    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    lock_path = Path(manifest['gpu_lock'])
    with lock_path.open('a') as gpu_lock, (local / 'driver.lock').open('a') as run_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (local / 'state.json').exists():
            raise RuntimeError('Local attempt already exists; inspect before authorizing another retry')
        state = dict(status='starting', host=socket.gethostname(), pid=os.getpid(),
                     started_utc=now(), task_order=manifest['task_order'], results=[])
        occupant = manifest['occupant']
        paused, child = False, None
        try:
            if (run / 'STOP').exists() or (local / 'STOP').exists():
                raise RuntimeError('Stop requested')
            assert process_start(occupant['pid']) == occupant['start_ticks']
            assert sha(occupant['script']) == occupant['script_sha256']
            active = subprocess.check_output([
                'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
            assert sorted(int(x) for x in active.split()) == occupant['gpu_pids'], active
            os.kill(occupant['pid'], signal.SIGSTOP)
            paused = True
            state.update(status='gpu_check', occupant_paused=True, updated_utc=now())
            write(local / 'state.json', state)
            gate = suite.check(run / 'checkpoint', 'cuda')
            write(local / 'gpu_adapter_checks.json', gate)
            # The driver must release its CUDA model before launching a task worker.
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            for task in manifest['task_order']:
                if stopped or (run / 'STOP').exists() or (local / 'STOP').exists():
                    raise RuntimeError('Stop requested')
                folder = run / task
                status_path = folder / 'status.json'
                prior = json.loads(status_path.read_text()) if status_path.exists() else None
                if prior and prior['status'] == 'success':
                    continue
                expected = manifest['prior_statuses'][task]
                assert prior == expected, 'Task changed after migration preparation: ' + task
                archive = local / 'prior_attempts' / task
                if task == 'ag-binding':
                    binding_cache(folder, suite.verify(run))
                    folder.rename(archive)
                else:
                    archive.mkdir()
                    for name in ('status.json', 'run.log'):
                        if (folder / name).exists():
                            (folder / name).rename(archive / name)
                folder.mkdir(exist_ok=True)
                state.update(status='running', current_task=task, updated_utc=now())
                with (folder / 'run.log').open('x') as log:
                    child = subprocess.Popen([
                        sys.executable, '-B', str(Path(__file__).resolve()),
                        '--run', str(run), '--task', task], stdout=log, stderr=subprocess.STDOUT)
                    state['child_pid'] = child.pid
                    write(local / 'state.json', state)
                    while child.poll() is None:
                        if stopped or (run / 'STOP').exists() or (local / 'STOP').exists():
                            child.terminate()
                            try:
                                child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                            raise RuntimeError('Stop requested; worker terminated')
                        state['updated_utc'] = now()
                        write(local / 'state.json', state)
                        time.sleep(2)
                result = dict(task=task, returncode=child.returncode, finished_utc=now())
                if not status_path.exists():
                    write(status_path, dict(status='failed', **result, error='Worker failed before task startup; inspect run.log'))
                task_state = json.loads(status_path.read_text())
                if task_state['status'] == 'running':
                    task_state.update(status='interrupted', **result)
                    write(status_path, task_state)
                result['status'] = task_state['status']
                state['results'].append(result)
                write(local / 'state.json', state)
                suite.report(run)
                print(json.dumps(result), flush=True)
                child = None
            state['status'] = 'success' if all(
                x['returncode'] == 0 and x['status'] == 'success' for x in state['results']) else 'failed'
        except BaseException as error:
            state.update(status='failed', error=repr(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if state.get('current_task'):
                status_path = run / state['current_task'] / 'status.json'
                if status_path.exists():
                    status = json.loads(status_path.read_text())
                    if status['status'] == 'running':
                        status.update(status='interrupted', error=state.get('error'), finished_utc=now())
                        write(status_path, status)
            if paused:
                try:
                    if process_start(occupant['pid']) == occupant['start_ticks']:
                        os.kill(occupant['pid'], signal.SIGCONT)
                        state['occupant_restored'] = True
                except (ProcessLookupError, FileNotFoundError):
                    state['occupant_restored'] = False
            state['finished_utc'] = now()
            write(local / 'state.json', state)
            suite.report(run)
        if state['status'] != 'success':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
