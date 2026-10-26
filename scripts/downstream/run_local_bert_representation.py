"""Run a prepared BERT suite locally, one task at a time on a shared GPU.

Activate pllm/protenix_abtcr and use the run's environment, then execute:
python -B scripts/downstream/run_local_bert_representation.py --run ABSOLUTE_RUN
Requires a bound local_20261018/manifest.json and terminal cloud tasks. This
driver leaves other local evaluations and their resource management untouched.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
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


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    local = run / 'local_20261018'
    manifest = json.loads((local / 'manifest.json').read_text())
    assert sha(__file__) == manifest['runner_sha256']
    assert sha(run / 'plan.json') == manifest['plan_sha256']
    assert sha(local / 'platform_terminal.json') == manifest['platform_sha256']
    terminal = json.loads((local / 'platform_terminal.json').read_text())
    assert {row['JobId'] for row in terminal} == set(manifest['cancelled_cloud_ids'])
    assert all(row['Status'] in ('Killed', 'Failed', 'Success') for row in terminal)
    spec = importlib.util.spec_from_file_location('pinned_bert_suite', run / 'run_suite.py')
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    plan = suite.verify(run)
    assert manifest['tasks'] == plan['tasks']
    assert os.environ['TCR_EVAL_STEP'] == str(plan['step'])
    assert os.environ['TCR_EVAL_SHA'] == plan['model_sha256']
    assert os.environ['TCR_EVAL_TAG'] == run.name
    assert Path(os.environ['TCR_EVAL_CKPT']) == run / 'checkpoint'
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def stop_requested():
        return stopped or (run / 'STOP').exists() or (local / 'STOP').exists()

    with (local / 'driver.lock').open('a') as lock:
        # Only prevent duplicate execution of this run; GPU sharing is intentional.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (local / 'state.json').exists():
            raise RuntimeError('An attempt already exists; inspect it before another retry')
        state = dict(status='starting', pid=os.getpid(), host=socket.gethostname(),
                     started_utc=now(), execution='local_shared_gpu', results=[],
                     task_order=plan['tasks'], device=os.environ.get('CUDA_VISIBLE_DEVICES'))
        child = None

        def run_child(arguments, log_path):
            nonlocal child
            if stop_requested():
                raise RuntimeError('Stop requested')
            # Wait only if actual free memory is insufficient; do not stop peer jobs.
            while True:
                memory = subprocess.check_output([
                    'nvidia-smi', '-i', str(manifest['gpu_index']), '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True)
                free_mib = int(memory.strip())
                state.update(free_memory_mib=free_mib, updated_utc=now())
                if free_mib >= manifest['minimum_free_mib']:
                    break
                state['status'] = 'waiting_for_memory'
                write(local / 'state.json', state)
                if stop_requested():
                    raise RuntimeError('Stop requested')
                time.sleep(2)
            state['status'] = 'running' if state.get('current_task') else 'gpu_check'
            with log_path.open('x') as log:
                child = subprocess.Popen([sys.executable, '-B', str(run / 'run_suite.py'),
                    '--run', str(run), *arguments], stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                state['child_pid'] = child.pid
                write(local / 'state.json', state)
                while child.poll() is None:
                    if stop_requested():
                        raise RuntimeError('Stop requested')
                    state['updated_utc'] = now()
                    write(local / 'state.json', state)
                    time.sleep(2)
                returncode = child.returncode
            child = None
            return returncode

        try:
            write(local / 'state.json', state)
            result = run_child(['--check', '--device', 'cuda'], local / 'gpu_check.log')
            if result:
                raise RuntimeError('GPU adapter check failed; inspect local gpu_check.log')
            assert json.loads((run / 'cuda_adapter_checks.json').read_text())['status'] == 'passed'
            state['gpu_gate'] = 'passed'
            for task in plan['tasks']:
                folder = run / task
                status_path = folder / 'status.json'
                if status_path.exists():
                    existing = json.loads(status_path.read_text())
                    if existing['status'] == 'success':
                        continue
                    raise RuntimeError('Task already attempted; inspect before retry: ' + task)
                folder.mkdir(exist_ok=True)
                state.update(current_task=task, updated_utc=now())
                result = run_child(['--task', task], folder / 'run.log')
                task_state = json.loads(status_path.read_text()) if status_path.exists() else {}
                if task_state.get('status') not in ('success', 'failed'):
                    task_state.update(status='failed', task=task, returncode=result,
                                      error='Worker exited without a completed task status', finished_utc=now())
                    write(status_path, task_state)
                outcome = dict(task=task, returncode=result, status=task_state['status'], finished_utc=now())
                state['results'].append(outcome)
                write(local / 'state.json', state)
                suite.report(run)
                print(json.dumps(outcome), flush=True)
            state['status'] = 'success' if all(
                row['returncode'] == 0 and row['status'] == 'success' for row in state['results']) else 'failed'
        except BaseException as error:
            state.update(status='failed', error=repr(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                # Terminate only the process group created by this driver.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            if state.get('current_task'):
                status_path = run / state['current_task'] / 'status.json'
                if status_path.exists():
                    task_state = json.loads(status_path.read_text())
                    if task_state['status'] == 'running':
                        task_state.update(status='interrupted', error=state.get('error'), finished_utc=now())
                        write(status_path, task_state)
            state['finished_utc'] = now()
            write(local / 'state.json', state)
            suite.report(run)
        if state['status'] != 'success':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
