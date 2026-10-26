"""Run a prepared diffusion/edit suite sequentially on the local shared GPU.

Activate pllm/protenix_abtcr; python -B THIS_FILE --run ABSOLUTE_RUN.
Requires CPU gates. STOP in the run/local directory stops only our child group.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
from scripts.downstream.run_bert_representation_suite import report, verify, write
from downstream.bert_full_token_eval import sha


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run, child = args.run.resolve(), None
    local = run / 'local'
    plan = verify(run)
    preparation = json.loads((run / 'preparation.json').read_text())
    assert preparation['status'] == 'cpu_passed'
    assert preparation['plan_sha256'] == sha(run / 'plan.json')
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
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (local / 'state.json').exists():
            raise RuntimeError('An attempt exists; inspect it before retrying')
        state = dict(status='starting', pid=os.getpid(), host=socket.gethostname(),
                     started_utc=now(), task_order=plan['tasks'], results=[],
                     execution='local_shared_gpu', plan_sha256=sha(run / 'plan.json'))

        def run_child(arguments, log_path):
            nonlocal child
            while True:
                if stop_requested():
                    raise RuntimeError('Stop requested')
                free = int(subprocess.check_output([
                    'nvidia-smi', '-i', str(plan['gpu_index']), '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip())
                if free >= plan['minimum_free_mib']:
                    break
                state.update(status='waiting_for_memory', free_mib=free, updated_utc=now())
                write(local / 'state.json', state)
                time.sleep(2)
            with log_path.open('x') as stream:
                child = subprocess.Popen(
                    [sys.executable, '-B', str(run / 'run_suite.py'), '--run', str(run), *arguments],
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status='running', child_pid=child.pid, updated_utc=now())
                write(local / 'state.json', state)
                while child.poll() is None:
                    if stop_requested():
                        raise RuntimeError('Stop requested')
                    state['updated_utc'] = now()
                    write(local / 'state.json', state)
                    time.sleep(3)
                returncode = child.returncode
            child = None
            return returncode

        try:
            write(local / 'state.json', state)
            if run_child(['--check', '--device', 'cuda'], local / 'gpu_checks.log'):
                raise RuntimeError('CUDA gate failed; inspect local/gpu_checks.log')
            state['cuda_gate'] = 'passed'
            for task in plan['tasks']:
                state['current_task'] = task
                folder = run / task
                if (folder / 'status.json').exists():
                    raise RuntimeError('Task already attempted: ' + task)
                result = run_child(['--task', task], folder / 'run.log')
                path = folder / 'status.json'
                task_state = json.loads(path.read_text()) if path.exists() else {}
                if task_state.get('status') not in ('success', 'failed'):
                    task_state.update(status='failed', task=task, returncode=result,
                                      error='Worker exited without a terminal task status')
                    write(path, task_state)
                state['results'].append(dict(task=task, returncode=result,
                                             status=task_state['status'], finished_utc=now()))
                write(local / 'state.json', state)
                report(run)
                # Publishing consumes completed artifacts only, with no model work.
                with (local / 'publication.log').open('a') as stream:
                    pub = subprocess.run(
                        [sys.executable, '-B', str(ROOT / 'scripts/downstream/publish_diffusion_edit_results.py'),
                         '--run', str(run)], stdout=stream, stderr=subprocess.STDOUT)
                state['publication_returncode'] = pub.returncode
                print(json.dumps(state['results'][-1]), flush=True)
            state['status'] = ('success' if all(r['returncode'] == 0 and r['status'] == 'success'
                                               for r in state['results']) else 'failed')
        except BaseException as error:
            state.update(status='failed', error=repr(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            state.update(finished_utc=now(), updated_utc=now())
            write(local / 'state.json', state)
            report(run)


if __name__ == '__main__':
    main()
