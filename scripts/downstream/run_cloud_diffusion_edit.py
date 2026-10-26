"""Run disjoint cloud groups from an already frozen diffusion/edit evaluation.

Activate pllm/protenix_abtcr, then python -B THIS_FILE --run ABS --group N.
Use --drain-local to supervise the in-flight worker after coordinator handoff.
The separate cloud manifest binds this launcher without changing the eval plan.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
from downstream.bert_full_token_eval import sha
from scripts.downstream.run_bert_representation_suite import report, verify, write


def now():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(start_ticks=stat[19], state=stat[0])
    except FileNotFoundError:
        return None


def publish(run, log):
    # Include collecting artifacts and writing the receipt in the lock.
    with (run / 'cloud/publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with log.open('a') as stream:
            result = subprocess.run(
                [sys.executable, '-B',
                 str(ROOT / 'scripts/downstream/publish_diffusion_edit_results.py'),
                 '--run', str(run)], stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Publication failed ({result.returncode}); inspect {log}')


def drain_local(run):
    local = run / 'local'
    receipt = json.loads((local / 'cloud_handoff.json').read_text())
    task, pid = receipt['task'], receipt['worker_pid']
    with (local / 'handoff_watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((local / 'state.json').read_text())
        state.update(status='draining_for_cloud', watcher_pid=os.getpid(),
                     execution='local_inflight_only; remaining tasks on cloud')
        while True:
            identity = process_identity(pid)
            if (not identity or identity['start_ticks'] != receipt['worker_start_ticks']
                    or identity['state'] == 'Z'):
                break
            state.update(updated_utc=now())
            write(local / 'state.json', state)
            time.sleep(3)
        task_state = json.loads((run / task / 'status.json').read_text())
        if task_state.get('status') not in ('success', 'failed'):
            task_state.update(status='failed',
                              error='In-flight worker exited without terminal status')
            write(run / task / 'status.json', task_state)
        state['results'].append(dict(task=task, status=task_state['status'],
                                     finished_utc=now(), returncode=None,
                                     exit_evidence='task status; worker was reparented'))
        state.update(status='handed_off' if task_state['status'] == 'success' else 'failed',
                     current_task=None, finished_utc=now(), updated_utc=now())
        write(local / 'state.json', state)
        report(run)
        publish(run, local / 'handoff_publication.log')
        write(local / 'handoff_finished.json',
              dict(status=task_state['status'], task=task, finished_utc=now(),
                   publication='success', pending_execution='cloud manifest'))


def cloud_group(run, manifest, group):
    tasks = manifest['groups'][group]
    folder = run / 'cloud' / f'g{group}'
    folder.mkdir(exist_ok=True)
    with (folder / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'state.json').exists():
            raise RuntimeError('Cloud group already attempted; inspect before retrying')
        state = dict(status='starting', group=group, tasks=tasks, results=[],
                     host=socket.gethostname(), pid=os.getpid(), started_utc=now(),
                     plan_sha256=manifest['plan_sha256'],
                     manifest_sha256=sha(run / 'cloud/manifest.json'))
        write(folder / 'state.json', state)
        try:
            import torch
            from downstream.diffusion_edit_eval import check
            torch.set_num_threads(4)
            assert torch.cuda.is_available(), 'CUDA unavailable on allocated worker'
            gate = check(run / 'checkpoint', 'cuda')
            assert gate['status'] == 'passed'
            assert gate['model']['model_sha256'] == manifest['model_sha256']
            gate.update(host=socket.gethostname(), gpu=torch.cuda.get_device_name(0),
                        checked_utc=now())
            write(folder / 'cuda_adapter_checks.json', gate)
            torch.cuda.empty_cache()
            for task in tasks:
                out = run / task
                if (out / 'status.json').exists():
                    raise RuntimeError('Task already attempted: ' + task)
                # Exclusive claim persists across failures; no automatic retry.
                with (out / 'cloud_claim.json').open('x') as claim:
                    json.dump(dict(group=group, started_utc=now(),
                                   manifest_sha256=state['manifest_sha256']), claim)
                state.update(status='running', current_task=task, updated_utc=now())
                write(folder / 'state.json', state)
                with (out / 'run.log').open('x') as log:
                    child = subprocess.run(
                        [sys.executable, '-B', str(run / 'run_suite.py'),
                         '--run', str(run), '--task', task],
                        stdout=log, stderr=subprocess.STDOUT)
                path = out / 'status.json'
                task_state = json.loads(path.read_text()) if path.exists() else {}
                if task_state.get('status') not in ('success', 'failed'):
                    task_state.update(status='failed', task=task,
                                      returncode=child.returncode,
                                      error='Worker exited without terminal status')
                    write(path, task_state)
                state['results'].append(dict(task=task, returncode=child.returncode,
                                             status=task_state['status']))
                write(folder / 'state.json', state)
                report(run)
                publish(run, folder / 'publication.log')
            if any(r['returncode'] or r['status'] != 'success' for r in state['results']):
                raise RuntimeError('One or more tasks failed; inspect individual logs')
            state.update(status='success', current_task=None)
        except BaseException as error:
            state.update(status='failed', error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            state.update(finished_utc=now(), updated_utc=now())
            write(folder / 'state.json', state)
            report(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--group', type=int)
    mode.add_argument('--drain-local', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'cloud/manifest.json').read_text())
    assert sha(run / 'cloud/manifest.json') == os.environ['DIFFEDIT_CLOUD_MANIFEST_SHA']
    assert sha(__file__) == manifest['launcher_sha256']
    assert sha(run / 'plan.json') == manifest['plan_sha256']
    plan = verify(run)
    assert plan['model_sha256'] == manifest['model_sha256']
    assert os.environ['TCR_EVAL_STEP'] == str(plan['step'])
    assert os.environ['TCR_EVAL_SHA'] == plan['model_sha256']
    assert os.environ['TCR_EVAL_TAG'] == run.name
    assert Path(os.environ['TCR_EVAL_CKPT']) == run / 'checkpoint'
    tasks = [task for group in manifest['groups'] for task in group]
    assert len(tasks) == len(set(tasks)) and set(tasks) <= set(plan['tasks'])
    assert manifest['local_inflight_task'] not in tasks
    if args.drain_local:
        drain_local(run)
    else:
        assert 0 <= args.group < len(manifest['groups'])
        cloud_group(run, manifest, args.group)


if __name__ == '__main__':
    main()
