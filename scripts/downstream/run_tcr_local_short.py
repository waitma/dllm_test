"""Run the approved short TCR jobs locally, serially, using unchanged YAMLs.

Activate pllm then protenix_abtcr. Run under a dedicated tmux session:
python scripts/downstream/run_tcr_local_short.py --plan /abs/local_plan.json
Never submits/cancels cloud jobs. Each matching cloud task must already be Killed.
Stops on failure or after a job exceeds ten minutes; never edits model/data/code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_cloud_killed(job):
    result = subprocess.run(['bash', str(ROOT / 'scripts/volc-no-proxy.sh'),
        'ml_task', 'get', '-i', job['cloud_task_id'], '-o', 'json',
        '--format', 'JobId,JobName,Status'], check=True, capture_output=True, text=True)
    # CLI upgrade notices can precede the JSON array.
    start = result.stdout.find('[{')
    if start < 0:
        raise ValueError('no machine-readable cloud status')
    rows, _ = json.JSONDecoder().raw_decode(result.stdout[start:])
    matches = [r for r in rows if r['JobId'] == job['cloud_task_id']]
    if len(matches) != 1 or matches[0]['Status'] != 'Killed' or matches[0]['JobName'] != job['task_name']:
        raise ValueError(f'cloud job not safely cancelled: {matches}')


def verify_plan(plan):
    allowed = {'t2a-v2', 't2b-v2', 't3broad-v2', 't1-cdr3ab',
               't1-others_cdr3b', 't1-others_longab'}
    keys = [j['key'] for j in plan['jobs']]
    if set(keys) != allowed or len(keys) != len(allowed):
        raise ValueError('unexpected local scope or duplicated tasks')
    for job in plan['jobs']:
        path = Path(job['yaml'])
        if sha(path) != job['yaml_sha256']:
            raise ValueError(f'YAML drift: {path}')
        config = yaml.safe_load(path.read_text())
        if config['TaskName'] != job['task_name'] or config['Preemptible'] is not False:
            raise ValueError('wrong task configuration')
        subprocess.run(['bash', '-n'], input=config['Entrypoint'], text=True, check=True)
        assert_cloud_killed(job)
        if Path(job['result_dir']).exists():
            raise FileExistsError(f'refuse to overwrite/resume an existing result: {job["result_dir"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    verify_plan(plan)
    from scripts.downstream.preflight_tcr_suite import verify_gates
    if verify_gates() != plan['input_gates']:
        raise ValueError('input gate identity changed')
    if args.check_only:
        print('LOCAL_PLAN_CHECK_PASSED', flush=True)
        return
    import torch
    assert torch.cuda.is_available(), 'local GPU required'
    output = Path(plan['runner_dir'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / 'status.json'
        if path.exists():
            raise FileExistsError('run already started; do not silently replay')
        state = {'status': 'running', 'host': socket.gethostname(), 'pid': os.getpid(),
            'start_utc': now(), 'plan': str(args.plan.resolve()), 'plan_sha256': sha(args.plan),
            'runner_sha256': sha(__file__), 'checkpoint': plan['checkpoint'],
            'global_step': 49000, 'mode': 'local_serial',
            'jobs': [dict(j, status='pending') for j in plan['jobs']]}
        def save():
            state['updated_utc'] = now()
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(path)
        save()
        try:
            for job in state['jobs']:
                assert_cloud_killed(job)
                free_mb = int(subprocess.check_output(['nvidia-smi', '--id=0',
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                if free_mb < 24 * 1024:
                    raise RuntimeError('less than 24 GiB free; do not contend with other GPU work')
                if sha(job['yaml']) != job['yaml_sha256']:
                    raise ValueError('YAML changed while local queue was running')
                if Path(job['result_dir']).exists():
                    raise FileExistsError(job['result_dir'])
                config = yaml.safe_load(Path(job['yaml']).read_text())
                job.update(status='running', start_utc=now(), log=str(output / (job['key'] + '.log')))
                save()
                started = time.monotonic()
                with Path(job['log']).open('x') as log:
                    # Same environment, seed, full data, epochs, batch size and
                    # output identity as the cancelled cloud job.
                    result = subprocess.run(['bash', '-c', config['Entrypoint']],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                job.update(exit_code=result.returncode, elapsed_seconds=round(time.monotonic()-started, 2), end_utc=now())
                if result.returncode:
                    job['status'] = 'failed'
                    raise RuntimeError(f'{job["key"]} exited {result.returncode}; see log')
                for name in job['expected_results']:
                    json.loads((Path(job['result_dir']) / name).read_text())
                job['status'] = 'executed_pending_result_audit'
                save()
                print(job['key'], job['status'], job['elapsed_seconds'], flush=True)
                if job['elapsed_seconds'] > 600 and job is not state['jobs'][-1]:
                    state['status'] = 'paused_after_slow_job'
                    state['reason'] = 'Measured runtime exceeded short-job budget; remaining jobs not started'
                    break
            else:
                state['status'] = 'executed_pending_result_audit'
        except Exception as exc:
            state.update(status='failed', error=repr(exc))
            raise
        finally:
            save()


if __name__ == '__main__':
    main()
