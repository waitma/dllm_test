"""Continue the approved 49000 TCR jobs locally after all six short jobs finish.

Activate pllm then protenix_abtcr and run in a dedicated tmux session:
python scripts/downstream/run_tcr_local_remaining.py --plan /abs/plan.json
Use --check-only before starting. This never cancels/submits cloud jobs or
changes evaluation parameters. Corresponding cloud jobs must already be Killed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import yaml

from scripts.downstream.run_tcr_local_short import ROOT, assert_cloud_killed, now, sha


def verify_prerequisite(plan):
    previous = json.loads(Path(plan['prerequisite_status']).read_text())
    expected = {'t2a-v2', 't2b-v2', 't3broad-v2', 't1-others_cdr3b',
                't1-cdr3ab', 't1-others_longab'}
    if (previous['status'] != 'executed_pending_result_audit'
            or {j['key'] for j in previous['jobs']} != expected
            or len(previous['jobs']) != len(expected)
            or previous['global_step'] != 49000
            or previous['checkpoint'] != plan['checkpoint']):
        raise ValueError('all six prerequisite jobs, including others, must finish first')
    for job in previous['jobs']:
        if job['status'] != 'executed_pending_result_audit' or job['exit_code'] != 0:
            raise ValueError('prerequisite job did not finish successfully')
        for name in job['expected_results']:
            json.loads((Path(job['result_dir']) / name).read_text())


def verify_plan(plan):
    verify_prerequisite(plan)
    allowed = {'t3deep-v2', 't4-benchmark14', 't4-held20', 't1-cdr3b'}
    keys = [j['key'] for j in plan['jobs']]
    if not keys or not set(keys) <= allowed or len(keys) != len(set(keys)):
        raise ValueError('unexpected or duplicate remaining task')
    if plan['global_step'] != 49000:
        raise ValueError('only the approved 49000 checkpoint is allowed')
    for job in plan['jobs']:
        path = Path(job['yaml'])
        if sha(path) != job['yaml_sha256']:
            raise ValueError(f'YAML drift: {path}')
        config = yaml.safe_load(path.read_text())
        if config['TaskName'] != job['task_name'] or config['Preemptible'] is not False:
            raise ValueError('wrong cloud task configuration')
        subprocess.run(['bash', '-n'], input=config['Entrypoint'], text=True, check=True)
        assert_cloud_killed(job)
        if Path(job['result_dir']).exists():
            raise FileExistsError(f'refuse overwrite/resume: {job["result_dir"]}')
    from scripts.downstream.preflight_tcr_suite import verify_gates
    if verify_gates() != plan['input_gates']:
        raise ValueError('input gate identity changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    verify_plan(plan)
    if args.check_only:
        print('REMAINING_LOCAL_PLAN_CHECK_PASSED', flush=True)
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('local GPU required')
    output = Path(plan['runner_dir'])
    output.mkdir(parents=True, exist_ok=True)
    # Hold the prerequisite runner lock as well: no concurrent local queues.
    prior_lock = Path(plan['prerequisite_status']).parent / 'run.lock'
    with prior_lock.open('a') as previous_lock, (output / 'run.lock').open('a') as lock:
        for handle in (previous_lock, lock):
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / 'status.json'
        if path.exists():
            raise FileExistsError('queue already started; do not silently replay')
        state = {'status': 'running', 'host': socket.gethostname(), 'pid': os.getpid(),
            'start_utc': now(), 'plan': str(args.plan.resolve()), 'plan_sha256': sha(args.plan),
            'runner_sha256': sha(__file__), 'helper_sha256': sha(ROOT / 'scripts/downstream/run_tcr_local_short.py'),
            'checkpoint': plan['checkpoint'], 'global_step': 49000, 'mode': 'local_serial',
            'prerequisite_status_sha256': sha(plan['prerequisite_status']),
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
                    raise RuntimeError('less than 24 GiB free; leave other GPU work untouched')
                if sha(job['yaml']) != job['yaml_sha256']:
                    raise ValueError('YAML changed during local execution')
                if Path(job['result_dir']).exists():
                    raise FileExistsError(job['result_dir'])
                config = yaml.safe_load(Path(job['yaml']).read_text())
                job.update(status='running', start_utc=now(), log=str(output / (job['key'] + '.log')))
                save()
                started = time.monotonic()
                with Path(job['log']).open('x') as log:
                    result = subprocess.run(['bash', '-c', config['Entrypoint']],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                job.update(exit_code=result.returncode, elapsed_seconds=round(time.monotonic()-started, 2), end_utc=now())
                if result.returncode:
                    job['status'] = 'failed'
                    raise RuntimeError(f'{job["key"]} exited {result.returncode}; see log')
                for name in job['expected_results']:
                    artifact = json.loads((Path(job['result_dir']) / name).read_text())
                    if name == 'run_manifest.json' and artifact.get('status') != 'success':
                        raise ValueError('evaluation manifest is not successful')
                job['status'] = 'executed_pending_result_audit'
                save()
                print(job['key'], job['status'], job['elapsed_seconds'], flush=True)
            state['status'] = 'executed_pending_result_audit'
        except Exception as exc:
            state.update(status='failed', error=repr(exc))
            raise
        finally:
            save()


if __name__ == '__main__':
    main()
