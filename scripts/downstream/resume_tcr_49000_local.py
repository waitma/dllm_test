"""Resume only held20 -> large T1 after audited serialization recovery.

Activate pllm then protenix_abtcr. Run --plan /abs/resume_plan.json --check-only,
then the same command without --check-only in tmux. No cloud writes/retries.
Historical failed states stay immutable; completed jobs need hashed audit proof.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import yaml

from scripts.downstream.run_tcr_local_short import ROOT, assert_cloud_killed, now, sha
from scripts.downstream.monitor_tcr_49000_local import DONE, atomic, verify_report
from scripts.downstream import run_tcr_native_generation as generation
from scripts.downstream.preflight_tcr_suite import verify_gates


def completed_jobs(plan):
    jobs = []
    for source in plan['completed_sources']:
        assert sha(source['status']) == source['status_sha256'], 'historical status changed'
        state = json.loads(Path(source['status']).read_text())
        assert state['global_step'] == 49000 and state['checkpoint'] == plan['checkpoint']
        for key in source['keys']:
            matches = [j for j in state['jobs'] if j['key'] == key]
            assert len(matches) == 1
            job = dict(matches[0])
            assert job['status'] == DONE and job['exit_code'] == 0
            job['inherited_from_status'] = source['status']
            jobs.append(job)
    recovery = plan['recovery']
    assert sha(recovery['status']) == recovery['status_sha256']
    state = json.loads(Path(recovery['status']).read_text())
    assert state['global_step'] == 49000 and state['checkpoint'] == plan['checkpoint']
    job = dict(next(j for j in state['jobs'] if j['key'] == 't4-benchmark14'))
    assert job['status'] == 'failed' and job['exit_code'] != 0
    output = generation.output_for('benchmark14')
    manifest = json.loads((output / 'run_manifest.json').read_text())
    assert manifest['status'] == 'success' and manifest['identity'] == generation.identity()
    assert manifest['recovery']['source'] == job['result_dir']
    for path, digest in manifest['recovery']['source_sha256'].items():
        assert sha(path) == digest
    job.update(status=DONE, exit_code=0, result_dir=str(output),
        original_execution_status='failed', completion_mode='cpu_rescore_no_regeneration',
        inherited_from_status=recovery['status'], recovery_manifest=str(output / 'run_manifest.json'))
    jobs.append(job)
    assert len(jobs) == len({j['key'] for j in jobs}) == 8
    covered = []
    for group in plan['completed_audits']:
        assert sha(group['report']) == group['report_sha256']
        verify_report(group)
        report = json.loads(Path(group['report']).read_text())
        artifacts = list(report.get('tracks', report.get('tasks', {})).values())
        if not artifacts:
            artifacts = [report['result']]
        outputs = {a['output'] for a in artifacts}
        assert {j['result_dir'] for j in jobs if j['key'] in group['jobs']} == outputs
        covered.extend(group['jobs'])
    assert len(covered) == 8 and set(covered) == {j['key'] for j in jobs}
    return jobs


def verify_plan(plan):
    assert plan['global_step'] == 49000 and plan['checkpoint'] == str(generation.CKPT)
    assert [j['key'] for j in plan['jobs']] == ['t4-held20', 't1-cdr3b']
    assert verify_gates(plan['generation_gate']) == plan['input_gates']
    completed = completed_jobs(plan)
    for job in completed + plan['jobs']:
        assert sha(job['yaml']) == job['yaml_sha256']
        assert job['cloud_status_at_migration'] == 'Killed'
    for job in plan['jobs']:
        config = yaml.safe_load(Path(job['yaml']).read_text())
        assert config['TaskName'] == job['task_name'] and config['Preemptible'] is False
        subprocess.run(['bash', '-n'], input=config['Entrypoint'], text=True, check=True)
        assert_cloud_killed(job)
        if Path(job['result_dir']).exists():
            raise FileExistsError(job['result_dir'])
    assert plan['jobs'][0]['result_dir'] == str(generation.output_for('held20'))
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    completed = verify_plan(plan)
    if args.check_only:
        print('TCR_REPAIR_RESUME_PLAN_PASSED', flush=True)
        return
    output = Path(plan['runner_dir'])
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for parent in [*[Path(s['status']).parent for s in plan['completed_sources']], output]:
            lock = stack.enter_context((parent / 'run.lock').open('a'))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / 'status.json'
        if path.exists():
            raise FileExistsError('never replay an already started resume queue')
        state = dict(status='running', host=socket.gethostname(), pid=os.getpid(), start_utc=now(),
            plan=str(args.plan.resolve()), plan_sha256=sha(args.plan), runner_sha256=sha(__file__),
            checkpoint=plan['checkpoint'], global_step=49000, mode='local_serial_audited_recovery',
            jobs=completed + [dict(j, status='pending') for j in plan['jobs']])
        def save():
            state['updated_utc'] = now()
            atomic(path, state)
        save()
        try:
            for job in state['jobs'][8:]:
                assert sha(args.plan) == state['plan_sha256'] and sha(__file__) == state['runner_sha256']
                assert verify_gates(plan['generation_gate']) == plan['input_gates']
                assert_cloud_killed(job)
                free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip())
                if free < 24 * 1024:
                    raise RuntimeError('less than 24 GiB free; leave unrelated GPU processes untouched')
                assert sha(job['yaml']) == job['yaml_sha256']
                if Path(job['result_dir']).exists():
                    raise FileExistsError(job['result_dir'])
                config = yaml.safe_load(Path(job['yaml']).read_text())
                job.update(status='running', start_utc=now(), log=str(output / (job['key'] + '.log')))
                save()
                started = time.monotonic()
                with Path(job['log']).open('x') as log:
                    process = subprocess.run(['bash', '-c', config['Entrypoint']], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT)
                job.update(exit_code=process.returncode, end_utc=now(),
                    elapsed_seconds=round(time.monotonic() - started, 2))
                if process.returncode:
                    job['status'] = 'failed'
                    raise RuntimeError(f'{job["key"]} failed; see {job["log"]}')
                for name in job['expected_results']:
                    artifact = json.loads((Path(job['result_dir']) / name).read_text())
                    if name == 'run_manifest.json':
                        assert artifact['status'] == 'success'
                job['status'] = DONE
                save()
                print(now(), job['key'], DONE, flush=True)
            state['status'] = DONE
        except Exception as exc:
            state.update(status='failed', error=repr(exc))
            raise
        finally:
            save()


if __name__ == '__main__':
    main()
