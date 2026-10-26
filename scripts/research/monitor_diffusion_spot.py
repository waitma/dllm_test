"""Observe one diffusion submission without submitting or cancelling jobs.

Activate pllm; python -B /ABS/scripts/research/monitor_diffusion_spot.py
--control /ABS/output/.../spot_20261022 [--once]. Polls every 60 seconds, writes
monitor_status.json/history.jsonl, and exits on STOP_MONITOR or a platform terminal
state. submission.json supplies the task ID after a successful submission.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from run_diffusion_spot import now, read, write


TERMINAL = {'Success', 'Failed', 'Killed'}


def platform(task_id):
    env = {k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
    result = subprocess.run(['volc', 'ml_task', 'list', '-n', task_id, '-s',
        'Initialized,Queue,Staging,Running,Killing,Success,Failed,Killed',
        '-o', 'json', '--limit', '20'], env=env, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr)[-2000:])
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\[', result.stdout):
        try:
            rows, _ = decoder.raw_decode(result.stdout[match.start():])
        except ValueError:
            continue
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get('JobId') == task_id:
                    return row
    raise RuntimeError('Exact submitted task ID was not returned by the platform')


def observe(control):
    control = Path(control).resolve()
    spec = read(control / 'spec.json')
    output = Path(spec['output_dir']).resolve()
    state = read(output / 'status.json')
    latest = read(output / 'latest.json')
    submission_path = control / 'submission.json'
    submission = read(submission_path) if submission_path.exists() else {}
    age = max(0, time.time() - state.get('time_unix', 0))
    result = dict(updated_utc=now(), task_id=submission.get('task_id'),
        preemptible=submission.get('preemptible', spec.get('preemptible', True)),
        platform_status='not_submitted', training_status='not_submitted',
        observed_step=state.get('step'), target_steps=state.get('target_steps'),
        latest_complete_step=latest.get('step'), heartbeat_age_seconds=round(age, 1),
        last_training_utc=datetime.fromtimestamp(state.get('time_unix', 0), timezone.utc).isoformat(),
        loss=state.get('loss'), seconds_per_step=state.get('seconds_per_step'),
        alerts=[], monitor_pid=os.getpid())
    if submission.get('task_id'):
        try:
            job = platform(submission['task_id'])
            status = job['Status']
            result.update(platform_status=status, platform=job)
            if status in TERMINAL:
                done_path = output / 'COMPLETE.json'
                done = read(done_path) if done_path.exists() else {}
                confirmed = status == 'Success' and done.get('status') == 'success' and done.get('step') == state.get('target_steps')
                result['training_status'] = 'completed' if confirmed else 'stopped_or_failed'
            elif status == 'Running':
                if state.get('status') == 'success' and state.get('step') == state.get('target_steps'):
                    result['training_status'] = 'completed_waiting_platform_exit'
                elif age < 600 and state.get('status') == 'training':
                    result['training_status'] = 'training'
                else:
                    runtime = read(control / 'runtime.json') if (control / 'runtime.json').exists() else {}
                    recent_start = time.time() - runtime.get('started_unix', 0) < 900
                    result['training_status'] = 'starting' if recent_start else 'stalled_or_starting'
                    if not recent_start:
                        result['alerts'].append('Running but no fresh training heartbeat for at least 10 minutes')
            else:
                result['training_status'] = status.lower()
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
            result.update(platform_status='query_error', training_status='unknown', platform_error=str(exc))
            result['alerts'].append('Platform query failed; cached training state is not proof of a live job')
    if result['training_status'] == 'training':
        for key in ('loss', 'grad_norm', 'learning_rate'):
            value = state.get(key)
            if value is not None and not math.isfinite(value):
                result['alerts'].append('Nonfinite training metric: ' + key)
    runtime_path = control / 'runtime.json'
    if runtime_path.exists():
        runtime = read(runtime_path)
        result['runtime'] = runtime
        attempt = Path(runtime['bundle']) if runtime.get('bundle') else None
        if attempt:
            restores = [read(p) for p in sorted(attempt.glob('restore_rank*.json'))]
            result['restored_ranks'] = len(restores)
            result['restore_verified'] = len(restores) == 8 and all(
                x.get('status') == 'passed' and x.get('restored_step') == runtime.get('starting_step') for x in restores)
    return result


def sync_milestone(control, result, previous):
    control = Path(control).resolve()
    task_id = result.get('task_id')
    status = result['platform_status']
    if not task_id or status in ('query_error', 'not_submitted'):
        return
    if (status, result.get('training_status')) == (previous.get('platform_status'), previous.get('training_status')):
        return
    spec = read(control / 'spec.json')
    preemptible = result.get('preemptible', True)
    mode = '闲时' if preemptible else '非闲时'
    doc = Path(spec['project_process']).resolve()
    with (control / 'project_process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        content = doc.read_text()
        content = re.sub(r'^Last updated \(UTC\): .*$', 'Last updated (UTC): ' + result['updated_utc'], content, count=1, flags=re.M)
        start = content.index('## Active Volc Training Tasks')
        end = content.find('\n## ', start + 4)
        if end < 0:
            end = len(content)
        section = content[start:end]
        section = re.sub(r'^Last updated \(UTC\): .*$', 'Last updated (UTC): ' + result['updated_utc'], section, count=1, flags=re.M)
        section = '\n'.join(line for line in section.split('\n') if not (line.startswith('|') and task_id in line))
        if status not in TERMINAL:
            row = (f"| diffusion/edit {mode}续训 | `{task_id}` | {result['updated_utc']} {status}；"
                   f"{result['training_status']}；最近日志 {result['observed_step']}/200000，完整保存 {result['latest_complete_step']} | "
                   f"[监控]({control / 'monitor_status.json'}) |")
            marker = '|---|---|---|---|'
            section = section.replace(marker, marker + '\n' + row, 1)
        content = content[:start] + section + content[end:]
        content += (f"\n\n## {result['updated_utc'][:10]} diffusion/edit {mode}任务状态变化\n\n"
                    f"{result['updated_utc']} task_id=`{task_id}`；平台`{status}`；"
                    f"训练观察`{result['training_status']}`；最后日志{result['observed_step']}，"
                    f"最新完整{result['latest_complete_step']}；Preemptible={str(preemptible).lower()}。"
                    f"[监控]({control / 'monitor_status.json'})。\n")
        temporary = doc.with_name(doc.name + '.spot.tmp')
        temporary.write_text(content)
        temporary.replace(doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    control = args.control.resolve()
    with (control / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (control / 'STOP_MONITOR').exists():
            path = control / 'monitor_status.json'
            previous = read(path) if path.exists() else {}
            try:
                result = observe(control)
                sync_milestone(control, result, previous)
            except Exception as exc:
                result = dict(updated_utc=now(), training_status='monitor_error', platform_status='unknown', error=repr(exc))
            write(path, result)
            with (control / 'monitor_history.jsonl').open('a') as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ('platform','runtime')}, ensure_ascii=False), flush=True)
            if args.once or result.get('platform_status') in TERMINAL:
                return
            for _ in range(60):
                if (control / 'STOP_MONITOR').exists():
                    return
                time.sleep(1)


if __name__ == '__main__':
    main()
