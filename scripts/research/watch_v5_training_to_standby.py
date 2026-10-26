"""Observe the existing V5 training and its automatic return to eight-GPU standby.

Activate pllm, then run python -B /absolute/path/watch_v5_training_to_standby.py.
This observer writes reports and Markdown only. It never starts/stops GPU work,
publishes tickets, removes control files, or declares final model acceptance.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
POOL = ROOT/'output/immune_training_container_c20250601_20261007'
EVIDENCE = ROOT/'output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008'
RESULT = POOL/'results/v5_scratch_launch_v3_20261007.json'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def standby_status():
    state = read(POOL/'state.json') or {}
    result = read(RESULT)
    pids = state.get('standby_pids', [])
    logs = []
    for rank in range(8):
        path = POOL/'logs'/f'standby{rank}.log'
        fresh = bool(result and path.exists() and path.stat().st_mtime >= result.get('finished', float('inf')))
        initialized = False
        if fresh:
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size-8192))
                tail = stream.read().decode(errors='replace')
            for line in tail.splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (item.get('kind') == 'maintenance_only' and rank < len(pids)
                        and item.get('pid') == pids[rank] and item.get('gpu')):
                    initialized = True
            initialized = (initialized and time.time()-path.stat().st_mtime < 120
                           and 'Traceback (most recent call last)' not in tail)
        logs.append({'rank': rank, 'updated_after_training_exit': fresh,
                     'current_pid_gpu_initialization_logged': initialized})
    restored = bool(result and state.get('state') == 'standby'
                    and time.time()-state.get('time', 0) < 15
                    and len(pids) == len(set(pids)) == 8
                    and all(item['current_pid_gpu_initialization_logged'] for item in logs))
    return {'container_state': state, 'training_result': result,
            'standby_log_updates': logs, 'eight_standby_observed': restored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=180000)
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError('Observation duration must be positive')
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('v5_existing_observer', ROOT/'scripts/research/watch_v5_direct.py')
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    observer.TASK = 't-20261007153431-vhx2f'
    observer.NAME = 'immune-v5-devpool-8gpu-20261007'
    observer.ATTEMPT = 'attempt2'
    started = datetime.now(timezone.utc).isoformat()
    deadline = time.monotonic()+args.seconds
    terminal_deadline = None
    final = None
    with (EVIDENCE/'observer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.monotonic() < deadline:
            now = datetime.now(timezone.utc).isoformat()
            value = {'at_utc': now, 'started_at': started, 'pid': os.getpid(),
                     'bounded_seconds': args.seconds, 'training_read_only': True}
            try:
                observed, platform_status = observer.observe()
                value.update({key: observed.get(key) for key in
                              ('training_step', 'training_status', 'latest_logged_loss', 'error_visible')})
                value['platform_status'] = platform_status
                value.update(standby_status())
                if value['training_result'] or observed.get('formal_complete'):
                    if terminal_deadline is None:
                        terminal_deadline = time.monotonic()+300
                    if value['eight_standby_observed']:
                        final = 'training_exited_standby_observed'
                    elif time.monotonic() >= terminal_deadline:
                        final = 'training_terminal_standby_not_confirmed'
                if platform_status in ('Success', 'Failed', 'Killed'):
                    final = final or 'platform_ended'
            except Exception as exc:
                value['observation_error'] = repr(exc)
            value['observer_status'] = final or 'running'
            observer.atomic(EVIDENCE/'latest.json', value)
            with (EVIDENCE/'observations.jsonl').open('a') as stream:
                stream.write(json.dumps(value, ensure_ascii=False)+'\n')
            print(json.dumps({key: value.get(key) for key in
                  ('at_utc', 'pid', 'observer_status', 'training_step', 'training_status',
                   'eight_standby_observed', 'error_visible', 'observation_error')}), flush=True)
            if final:
                break
            time.sleep(min(30, max(0, deadline-time.monotonic())))
        value['observer_status'] = final or 'observation_window_expired'
        value['closed_at'] = datetime.now(timezone.utc).isoformat()
        value['final_model_acceptance_performed'] = False
        observer.atomic(EVIDENCE/'latest.json', value)
        observer.atomic(EVIDENCE/'exit.json', value)
        note = (f"\n\n## {value['closed_at']} 训练至占卡只读监督结束\n\n"
                f"监督结果={value['observer_status']}；训练票据状态={value.get('training_status')}；"
                f"八卡占卡恢复观察={value.get('eight_standby_observed', False)}。"
                f"证据：{EVIDENCE}/exit.json。未变更GPU进程或票据；完整训练产物最终验收另行核对。\n")
        for path in (ROOT/'PROJECT_PROCESS.md', ROOT/'docs/EXPERIMENT_LOG.md'):
            with path.open('a') as stream:
                stream.write(note)
        progress = ROOT/'PROJECT_PROCESS.md'
        progress.write_text(re.sub(r'Last updated: .*', 'Last updated: '+value['closed_at'],
                                   progress.read_text(), count=1))


if __name__ == '__main__':
    main()
