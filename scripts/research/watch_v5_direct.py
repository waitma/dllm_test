"""Read-only job supervision with compact status updates; no submissions/retries.

Activate pllm, then python -B /absolute/path/watch_v5_direct.py --seconds 900.
Reads the existing task and shared reports; writes only supervision evidence and
the matching project status rows. It never changes a training process or data.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
SUPERVISION = ROOT/'output/immune_v5_object302_noesmc_300k_b512_supervision'
RUN = ROOT/'output/immune_v5_object302_noesmc_300k_b512'
TASK = 't-20261007151807-cq4fb'
NAME = 'immune-v5-object302-scratch-8gpu-20261007'
ATTEMPT = 'attempt1'
CONTAINER = ROOT/'output/immune_training_container_c20250601_20261007'
CONTAINER_TASK = 't-20261007153431-vhx2f'
TICKET = 'v5_scratch_launch_v3_20261007.json'


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def training_outcome():
    """Separate persistent resource state from its one authorized training ticket."""
    if TASK != CONTAINER_TASK:
        return {}
    state_path = CONTAINER/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    result_path = CONTAINER/'results'/TICKET
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    if result is not None:
        status = result.get('state', 'unknown')
    elif state and state.get('state') == 'training' and Path(state.get('ticket', '')).name == TICKET:
        status = 'running'
    else:
        status = 'not_confirmed_running'
    return {'container_state': state, 'training_status': status,
            'training_result': result, 'training_result_path': str(result_path)}


def observe():
    now = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(['volc', 'ml_task', 'list', '-n', NAME, '--status',
        'Initialized,Queue,Staging,Running,Killing,Success,Failed,Killed', '-o', 'json', '--limit', '20'],
        text=True, capture_output=True, timeout=30)
    value = {'at_utc': now, 'task_id': TASK, 'query_returncode': result.returncode,
             'query_output': result.stdout+result.stderr, 'training_step': 0,
             'initialization': 'scratch', 'model_parameters_frozen': False}
    records = []
    for line in result.stdout.splitlines():
        if line.startswith('['):
            records = json.loads(line)
    matches = [r for r in records if r['JobId'] == TASK]
    value['platform'] = matches[0] if len(matches) == 1 else None
    profiles = [json.loads(p.read_text()) for p in sorted((SUPERVISION/ATTEMPT).glob('profile_rank*.json'))]
    value['synthetic_profiles'] = profiles
    value['profile_passed_ranks'] = sum(p.get('status') == 'passed' for p in profiles)
    timing = RUN/'step_timing.jsonl'
    rows = []
    if timing.exists():
        for line in timing.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # An in-flight last line is not a completed step.
    if rows:
        value['training_step'] = rows[-1]['step']
        value['latest_step'] = rows[-1]
        steady = [r['interval_seconds'] for r in rows[-50:] if not r['first_or_after_save']]
        value['recent_median_step_seconds'] = statistics.median(steady) if steady else None
    log = SUPERVISION/('torchrun_'+ATTEMPT+'.log')
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-30000))
            tail = stream.read().decode(errors='replace')
        value['log_tail'] = tail[-12000:]
        value['error_visible'] = any(x in tail for x in ('Traceback (most recent call last)', 'CUDA out of memory', 'ChildFailedError'))
        losses = re.findall(r"'loss': ([+-]?(?:nan|inf|[0-9.eE+\-]+))", tail, re.I)
        loss = float(losses[-1]) if losses else None
        value['latest_logged_loss'] = loss if loss is None or math.isfinite(loss) else losses[-1]
        if loss is not None:
            value['loss_is_finite'] = math.isfinite(loss)
            value['error_visible'] |= not value['loss_is_finite']
    value.update(training_outcome())
    if value.get('training_status') == 'failed':
        value['error_visible'] = True
    value['formal_complete'] = (RUN/'COMPLETE.json').exists()
    atomic(SUPERVISION/'latest.json', value)
    with (SUPERVISION/'observations.jsonl').open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
    status = value['platform']['Status'] if value['platform'] else 'query_unresolved'
    progress = f"{status}；synthetic {value['profile_passed_ranks']}/8；正式{value['training_step']}/300000步"
    if 'training_status' in value:
        progress = '容器'+progress+'；训练'+value['training_status']
    if value.get('error_visible'):
        progress += '；训练子任务报错，检查原日志'
    path = ROOT/'PROJECT_PROCESS.md'
    text = path.read_text()
    text = re.sub(r'Last updated: .*', 'Last updated: '+now, text, count=1)
    row = f'| {NAME} | {TASK} | {now} {progress} | [状态证据]({SUPERVISION}/latest.json) |'
    text = re.sub(r'^\| '+re.escape(NAME)+r' \|.*$',
                  '' if status in ('Success', 'Failed', 'Killed') else row, text, flags=re.M)
    next_save = min(300000, (value['training_step']//1000+1)*1000)
    text = re.sub(r'^\| 新版本模型／生成实现 \|.*$',
                  f'| 新版本模型／生成实现 | {progress} | 核实已保存产物，并监督{next_save}步验证/保存 |', text, flags=re.M)
    path.write_text(text)
    state = ROOT/'output/v7_formal_supervision_20261005/root_observation_latest.json'
    current = json.loads(state.read_text())
    current['current_snapshot'] = {'scope': 'new_v5_scratch_formal', 'at_utc': now,
        'task_id': TASK, 'status': status, 'step': value['training_step'],
        'training_status': value.get('training_status'),
        'ablation_status': 'paused_by_user', 'evidence': str(SUPERVISION/'latest.json')}
    observation = current.setdefault('active_bounded_observation', {})
    observation.update(scope='persistent_v5_container' if 'devpool' in NAME else 'new_v5_direct',
                       task_id=TASK, attempt=ATTEMPT, observed_at=now,
                       evidence=str(SUPERVISION/'observations.jsonl'))
    atomic(state, current)
    return value, status


def main():
    global TASK, NAME, ATTEMPT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--task-id', default=TASK)
    parser.add_argument('--task-name', default=NAME)
    parser.add_argument('--attempt', default=ATTEMPT)
    args = parser.parse_args()
    TASK, NAME, ATTEMPT = args.task_id, args.task_name, args.attempt
    SUPERVISION.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic()+args.seconds
    while True:
        value, status = observe()
        print(json.dumps({k:value.get(k) for k in ('at_utc','training_step','profile_passed_ranks',
              'recent_median_step_seconds','latest_logged_loss','error_visible','training_status')} | {'status':status}), flush=True)
        if (status in ('Success','Failed','Killed') or value.get('error_visible')
                or value.get('training_status') in ('succeeded','failed')
                or value['formal_complete'] or time.monotonic() >= deadline):
            return
        time.sleep(min(30, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
