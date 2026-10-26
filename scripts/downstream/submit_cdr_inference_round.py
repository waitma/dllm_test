"""Activate pllm; python -B FILE --round 1|2|3 [--poll].

Submit one frozen inference-only round, immediately recording the returned
task ID in PROJECT_PROCESS.md. Poll the exact ID; never relaunch on timeout.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
OUT = ROOT/'output/downstream_generation/cdr_inference_rounds_20261015'
CLI = '/root/.volc/bin/volc'


def read_json_output(value):
    start = value.find('[{')
    if start < 0:
        raise ValueError('No task JSON in CLI output')
    result, _ = json.JSONDecoder().raw_decode(value[start:])
    assert len(result) == 1
    return result[0]


def update_log(n, task, status, event, run=None, config=None, label=None, event_date=None):
    run = OUT if run is None else Path(run)
    label = f'CDR免训练第{n}轮' if label is None else label
    now = datetime.now(timezone.utc).isoformat()
    path = ROOT/'PROJECT_PROCESS.md'
    text = path.read_text()
    text = re.sub(r'Last updated \(UTC\): .*', 'Last updated (UTC): '+now, text, count=1)
    head, rest = text.split('## Active Volc Training Tasks', 1)
    active, tail = rest.split('## 下一步', 1)
    active = re.sub(r'^.*\| '+re.escape(task)+r' \|.*\n', '', active, flags=re.M)
    active = re.sub(r'Last updated[^\n]*', 'Last updated（CDR免训练评测）：'+now+'。', active, count=1)
    if status in ('Initialized', 'Waiting', 'Queue', 'Staging', 'Running', 'Killing'):
        row = f'| {label} | {task} | {now} {status} | [平台记录]({run}/round{n}_platform.json) |\n'
        anchor = '|---|---|---|---|\n'
        assert anchor in active
        active = active.replace(anchor, anchor+row, 1)
    text = head+'## Active Volc Training Tasks'+active+'## 下一步'+tail
    if event:
        yaml = ROOT/f'eval_jobs/cdr_inference_round{n}_20261015.yml' if config is None else Path(config)
        text += (f'\n## {event_date or "2026-10-15"} {label} {event}\n\n'
                 f'- task_id=`{task}`；状态 `{status}`；记录时间 `{now}`；`Preemptible: false`。\n'
                 f'- 配置：[{yaml.name}]({yaml})；执行产物：[{run.name}]({run})。\n')
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, required=True, choices=(1, 2, 3))
    parser.add_argument('--poll', action='store_true')
    args = parser.parse_args()
    n = args.round
    record = OUT/f'round{n}_submission.json'
    platform = OUT/f'round{n}_platform.json'
    if args.poll:
        task = json.loads(record.read_text())['task_id']
        result = subprocess.run([CLI, 'ml_task', 'get', '-i', task, '-o', 'json',
                                 '--format', 'JobId,JobName,Status,Start,End'],
                                capture_output=True, text=True, check=True)
        state = read_json_output(result.stdout)
        assert state['JobId'] == task
        previous = json.loads(platform.read_text()) if platform.exists() else {}
        state['observed_utc'] = datetime.now(timezone.utc).isoformat()
        platform.write_text(json.dumps(state, indent=2)+'\n')
        if previous.get('Status') != state['Status']:
            update_log(n, task, state['Status'], '终态' if state['Status'] in ('Success', 'Failed', 'Killed', 'Succeeded') else None)
        print(json.dumps(state), flush=True)
        return
    assert not record.exists(), 'Already submitted; poll existing ID'
    assert json.loads((OUT/'gate_cpu.json').read_text())['status'] == 'passed'
    if n > 1:
        # Both confirmation rounds depend on the same frozen R1 selection,
        # not on each other's scores, so they may run independently.
        assert json.loads((OUT/'round1/audit.json').read_text())['status'] == 'passed'
        assert json.loads((OUT/'round1/independent_audit.json').read_text())['status'] == 'passed'
        assert (OUT/'selection.json').exists()
    conf = ROOT/f'eval_jobs/cdr_inference_round{n}_20261015.yml'
    result = subprocess.run([CLI, 'ml_task', 'submit', '--conf', str(conf)], capture_output=True, text=True)
    (OUT/f'round{n}_submit.log').write_text(result.stdout+'\n'+result.stderr)
    result.check_returncode()
    ids = set(re.findall(r't-\d{14}-[a-z0-9]+', result.stdout))
    assert len(ids) == 1, 'Inspect raw submission log before doing anything else'
    task = ids.pop()
    record.write_text(json.dumps(dict(task_id=task, config=str(conf),
        submitted_utc=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
    update_log(n, task, '提交成功，平台状态待查询', 'submit')
    for attempt in range(5):
        queried = subprocess.run([CLI, 'ml_task', 'get', '-i', task, '-o', 'json',
                                  '--format', 'JobId,JobName,Status,Start,End'],
                                 capture_output=True, text=True, check=True)
        try:
            state = read_json_output(queried.stdout)
            break
        except ValueError:
            if attempt == 4: raise
            time.sleep(1)
    state['observed_utc'] = datetime.now(timezone.utc).isoformat()
    platform.write_text(json.dumps(state, indent=2)+'\n')
    update_log(n, task, state['Status'], '初始状态确认')
    print(task, flush=True)


if __name__ == '__main__':
    main()
