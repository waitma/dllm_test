"""Activate pllm; python -B FILE --checkpoint 112000|50000 [--poll].

Submit or query an exact recorded Volc job for the frozen left-to-right follow-up.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from submit_cdr_inference_round import ROOT, CLI, read_json_output, update_log

OUT = ROOT / 'output/downstream_generation/cdr_left_to_right_20261016'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=('112000', '50000'), required=True)
    parser.add_argument('--poll', action='store_true')
    args = parser.parse_args()
    cp = args.checkpoint
    n = 1 if cp == '112000' else 2
    config = ROOT / f'eval_jobs/cdr_left_to_right_{cp}_20261016.yml'
    record, platform = OUT / f'round{n}_submission.json', OUT / f'round{n}_platform.json'
    def log(task, status, event):
        update_log(n, task, status, event, run=OUT, config=config,
                   label='CDR从左到右解码' + cp, event_date='2026-10-16')
    if args.poll:
        task = json.loads(record.read_text())['task_id']
    else:
        assert not record.exists(), 'Already submitted; poll the recorded ID'
        gate = json.loads((OUT / 'gate_cpu_112000.json').read_text())
        assert gate['status'] == 'passed'
        assert gate['plan_sha256'] == hashlib.sha256((OUT / 'plan.json').read_bytes()).hexdigest()
        result = subprocess.run([CLI, 'ml_task', 'submit', '--conf', str(config)], capture_output=True, text=True)
        (OUT / f'round{n}_submit.log').write_text(result.stdout + '\n' + result.stderr)
        result.check_returncode()
        ids = set(re.findall(r't-\d{14}-[a-z0-9]+', result.stdout))
        assert len(ids) == 1
        task = ids.pop()
        record.write_text(json.dumps(dict(task_id=task, checkpoint=cp, config=str(config),
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), plan_sha256=gate['plan_sha256'],
            submitted_utc=datetime.now(timezone.utc).isoformat()), indent=2) + '\n')
        log(task, 'Initialized', 'submit')
    for attempt in range(5):
        result = subprocess.run([CLI, 'ml_task', 'get', '-i', task, '-o', 'json',
            '--format', 'JobId,JobName,Status,Start,End'], capture_output=True, text=True, check=True)
        try:
            state = read_json_output(result.stdout)
            break
        except ValueError:
            if attempt == 4:
                raise
            time.sleep(1)
    assert state['JobId'] == task
    previous = json.loads(platform.read_text()) if platform.exists() else {}
    state['observed_utc'] = datetime.now(timezone.utc).isoformat()
    platform.write_text(json.dumps(state, indent=2) + '\n')
    if previous.get('Status') != state['Status']:
        log(task, state['Status'], '终态' if state['Status'] in ('Success', 'Failed', 'Killed') else '初始状态确认' if not args.poll else None)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
