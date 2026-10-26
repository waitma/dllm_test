"""Activate pllm; python -B FILE --phase explore|confirm [--poll].

Submit frozen eval_jobs YAML and immediately record the exact Volc task ID.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

from submit_cdr_inference_round import ROOT, CLI, read_json_output, update_log

OUT = ROOT/'output/downstream_generation/cdr_block_context_20261015'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('explore', 'confirm'), required=True); p.add_argument('--poll', action='store_true')
    args = p.parse_args(); n = 1 if args.phase == 'explore' else 2
    config = ROOT/f'eval_jobs/cdr_block_{args.phase}_20261015.yml'
    record, platform = OUT/f'round{n}_submission.json', OUT/f'round{n}_platform.json'
    def log(task, status, event):
        update_log(n, task, status, event, run=OUT, config=config, label='CDR分组软上下文'+('开发探索' if n == 1 else '冻结OAS确认'))
    if args.poll:
        task = json.loads(record.read_text())['task_id']
    else:
        assert not record.exists(), 'Already submitted; poll existing task'
        assert json.loads((OUT/'gate_cpu_112000.json').read_text())['status'] == 'passed'
        if n == 2:
            assert json.loads((OUT/'selection.json').read_text())['eligible']
            assert json.loads((OUT/'explore_112000/independent_audit.json').read_text())['status'] == 'passed'
        result = subprocess.run([CLI, 'ml_task', 'submit', '--conf', str(config)], capture_output=True, text=True)
        (OUT/f'round{n}_submit.log').write_text(result.stdout+'\n'+result.stderr); result.check_returncode()
        ids = set(re.findall(r't-\d{14}-[a-z0-9]+', result.stdout)); assert len(ids) == 1
        task = ids.pop()
        record.write_text(json.dumps(dict(task_id=task, config=str(config), submitted_utc=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
        log(task, 'Initialized', 'submit')
    for attempt in range(5):
        query = subprocess.run([CLI, 'ml_task', 'get', '-i', task, '-o', 'json', '--format', 'JobId,JobName,Status,Start,End'], capture_output=True, text=True, check=True)
        try:
            state = read_json_output(query.stdout); break
        except ValueError:
            if attempt == 4: raise
            time.sleep(1)
    assert state['JobId'] == task
    previous = json.loads(platform.read_text()) if platform.exists() else {}
    state['observed_utc'] = datetime.now(timezone.utc).isoformat()
    platform.write_text(json.dumps(state, indent=2)+'\n')
    if previous.get('Status') != state['Status']:
        log(task, state['Status'], '终态' if state['Status'] in ('Success', 'Failed', 'Killed') else '初始状态确认' if not args.poll else None)
    print(json.dumps(state), flush=True)


if __name__ == '__main__': main()
