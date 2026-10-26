"""Activate pllm; python -B FILE [--poll] to submit/poll the new probability exploration.

Use the shared task-log writer, preserving exact IDs and avoiding resubmission
on eventually-consistent platform queries. No training or external messaging.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

from submit_cdr_inference_round import ROOT, CLI, read_json_output, update_log

OUT = ROOT/'output/downstream_generation/cdr_marginal_exploration_20261015'
CONFIG = ROOT/'eval_jobs/cdr_marginal_explore112000_20261015.yml'


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--poll', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--ensemble', action='store_true')
    mode.add_argument('--contrast', action='store_true')
    mode.add_argument('--soft', action='store_true')
    mode.add_argument('--confirm', action='store_true')
    args = parser.parse_args(); n = 5 if args.confirm else 4 if args.soft else 3 if args.contrast else 2 if args.ensemble else 1
    config = ROOT/'eval_jobs/cdr_new_inference_confirmation_20261015.yml' if args.confirm else ROOT/'eval_jobs/cdr_soft_context_probe_20261015.yml' if args.soft else ROOT/'eval_jobs/cdr_context_contrast_probe_20261015.yml' if args.contrast else ROOT/'eval_jobs/cdr_checkpoint_ensemble_probe_20261015.yml' if args.ensemble else CONFIG
    record = OUT/f'round{n}_submission.json'; platform = OUT/f'round{n}_platform.json'
    def log(task, status, event):
        update_log(n, task, status, event, run=OUT, config=config,
                   label='CDR新推理冻结独立确认' if args.confirm else 'CDR软embedding上下文探索' if args.soft else 'CDR上下文强度探索' if args.contrast else 'CDR checkpoint概率集成探索' if args.ensemble else 'CDR概率聚合新探索112000')
    if args.poll:
        task = json.loads(record.read_text())['task_id']
    else:
        assert not record.exists(), 'Already submitted; poll the recorded ID'
        assert json.loads((OUT/'gate_cpu.json').read_text())['status'] == 'passed'
        if args.confirm:
            assert (OUT/'confirmation_plan.json').is_file()
            assert json.loads((OUT/'soft_context/independent_audit.json').read_text())['status'] == 'passed'
        if args.ensemble:
            assert json.loads((OUT/'explore_112000/independent_audit.json').read_text())['status'] == 'passed'
        result = subprocess.run([CLI, 'ml_task', 'submit', '--conf', str(config)], capture_output=True, text=True)
        (OUT/f'round{n}_submit.log').write_text(result.stdout+'\n'+result.stderr); result.check_returncode()
        ids = set(re.findall(r't-\d{14}-[a-z0-9]+', result.stdout)); assert len(ids) == 1
        task = ids.pop()
        record.write_text(json.dumps(dict(task_id=task, config=str(config), submitted_utc=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
        log(task, '提交成功，平台状态待查询', 'submit')
    for attempt in range(5):
        result = subprocess.run([CLI, 'ml_task', 'get', '-i', task, '-o', 'json',
                                 '--format', 'JobId,JobName,Status,Start,End'], capture_output=True, text=True, check=True)
        try:
            state = read_json_output(result.stdout); break
        except ValueError:
            if attempt == 4: raise
            time.sleep(1)
    assert state['JobId'] == task
    previous = json.loads(platform.read_text()) if platform.exists() else {}
    state['observed_utc'] = datetime.now(timezone.utc).isoformat(); platform.write_text(json.dumps(state, indent=2)+'\n')
    if previous.get('Status') != state['Status']:
        event = '终态' if state['Status'] in ('Success', 'Failed', 'Killed') else '初始状态确认' if not args.poll else None
        log(task, state['Status'], event)
    print(json.dumps(state), flush=True)


if __name__ == '__main__': main()
