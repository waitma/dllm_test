"""Evaluate both feasible-target pilot arms after the fixed1000 training passes.

Activate pllm/protenix_abtcr; from repository root run two local lanes:
python -B scripts/downstream/run_feasible_final_cross_tasks.py --lane tcr_cdr --training-pid PID
python -B scripts/downstream/run_feasible_final_cross_tasks.py --lane ab --training-pid PID
Waits on a verified live training process; never restarts training/evaluations.
Both arms receive actual TCR(two attention modes), CDR, Specificity and m396.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / 'output/diagnostics'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', required=True, choices=['tcr_cdr', 'ab'])
    parser.add_argument('--training-pid', required=True, type=int)
    args = parser.parse_args()
    out = DIAG / 'feasible_final_cross_task_20261019'
    out.mkdir(exist_ok=True)
    lane = out / ('driver_' + args.lane)
    lane.mkdir(exist_ok=False)
    run = DIAG / 'feasible_final_pilot_20261019'
    audit = run
    state = dict(status='running', stage='waiting_for_fixed_endpoint_training', pid=os.getpid(),
                 training_pid=args.training_pid, started_utc=datetime.now(timezone.utc).isoformat(),
                 completed=[], source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def save():
        temporary = lane / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(lane / 'status.json')

    def invoke(script, argv, label):
        command = [sys.executable, '-B', str(ROOT / 'scripts/downstream' / script), *argv]
        with (lane / (label + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state.update(stage=label, child_pid=child.pid, command=command)
            save()
            print(label, 'PID', child.pid, flush=True)
            code = child.wait()
        state.pop('child_pid', None)
        if code:
            raise RuntimeError(f'{label} exited {code}; inspect {log.name}')
        state['completed'].append(label)
        save()

    save()
    try:
        while True:
            try:
                ready = json.loads((audit / 'status.json').read_text())['status'] == 'success'
            except (FileNotFoundError, json.JSONDecodeError):
                ready = False
            if ready:
                break
            try:
                command = Path(f'/proc/{args.training_pid}/cmdline').read_bytes()
            except FileNotFoundError:
                command = b''
            if b'train_feasible_final_pilot.py' not in command:
                raise RuntimeError('Original training process is no longer live and has not succeeded; no restart')
            time.sleep(10)
        assert json.loads((run / 'status.json').read_text())['status'] == 'success'
        plan = json.loads((run / 'plan.json').read_text())
        assert plan['steps'] == 1000 and plan['arms'] == ['pool_only', 'feasible_token']
        for path,digest in plan['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        for arm in plan['arms']:
            verified = json.loads((run / arm / 'verification.json').read_text())
            assert verified['status'] == 'passed' and len(verified['oof_checks']) == 10
            assert verified['reloaded_features_bitwise_equal'] and verified['final_norm_bitwise_original']
            checkpoint = json.loads((run / arm / 'checkpoint.json').read_text())
            assert checkpoint['steps'] == 1000
            assert hashlib.sha256(Path(checkpoint['parameter_file']).read_bytes()).hexdigest() == checkpoint['sha256']
        state['endpoint_plan_sha256'] = hashlib.sha256((run / 'plan.json').read_bytes()).hexdigest()
        jobs = [('t3broad', 'native'), ('t3broad', 'no_mask_keys'), ('cdr', 'native')] if args.lane == 'tcr_cdr' else [('specificity', 'no_mask_keys'), ('m396', 'no_mask_keys')]
        for task, mode in jobs:
            for arm in ['pool_only', 'feasible_token']:
                target = out / arm / (task + '_' + mode)
                target.parent.mkdir(exist_ok=True)
                assert not target.exists(), target
                label = arm + '_' + task + '_' + mode
                arguments = ['--checkpoint-dir', str(run / arm), '--out', str(target)]
                if task == 'cdr':
                    invoke('evaluate_final_preservation_cdr.py', arguments, label)
                    invoke('compare_final_cdr.py', ['--run', str(target)], label + '_verify')
                else:
                    invoke('evaluate_final_attention_probe.py', arguments + ['--task', task, '--attention-mode', mode], label)
                    if task in ('specificity', 'm396'):
                        invoke('verify_final_preservation_probe.py', ['--task', task, '--run', str(target)], label + '_rescore')
                    invoke('compare_final_cross_task.py', ['--task', task, '--run', str(target)], label + '_compare')
        state.update(status='success', stage='complete')
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
