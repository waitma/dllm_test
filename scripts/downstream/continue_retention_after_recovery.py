"""Audit the recovered fixed6000 pilot and launch its interrupted task queues.

Activate pllm/protenix_abtcr and run from repository root:
python -B scripts/downstream/continue_retention_after_recovery.py --training-pid PID
Waits only on the specified recovery process; fails if it ends unsuccessfully.
Archives only old, empty queues whose original processes are absent. Each new
queue uses the existing evaluator and a detached process with persistent logs.
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', required=True, type=int)
    args = parser.parse_args()
    run = DIAG / 'final_layer_retention_pilot_20261018'
    audit = DIAG / 'final_layer_retention_audit_20261018'
    cross = DIAG / 'final_layer_retention_cross_task_20261018'
    state_path = DIAG / 'final_layer_retention_recovery_supervision_20261019.json'
    assert not state_path.exists()
    state = dict(status='running', stage='wait_recovery', pid=os.getpid(), training_pid=args.training_pid,
                 started_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__))

    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)

    save()
    try:
        while True:
            current = json.loads((run / 'status.json').read_text())
            if current['status'] == 'success':
                assert current['pid'] == args.training_pid
                break
            command_path = Path(f'/proc/{args.training_pid}/cmdline')
            if not command_path.exists() or b'resume_final_layer_retention_pilot.py' not in command_path.read_bytes():
                raise RuntimeError('Recovery process is absent without a successful terminal status; no automatic restart')
            time.sleep(10)
        recovery_file = run / 'recovery_audit_20261019/recovery.json'
        recovery = json.loads(recovery_file.read_text())
        assert recovery['status'] == 'success' and recovery['overlapping5520_bitwise_equal']
        assert recovery['control_unchanged'] and recovery['restored_optimizer_steps'] == [5200]
        assert sha(recovery['source']) == recovery['source_sha256']
        for key in ['archived_sha256', 'control_sha256']:
            for path, digest in recovery[key].items():
                assert sha(path) == digest, path
        assert sha(recovery['recovery_bundle']['bundle']) == recovery['recovery_bundle']['sha256']
        state.update(stage='endpoint_audit', recovery_sha256=sha(recovery_file))
        save()
        assert not audit.exists()
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/downstream/audit_final_retention_pilot.py'),
                        '--out', str(audit)], cwd=ROOT, check=True)
        state['queue_pids'] = {}
        for lane in ['tcr_cdr', 'ab']:
            old = cross / ('driver_' + lane)
            prior = json.loads((old / 'status.json').read_text())
            assert prior['completed'] == [] and prior['stage'] == 'waiting_for_endpoint_audit'
            assert not Path(f"/proc/{prior['pid']}/cmdline").exists(), 'Old queue PID still present'
            destination = cross / ('driver_' + lane + '_interrupted_20261019')
            assert not destination.exists()
            old.rename(destination)
            log_path = DIAG / ('final_layer_retention_cross_' + lane + '_20261019.log')
            with log_path.open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', '-B',
                    str(ROOT / 'scripts/downstream/run_final_retention_cross_tasks.py'),
                    '--lane', lane, '--audit-pid', str(os.getpid())], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['queue_pids'][lane] = child.pid
            save()
        state.update(status='success', stage='queues_launched')
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
