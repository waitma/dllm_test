"""Run the fixed-chain50000 layer-two suite directly on the local GPU.

Activate pllm/protenix_abtcr and the prepared GPU environment, then run:
python -B scripts/downstream/run_local_fixed_layer_suite.py --run ABSOLUTE_RUN
The existing group runner performs GPU gates and retains task-level outputs.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / 'local_driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((run / 'plan.json').read_text())
        assert plan['execution_mode'] == 'local_single_gpu'
        if (run / 'submissions.json').exists():
            raise RuntimeError('This local-only run must not have remote submissions')
        for task in plan['tasks']:
            status = run / task / 'status.json'
            if status.exists() and json.loads(status.read_text())['status'] == 'running':
                raise RuntimeError('Another worker is marked running: ' + task)
        state = dict(status='running', pid=os.getpid(), host=socket.gethostname(),
                     device=os.environ.get('CUDA_VISIBLE_DEVICES'),
                     started_utc=datetime.now(timezone.utc).isoformat(),
                     group_order=[0, 4, 2, 1, 3, 5], results=[])
        try:
            for group in state['group_order']:
                if (run / 'STOP').exists():
                    raise RuntimeError('Stop requested')
                state.update(current_group=group, updated_utc=datetime.now(timezone.utc).isoformat())
                with (run / f'local_group_{group}.log').open('a') as log:
                    process = subprocess.Popen(
                        [sys.executable, '-B', str(run / 'run_suite.py'),
                         '--run', str(run), '--group', str(group)],
                        stdout=log, stderr=subprocess.STDOUT)
                    state['child_pid'] = process.pid
                    write(run / 'local_driver.json', state)
                    result = process.wait()
                state['results'].append(dict(group=group, returncode=result))
                write(run / 'local_driver.json', state)
            state['status'] = 'success' if all(r['returncode'] == 0 for r in state['results']) else 'failed'
        except BaseException as error:
            state.update(status='failed', error=repr(error))
            raise
        finally:
            state['finished_utc'] = datetime.now(timezone.utc).isoformat()
            write(run / 'local_driver.json', state)
        if state['status'] != 'success':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
